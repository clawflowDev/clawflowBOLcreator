import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image

app = FastAPI(title="CLAWFLOW Dynamic Geometric Engine", version="35.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Multi-Level Dynamic Engine Active",
        "mode": "Otsu + Convex Hull + MinAreaRect Fallback"
    }, 200

def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos en formato: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transformación de perspectiva matemática basada en los 4 puntos detectados"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    maxWidth = max(maxWidth, 900)
    maxHeight = max(maxHeight, 1200)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_corners_multilevel(img: np.ndarray) -> np.ndarray:
    """
    Estrategia Multinivel de Visión Computacional:
    Nivel 1: Otsu + Canny + Polígono de 4 lados (Ideal págs. 1 y 2).
    Nivel 2: Cierre Morfológico + Convex Hull (Ideal pág. 3 / recibos arrugados).
    Nivel 3: Caja Orientada de Menor Área (MinAreaRect - Respaldo dinámico infalible).
    """
    h, w = img.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # -------------------------------------------------------------
    # NIVEL 1: Canny con Umbralización Adaptativa Otsu
    # -------------------------------------------------------------
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.5
    edged = cv2.Canny(blurred, low_thresh, high_thresh)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for c in contours:
        area = cv2.contourArea(c)
        if area > (total_area * 0.20):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                print("-> [VISION LOG] Nivel 1 Exitoso: Cuadrilátero convexo directo detectado.")
                return approx.reshape(4, 2).astype("float32")

    # -------------------------------------------------------------
    # NIVEL 2: Cierre Morfológico + Envolvente Convexa (Convex Hull)
    # -------------------------------------------------------------
    print("-> [VISION LOG] Ejecutando Nivel 2: Cierre morfológico y Convex Hull para bordes difíciles...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours_m, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_m = sorted(contours_m, key=cv2.contourArea, reverse=True)[:10]

    for c in contours_m:
        area = cv2.contourArea(c)
        if area > (total_area * 0.18):
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
            if len(approx) == 4:
                print("-> [VISION LOG] Nivel 2 Exitoso: Envolvente convexa ajustada a 4 esquinas.")
                return approx.reshape(4, 2).astype("float32")

    # -------------------------------------------------------------
    # NIVEL 3: Caja Orientada de Menor Área (MinAreaRect)
    # -------------------------------------------------------------
    print("-> [VISION LOG] Ejecutando Nivel 3: Caja delimitadora orientada de precisión...")
    for c in contours_m:
        area = cv2.contourArea(c)
        if area > (total_area * 0.15):
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            print("-> [VISION LOG] Nivel 3 Exitoso: Esquinas calculadas vía MinAreaRect.")
            return box.astype("float32")

    # Respaldo Geométrico Proporcional Limpio (Si la foto carece por completo de contraste)
    print("-> [VISION LOG WARNING] Sin contraste suficiente. Aplicando margen de seguridad del 2.5%.")
    mx, my = int(w * 0.025), int(h * 0.025)
    return np.array([
        [mx, my],
        [w - mx, my],
        [w - mx, h - my],
        [mx, h - my]
    ], dtype="float32")

def process_document_pipeline(image_bytes: bytes) -> Image.Image:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen binaria.")

        original_h, original_w = img.shape[:2]

        # Tratamiento de imágenes horizontales (fotos de remolques/carga)
        if original_w > original_h:
            print("-> [LOG] Imagen horizontal. Preservando relación de aspecto...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # 1. Extracción de las 4 esquinas mediante el motor multinivel
        corner_points = find_corners_multilevel(img)

        # 2. Transformación de perspectiva
        warped = four_point_transform(img, corner_points)

        # 3. Estandarización de tamaño (Ancho fijo de 1600px para legibilidad uniforme)
        target_width = 1600
        ch, cw = warped.shape[:2]
        if cw != target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            warped = cv2.resize(warped, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # 4. Filtro bilateral anti-ruido + Realce de legibilidad de tinta (CLAHE)
        smoothed = cv2.bilateralFilter(warped, 9, 75, 75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # 5. Enmarcado en Lienzo Blanco Corporativo A4 (30px de margen estético)
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        return canvas

    except Exception as e:
        print(f"🚨 [LOG ERROR]: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        print(f"-> [LOG] Procesando lote de {len(payload.image_urls)} imágenes...")
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes.")

        pil_images = []
        for index, img_uri in enumerate(payload.image_urls):
            try:
                if img_uri.startswith("data:"):
                    header, encoded = img_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    response = requests.get(img_uri, timeout=15)
                    response.raise_for_status()
                    image_bytes = response.content

                processed_page = process_document_pipeline(image_bytes)
                pil_images.append(processed_page)
            except Exception as img_err:
                print(f"🚨 [LOG ERROR] Imagen {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Error en procesamiento del lote.")

        pdf_output = io.BytesIO()
        first_image = pil_images[0]
        remaining_images = pil_images[1:] if len(pil_images) > 1 else []

        first_image.save(
            pdf_output,
            format="PDF",
            save_all=True,
            append_images=remaining_images,
            quality=92,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF procesado exitosamente mediante motor multinivel OpenCV."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
