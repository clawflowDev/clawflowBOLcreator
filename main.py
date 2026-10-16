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

app = FastAPI(title="CLAWFLOW Universal Isolation Engine", version="38.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Universal 5-Layer Engine Active",
        "mode": "Sheet Isolation + Convex Hull + MinAreaRect + CLAHE Dynamic"
    }, 200

def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos en sentido horario: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica la transformación de perspectiva matemática sin deformar proporciones"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Limites mínimos para preservar resolución A4 estandarizada
    maxWidth = max(maxWidth, 1000)
    maxHeight = max(maxHeight, 1400)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def extract_universal_paper_corners(img: np.ndarray) -> np.ndarray:
    """
    IMPLEMENTACIÓN DE CAPAS 1 A 4:
    Aísla la masa luminosa de la hoja completa ignorando tablas internas,
    hojas apiladas debajo ("milhojas") y sombras parciales.
    """
    h, w = img.shape[:2]
    total_area = h * w

    # CAPA 1: Escala de Grises + Blur Intenso para unificar reflectancia
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)

    # Binarización de Otsu para separar Papel (Brillante) vs Fondo (Oscuro)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir si el fondo resultara blanco por algún reflejo extraño
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # CAPA 2: Cierre Morfológico con Kernel Grande (25x25)
    # Rellena cajas, textos, firmas y pega las hojas de atrás como una sola masa
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Obtener el contorno externo más grande
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        # Si la masa ocupa al menos el 20% de la foto
        if area > (total_area * 0.20):
            # CAPA 3: Convex Hull (Elimina ondulaciones y esquinas arrugadas)
            hull = cv2.convexHull(c)

            # Intentar polígono directo de 4 esquinas
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                print("-> [ENGINE LOG] Capa 3: 4 Esquinas directas extraídas de la masa de papel.")
                return approx.reshape(4, 2).astype("float32")

            # CAPA 4: Si hay más de 4 puntos por arrugas, extrae el Bounding Box Rotado Mínimo
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            print("-> [ENGINE LOG] Capa 4: 4 Esquinas calculadas vía MinAreaRect envolvente.")
            return box.astype("float32")

    # Resguardo proporcional si no existe contraste (ej. hoja blanca sobre mesa blanca)
    print("-> [ENGINE WARNING] Sin contraste suficiente con el fondo. Aplicando resguardo perimetral 2.5%.")
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

        # 🛡️ PROTECCIÓN ANTI-OOM (502): Reducción preventiva a max 1600px
        max_dimension = 1600
        if original_w > max_dimension or original_h > max_dimension:
            scaling_factor = max_dimension / float(max(original_w, original_h))
            new_w = int(original_w * scaling_factor)
            new_h = int(original_h * scaling_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            original_h, original_w = img.shape[:2]

        # Tratamiento de imágenes horizontales (Fotos de remolques / carga)
        if original_w > original_h:
            print("-> [ENGINE LOG] Imagen horizontal detectada. Preservando original...")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # 1 a 4. Extracción de las 4 esquinas de la masa completa
        corner_points = extract_universal_paper_corners(img)

        # Transformación de perspectiva
        warped = four_point_transform(img, corner_points)

        # Estandarización de tamaño (Ancho objetivo 1500px para nitidez de impresión)
        target_width = 1500
        ch, cw = warped.shape[:2]
        if cw != target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            warped = cv2.resize(warped, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # CAPA 5: Restauración de Tinta y Remoción de Sombras
        # 5.1 Filtro Bilateral: Mantiene bordes de tinta afilados mientras suaviza la textura del papel
        smoothed = cv2.bilateralFilter(warped, 9, 75, 75)

        # 5.2 Normalización de Iluminación Local (CLAHE en espacio LAB)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Lienzo A4 Corporativo con margen de resguardo de 30px
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        return canvas

    except Exception as e:
        print(f"🚨 [ENGINE ERROR]: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        print(f"-> [LOG 1.0] Petición /process-bol recibida. Procesando {len(payload.image_urls)} imágenes.")
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes en el payload.")

        pil_images = []
        for index, img_uri in enumerate(payload.image_urls):
            try:
                print(f"-> [LOG 1.1] Procesando imagen {index+1} de {len(payload.image_urls)}...")
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
                print(f"🚨 [LOG ERROR] Error procesando imagen {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        print("-> [LOG 2.0] Compilando PDF final unificado...")
        pdf_output = io.BytesIO()
        first_image = pil_images[0]
        remaining_images = pil_images[1:] if len(pil_images) > 1 else []

        first_image.save(
            pdf_output,
            format="PDF",
            save_all=True,
            append_images=remaining_images,
            quality=90,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        print("-> [LOG 2.1] ¡PDF unificado generado exitosamente con el motor de 5 capas!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado exitosamente usando Universal Isolation Engine v38.0."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
