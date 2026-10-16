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

app = FastAPI(title="CLAWFLOW Autonomous Local Geometric Engine", version="33.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Autonomous Geometric Engine Active",
        "mode": "Local Contour Intelligence + Perspective Warp + CLAHE"
    }, 200

def order_points(pts):
    """Ordena dinámicamente 4 puntos: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Ejecuta el recorte de perspectiva milimétrico basado en coordenadas geométricas"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

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

def get_autonomous_document_corners(img: np.ndarray) -> np.ndarray:
    """
    Motor local autónomo: Detecta las 4 esquinas reales del documento analizando 
    la geometría y los contornos de la imagen de forma inteligente y robusta.
    """
    original_h, original_w = img.shape[:2]
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Filtro bilateral para suavizar ruidos sin perder definición de bordes
        filtered = cv2.bilateralFilter(gray, 11, 75, 75)
        
        # Detección de bordes adaptativa optimizada para documentos de transporte (BOL/Recibos)
        edged = cv2.Canny(filtered, 50, 150)
        
        # Operación morfológica para cerrar pequeñas brechas en los bordes del papel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(c)
            
            # Validar que el contorno sea un cuadrilátero válido que ocupe entre el 15% y el 98% de la foto
            if len(approx) == 4 and (original_h * original_w * 0.15) < area < (original_h * original_w * 0.98):
                if cv2.isContourConvex(approx):
                    screenCnt = approx
                    break

        if screenCnt is not None:
            print("-> [LOG AUTO] ¡Cuadrilátero del documento detectado localmente con éxito!")
            return screenCnt.reshape(4, 2).astype("float32")
        
        raise ValueError("Contorno geométrico estricto no cerrado. Activando calibrador de respaldo.")

    except Exception as err:
        print(f"-> [LOG AUTO FALLBACK] {str(err)} - Aplicando márgenes calibrados de precisión.")
        
        # Márgenes milimétricos calibrados de respaldo seguro
        margin_x = int(original_w * 0.035)
        margin_y = int(original_h * 0.035)
        
        calibrated_pts = np.array([
            [margin_x, margin_y],                        # Top-Left
            [original_w - margin_x, margin_y],           # Top-Right
            [original_w - margin_x, original_h - margin_y],# Bottom-Right
            [margin_x, original_h - margin_y]            # Bottom-Left
        ], dtype="float32")
        
        return calibrated_pts

def process_document_pipeline(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen binaria.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales: {original_w}x{original_h} px")

        # Validación para imágenes horizontales (fotos de carga/remolque) -> Se preservan intactas a color
        if original_w > original_h:
            print("-> [LOG 4.2] Imagen horizontal detectada. Preservando a color sin recortes...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # -------------------------------------------------------------
        # FASE 1: OBTENCIÓN AUTÓNOMA DE COORDENADAS
        # -------------------------------------------------------------
        print("-> [LOG 4.3] Analizando y extrayendo esquinas del documento autónomamente...")
        corner_points = get_autonomous_document_corners(img)

        # -------------------------------------------------------------
        # FASE 2: EJECUCIÓN DE PERSPECTIVA MILIMÉTRICA
        # -------------------------------------------------------------
        print("-> [LOG 4.4] Ejecutando transformación geométrica de perspectiva...")
        warped_doc = four_point_transform(img, corner_points)

        # -------------------------------------------------------------
        # FASE 3: ESTANDARIZACIÓN Y REALCE DE TINTA (OCR-READY CORPORATIVO)
        # -------------------------------------------------------------
        print("-> [LOG 4.5] Estandarizando y aplicando motor CLAHE de alta legibilidad...")
        target_width = 1600
        ch, cw = warped_doc.shape[:2]
        if cw != target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            warped_doc = cv2.resize(warped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # Filtro bilateral para limpieza de fondo y realce de luminancia LAB + CLAHE
        smoothed = cv2.bilateralFilter(warped_doc, 9, 75, 75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Lienzo blanco corporativo unificado con margen estético de 30px
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.6] Página procesada, recortada con precisión y enmarcada impecablemente.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.7 ERROR EN PIPELINE AUTÓNOMO]: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        print(f"-> [LOG 5.0] Petición /process-bol recibida con {len(payload.image_urls)} imágenes.")
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes en el payload.")

        pil_images = []
        for index, img_uri in enumerate(payload.image_urls):
            try:
                print(f"-> [LOG 5.1] Procesando imagen {index+1} de {len(payload.image_urls)}...")
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
                print(f"🚨 [LOG 5.5] Error aislado en imagen {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada en el lote.")

        print("-> [LOG 6.0] Compilando PDF final unificado en alta resolución...")
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

        print("-> [LOG 6.1] ¡PDF unificado generado exitosamente con motor autónomo local!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado exitosamente con motor geométrico autónomo local."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO EN ENDPOINT]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
