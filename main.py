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
import os

app = FastAPI(title="CLAWFLOW BOL/POD Pro Cropper Engine", version="5.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Pro Cropper Engine Active"}

def resize_image_if_needed(np_img: np.ndarray, max_width: int = 1600) -> np.ndarray:
    h, w = np_img.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        np_img = cv2.resize(np_img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return np_img

def auto_crop_document(np_img: np.ndarray) -> np.ndarray:
    """
    Detecta los bordes externos de la hoja de papel en la foto para recortar
    el fondo (asientos, mesa) y dejar únicamente el documento limpio.
    """
    try:
        # Copia para trabajar
        orig = np_img.copy()
        h, w = orig.shape[:2]

        # 1. Escala de grises y desenfoque para mitigar ruido del fondo
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Detección de bordes con Canny
        edged = cv2.Canny(blurred, 75, 200)

        # 3. Encontrar contornos
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        # Buscar un contorno de 4 esquinas que ocupe una buena porción de la imagen
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Si el contorno tiene 4 puntos y el área es significativa (al 15% de la imagen)
            if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.15):
                screenCnt = approx
                break

        # Si encontró la hoja con éxito, procedemos a aislarla
        if screenCnt is not None:
            # Extraer coordenadas y ordenar puntos para perspectiva (opcional: devolver caja contenedora)
            x, y, w_box, h_box = cv2.boundingRect(screenCnt)
            # Agregar un pequeño margen de seguridad (padding de 15 píxeles) para no cortar texto al borde
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + w_box + pad)
            y2 = min(h, y + h_box + pad)
            
            cropped = orig[y1:y2, x1:x2]
            if cropped.shape[0] > 100 and cropped.shape[1] > 100:
                return cropped

    except Exception as e:
        print(f"⚠️ Aviso en auto-crop (usando imagen completa de respaldo): {str(e)}")

    return np_img

def process_document_pro_scanner(image_bytes: bytes) -> Image.Image:
    """
    Motor Pro: Recorta el fondo no deseado y aplica un contraste óptimo y elegante.
    """
    try:
        print("-> Procesando documento con motor Pro Scanner (Crop + Contraste ajustado)...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 1. Redimensionar primero para agilizar el cálculo de bordes
        img = resize_image_if_needed(img, max_width=1600)

        # 2. Intentar recortar los bordes de la hoja (eliminar asientos/mesa)
        cropped_img = auto_crop_document(img)

        # 3. Escala de grises
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # 4. CLAHE con un toque más de fuerza (clipLimit=3.0) para el contraste adicional que pediste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. Reducción de ruido sutil
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Éxito! Documento recortado y contrastado con éxito.")
        return processed_image

    except Exception as e:
        print(f"❌ Error en procesamiento Pro: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray).convert("RGB")

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        if not payload.image_urls or len(payload.image_urls) == 0:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes en el lote.")

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

                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                h, w, _ = np_img.shape

                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    np_img_resized = resize_image_if_needed(np_img, max_width=1600)
                    color_image = Image.fromarray(cv2.cvtColor(np_img_resized, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando recorte y contraste Pro.")
                    clean_doc = process_document_pro_scanner(image_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"Advertencia procesando imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

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

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF unificado con motor Pro Cropper generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
