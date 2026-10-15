import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

app = FastAPI(title="CLAWFLOW BOL/POD Advanced Processor", version="1.0")

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW BOL Batch Microservice with Auto-Crop"}

def auto_crop_and_enhance(np_image):
    """
    Detecta los bordes del papel en la imagen, recorta el fondo y aplica efecto escáner.
    """
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar desenfoque para reducir ruido
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detección de bordes (Canny)
        edged = cv2.Canny(blur, 75, 200)
        
        # Encontrar contornos en la imagen de bordes
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in contours:
            # Aproximar el contorno
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Si el contorno aproximado tiene 4 puntos, asumimos que es la hoja del documento
            if len(approx) == 4:
                screen_cnt = approx
                break
                
        # Si encuentra un contorno rectangular válido de hoja, hace el recorte y perspectiva
        if screen_cnt is not None:
            # Reorganizar puntos para la perspectiva
            pts = screen_cnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
                
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(np_image, M, (maxWidth, maxHeight))
            
            # Si el recorte fue exitoso y tiene dimensiones lógicas, lo usamos
            if warped.shape[0] > 100 and warped.shape[1] > 100:
                np_image = warped
        
        # --- EFECTO ESCÁNER PROFESIONAL ---
        # Convertir a PIL para aplicar mejoras de contraste y nitidez fina
        image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
        image = image.convert("L")
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.9) # Alto contraste para blanquear fondo
        image = image.filter(ImageFilter.SHARPEN)
        
        image = image.convert("RGB")
        return image
        
    except Exception as e:
        print(f"Aviso en auto-crop, usando imagen original mejorada: {str(e)}")
        # Fallback de seguridad si el contorno no es claro: procesa la imagen completa
        image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)).convert("L")
        image = ImageEnhance.Contrast(image).enhance(1.8)
        return image.convert("RGB")

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

                # Cargar imagen con OpenCV en memoria
                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                # Procesar: Recorte automático, enderezamiento y efecto escáner
                processed_pil_image = auto_crop_and_enhance(np_img)
                pil_images.append(processed_pil_image)
                
            except Exception as img_err:
                print(f"Advertencia procesando imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        # Compilar en PDF unificado
        pdf_output = io.BytesIO()
        first_image = pil_images[0]
        remaining_images = pil_images[1:] if len(pil_images) > 1 else []

        first_image.save(
            pdf_output,
            format="PDF",
            save_all=True,
            append_images=remaining_images
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF con recorte inteligente y efecto escáner generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
