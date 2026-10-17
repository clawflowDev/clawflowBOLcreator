import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image

app = FastAPI(title="CLAWFLOW Pro-Deskew Engine", version="43.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Balanced + Deskew & Fill-to-Fit Engine",
    }, 200

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detecta el ángulo de inclinación del documento recortado y lo endereza
    para que quede perfectamente vertical.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invertir para que el fondo blanco sea negro y el texto/bordes blancos
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Encontrar coordenadas de los píxeles no cero (el documento)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image # Si no hay suficientes puntos, retorna original
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Ajustar el ángulo de rotación estándar de OpenCV
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Solo enderezar si la inclinación es perceptible (> 0.5 grados y < 45 grados)
    if abs(angle) > 0.5 and abs(angle) < 45:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return corrected
        
    return image

def process_pro_deskew_document(image_bytes: bytes) -> Image.Image:
    """
    Aplica enderezado (deskew), mejora de grises equilibrada (CLAHE suave) 
    y montaje Fill-to-Fit sobre A4.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    # Si es horizontal (foto de carga), no se endereza ni binariza texto
    if img.shape[1] > img.shape[0]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(rgb)
    else:
        # 1. Enderezar la página automáticamente (Deskew)
        straight_img = deskew_image(img)

        # 2. Filtro bilateral moderado para limpiar ruido
        smoothed = cv2.bilateralFilter(straight_img, 9, 50, 50)

        # 3. Contraste equilibrado (CLAHE suave)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(rgb)

    # 4. Fill-to-Fit A4 Proporcional
    target_w, target_h = 1650, 2339 
    dw, dh = doc_pil.size
    ratio = min((target_w - 60) / dw, (target_h - 60) / dh)
    new_w = int(dw * ratio)
    new_h = int(dh * ratio)
    
    doc_pil = doc_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2
    
    canvas.paste(doc_pil, (pos_x, pos_y))
    return canvas

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="Sin imágenes en payload.")

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

                processed_page = process_pro_deskew_document(image_bytes)
                pil_images.append(processed_page)
            except Exception as img_err:
                print(f"🚨 [ERROR IMG {index+1}]: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Sin páginas procesadas.")

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
            "message": "PDF optimizado con Enderezado geométrico (Deskew) y Fill-to-Fit A4."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
