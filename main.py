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

app = FastAPI(title="CLAWFLOW Pro-Scan Engine", version="41.0")

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
        "service": "CLAWFLOW Strict Fill-to-Fit & High Contrast Tinta Engine",
    }, 200

def process_pro_document(image_bytes: bytes) -> Image.Image:
    """
    Aplica binarización de alto contraste para eliminar sombras y realiza 
    un Fill-to-Fit estricto al máximo espacio de la hoja A4.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    # Si la imagen es horizontal (ej. la foto del camión/remolque), la dejamos pasar limpia sin binarizar texto
    if img.shape[1] > img.shape[0]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(rgb)
    else:
        # 1. Escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Reducción de ruido avanzada manteniendo bordes de texto
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 3. Umbral Adaptativo para eliminar sombras del asiento y blanquear el fondo
        # Esto convierte las sombras oscuras a blanco y deja la tinta ultra nítida
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        
        # Convertir a formato RGB de PIL
        rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        doc_pil = Image.fromarray(rgb)

    # Dimensiones exactas proporcionales a una hoja A4 a alta resolución (300 DPI aprox: 2480 x 3508 o versión estándar 1650 x 2339)
    target_w, target_h = 1650, 2339 
    
    # Calcular escala de "Fill-to-Fit" maximizada
    dw, dh = doc_pil.size
    ratio = min((target_w - 40) / dw, (target_h - 40) / dh)
    new_w = int(dw * ratio)
    new_h = int(dh * ratio)
    
    # Redimensionar con la máxima calidad de interpolación
    doc_pil = doc_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Crear lienzo A4 100% blanco puro
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    
    # Centrar perfectamente el documento escalado
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

                processed_page = process_pro_document(image_bytes)
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
            quality=95,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF optimizado con Fill-to-Fit y binarización de alta nitidez."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
