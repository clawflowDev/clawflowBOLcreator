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

app = FastAPI(title="CLAWFLOW BOL/POD Natural Balance Engine", version="4.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Natural Balance Engine Active"}

def resize_image_if_needed(np_img: np.ndarray, max_width: int = 1600) -> np.ndarray:
    """
    Redimensiona la imagen proporcionalmente si supera el ancho máximo para aligerar
    el procesamiento y garantizar velocidad en el servidor.
    """
    h, w = np_img.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        np_img = cv2.resize(np_img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return np_img

def process_document_natural_balance(image_bytes: bytes) -> Image.Image:
    """
    Motor de escáner de balance natural (CLAHE + Escala de Grises Optimizada):
    Blanquea el fondo y oscurece el texto sutilmente sin destruir los trazos finos ni las firmas.
    """
    try:
        print("-> Procesando documento con balance natural y CLAHE...")
        
        # 1. Decodificar bytes a matriz OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 2. Redimensionar para optimizar peso
        img = resize_image_if_needed(img, max_width=1600)

        # 3. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Esto realza los contrastes locales de forma inteligente: aclara el fondo gris y oscurece las letras limpiamente.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. Suavizado muy sutil para eliminar ruido sin perder nitidez en las letras
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Convertir a PIL RGB
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Éxito! Documento procesado con balance natural.")
        return processed_image

    except Exception as e:
        print(f"❌ Error en el procesamiento natural: {str(e)}")
        # Fallback de seguridad local
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

                # Análisis rápido de dimensiones para distinguir Carga vs Documento
                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                h, w, _ = np_img.shape

                # REGLA: Carga horizontal a color (optimizada de peso), Documentos verticales con balance natural
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    np_img_resized = resize_image_if_needed(np_img, max_width=1600)
                    color_image = Image.fromarray(cv2.cvtColor(np_img_resized, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando balance natural.")
                    clean_doc = process_document_natural_balance(image_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"Advertencia procesando imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        # Compilar en PDF unificado de alta calidad
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
            "message": "PDF unificado con motor de balance natural generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
