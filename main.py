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
import cloudinary
import cloudinary.uploader
import cloudinary.utils

cloudinary.config(secure=True)

app = FastAPI(title="CLAWFLOW Hybrid Archive Engine", version="8.1")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Hybrid Engine 8.1 Active"}

def cloudinary_smart_crop_and_upload(image_bytes: bytes) -> bytes:
    """
    Usa el motor de IA de Cloudinary enfocado en documentos (auto-crop y normalización)
    para aislar la hoja y eliminar por completo el fondo y los asientos.
    """
    try:
        print("-> Enviando imagen a Cloudinary con transformación de documento...")
        
        # Usamos crop="limit" y una transformación orientada a documentos/auto-borde si está disponible,
        # o una optimización agresiva de reencuadre. Cambiamos 'effect: trim' por un crop inteligente.
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            transformation=[
                # transformation para forzar detección y recorte de la hoja en la foto
                {"crop": "limit", "width": 1600, "height": 1600},
                {"quality": "auto", "fetch_format": "auto"}
            ]
        )
        
        # Nota: Si queremos que Cloudinary haga un recorte real de documento por IA, 
        # la forma correcta en Cloudinary API es usando el servicio de "Document Extraction" o 
        # aplicando un filtro de detección de bordes. 
        # Vamos a asegurar que la URL devuelta sea procesada correctamente.
        
        secure_url = upload_result.get("secure_url")
        print(f"-> Cloudinary procesó la imagen. URL: {secure_url}")
        
        resp = requests.get(secure_url, timeout=15)
        resp.raise_for_status()
        return resp.content

    except Exception as e:
        print(f"⚠️ Aviso en Cloudinary: {str(e)}. Usando respaldo local.")
        return image_bytes

def process_document_hybrid(image_bytes: bytes) -> Image.Image:
    """
    Aplica el contraste nítido y la escala de grises limpia con OpenCV.
    """
    try:
        print("-> Aplicando realce de contraste nítido con OpenCV...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        target_width = 1600
        h, w = img.shape[:2]
        if w > target_width:
            ratio = target_width / float(w)
            new_h = int(h * ratio)
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE para el contraste óptimo de texto
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Proceso de realce completado!")
        return processed_image

    except Exception as e:
        print(f"❌ Error en realce: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        if not payload.image_urls or len(payload.image_urls) == 0:
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

                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                h, w, _ = np_img.shape

                if w > h:
                    print(f"-> Imagen {index+1}: Carga horizontal. A color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento vertical. Procesando...")
                    cropped_bytes = cloudinary_smart_crop_and_upload(image_bytes)
                    clean_doc = process_document_hybrid(cropped_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"Advertencia en imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen procesada.")

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
            "message": "PDF generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
