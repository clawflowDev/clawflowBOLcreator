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

# Configuración automática de Cloudinary desde las variables de entorno de Render
# Asegúrate de tener configurado CLOUDINARY_URL en tus Environment Variables de Render
cloudinary.config(secure=True)

app = FastAPI(title="CLAWFLOW Hybrid Archive Engine", version="8.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Hybrid Engine (Cloudinary Crop + OpenCV Enhancer) Active"}

def cloudinary_smart_crop_and_upload(image_bytes: bytes) -> bytes:
    """
    Envía la imagen a Cloudinary para que su IA haga el recorte inteligente del documento
    y elimine los fondos (asientos, mesas), devolviendo la imagen limpia.
    """
    try:
        print("-> Enviando imagen a Cloudinary para recorte inteligente...")
        
        # Subir la imagen a Cloudinary aplicando transformaciones de recorte automático y detección de bordes
        # 'e_trim' o 'crop: limit' con optimización de documento
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            transformation=[
                {"effect": "trim", "f_auto": "true"}, # Recorte automático de bordes por IA de Cloudinary
                {"quality": "auto"}
            ]
        )
        
        secure_url = upload_result.get("secure_url")
        print(f"-> Cloudinary recortó con éxito. URL optimizada: {secure_url}")
        
        # Descargar la imagen ya recortada por Cloudinary
        resp = requests.get(secure_url, timeout=15)
        resp.raise_for_status()
        return resp.content

    except Exception as e:
        print(f"⚠️ Aviso: Cloudinary falló en el recorte ({str(e)}). Usando bytes originales de respaldo.")
        return image_bytes

def process_document_hybrid(image_bytes: bytes) -> Image.Image:
    """
    Fase 2 del Híbrido: Toma la imagen ya recortada por Cloudinary y le aplica
    el realce de contraste local perfecto con OpenCV.
    """
    try:
        print("-> Aplicando realce de contraste local OpenCV sobre imagen recortada...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 1. Redimensionar estándar a 1600px de ancho
        target_width = 1600
        h, w = img.shape[:2]
        if w > target_width:
            ratio = target_width / float(w)
            new_h = int(h * ratio)
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # 2. Escala de grises y CLAHE (el contraste nítido que te gusta)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Reducción de ruido sutil
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Proceso híbrido completado con éxito!")
        return processed_image

    except Exception as e:
        print(f"❌ Error en procesamiento híbrido: {str(e)}")
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

                # REGLA: Carga horizontal a color optimizado; Documentos verticales pasan por el motor Híbrido (Cloudinary + OpenCV)
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Iniciando pipeline Híbrido.")
                    # Paso 1: Recorte inteligente en Cloudinary
                    cropped_bytes = cloudinary_smart_crop_and_upload(image_bytes)
                    # Paso 2: Realce estético local con OpenCV
                    clean_doc = process_document_hybrid(cropped_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"Advertencia procesando imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        # Compilar en PDF unificado de alta calidad y peso optimizado
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
            "message": "PDF unificado con motor Híbrido (Cloudinary + OpenCV) generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
