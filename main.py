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
import cloudinary
import cloudinary.uploader
import os

app = FastAPI(title="CLAWFLOW BOL/POD Cloudinary Engine", version="2.1")

# Configurar Cloudinary usando las variables de entorno de Render
cloudinary.config(
  cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
  api_key = os.environ.get("CLOUDINARY_API_KEY"),
  api_secret = os.environ.get("CLOUDINARY_API_SECRET")
)

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Cloudinary Microservice Active"}

def process_document_with_cloudinary(image_bytes: bytes) -> Image.Image:
    """
    Sube la imagen a Cloudinary y aplica transformaciones de nivel 'CamScanner'
    (blanqueo total de fondo, eliminación de sombras y realce de tinta negra).
    """
    try:
        print("-> Subiendo documento a Cloudinary para efecto escáner...")
        # Subir los bytes de la imagen aplicando el perfil de documento de alta definición
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            transformation=[
                {"effect": "blackwhite:threshold:65"},     # Binarización agresiva: limpia el fondo gris a blanco puro y oscurece la tinta
                {"effect": "sharpen:150"},                # Nitidez extrema para que las letras y firmas no pierdan definición
                {"fetch_format": "jpg"}
            ]
        )
        
        processed_url = upload_result.get("secure_url")
        print(f"-> ¡Éxito en Cloudinary! URL de escáner: {processed_url}")

        # Descargar la imagen ya procesada
        resp = requests.get(processed_url, timeout=20)
        resp.raise_for_status()
        
        processed_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return processed_image

    except Exception as e:
        print(f"❌ ERROR CRÍTICO procesando con Cloudinary: {str(e)}")
        # Fallback de seguridad local si llegara a fallar la conexión
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray).convert("RGB")

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

                # Análisis de orientación (ancho vs alto) para diferenciar Carga vs Documento
                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                h, w, _ = np_img.shape

                # REGLA: Carga horizontal a color, Documentos verticales procesados con Cloudinary
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - Ancho {w} > Alto {h}). A todo color.")
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - Alto {h} >= Ancho {w}). Enviando a Cloudinary.")
                    clean_doc = process_document_with_cloudinary(image_bytes)
                    pil_images.append(clean_doc)
                
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
            "message": "PDF unificado con motor de escáner Cloudinary generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
