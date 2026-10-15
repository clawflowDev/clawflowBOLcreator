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

app = FastAPI(title="CLAWFLOW BOL/POD Pro Scanner Engine", version="3.5")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Pro Scanner Engine Active"}

def resize_image_if_needed(np_img: np.ndarray, max_width: int = 1600) -> np.ndarray:
    """
    Redimensiona la imagen proporcionalmente si supera el ancho máximo para aligerar
    el procesamiento y garantizar velocidad y estabilidad en el servidor.
    """
    h, w = np_img.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        np_img = cv2.resize(np_img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return np_img

def process_document_balanced_scanner(image_bytes: bytes) -> Image.Image:
    """
    Motor de escáner balanceado (Estilo CamScanner Pro):
    - Redimensiona para optimizar peso.
    - Suaviza el ruido de fondo respetando las líneas de texto.
    - Aplica umbral adaptativo calibrado para evitar deformar o fragmentar las letras.
    """
    try:
        print("-> Procesando documento con motor de escáner balanceado...")
        
        # 1. Decodificar bytes a matriz OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 2. Redimensionar para evitar imágenes gigantescas de 5MB+
        img = resize_image_if_needed(img, max_width=1600)

        # 3. Escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4. Filtro bilateral: Elimina texturas y sombras de la mesa manteniendo el texto ultra nítido
        denoised = cv2.bilateralFilter(gray, 11, 85, 85)
        
        # 5. Umbral adaptativo calibrado (Offset en 12 para evitar que las letras se 'coman' o se rompan)
        thresh = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15,  # Vecindad óptima para documentos tamaño carta/factura
            12   # Offset equilibrado para conservar el grosor de las letras
        )
        
        # Convertir a PIL RGB
        processed_image = Image.fromarray(thresh).convert("RGB")
        print("-> ¡Éxito! Documento procesado con balance perfecto.")
        return processed_image

    except Exception as e:
        print(f"❌ Error en el escáner balanceado: {str(e)}")
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

                # REGLA: Carga horizontal a color (optimizada de peso), Documentos verticales con escáner balanceado
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    np_img_resized = resize_image_if_needed(np_img, max_width=1600)
                    color_image = Image.fromarray(cv2.cvtColor(np_img_resized, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando escáner balanceado.")
                    clean_doc = process_document_balanced_scanner(image_bytes)
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
            quality=85,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF unificado con motor Pro Scanner generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
