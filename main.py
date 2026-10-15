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

app = FastAPI(title="CLAWFLOW Clean Archive Engine", version="6.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Clean Archive Engine Active"}

def resize_and_clean_document(image_bytes: bytes) -> Image.Image:
    """
    Procesa la imagen para archivo limpio:
    - Normaliza el tamaño y el encuadre.
    - Aplica escala de grises y contraste natural balanceado (legible y uniforme).
    - Recorta los márgenes exteriores para eliminar fondos no deseados (asientos/mesas).
    """
    try:
        print("-> Procesando documento para archivo limpio y uniforme...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        h, w = img.shape[:2]

        # 1. Recorte perimetral de seguridad (Trim): Remueve un 5% de los bordes exteriores 
        # donde usualmente se cuelan los asientos, mesas o dedos, aislando el cuerpo principal del documento.
        crop_y = int(h * 0.04)
        crop_x = int(w * 0.04)
        img_cropped = img[crop_y:h-crop_y, crop_x:w-crop_x]

        # 2. Redimensionar de forma estándar para mantener uniformidad en el archivo final
        target_width = 1600
        ch, cw = img_cropped.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            img_resized = cv2.resize(img_cropped, (target_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_cropped

        # 3. Convertir a escala de grises profesional
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 4. Contraste equilibrado (CLAHE suave) para que la lectura sea perfecta y nítida sin deformar texto
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. Reducción de ruido sutil para dejar el fondo blanco/gris claro limpio
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Documento procesado y unificado con éxito!")
        return processed_image

    except Exception as e:
        print(f"❌ Error en procesamiento de archivo: {str(e)}")
        # Fallback de seguridad
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

                # REGLA: Carga horizontal se mantiene a color optimizado; Documentos verticales se limpian para archivo
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando limpieza de archivo.")
                    clean_doc = resize_and_clean_document(image_bytes)
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
            "message": "PDF unificado con motor Clean Archive generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
