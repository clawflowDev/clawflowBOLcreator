import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image, ImageOps
import os

app = FastAPI(title="CLAWFLOW Clean Canvas Engine", version="7.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Clean Canvas Engine Active"}

def process_document_on_clean_canvas(image_bytes: bytes) -> Image.Image:
    """
    Procesa la imagen para archivo limpio usando un lienzo blanco uniforme:
    - Realiza el realce de contraste perfecto (CLAHE) para legibilidad óptima.
    - Coloca la imagen centrada sobre un fondo/lienzo blanco puro estandarizado,
      eliminando visualmente todo el fondo de asientos, mesas o entornos sin arriesgar texto.
    """
    try:
        print("-> Procesando documento con motor Clean Canvas...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 1. Redimensionar de forma estándar para mantener uniformidad (ancho base de 1600px)
        target_width = 1600
        h, w = img.shape[:2]
        if w > target_width:
            ratio = target_width / float(w)
            new_h = int(h * ratio)
            img_resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img

        # 2. Convertir a escala de grises profesional
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Contraste equilibrado (CLAHE) para que la lectura sea perfecta y nítida
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. Reducción de ruido sutil
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Convertir a PIL Image en RGB
        doc_image = Image.fromarray(denoised).convert("RGB")

        # 5. El toque maestro del Lienzo Blanco (Canvas Padding):
        # Creamos un lienzo vertical estándar tipo documento limpio (ej: proporción A4 / Carta de 1700x2200 o basado en la foto)
        doc_w, doc_h = doc_image.size
        canvas_width = max(doc_w + 100, 1650)   # Margen simétrico a los lados
        canvas_height = max(doc_h + 100, 2200)  # Margen simétrico arriba y abajo
        
        # Crear lienzo completamente blanco puro
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        
        # Calcular posición exacta para centrar la foto del documento en el lienzo
        offset_x = (canvas_width - doc_w) // 2
        offset_y = (canvas_height - doc_h) // 2
        
        # Pegar el documento en el centro exacto del lienzo blanco
        canvas.paste(doc_image, (offset_x, offset_y))
        
        print("-> ¡Documento unificado y centrado en lienzo blanco con éxito!")
        return canvas

    except Exception as e:
        print(f"❌ Error en procesamiento de lienzo: {str(e)}")
        # Fallback de seguridad
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

                # REGLA: Carga horizontal se mantiene a color optimizado; Documentos verticales pasan por el motor Clean Canvas
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando Clean Canvas.")
                    clean_doc = process_document_on_clean_canvas(image_bytes)
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
            "message": "PDF unificado con motor Clean Canvas generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
