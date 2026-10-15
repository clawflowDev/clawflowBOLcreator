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

app = FastAPI(title="CLAWFLOW Groq Vision Engine", version="10.1")

# Obtener la API Key de Groq desde las variables de entorno de Render
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Groq Vision & OpenCV Engine Active"}

def get_document_bounding_box_from_groq(image_bytes: bytes, original_w: int, original_h: int) -> list:
    """
    Envía la imagen a Groq Vision para que identifique las coordenadas [ymin, xmin, ymax, xmax] 
    de la hoja de papel (escala de 0 a 1000).
    """
    if not GROQ_API_KEY:
        print("⚠️ Advertencia: GROQ_API_KEY no está configurada en Render.")
        return None

    try:
        print("-> Consultando a Groq Vision para detectar los límites exactos de la hoja...")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analiza esta imagen y detecta estrictamente las coordenadas de los límites del documento de papel (Bill of Lading / página) ignorando por completo el fondo, los asientos o la mesa. Devuelve SOLAMENTE un objeto JSON válido con este formato exacto, sin texto markdown adicional: {\"box_2d\": [ymin, xmin, ymax, xmax]} donde los valores estén en una escala de 0 a 1000."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"].strip()
        
        # Limpiar posibles bloques markdown de la respuesta
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        box = parsed.get("box_2d")

        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            # Traducir de escala 0-1000 a píxeles reales
            pixel_ymin = int((ymin / 1000.0) * original_h)
            pixel_xmin = int((xmin / 1000.0) * original_w)
            pixel_ymax = int((ymax / 1000.0) * original_h)
            pixel_xmax = int((xmax / 1000.0) * original_w)
            
            print(f"-> Groq Vision detectó el documento en: ymin={pixel_ymin}, xmin={pixel_xmin}, ymax={pixel_ymax}, xmax={pixel_xmax}")
            return [pixel_ymin, pixel_xmin, pixel_ymax, pixel_xmax]

    except Exception as e:
        print(f"⚠️ Error al consultar Groq Vision: {str(e)}")

    return None

def process_document_with_groq_vision(image_bytes: bytes) -> Image.Image:
    """
    Procesa el documento usando las coordenadas de Groq Vision + realce OpenCV.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        original_h, original_w = img.shape[:2]

        # 1. Obtener coordenadas de recorte mediante Groq Vision
        box = get_document_bounding_box_from_groq(image_bytes, original_w, original_h)

        if box:
            ymin, xmin, ymax, xmax = box
            # Margen de seguridad interno del 1% para evitar bordes sucios
            margin_y = int((ymax - ymin) * 0.01)
            margin_x = int((xmax - xmin) * 0.01)
            
            y1 = max(0, ymin + margin_y)
            x1 = max(0, xmin + margin_x)
            y2 = min(original_h, ymax - margin_y)
            x2 = min(original_w, xmax - margin_x)
            
            cropped_doc = img[y1:y2, x1:x2]
            print("-> Recorte milimétrico aplicado con éxito mediante Groq Vision.")
        else:
            print("-> Aviso: Usando recorte de respaldo seguro.")
            crop_y = int(original_h * 0.05)
            crop_x = int(original_w * 0.05)
            cropped_doc = img[crop_y:original_h-crop_y, crop_x:original_w-crop_x]

        # 2. Redimensionar estándar a 1600px de ancho
        target_width = 1600
        ch, cw = cropped_doc.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            cropped_doc = cv2.resize(cropped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # 3. Realce de contraste corporativo (CLAHE)
        gray = cv2.cvtColor(cropped_doc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        doc_pil = Image.fromarray(denoised).convert("RGB")

        # 4. Lienzo blanco estandarizado (Padding simétrico)
        dw, dh = doc_pil.size
        canvas_w = dw + 80
        canvas_h = dh + 80
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (40, 40))

        return canvas

    except Exception as e:
        print(f"❌ Error crítico en procesamiento: {str(e)}")
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

                # REGLA: Horizontal a color; Vertical procesado con Groq Vision
                if w > h:
                    print(f"-> Imagen {index+1}: Carga horizontal. A color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento vertical. Procesando con Groq Vision.")
                    clean_doc = process_document_with_groq_vision(image_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"Advertencia procesando imagen {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

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
            "message": "PDF unificado con Groq Vision generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
