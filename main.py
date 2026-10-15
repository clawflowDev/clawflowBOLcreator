import io
import json
import base64
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image
import os

app = FastAPI(title="CLAWFLOW Dynamic Vision Engine", version="13.0")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@app.get("/")
def health_check():
    key_status = "CONFIGURADA (Longitud: " + str(len(GROQ_API_KEY)) + ")" if GROQ_API_KEY else "NO CONFIGURADA ❌"
    return {"status": "online", "service": "CLAWFLOW Dynamic Vision Engine Active", "groq_key_status": key_status}

def get_active_vision_model() -> str:
    """
    Consulta dinámicamente a la API de Groq los modelos disponibles en tiempo real
    y selecciona automáticamente el mejor modelo de visión activo.
    """
    default_fallback = "llama-3.2-11b-vision-preview"
    
    if not GROQ_API_KEY:
        return default_fallback

    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            # Buscar modelos que contengan 'vision' en su ID
            vision_models = [m["id"] for m in models_data if "vision" in m["id"].lower() and m.get("active", True)]
            
            if vision_models:
                # Priorizar el modelo de 11b o el primero disponible que encuentre
                selected = next((m for m in vision_models if "11b" in m), vision_models[0])
                print(f"-> [Dinámico] Modelo de visión seleccionado automáticamente: {selected}")
                return selected
    except Exception as e:
        print(f"⚠️ No se pudo consultar la lista de modelos dinámicos: {str(e)}. Usando respaldo: {default_fallback}")

    return default_fallback

def get_document_corners_from_groq(image_bytes: bytes, original_w: int, original_h: int) -> list:
    if not GROQ_API_KEY:
        print("🚨 ERROR CRÍTICO: GROQ_API_KEY está vacía en las variables de entorno de Render.")
        return None

    try:
        # Descubrir dinámicamente qué modelo de visión está activo hoy
        active_model = get_active_vision_model()
        
        print(f"-> [Groq Vision] Usando modelo dinámico: {active_model}")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt_text = (
            "Eres un motor experto en visión por computadora para escaneo de documentos. "
            "Inspecciona esta foto a color y detecta estrictamente las 4 esquinas de la hoja de papel (el Bill of Lading / recibo dentro del marco). "
            "Ignora por completo el fondo, asientos de camión, sombras u otros objetos. "
            "Devuelve un objeto JSON válido con los 4 vértices exactos en orden horario: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]. "
            "Cada vértice debe tener sus coordenadas [x, y] escaladas de 0 a 1000. "
            "Formato estrictamente requerido, sin texto markdown adicional: "
            "{\"corners\": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}"
        )

        payload = {
            "model": active_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 200
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
        
        print(f"-> [Groq Vision] Código de respuesta HTTP: {response.status_code}")
        if response.status_code != 200:
            print(f"-> [Groq Vision Error Body]: {response.text}")
            return None

        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"].strip()
        print(f"-> [Groq Vision Raw Response]: {content}")
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        corners = parsed.get("corners")

        if corners and len(corners) == 4:
            pixel_corners = []
            for pt in corners:
                x = int((pt[0] / 1000.0) * original_w)
                y = int((pt[1] / 1000.0) * original_h)
                pixel_corners.append([x, y])
            print(f"-> [Groq Vision] Esquinas convertidas a píxeles: {pixel_corners}")
            return pixel_corners

    except Exception as e:
        print(f"🚨 Excepción atrapada en get_document_corners_from_groq: {str(e)}")

    return None

def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_document_pipeline(image_bytes: bytes) -> Image.Image:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        original_h, original_w = img.shape[:2]

        corners = get_document_corners_from_groq(image_bytes, original_w, original_h)

        if corners:
            warped_doc = four_point_transform(img, corners)
            print("-> [OpenCV] Recorte de perspectiva aplicado con éxito.")
        else:
            print("🚨 [ALERTA] Groq Vision no entregó esquinas. Usando respaldo de emergencia (Margen 4%).")
            margin_y = int(original_h * 0.04)
            margin_x = int(original_w * 0.04)
            warped_doc = img[margin_y:original_h-margin_y, margin_x:original_w-margin_x]

        target_width = 1600
        ch, cw = warped_doc.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            warped_doc = cv2.resize(warped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(warped_doc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        dw, dh = doc_pil.size
        canvas_w = dw + 60
        canvas_h = dh + 60
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        return canvas

    except Exception as e:
        print(f"❌ Error crítico en process_document_pipeline: {str(e)}")
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
                    print(f"-> Imagen {index+1}: Horizontal. A color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Vertical. Procesando con Groq Vision.")
                    clean_doc = process_document_pipeline(image_bytes)
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
            "message": "PDF generado con descubrimiento dinámico de modelos de visión."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
