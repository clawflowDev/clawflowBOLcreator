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

app = FastAPI(title="CLAWFLOW Diagnostic Engine", version="13.1")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@app.get("/")
def health_check():
    key_status = "CONFIGURADA (Longitud: " + str(len(GROQ_API_KEY)) + ")" if GROQ_API_KEY else "NO CONFIGURADA ❌"
    return {"status": "online", "service": "CLAWFLOW Diagnostic Engine Active", "groq_key_status": key_status}

def get_active_vision_model() -> str:
    default_fallback = "llama-3.2-11b-vision-preview"
    if not GROQ_API_KEY:
        print("🚨 [LOG 1.0] GROQ_API_KEY vacía. Usando fallback por defecto.")
        return default_fallback

    try:
        print("-> [LOG 1.1] Consultando endpoint de modelos activos a Groq...")
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        
        print(f"-> [LOG 1.2] Status HTTP de /models: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            vision_models = [m["id"] for m in models_data if "vision" in m["id"].lower()]
            print(f"-> [LOG 1.3] Modelos de visión detectados en la cuenta: {vision_models}")
            
            if vision_models:
                selected = next((m for m in vision_models if "11b" in m), vision_models[0])
                print(f"-> [LOG 1.4] Modelo seleccionado dinámicamente: {selected}")
                return selected
        else:
            print(f"-> [LOG 1.5] Error listando modelos: {response.text}")
    except Exception as e:
        print(f"🚨 [LOG 1.6 Excepción en get_active_vision_model]: {str(e)}")

    print(f"-> [LOG 1.7] Usando fallback de seguridad: {default_fallback}")
    return default_fallback

def get_document_corners_from_groq(image_bytes: bytes, original_w: int, original_h: int) -> list:
    if not GROQ_API_KEY:
        print("🚨 [LOG 2.0] GROQ_API_KEY no disponible para análisis de esquinas.")
        return None

    try:
        active_model = get_active_vision_model()
        print(f"-> [LOG 2.1] Iniciando payload para Groq Vision con modelo: {active_model}")
        
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

        print("-> [LOG 2.2] Enviando POST request a Groq Chat Completions...")
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
        
        print(f"-> [LOG 2.3] Código HTTP de Groq Vision: {response.status_code}")
        if response.status_code != 200:
            print(f"🚨 [LOG 2.4 ERROR BODY]: {response.text}")
            return None

        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"].strip()
        print(f"-> [LOG 2.5] Respuesta cruda de la IA: {content}")
        
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
            print(f"-> [LOG 2.6] Esquinas convertidas exitosamente a píxeles: {pixel_corners}")
            return pixel_corners
        else:
            print(f"🚨 [LOG 2.7] El JSON parseado no contiene 'corners' válidos: {parsed}")

    except Exception as e:
        print(f"🚨 [LOG 2.8 EXCEPCIÓN CRÍTICA en get_document_corners]: {str(e)}")

    return None

def four_point_transform(image, pts):
    print(f"-> [LOG 3.0] Ejecutando four_point_transform con puntos: {pts}")
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
    print(f"-> [LOG 3.1] Perspectiva aplicada. Dimensiones resultantes: {warped.shape}")
    return warped

def process_document_pipeline(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar el buffer de imagen.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales de la foto: {original_w}x{original_h}")

        corners = get_document_corners_from_groq(image_bytes, original_w, original_h)

        if corners:
            print("-> [LOG 4.2] Llamando a transformación geométrica con esquinas de la IA...")
            warped_doc = four_point_transform(img, corners)
        else:
            print("🚨 [LOG 4.3 ALERTA] Groq Vision no devolvió esquinas. Activando respaldo de emergencia (Margen 4%).")
            margin_y = int(original_h * 0.04)
            margin_x = int(original_w * 0.04)
            warped_doc = img[margin_y:original_h-margin_y, margin_x:original_w-margin_x]

        print("-> [LOG 4.4] Redimensionando y aplicando mejora de contraste CLAHE a color...")
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

        print("-> [LOG 4.5] Armando lienzo blanco corporativo estándar...")
        dw, dh = doc_pil.size
        canvas_w = dw + 60
        canvas_h = dh + 60
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.6] Documento procesado y enmarcado con éxito.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.7 EXCEPCIÓN CRÍTICA en process_document_pipeline]: {str(e)}")
        nparr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        print(f"-> [LOG 5.0] Petición /process-bol recibida con {len(payload.image_urls)} imágenes.")
        if not payload.image_urls or len(payload.image_urls) == 0:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes.")

        pil_images = []
        
        for index, img_uri in enumerate(payload.image_urls):
            try:
                print(f"-> [LOG 5.1] Descargando/decodificando imagen {index+1}...")
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
                    print(f"🚨 [LOG 5.2] No se pudo decodificar la imagen {index+1}")
                    continue

                h, w, _ = np_img.shape

                if w > h:
                    print(f"-> [LOG 5.3] Imagen {index+1} es HORIZONTAL ({w}x{h}). Manteniendo a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> [LOG 5.4] Imagen {index+1} es VERTICAL ({w}x{h}). Entrando a pipeline de visión...")
                    clean_doc = process_document_pipeline(image_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"🚨 [LOG 5.5] Error procesando imagen {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        print("-> [LOG 6.0] Compilando PDF final unificado...")
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

        print("-> [LOG 6.1] ¡PDF generado y codificado exitosamente!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado con logs quirúrgicos detallados."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO GENERAL]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
