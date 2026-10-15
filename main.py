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

app = FastAPI(title="CLAWFLOW Router HF Vision Engine", version="17.0")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Modelos en cascada usando el router oficial de Hugging Face
VISION_MODELS_CANDIDATES = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "microsoft/Phi-3-vision-128k-instruct"
]

@app.get("/")
def health_check():
    token_status = "CONFIGURADA (Longitud: " + str(len(HF_TOKEN)) + ")" if HF_TOKEN else "NO CONFIGURADA ❌"
    return {"status": "online", "service": "CLAWFLOW Router HF Vision Engine Active", "hf_token_status": token_status}

def get_document_corners_from_hf(image_bytes: bytes, original_w: int, original_h: int) -> list:
    if not HF_TOKEN:
        print("🚨 [LOG 2.0] ERROR CRÍTICO: HF_TOKEN no está configurado en Render.")
        return None

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Endpoint oficial del router compatible con OpenAI
    api_url = "https://router.huggingface.co/v1/chat/completions"

    prompt_text = (
        "Analiza esta imagen a color y localiza estrictamente los límites del documento de papel (la hoja de Bill of Lading / recibo). "
        "Ignora por completo el fondo oscuro, los asientos, las manos o la mesa. "
        "Devuelve SOLAMENTE un objeto JSON válido con este formato exacto, sin texto adicional ni bloques markdown: "
        "{\"box_2d\": [ymin, xmin, ymax, xmax]} donde los valores estén en una escala de 0 a 1000."
    )

    for index, model_name in enumerate(VISION_MODELS_CANDIDATES):
        try:
            print(f"-> [LOG 2.1.{index}] Intentando con router oficial para [{model_name}]...")
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
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

            response = requests.post(api_url, headers=headers, json=payload, timeout=25)
            
            print(f"-> [LOG 2.2.{index}] HTTP Status del router para {model_name}: {response.status_code}")
            if response.status_code != 200:
                print(f"🚨 [LOG 2.3.{index} ERROR BODY]: {response.text}")
                print(f"-> [LOG 2.4.{index}] Modelo {model_name} falló. Pasando al siguiente en la cadena...")
                continue

            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"].strip()
            print(f"-> [LOG 2.5.{index}] Respuesta obtenida de {model_name}: {content}")
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            box = parsed.get("box_2d")

            if box and len(box) == 4:
                ymin, xmin, ymax, xmax = box
                pixel_ymin = int((ymin / 1000.0) * original_h)
                pixel_xmin = int((xmin / 1000.0) * original_w)
                pixel_ymax = int((ymax / 1000.0) * original_h)
                pixel_xmax = int((xmax / 1000.0) * original_w)
                
                print(f"-> [LOG 2.6] ¡Éxito con el modelo {model_name}! Coordenadas: ymin={pixel_ymin}, xmin={pixel_xmin}, ymax={pixel_ymax}, xmax={pixel_xmax}")
                return [pixel_ymin, pixel_xmin, pixel_ymax, pixel_xmax]

        except Exception as e:
            print(f"🚨 [LOG 2.7 Excepción con modelo {model_name}]: {str(e)}")
            continue

    print("🚨 [LOG 2.8 ALERTA TOTAL] Todos los modelos del router fallaron.")
    return None

def process_document_with_hf(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales: {original_w}x{original_h}")

        box = get_document_corners_from_hf(image_bytes, original_w, original_h)

        if box:
            ymin, xmin, ymax, xmax = box
            margin_y = int((ymax - ymin) * 0.01)
            margin_x = int((xmax - xmin) * 0.01)
            
            y1 = max(0, ymin + margin_y)
            x1 = max(0, xmin + margin_x)
            y2 = min(original_h, ymax - margin_y)
            x2 = min(original_w, xmax - margin_x)
            
            cropped_doc = img[y1:y2, x1:x2]
            print("-> [LOG 4.2] Recorte milimétrico aplicado con éxito usando el router.")
        else:
            print("🚨 [LOG 4.3 ALERTA] Usando respaldo de emergencia (Margen 4%).")
            crop_y = int(original_h * 0.04)
            crop_x = int(original_w * 0.04)
            cropped_doc = img[crop_y:original_h-crop_y, crop_x:original_w-crop_x]

        print("-> [LOG 4.4] Redimensionando y aplicando CLAHE a color...")
        target_width = 1600
        ch, cw = cropped_doc.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            cropped_doc = cv2.resize(cropped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(cropped_doc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        print("-> [LOG 4.5] Armando lienzo blanco corporativo...")
        dw, dh = doc_pil.size
        canvas_w = dw + 60
        canvas_h = dh + 60
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.6] Documento procesado y enmarcado con éxito.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.7 EXCEPCIÓN CRÍTICA]: {str(e)}")
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
                print(f"-> [LOG 5.1] Procesando imagen {index+1}...")
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
                    print(f"-> [LOG 5.3] Imagen {index+1} horizontal. Manteniendo a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> [LOG 5.4] Imagen {index+1} vertical. Enviando al router oficial de visión...")
                    clean_doc = process_document_with_hf(image_bytes)
                    pil_images.append(clean_doc)
                
            except Exception as img_err:
                print(f"🚨 [LOG 5.5] Error en imagen {index+1}: {str(img_err)}")

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
            "message": "PDF generado utilizando el router oficial de Hugging Face."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
