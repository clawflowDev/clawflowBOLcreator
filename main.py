import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image

app = FastAPI(title="CLAWFLOW Manual Crop Engine", version="40.0")

# Habilitar CORS para que Apps Script / WebApp pueda comunicarse sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Manual Crop & A4 Formatting Engine",
        "mode": "Manual Crop Payload + CLAHE Enhancement + Proportional A4 Canvas"
    }, 200

def process_cropped_image(image_bytes: bytes) -> Image.Image:
    """Aplica realce de tinta y centra proporcionalmente la imagen en un lienzo A4"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen recortada.")

    # 1. Filtro Bilateral: suaviza grano de papel conservando nitidez de tinta
    smoothed = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. Realce de Iluminación Local (CLAHE)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Convertir a PIL
    doc_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    # 3. Montaje A4 Proporcional (1480px x 2100px)
    target_w, target_h = 1480, 2100
    
    # Escalar para cubrir la mayor área posible conservando aspecto
    doc_pil.thumbnail((target_w - 60, target_h - 60), Image.Resampling.LANCZOS)
    
    # Crear lienzo A4 totalmente blanco
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    
    # Calcular coordenadas para centrado perfecto
    dw, dh = doc_pil.size
    pos_x = (target_w - dw) // 2
    pos_y = (target_h - dh) // 2
    
    canvas.paste(doc_pil, (pos_x, pos_y))
    return canvas

class ManualCropRequest(BaseModel):
    image_urls: List[str] # Lista de imágenes en Base64 o URLs recortadas
    metadata: Dict[str, Any] = {}

@app.post("/process-bol")
async def process_bol(payload: ManualCropRequest):
    try:
        print(f"-> [LOG 1.0] Petición recibida con {len(payload.image_urls)} recortes manuales.")
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes en el payload.")

        pil_images = []
        for index, img_uri in enumerate(payload.image_urls):
            try:
                print(f"-> [LOG 1.1] Procesando recorte de página {index+1}...")
                if img_uri.startswith("data:"):
                    header, encoded = img_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    response = requests.get(img_uri, timeout=15)
                    response.raise_for_status()
                    image_bytes = response.content

                processed_page = process_cropped_image(image_bytes)
                pil_images.append(processed_page)
            except Exception as img_err:
                print(f"🚨 [LOG ERROR] Error en página {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen recortada pudo ser procesada.")

        print("-> [LOG 2.0] Consolidando PDF A4 final...")
        pdf_output = io.BytesIO()
        first_image = pil_images[0]
        remaining_images = pil_images[1:] if len(pil_images) > 1 else []

        first_image.save(
            pdf_output,
            format="PDF",
            save_all=True,
            append_images=remaining_images,
            quality=92,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        print("-> [LOG 2.1] ¡PDF A4 compilado exitosamente!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF consolidado con recortes manuales en formato A4."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
