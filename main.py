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

app = FastAPI(title="CLAWFLOW BOL/POD Smart Processor - Step 1", version="1.0")

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW BOL Microservice Active"}

def enhance_document_with_clahe(np_image):
    """
    Aplica mejora tipo escáner profesional usando CLAHE para eliminar sombras
    y resaltar el texto de los documentos de forma impecable.
    """
    try:
        # 1. Convertir a escala de grises
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Esto equilibra las sombras locales (como los dobleces del papel o la luz desigual)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 3. Convertir de vuelta a imagen PIL RGB para el PDF
        image = Image.fromarray(enhanced_gray).convert("RGB")
        return image
    except Exception as e:
        print(f"Aviso en CLAHE, usando grises estándar: {str(e)}")
        # Fallback de seguridad
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
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

                nparr = np.frombuffer(image_bytes, np.uint8)
                np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if np_img is None:
                    continue

                h, w, _ = np_img.shape

                # --- CLASIFICACIÓN DINÁMICA ---
                # Analizamos si es una foto de carga (horizontal / landscape) o un documento (vertical / portrait).
                # (Más adelante en el paso 2 afinaremos esto con análisis de color y saturación).
                if w > h:
                    print(f"-> Imagen {index+1}: Detectada como Carga (Horizontal). Manteniendo a color.")
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Detectada como Documento (Vertical). Aplicando efecto escáner CLAHE.")
                    processed_doc = enhance_document_with_clahe(np_img)
                    pil_images.append(processed_doc)
                
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
            "message": "Paso 1 completado: Carga a color y documentos con escáner inteligente CLAHE."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
