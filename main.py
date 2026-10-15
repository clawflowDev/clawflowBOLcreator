import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image

app = FastAPI(title="CLAWFLOW BOL/POD Batch Processor", version="1.0")

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW BOL Batch Microservice"}

@app.post("/process-bol")
async def process_bol(payload: BOLRequest):
    try:
        if not payload.image_urls or len(payload.image_urls) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No se recibieron imágenes en el lote."
            )

        pil_images = []
        
        # Descargar y convertir cada imagen de Data-URI / URL a objeto PIL Image
        for index, img_uri in enumerate(payload.image_urls):
            try:
                if img_uri.startswith("data:"):
                    # Extraer base64 de un Data URI (ej: data:image/jpeg;base64,...)
                    header, encoded = img_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    # Descarga por URL HTTP tradicional si aplica
                    response = requests.get(img_uri, timeout=15)
                    response.raise_for_status()
                    image_bytes = response.content

                image = Image.open(io.BytesIO(image_bytes))
                
                # Convertir a RGB (necesario si vienen en RGBA o P para guardar como PDF)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                pil_images.append(image)
            except Exception as img_err:
                print(f"Advertencia: No se pudo procesar la imagen índice {index}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(
                status_code=400, 
                detail="Ninguna de las imágenes del lote pudo ser procesada correctamente."
            )

        # Compilar todas las imágenes en un solo archivo PDF unificado en memoria
        pdf_output = io.BytesIO()
        
        # La primera imagen guarda el PDF y el resto se añaden como páginas apendadas
        first_image = pil_images[0]
        remaining_images = pil_images[1:] if len(pil_images) > 1 else []

        first_image.save(
            pdf_output,
            format="PDF",
            save_all=True,
            append_images=remaining_images
        )
        
        pdf_output.seek(0)
        pdf_bytes = pdf_output.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF unificado generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=json.dumps({"error": f"Error interno en el servidor: {str(e)}"})
        )
