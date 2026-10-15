import os
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
import uvicorn

app = FastAPI(title="CLAWFLOW BOL & POD Creator Engine", version="2.0")

class BolPayload(BaseModel):
    image_urls: List[str] # Lista de Data URIs o URLs públicas de las imágenes/páginas
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW BOL Engine - Batch Ready"}

@app.post("/process-bol")
def process_bol(payload: BolPayload):
    try:
        if not payload.image_urls:
            raise HTTPException(status_code=400, status.dumps({"error": "No se recibieron imágenes en el lote."}))
        
        processed_images = []
        
        for idx, img_source in enumerate(payload.image_urls):
            # Soportar Data URI (Base64) o URL estándar
            if img_source.startswith("data:"):
                header, encoded = img_source.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                import requests
                resp = requests.get(img_source)
                image_bytes = resp.content
            
            # Abrir imagen con PIL y asegurar formato RGB (ideal para PDF)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            processed_images.append(image)
        
        # Unir todas las imágenes en un único PDF consolidado
        pdf_output_stream = io.BytesIO()
        if processed_images:
            first_img = processed_images[0]
            rest_imgs = processed_images[1:] if len(processed_images) > 1 else []
            
            first_img.save(
                pdf_output_stream,
                format="PDF",
                save_all=True,
                append_images=rest_imgs
            )
        
        pdf_bytes = pdf_output_stream.getvalue()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        return {
            "status": "success",
            "total_pages": len(processed_images),
            "pdf_base64": pdf_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
