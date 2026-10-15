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

app = FastAPI(title="CLAWFLOW BOL/POD Final Cropper Engine", version="5.1")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Final Cropper Engine Active"}

def resize_image_if_needed(np_img: np.ndarray, max_width: int = 1600) -> np.ndarray:
    h, w = np_img.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        np_img = cv2.resize(np_img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return np_img

def auto_crop_document(np_img: np.ndarray) -> np.ndarray:
    """
    Motor de recorte optimizado (Versión 5.1):
    Encuentra los contornos del papel de forma estricta para eliminar asientos,
    mesas y fondos, dejando únicamente la hoja limpia.
    """
    try:
        orig = np_img.copy()
        h, w = orig.shape[:2]

        # 1. Escala de grises y desenfoque gaussiano
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. Umbralización Otsu para separar el papel blanco del fondo oscuro de la mesa/asiento
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Encontrar contornos externos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno con mayor área (el papel)
            max_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_c)
            
            # Si el área ocupa al menos el 10% de la imagen, es un documento válido
            if area > (h * w * 0.10):
                x, y, w_box, h_box = cv2.boundingRect(max_c)
                
                # Margen de seguridad ultra fino (5 píxeles) para no recortar texto pero eliminar todo el fondo
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + w_box + pad)
                y2 = min(h, y + h_box + pad)
                
                cropped = orig[y1:y2, x1:x2]
                if cropped.shape[0] > 100 and cropped.shape[1] > 100:
                    print("-> ¡Auto-crop exitoso! Fondo eliminado correctamente.")
                    return cropped

    except Exception as e:
        print(f"⚠️ Aviso en auto-crop preciso: {str(e)}")

    # Fallback: Si por alguna razón extrema no halla el contorno, devuelve la imagen original
    return np_img

def process_document_final(image_bytes: bytes) -> Image.Image:
    """
    Combina el recorte estricto de la hoja con el contraste CLAHE perfecto.
    """
    try:
        print("-> Procesando documento con motor Final (Crop estricto + Contraste óptimo)...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 1. Redimensionar para velocidad y precisión
        img = resize_image_if_needed(img, max_width=1600)

        # 2. Recorte estricto del papel
        cropped_img = auto_crop_document(img)

        # 3. Escala de grises
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # 4. CLAHE con el nivel de contraste exacto que te gustó (clipLimit=3.0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 5. Reducción de ruido sutil
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        processed_image = Image.fromarray(denoised).convert("RGB")
        print("-> ¡Éxito total en el procesamiento del documento!")
        return processed_image

    except Exception as e:
        print(f"❌ Error en procesamiento Final: {str(e)}")
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

                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    np_img_resized = resize_image_if_needed(np_img, max_width=1600)
                    color_image = Image.fromarray(cv2.cvtColor(np_img_resized, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Aplicando recorte estricto y contraste.")
                    clean_doc = process_document_final(image_bytes)
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
            "message": "PDF unificado con motor Final Cropper generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
