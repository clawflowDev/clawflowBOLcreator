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

app = FastAPI(title="CLAWFLOW Robust Document Engine", version="9.0")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Robust Document Engine Active"}

def order_points(pts):
    """Ordena los 4 puntos cardinales de un contorno: superior-izq, superior-der, inferior-der, inferior-izq."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Realiza la corrección de perspectiva (Enderezar y aislar la hoja)."""
    rect = order_points(pts)
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

def process_document_robust(image_bytes: bytes) -> Image.Image:
    """
    Motor Robusto Local:
    1. Detecta la hoja mediante umbral adaptativo y contornos.
    2. Si halla la hoja, la extrae y endereza con perspectiva.
    3. Aplica contraste CLAHE nítido para lectura perfecta.
    4. Si el entorno es muy irregular, enmarca sobre un fondo limpio uniforme.
    """
    try:
        print("-> Procesando documento con motor robusto local...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        h, w = img.shape[:2]
        
        # Redimensionar para procesamiento rápido y estable
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Umbral adaptativo para separar el papel blanco del fondo del asiento/mesa
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        warped = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Si el contorno tiene 4 esquinas y ocupa una porción razonable de la imagen (más del 15%)
            if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.15):
                try:
                    warped = four_point_transform(img, approx.reshape(4, 2))
                    break
                except Exception:
                    continue

        # Si el motor geométrico logró aislar la hoja con éxito
        if warped is not None and warped.shape[0] > 200 and warped.shape[1] > 200:
            print("-> ¡Detección geométrica exitosa! Hoja aislada y enderezada.")
            target_doc = warped
        else:
            print("-> Aviso: Detección estricta no concluyente. Aplicando recorte perimetral seguro de respaldo.")
            # Respaldo seguro: recortar un 6% perimetral para quitar bordes toscos sin morder texto
            pad_y = int(h * 0.06)
            pad_x = int(w * 0.06)
            target_doc = img[pad_y:h-pad_y, pad_x:w-pad_x]

        # Convertir a escala de grises y aplicar el contraste CLAHE perfecto que te gusta
        gray_doc = cv2.cvtColor(target_doc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_doc)
        
        # Reducción de ruido sutil
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        
        doc_pil = Image.fromarray(denoised).convert("RGB")

        # Lienzo de estandarización final (Padding simétrico elegante para uniformidad en el PDF)
        dw, dh = doc_pil.size
        canvas_w = dw + 80
        canvas_h = dh + 80
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (40, 40))

        print("-> ¡Documento procesado, nítido y unificado con éxito!")
        return canvas

    except Exception as e:
        print(f"❌ Error crítico en procesamiento robusto: {str(e)}")
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

                # REGLA: Carga horizontal a color optimizado; Documentos verticales pasan por el motor robusto
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal - {w}x{h}). Optimizando a color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical - {w}x{h}). Procesando con motor robusto.")
                    clean_doc = process_document_robust(image_bytes)
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
            "message": "PDF unificado con motor Robusto Local generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
