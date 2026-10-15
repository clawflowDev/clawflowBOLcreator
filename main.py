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

app = FastAPI(title="CLAWFLOW Color-Aided Edge Engine", version="9.1")

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW Color-Aided Edge Engine Active"}

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

def process_document_color_guided(image_bytes: bytes) -> Image.Image:
    """
    Motor con Detección a Color:
    1. Analiza los canales de color originales para encontrar los bordes de la hoja sin confundirse con grises.
    2. Aisla y endereza la perspectiva.
    3. Realza el texto para máxima legibilidad de archivo.
    """
    try:
        print("-> Procesando documento con guía de color para bordes...")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        h, w = img.shape[:2]
        
        # Redimensionar para estabilidad
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # Convertir a espacio de color LAB o HSV para potenciar la separación de tonos (Color-based edge detection)
        # El canal L (Luminancia) combinado con saturación ayuda muchísimo a aislar el papel blanco de asientos oscuros.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Desenfoque suave para eliminar texturas de la tela del asiento
        blurred = cv2.GaussianBlur(l_channel, (7, 7), 0)
        
        # Umbralización Otsu sobre el canal de luminancia optimizado
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        warped = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Si tiene 4 esquinas y ocupa más del 15% de la imagen
            if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.15):
                try:
                    warped = four_point_transform(img, approx.reshape(4, 2))
                    break
                except Exception:
                    continue

        if warped is not None and warped.shape[0] > 200 and warped.shape[1] > 200:
            print("-> ¡Detección guiada por color exitosa! Hoja aislada.")
            target_doc = warped
        else:
            print("-> Aviso: Detección estricta no concluyente. Aplicando recorte de respaldo seguro.")
            pad_y = int(h * 0.05)
            pad_x = int(w * 0.05)
            target_doc = img[pad_y:h-pad_y, pad_x:w-pad_x]

        # Post-procesamiento enfocado en nitidez de texto manteniendo escala de grises limpia
        gray_doc = cv2.cvtColor(target_doc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_doc)
        
        denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
        doc_pil = Image.fromarray(denoised).convert("RGB")

        # Lienzo de estandarización final (Padding simétrico elegante)
        dw, dh = doc_pil.size
        canvas_w = dw + 80
        canvas_h = dh + 80
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(doc_pil, (40, 40))

        print("-> ¡Documento procesado con éxito!")
        return canvas

    except Exception as e:
        print(f"❌ Error en procesamiento guiado por color: {str(e)}")
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
                    print(f"-> Imagen {index+1}: Carga horizontal. A color.")
                    target_w = 1600
                    if w > target_w:
                        ratio = target_w / float(w)
                        np_img = cv2.resize(np_img, (target_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento vertical. Procesando con guía de color.")
                    clean_doc = process_document_color_guided(image_bytes)
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
            "message": "PDF generado exitosamente."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
