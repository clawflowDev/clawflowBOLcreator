import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from PIL import Image, ImageOps

app = FastAPI(title="CLAWFLOW BOL/POD CamScanner Engine", version="1.0")

class BOLRequest(BaseModel):
    image_urls: List[str]
    metadata: Dict[str, Any] = {}

@app.get("/")
def health_check():
    return {"status": "online", "service": "CLAWFLOW CamScanner Microservice Active"}

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
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

def camscanner_process(np_image):
    """
    Aplica binarización avanzada estilo CamScanner: fondo blanco puro y texto ultra nítido,
    garantizando que nunca falle ni baje la calidad.
    """
    try:
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        
        # Intento de detección de bordes para recorte inteligente
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        screen_cnt = None
        orig_area = np_image.shape[0] * np_image.shape[1]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (orig_area * 0.15):
                screen_cnt = approx
                break

        working_img = np_image
        if screen_cnt is not None:
            working_img = four_point_transform(np_image, screen_cnt.squeeze())
        else:
            # Fallback seguro: Si no encuentra los 4 puntos exactos, recortamos un 3% de los bordes 
            # exteriores para limpiar la mesa sin deformar ni manchar la hoja.
            h, w = gray.shape
            margin_h, margin_w = int(h * 0.03), int(w * 0.03)
            working_img = np_image[margin_h:h-margin_h, margin_w:w-margin_w]

        # --- MAGIA ESTILO CAMSCANNER (Threshold Adaptativo) ---
        # Convierte a escala de grises la imagen de trabajo
        gray_work = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        
        # Elimina sombras de iluminación despareja y vuelve el fondo blanco absoluto con texto negro nítido
        denoised = cv2.fastNlMeansDenoising(gray_work, h=10)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 10
        )

        return Image.fromarray(thresh).convert("RGB")

    except Exception as e:
        print(f"Aviso en procesamiento CamScanner: {str(e)}")
        # Fallback ultra seguro en grises limpios
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresh).convert("RGB")

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

                # REGLA: Carga horizontal a color, Documentos verticales con filtro CamScanner
                if w > h:
                    print(f"-> Imagen {index+1}: Carga detectada (Horizontal). A todo color.")
                    color_image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                    pil_images.append(color_image)
                else:
                    print(f"-> Imagen {index+1}: Documento detectado (Vertical). Aplicando motor CamScanner.")
                    clean_doc = camscanner_process(np_img)
                    pil_images.append(clean_doc)
                
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
            "message": "PDF generado con éxito bajo motor CamScanner (fondos blancos y máxima legibilidad)."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error crítico en /process-bol: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
