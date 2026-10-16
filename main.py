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

app = FastAPI(title="CLAWFLOW Robust Document Warp Engine", version="23.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Robust Document Warp Engine Active",
        "mode": "Adaptive Perspective Warp + Fallback Perimetric Crop"
    }, 200

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

    maxWidth = max(maxWidth, 800)
    maxHeight = max(maxHeight, 1100)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_document_with_warp(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen binaria.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales: {original_w}x{original_h} px")

        # Si la imagen es horizontal (fotos de carga), la preservamos a color
        if original_w > original_h:
            print("-> [LOG 4.2] Imagen horizontal detectada. Preservando a color...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # -------------------------------------------------------------
        # DETECCIÓN ADAPTATIVA (WARP + RECÓRTE PERIMETRAL SEGURO)
        # -------------------------------------------------------------
        print("-> [LOG 4.3] Buscando esquinas y contorno del documento...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            area = cv2.contourArea(c)
            # Aceptar contornos de 4 esquinas que ocupen entre el 20% y el 95% de la foto
            if len(approx) == 4 and (original_h * original_w * 0.20) < area < (original_h * original_w * 0.95):
                screenCnt = approx
                break

        if screenCnt is not None:
            print("-> [LOG 4.4] ¡Contorno de 4 esquinas detectado con éxito! Aplicando Warp...")
            warped = four_point_transform(img, screenCnt.reshape(4, 2))
        else:
            print("-> [LOG 4.4 WARNING] Contorno no cerrado. Aplicando recorte perimetral inteligente del 5% para eliminar fondo.")
            # Recorte de seguridad del 5% en cada lado para garantizar que la página 3 elimine el fondo oscuro o mesa
            crop_y = int(original_h * 0.05)
            crop_x = int(original_w * 0.05)
            warped = img[crop_y:original_h-crop_y, crop_x:original_w-crop_x]

        # Estandarizar ancho a 1600px para máxima nitidez
        target_width = 1600
        ch, cw = warped.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            warped = cv2.resize(warped, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # Realce de contraste CLAHE
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Lienzo blanco corporativo con margen estético de resguardo
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.5] Documento procesado, recortado y enmarcado con éxito.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.6 ERROR EN PROCESAMIENTO]: {str(e)}")
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
        if not payload.image_urls:
            raise HTTPException(status_code=400, detail="No se recibieron imágenes.")

        pil_images = []
        for index, img_uri in enumerate(payload.image_urls):
            try:
                print(f"-> [LOG 5.1] Procesando imagen {index+1} de {len(payload.image_urls)}...")
                if img_uri.startswith("data:"):
                    header, encoded = img_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                else:
                    response = requests.get(img_uri, timeout=15)
                    response.raise_for_status()
                    image_bytes = response.content

                processed_page = process_document_with_warp(image_bytes)
                pil_images.append(processed_page)
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

        print("-> [LOG 6.1] ¡PDF generado exitosamente!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado exitosamente con respaldo de recorte perimetral adaptativo."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
