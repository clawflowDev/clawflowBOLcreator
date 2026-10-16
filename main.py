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

app = FastAPI(title="CLAWFLOW Guaranteed Crop Engine", version="26.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Guaranteed Crop Engine Active",
        "mode": "Guaranteed Perimetric Crop + Bilateral CLAHE Enhancement"
    }, 200

def process_document_guaranteed(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen binaria.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales: {original_w}x{original_h} px")

        # Si la imagen es horizontal (fotos de la carga/remolque), se preserva a color intacta
        if original_w > original_h:
            print("-> [LOG 4.2] Imagen horizontal detectada. Preservando a color sin recortes...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # -------------------------------------------------------------
        # RECORTE PERIMETRAL GARANTIZADO (ELIMINACIÓN DE FONDOS / MESA)
        # -------------------------------------------------------------
        print("-> [LOG 4.3] Aplicando recorte perimetral simétrico garantizado para páginas verticales...")
        
        # Margenes porcentuales precisos para eliminar el fondo de las fotos de WhatsApp en cualquier página
        # 6% en los laterales (izq/der) y 7% en vertical (arriba/abajo)
        crop_y_top = int(original_h * 0.07)
        crop_y_bottom = int(original_h * 0.93)
        crop_x_left = int(original_w * 0.06)
        crop_x_right = int(original_w * 0.94)

        cropped_doc = img[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]
        print(f"-> [LOG 4.4] Recorte aplicado: de {original_w}x{original_h} a {cropped_doc.shape[1]}x{cropped_doc.shape[0]} px")

        # -------------------------------------------------------------
        # ESTANDARIZACIÓN Y REALCE DE TINTA (OCR-READY)
        # -------------------------------------------------------------
        print("-> [LOG 4.5] Aplicando redimensionamiento y realce CLAHE para máxima nitidez...")
        target_width = 1600
        ch, cw = cropped_doc.shape[:2]
        if cw != target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            cropped_doc = cv2.resize(cropped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # Filtro bilateral para suavizar ruidos de fondo manteniendo los textos súper nítidos
        smoothed = cv2.bilateralFilter(cropped_doc, 9, 75, 75)

        # Espacio de color LAB para destacar la luminancia del texto sobre el papel
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Enmarcado en lienzo blanco corporativo (30px de margen estético)
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.6] Página procesada, recortada y enmarcada con éxito.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.7 ERROR EN PROCESAMIENTO]: {str(e)}")
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

                processed_page = process_document_guaranteed(image_bytes)
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
            quality=92,
            optimize=True
        )
        
        pdf_output.seek(0)
        pdf_base64 = base64.b64encode(pdf_output.read()).decode("utf-8")

        print("-> [LOG 6.1] ¡PDF generado exitosamente!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado exitosamente con el motor de recorte perimetral garantizado."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
