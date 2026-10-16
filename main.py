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

app = FastAPI(title="CLAWFLOW Ink Density & Margin Engine", version="21.1")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Ink Density Engine Active",
        "mode": "Filter Shadows + 1cm Margin Ink Detection + A4 Canvas Composition"
    }

def process_image_by_ink_density(image_bytes: bytes) -> Image.Image:
    try:
        print("-> [LOG 4.0] Decodificando imagen con OpenCV...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen binaria.")

        original_h, original_w = img.shape[:2]
        print(f"-> [LOG 4.1] Dimensiones originales: {original_w}x{original_h} px")

        # Si la imagen es horizontal (fotos de la carga/remolque), la dejamos intacta a color
        if original_w > original_h:
            print("-> [LOG 4.2] Imagen horizontal detectada. Preservando a color sin recortes...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # -------------------------------------------------------------
        # FILTRADO DE SOMBRAS Y DETECCIÓN DE TINTA EFECTIVA
        # -------------------------------------------------------------
        print("-> [LOG 4.3] Filtrando ruido/sombras y analizando límites reales de tinta...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Suavizado para reducir el ruido de la cámara
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. Binarización adaptativa para aislar texto local e ignorar sombras de fondo
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # 3. Limpieza morfológica: conectar letras/tablas y eliminar puntos aislados del fondo
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 4. Obtener contornos de los bloques de tinta
        contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_x, min_y = original_w, original_h
        max_x, max_y = 0, 0
        valid_ink_found = False

        # Descartamos ruido pequeño (manchas, sombras lejanas) midiendo áreas de los contornos
        min_contour_area = (original_w * original_h) * 0.0005  # Al menos 0.05% del área total

        for c in contours:
            if cv2.contourArea(c) > min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                # Ignorar si el contorno es sospechosamente una línea gigante en todo el borde de la foto
                if w > original_w * 0.98 and h > original_h * 0.98:
                    continue
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                valid_ink_found = True

        if valid_ink_found and (max_x > min_x) and (max_y > min_y):
            ink_w = max_x - min_x
            ink_h = max_y - min_y
            print(f"-> [LOG 4.4] Área de tinta consolidada: x={min_x}, y={min_y}, w={ink_w}, h={ink_h}")

            # ---------------------------------------------------------
            # CÁLCULO DINÁMICO DE 1 CM DE MARGEN (PADDING)
            # ---------------------------------------------------------
            px_per_cm = int(original_h / 29.7)  # Basado en A4 (~29.7 cm)
            if px_per_cm < 25:
                px_per_cm = 35

            print(f"-> [LOG 4.5] Aplicando padding de 1 cm ({px_per_cm} px)...")

            x1 = max(0, min_x - px_per_cm)
            y1 = max(0, min_y - px_per_cm)
            x2 = min(original_w, max_x + px_per_cm)
            y2 = min(original_h, max_y + px_per_cm)

            cropped_doc = img[y1:y2, x1:x2]
            print(f"-> [LOG 4.6] Recorte efectivo realizado: de ({original_w}x{original_h}) a ({x2-x1}x{y2-y1}) px.")
        else:
            print("-> [LOG 4.4 WARNING] No se aislaron bloques de tinta limpios. Usando imagen con margen de resguardo.")
            margin_y = int(original_h * 0.02)
            margin_x = int(original_w * 0.02)
            cropped_doc = img[margin_y:original_h-margin_y, margin_x:original_w-margin_x]

        # Redimensionar estándar y CLAHE
        target_width = 1600
        ch, cw = cropped_doc.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            cropped_doc = cv2.resize(cropped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(cropped_doc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Lienzo blanco corporativo
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.7] Documento procesado y enmarcado con éxito.")
        return canvas

    except Exception as e:
        print(f"🚨 [LOG 4.8 ERROR EN PROCESAMIENTO LOCAL]: {str(e)}")
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
            raise HTTPException(status_code=400, detail="No se recibieron URLs de imágenes.")

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

                processed_page = process_image_by_ink_density(image_bytes)
                pil_images.append(processed_page)
                
            except Exception as img_err:
                print(f"🚨 [LOG 5.5] Error procesando imagen {index+1}: {str(img_err)}")

        if not pil_images:
            raise HTTPException(status_code=400, detail="Ninguna imagen pudo ser procesada.")

        print("-> [LOG 6.0] Compilando PDF final unificado en alta resolución...")
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

        print("-> [LOG 6.1] ¡PDF unificado generado exitosamente sin pérdidas de contenido!")
        return {
            "status": "success",
            "total_pages": len(pil_images),
            "pdf_base64": pdf_base64,
            "message": "PDF generado con éxito usando detección de tinta y padding de 1cm."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"🚨 [LOG 6.2 ERROR CRÍTICO]: {str(e)}")
        raise HTTPException(status_code=500, detail=json.dumps({"error": str(e)}))
