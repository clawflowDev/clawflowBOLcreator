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

app = FastAPI(title="CLAWFLOW Ink Density & Margin Engine", version="21.0")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "CLAWFLOW Ink Density Engine Active",
        "mode": "1cm Margin Post-Ink Detection + A4 Canvas"
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

        # Si la imagen es claramente horizontal (fotos de la carga/remolque), la dejamos intacta a color
        if original_w > original_h:
            print("-> [LOG 4.2] Imagen horizontal detectada. Preservando a color sin recortes...")
            target_w = 1600
            if original_w > target_w:
                ratio = target_w / float(original_w)
                img = cv2.resize(img, (target_w, int(original_h * ratio)), interpolation=cv2.INTER_AREA)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_img)

        # -------------------------------------------------------------
        # DETECCIÓN BASADA EN DENSIDAD DE TINTA Y ELEMENTOS GRÁFICOS
        # -------------------------------------------------------------
        print("-> [LOG 4.3] Analizando límites de tinta y elementos gráficos...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicamos un umbral para resaltar los elementos oscuros (texto, líneas, sellos)
        # Binarización invertida: Tinta = Blanco (255), Fondo = Negro (0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Encontrar todas las coordenadas donde hay tinta / elementos gráficos
        points = cv2.findNonZero(thresh)

        if points is not None:
            # Obtener el rectángulo delimitador exacto de la tinta
            x, y, w, h = cv2.boundingRect(points)
            print(f"-> [LOG 4.4] Bounding Box de tinta encontrado: x={x}, y={y}, w={w}, h={h}")

            # ---------------------------------------------------------
            # CÁLCULO DINÁMICO DE 1 CM DE MARGEN (PADDING)
            # ---------------------------------------------------------
            # Asumiendo una resolución promedio estándar de 150 a 300 DPI en fotos de móvil:
            # 1 pulgada = 2.54 cm. Estimamos los px/cm proporcionalmente a la altura de la imagen.
            px_per_cm = int(original_h / 29.7)  # Basado en la altura de una hoja A4 (~29.7 cm)
            if px_per_cm < 20:
                px_per_cm = 30  # Valor mínimo de resguardo
                
            print(f"-> [LOG 4.5] Margen de 1 cm calculado: {px_per_cm} píxeles de padding.")

            # Aplicar el margen de 1 cm hacia afuera de los límites de la tinta
            x1 = max(0, x - px_per_cm)
            y1 = max(0, y - px_per_cm)
            x2 = min(original_w, x + w + px_per_cm)
            y2 = min(original_h, y + h + px_per_cm)

            cropped_doc = img[y1:y2, x1:x2]
            print(f"-> [LOG 4.6] Recorte final aplicado con 1 cm de distancia alrededor de la tinta.")
        else:
            print("-> [LOG 4.4 WARNING] No se detectó tinta clara. Usando imagen completa sin recorte.")
            cropped_doc = img

        # Redimensionar estándar y aplicar CLAHE para nitidez de texto
        target_width = 1600
        ch, cw = cropped_doc.shape[:2]
        if cw > target_width:
            ratio = target_width / float(cw)
            new_h = int(ch * ratio)
            cropped_doc = cv2.resize(cropped_doc, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # Realce de contraste CLAHE (resalta la tinta sobre papel arrugado o sombreado)
        lab = cv2.cvtColor(cropped_doc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_color = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        doc_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        doc_pil = Image.fromarray(doc_rgb)

        # Lienzo blanco corporativo con margen estético adicional
        dw, dh = doc_pil.size
        canvas = Image.new("RGB", (dw + 60, dh + 60), (255, 255, 255))
        canvas.paste(doc_pil, (30, 30))

        print("-> [LOG 4.7] Documento enmarcado exitosamente sobre lienzo corporativo.")
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
