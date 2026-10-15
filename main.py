from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import uvicorn

app = FastAPI(title="CLAWFLOW BOL & POD Master Engine")

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def aplicar_efecto_escaner(imagen_bytes):
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break
            
    if doc_cnt is not None:
        pts = doc_cnt.reshape(4, 2)
        rect = ordenar_puntos(pts)
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
        warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
    else:
        warped = gray
        
    scanner_effect = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    
    return Image.fromarray(scanner_effect).convert("RGB")

@app.get("/")
def read_root():
    return {"status": "online", "system": "CLAWFLOW BOL & POD Master Engine"}

@app.post("/procesar-bol-pod")
async def procesar_bol_pod(files: list[UploadFile] = File(...)):
    imagenes_bol_procesadas = []
    imagenes_carga_anexos = []
    
    for file in files:
        contents = await file.read()
        filename_lower = file.filename.lower()
        
        if any(term in filename_lower for term in ["carga", "cargo", "truck", "pallet", "anexo"]):
            img_carga = Image.open(io.BytesIO(contents)).convert("RGB")
            imagenes_carga_anexos.append(img_carga)
        else:
            img_procesada = aplicar_efecto_escaner(contents)
            if img_procesada:
                imagenes_bol_procesadas.append(img_procesada)
            else:
                img_fallback = Image.open(io.BytesIO(contents)).convert("RGB")
                imagenes_bol_procesadas.append(img_fallback)

    todas_las_paginas = imagenes_bol_procesadas + imagenes_carga_anexos
    
    if not todas_las_paginas:
        return JSONResponse(status_code=400, content={"error": "No se procesó ninguna imagen válida."})
        
    pdf_buffer = io.BytesIO()
    primera_pagina = todas_las_paginas[0]
    resto_paginas = todas_las_paginas[1:] if len(todas_las_paginas) > 1 else []
    
    primera_pagina.save(
        pdf_buffer, 
        format="PDF", 
        save_all=True, 
        append_images=resto_paginas,
        resolution=100.0,
        quality=85
    )
    pdf_bytes = pdf_buffer.getvalue()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    return {
        "status": "success",
        "message": "PDF unificado y optimizado con éxito",
        "total_paginas_bol": len(imagenes_bol_procesadas),
        "total_anexos_carga": len(imagenes_carga_anexos),
        "pdf_base64": pdf_base64
    }
