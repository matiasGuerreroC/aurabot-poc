import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def pipeline_aurabot(image_path):
    print("="*60)
    print(" INICIANDO PIPELINE DE VISIÓN MULTIMODAL - AURABOT ")
    print("="*60)
    print("[AURABOT] Módulo B: Cargando motor de inferencia YOLOv8...")
    
    # 1. Cargar modelo YOLOv8 Nano (se descarga solo la primera vez que lo corres)
    model = YOLO('yolov8n.pt') 
    
    print(f"[AURABOT] Escaneando imagen: {image_path}")
    # 2. Inferencia: YOLO detecta los objetos en la foto original
    resultados = model(image_path)
    
    # Cargar la imagen con OpenCV para manipularla
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extraer las cajas delimitadoras (Bounding Boxes)
    cajas = resultados[0].boxes
    if len(cajas) == 0:
        print("[Error] YOLOv8 no detectó a ninguna persona. Intenta con otra foto.")
        return

    # Tomamos la primera detección (asumiendo que es la persona mayor)
    # xyxy saca las coordenadas: X_inicial, Y_inicial, X_final, Y_final
    x1, y1, x2, y2 = map(int, cajas[0].xyxy[0])
    
    print("[AURABOT] -> Sujeto localizado. Extrayendo Región de Interés (ROI)...")
    
    # 3. Dibujar el Bounding Box de YOLO en la foto original para la demo
    img_yolo = img_rgb.copy()
    cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (255, 0, 0), 3) # Caja roja
    cv2.putText(img_yolo, "AURABOT: Residente", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 4. Recortar solo el rostro/cuerpo detectado (El ROI)
    roi_recortada = img_rgb[y1:y2, x1:x2]
    roi_gris = cv2.cvtColor(roi_recortada, cv2.COLOR_RGB2GRAY)
    
    print("[AURABOT] Módulo A: Aplicando mitigación de sesgo geriátrico al ROI...")
    
    # 5. Tu código estrella: Filtro Bilateral y Gradientes de Sobel (Umbral 25)
    roi_filtrada = cv2.bilateralFilter(roi_gris, 9, 75, 75)
    grad_x = cv2.Sobel(roi_filtrada, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_filtrada, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(grad_x, grad_y)
    magnitud_norm = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Aplicando tu ajuste maestro de 25
    _, mascara_estructural = cv2.threshold(magnitud_norm, 25, 255, cv2.THRESH_BINARY)
    
    print("[AURABOT] Procesamiento exitoso. Generando dashboard visual...")

    # ================= VISUALIZACIÓN DE ALTO IMPACTO =================
    plt.figure(figsize=(15, 5))
    plt.suptitle("AURABOT PIPELINE: YOLOv8 + Mitigación de Sesgo Geriátrico", fontsize=16, fontweight='bold')

    # Imagen 1: YOLO actuando
    plt.subplot(1, 3, 1)
    plt.title("1. Detección Espacial (YOLOv8)")
    plt.imshow(img_yolo)
    plt.axis('off')

    # Imagen 2: El recorte
    plt.subplot(1, 3, 2)
    plt.title("2. Extracción de Datos (ROI)")
    plt.imshow(roi_gris, cmap='gray')
    plt.axis('off')

    # Imagen 3: Tu filtro de gradientes
    plt.subplot(1, 3, 3)
    plt.title("3. Tensor Estructural Limpio")
    plt.imshow(mascara_estructural, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Usa la misma imagen original que tenías (no la recortada por ti, deja que YOLO la recorte)
    pipeline_aurabot("rostro_anciano.jpg")