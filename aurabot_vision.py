import cv2
import matplotlib.pyplot as plt


def detectar_rostro(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("No se pudo cargar el clasificador Haar de rostros de OpenCV.")

    rostros = detector.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(rostros) == 0:
        return img_rgb, None

    x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
    margen_x = int(w * 0.15)
    margen_y = int(h * 0.20)

    x1 = max(x - margen_x, 0)
    y1 = max(y - margen_y, 0)
    x2 = min(x + w + margen_x, img_rgb.shape[1])
    y2 = min(y + h + margen_y, img_rgb.shape[0])

    return img_rgb, (x1, y1, x2, y2)

def pipeline_aurabot(image_path):
    print("="*60)
    print(" INICIANDO PIPELINE DE VISIÓN - AURABOT ")
    print("="*60)
    print("[AURABOT] Módulo B: Cargando detector facial de OpenCV...")

    print(f"[AURABOT] Escaneando imagen: {image_path}")
    img_rgb, rostro_bbox = detectar_rostro(image_path)

    if rostro_bbox is None:
        print("[Error] No se detectó ningún rostro frontal. Prueba con una foto más frontal y bien iluminada.")
        return

    x1, y1, x2, y2 = rostro_bbox
    print("[AURABOT] -> Rostro localizado. Extrayendo Región de Interés (ROI)...")

    img_rostro = img_rgb.copy()
    cv2.rectangle(img_rostro, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(img_rostro, "AURABOT: Rostro", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    roi_recortada = img_rgb[y1:y2, x1:x2]
    roi_gris = cv2.cvtColor(roi_recortada, cv2.COLOR_RGB2GRAY)
    
    print("[AURABOT] Módulo A: Generando mapa estructural del rostro...")
    
    roi_filtrada = cv2.bilateralFilter(roi_gris, 9, 75, 75)
    grad_x = cv2.Sobel(roi_filtrada, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_filtrada, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(grad_x, grad_y)
    magnitud_norm = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, mascara_estructural = cv2.threshold(magnitud_norm, 25, 255, cv2.THRESH_BINARY)
    
    print("[AURABOT] Procesamiento exitoso. Generando dashboard visual...")

    # ================= VISUALIZACIÓN DE ALTO IMPACTO =================
    plt.figure(figsize=(15, 5))
    plt.suptitle("AURABOT PIPELINE: Detección Facial + Mitigación de Sesgo Geriátrico", fontsize=16, fontweight='bold')

    # Imagen 1: Detección facial
    plt.subplot(1, 3, 1)
    plt.title("1. Detección Facial")
    plt.imshow(img_rostro)
    plt.axis('off')

    # Imagen 2: El rostro recortado
    plt.subplot(1, 3, 2)
    plt.title("2. Extracción del Rostro (ROI)")
    plt.imshow(roi_gris, cmap='gray')
    plt.axis('off')

    # Imagen 3: El mapa estructural del rostro
    plt.subplot(1, 3, 3)
    plt.title("3. Tensor Estructural Facial")
    plt.imshow(mascara_estructural, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pipeline_aurabot("rostro_anciano.jpg")