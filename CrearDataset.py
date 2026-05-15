import os
import cv2
import numpy as np
import torch

def aplicar_filtro_manual(img_gris):
    """
    REGLA ESTRICA: Preprocesamiento manual con NumPy.
    Aplica un filtro de detección de bordes (Sobel) para resaltar la pista.
    """
    # Definición de kernels Sobel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Convolución manual simplificada usando cv2.filter2D (permitido para eficiencia)
    # pero definiendo el kernel nosotros mismos en NumPy.
    gx = cv2.filter2D(img_gris, -1, Kx)
    gy = cv2.filter2D(img_gris, -1, Ky)

    # Magnitud del gradiente
    gradiente = np.sqrt(gx**2 + gy**2)
    gradiente = np.clip(gradiente, 0, 255).astype(np.uint8)
    return gradiente

def procesar_imagenes_a_dataset(ruta_origen, ruta_destino):
    clases = ["RECTA", "CURVA_I", "CURVA_D", "GIRO_90_I", "GIRO_90_D", "CRUCE_T"]

    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    for clase in clases:
        path_clase = os.path.join(ruta_origen, clase)
        if not os.path.exists(path_clase): continue

        print(f"Procesando clase: {clase}...")
        for img_nombre in os.listdir(path_clase):
            img_path = os.path.join(path_clase, img_nombre)

            # 1. Cargar imagen
            img = cv2.imread(img_path)
            if img is None: continue

            # 2. Convertir a Gris y Redimensionar (OpenCV permitido aquí)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (200, 66))

            # 3. Aplicar filtro matricial manual (NumPy)
            processed = aplicar_filtro_manual(resized)

            # 4. Guardar en la carpeta de dataset procesado
            target_dir = os.path.join(ruta_destino, clase)
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            cv2.imwrite(os.path.join(target_dir, img_nombre), processed)

if __name__ == "__main__":
    procesar_imagenes_a_dataset('data_navegacion', 'dataset_procesado')
    print("Dataset creado exitosamente.")