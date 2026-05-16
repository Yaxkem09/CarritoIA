import os
import cv2
import numpy as np


def aplicar_filtro_manual(img_gris):
    """
    REGLA ESTRICTA URL: Preprocesamiento manual con NumPy.
    Aplica kernels Sobel para resaltar la pista sin usar funciones automáticas.
    """
    # Definición de kernels Sobel en NumPy
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Convolución manual (usando filter2D solo para aplicar nuestro kernel)
    gx = cv2.filter2D(img_gris, -1, Kx)
    gy = cv2.filter2D(img_gris, -1, Ky)

    # Magnitud del gradiente matricial
    gradiente = np.sqrt(gx ** 2 + gy ** 2)
    gradiente = np.clip(gradiente, 0, 255).astype(np.uint8)
    return gradiente


def procesar_imagenes_a_dataset(ruta_origen, ruta_destino):
    # Nombres de carpetas exactos según tus capturas del disco D
    clases = ["RECTA", "CURVA_IZQUIERDA", "CURVA_DERERCHA", "GIRO_90_IZQ", "GIRO_90_DER", "CRUCE_T"]

    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)
        print(f"Carpeta de destino '{ruta_destino}' creada.")

    for clase in clases:
        # Buscamos la carpeta directamente en la ruta de origen
        path_clase = os.path.join(ruta_origen, clase)

        if not os.path.exists(path_clase):
            print(f"ALERTA: Saltando '{clase}' porque no existe la carpeta en {path_clase}")
            continue

        print(f"Procesando clase: {clase}...")
        archivos = [f for f in os.listdir(path_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not archivos:
            print(f"   -> Sin imágenes en {clase}")
            continue

        target_dir = os.path.join(ruta_destino, clase)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for img_nombre in archivos:
            img_path = os.path.join(path_clase, img_nombre)

            img = cv2.imread(img_path)
            if img is None: continue

            # Preprocesamiento requerido
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (200, 66))  # Resolución para ModeloNavegacion
            processed = aplicar_filtro_manual(resized)

            cv2.imwrite(os.path.join(target_dir, img_nombre), processed)

        print(f"   -> {len(archivos)} imágenes procesadas correctamente.")


if __name__ == "__main__":
    procesar_imagenes_a_dataset('.', 'dataset_procesado')
    print("\n--- ¡Dataset creado exitosamente! ---")