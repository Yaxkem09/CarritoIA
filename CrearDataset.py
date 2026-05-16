import os
import cv2
import numpy as np


def aplicar_filtro_sobel_normalizado(img_gris):
    """
    Preprocesamiento manual con NumPy.
    Aplica kernels Sobel y normaliza el gradiente para evitar imágenes negras.
    """
    # Definición de kernels Sobel en NumPy
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    # Convolución manual
    gx = cv2.filter2D(img_gris, cv2.CV_32F, Kx)
    gy = cv2.filter2D(img_gris, cv2.CV_32F, Ky)

    # Magnitud del gradiente
    gradiente = np.sqrt(gx ** 2 + gy ** 2)

    # Normalización para mejorar contraste
    gradiente = cv2.normalize(gradiente, None, 0, 255, cv2.NORM_MINMAX)

    return gradiente.astype(np.uint8)


def procesar_imagenes_a_dataset(ruta_origen, ruta_destino):
    clases = ["RECTA", "CURVA_IZQUIERDA", "CURVA_DERECHA",
              "GIRO_90_IZQ", "GIRO_90_DER", "CRUCE_T"]

    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)
        print(f"Carpeta de destino '{ruta_destino}' creada.")

    for clase in clases:
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
            if img is None:
                continue

            # Preprocesamiento requerido
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (200, 66))  # Resolución para ModeloNavegacion
            processed = aplicar_filtro_sobel_normalizado(resized)

            cv2.imwrite(os.path.join(target_dir, img_nombre), processed)

        print(f"   -> {len(archivos)} imágenes procesadas correctamente.")


if __name__ == "__main__":
    procesar_imagenes_a_dataset('.', 'dataset_procesado')
    print("\n--- ¡Dataset creado exitosamente! ---")