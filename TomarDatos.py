import os

import cv2

from Config import (
    ALTO_CAMARA,
    ANCHO_CAMARA,
    CADA_CUANTOS_FRAMES_GUARDAR,
    CLASES_BONUS,
    CLASES_NAVEGACION,
    INDICE_CAMARA,
    RUTA_DATA_BONUS,
    RUTA_DATA_NAVEGACION,
    crear_carpetas,
)


def mostrar_menu_principal():
    print("\n=== TOMAR DATOS ===")
    print("1. Capturar datos de navegacion")
    print("2. Capturar datos del bonus")
    print("0. Salir")
    return input("Seleccione una opcion: ").strip()


def elegir_clase(titulo, clases):
    while True:
        print(f"\n=== {titulo} ===")
        for i, clase in enumerate(clases, start=1):
            print(f"{i}. {clase}")
        print("0. Volver")

        opcion = input("Seleccione una clase: ").strip()

        if opcion == "0":
            return None

        if opcion.isdigit():
            indice = int(opcion) - 1
            if 0 <= indice < len(clases):
                return clases[indice]

        print("Opcion invalida. Intente de nuevo.")


def contar_imagenes(carpeta):
    extensiones = (".jpg", ".jpeg", ".png", ".bmp")
    total = 0

    if not os.path.exists(carpeta):
        return 0

    for nombre in os.listdir(carpeta):
        if nombre.lower().endswith(extensiones):
            total += 1

    return total


def siguiente_numero(carpeta, clase):
    mayor = 0

    if not os.path.exists(carpeta):
        return 1

    for nombre in os.listdir(carpeta):
        if not nombre.lower().endswith(".jpg"):
            continue

        inicio = f"{clase}_"
        if not nombre.startswith(inicio):
            continue

        numero_texto = nombre.replace(inicio, "").replace(".jpg", "")
        if numero_texto.isdigit():
            mayor = max(mayor, int(numero_texto))

    return mayor + 1


def dibujar_texto(frame, tipo_dataset, clase, estado, existentes, nuevas, total):
    textos = [
        f"Dataset: {tipo_dataset}",
        f"Clase: {clase}",
        f"Estado: {estado}",
        f"Existian antes: {existentes}",
        f"Guardadas esta sesion: {nuevas}",
        f"Total actual: {total}",
        "espacio = pausar/reanudar",
        "q = terminar captura",
    ]

    y = 25
    for texto in textos:
        cv2.putText(
            frame,
            texto,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y += 28


def capturar_clase(tipo_dataset, ruta_base, clase):
    carpeta_clase = os.path.join(ruta_base, clase)
    os.makedirs(carpeta_clase, exist_ok=True)

    existentes = contar_imagenes(carpeta_clase)
    numero_actual = siguiente_numero(carpeta_clase, clase)
    nuevas = 0
    frame_contador = 0
    pausado = False

    camara = cv2.VideoCapture(INDICE_CAMARA)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)

    if not camara.isOpened():
        print(f"Error: no se pudo abrir la camara con indice {INDICE_CAMARA}.")
        return

    print(f"\nCapturando dataset: {tipo_dataset}")
    print(f"Clase seleccionada: {clase}")
    print(f"Carpeta: {carpeta_clase}")
    print("Presione espacio para pausar/reanudar.")
    print("Presione q para terminar la captura.\n")

    while True:
        correcto, frame = camara.read()

        if not correcto:
            print("Error: no se pudo leer imagen desde la camara.")
            break

        total_actual = existentes + nuevas

        if not pausado:
            frame_contador += 1

            if frame_contador % CADA_CUANTOS_FRAMES_GUARDAR == 0:
                nombre_archivo = f"{clase}_{numero_actual:06d}.jpg"
                ruta_archivo = os.path.join(carpeta_clase, nombre_archivo)
                cv2.imwrite(ruta_archivo, frame)

                nuevas += 1
                total_actual = existentes + nuevas
                print(f"Imagen guardada: {ruta_archivo}")

                numero_actual += 1

        estado = "pausado" if pausado else "capturando"
        frame_mostrar = frame.copy()
        dibujar_texto(
            frame_mostrar,
            tipo_dataset,
            clase,
            estado,
            existentes,
            nuevas,
            total_actual,
        )

        cv2.imshow("Tomar Datos", frame_mostrar)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == ord(" "):
            pausado = not pausado

    camara.release()
    cv2.destroyAllWindows()

    print("\n=== RESUMEN DE CAPTURA ===")
    print(f"Tipo de dataset: {tipo_dataset}")
    print(f"Clase capturada: {clase}")
    print(f"Imagenes que ya existian antes: {existentes}")
    print(f"Imagenes nuevas capturadas: {nuevas}")
    print(f"Total final de esa clase: {existentes + nuevas}")
    print(f"Carpeta donde se guardaron: {carpeta_clase}")


def main():
    crear_carpetas()

    while True:
        opcion = mostrar_menu_principal()

        if opcion == "0":
            print("Saliendo de TomarDatos.py.")
            break

        if opcion == "1":
            clase = elegir_clase(
                "CLASE DE NAVEGACION A CAPTURAR",
                CLASES_NAVEGACION,
            )
            if clase is not None:
                capturar_clase("navegacion", RUTA_DATA_NAVEGACION, clase)

        elif opcion == "2":
            clase = elegir_clase(
                "CLASE DEL BONUS A CAPTURAR",
                CLASES_BONUS,
            )
            if clase is not None:
                capturar_clase("bonus", RUTA_DATA_BONUS, clase)

        else:
            print("Opcion invalida. Intente de nuevo.")


if __name__ == "__main__":
    main()
