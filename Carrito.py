import time
from collections import deque, Counter

import cv2
import numpy as np
import torch

from Bluetooth import BluetoothCarrito
from CrearDataset import aplicar_filtro_sobel_normalizado
from Config import (
    CADA_CUANTOS_FRAMES_GUARDAR,
    CLASES_NAVEGACION,
    COMANDO_AVANZAR,
    COMANDO_DERECHA,
    COMANDO_DETENER,
    COMANDO_GIRO_DERECHA,
    COMANDO_GIRO_IZQUIERDA,
    COMANDO_IZQUIERDA,
    INDICE_CAMARA,
    RUTA_MODELOS,
)
from ModeloNavegacion import ModeloNavegacion


USAR_BLUETOOTH = True    # True para la prueba con el robot
TAM_VENTANA = 3          # frames por ventana del modelo (= entrenamiento)

# --- Control del CRUCE_T (lo importante) ---
TAM_VOTO = 3             # voto corto para la direccion (responsivo)
VENTANA_CRUCE = 6        # predicciones recientes que se vigilan para el cruce
MIN_FRAMES_CRUCE = 4     # cuantas de esas deben ser CRUCE_T para CONFIRMAR
CONF_MIN_CRUCE = 0.70    # confianza minima para que un frame cuente como cruce
CONFIRMACIONES_CURVA = 4  # para cambiar a una curva, se necesitan esta cantidad de predicciones seguidass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ModeloNavegacion(num_clases=len(CLASES_NAVEGACION)).to(device)
modelo.load_state_dict(
    torch.load(f"{RUTA_MODELOS}/modelo_final.pth", map_location=device)
)
modelo.eval()


MAPEO_COMANDOS = {
    "RECTA": COMANDO_AVANZAR,
    "CURVA_IZQUIERDA": COMANDO_IZQUIERDA,
    "CURVA_DERECHA": COMANDO_DERECHA,
    "GIRO_90_IZQ": COMANDO_GIRO_IZQUIERDA,
    "GIRO_90_DER": COMANDO_GIRO_DERECHA,
}


def preprocesar_frame(frame):
    """MISMO preprocesamiento que el entrenamiento (sin augmentation)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 66))
    img_filtrada = aplicar_filtro_sobel_normalizado(resized)
    return np.array(img_filtrada, dtype=np.float32) / 255.0


def ejecutar_accion(bt, clase_predicha):
    if clase_predicha == "CRUCE_T":
        print("Cruce en T detectado. Deteniendo 2 segundos...")
        t_fin = time.time() + 2.0
        while time.time() < t_fin:
            if USAR_BLUETOOTH:
                bt.enviar(COMANDO_DETENER)
            time.sleep(0.1)

        decision = np.random.choice([COMANDO_GIRO_IZQUIERDA, COMANDO_GIRO_DERECHA])
        print(f"Decision tomada: {decision}")
        # IMPORTANTE: reenviar el giro continuamente. Si se envia una sola
        # vez, el timeout de seguridad del Arduino (500 ms) lo corta.
        t_fin = time.time() + 1.5
        while time.time() < t_fin:
            if USAR_BLUETOOTH:
                bt.enviar(decision)
            time.sleep(0.1)
        return

    comando = MAPEO_COMANDOS.get(clase_predicha, COMANDO_DETENER)
    if USAR_BLUETOOTH:
        bt.enviar(comando)


def main():
    bt = BluetoothCarrito()
    if USAR_BLUETOOTH and not bt.conectar():
        print("Error: No se pudo establecer conexion Bluetooth. Abortando...")
        return

    cap = cv2.VideoCapture(INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la camara con indice {INDICE_CAMARA}.")
        if USAR_BLUETOOTH:
            bt.cerrar()
        return

    fifo_frames = deque(maxlen=TAM_VENTANA)
    hist_cruce = deque(maxlen=VENTANA_CRUCE)   # ventana larga para el CRUCE_T

    # Debounce de la direccion.
    clase_manejo = "RECTA"      # comando que se esta ejecutando
    clase_candidata = "RECTA"   # clase nueva que aun no se confirma
    contador_candidata = 0      # veces seguidas que aparecio la candidata

    confianza_val = 0.0
    contador_frames = 0

    print("Sistema iniciado. Presione 'q' para salir.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la camara.")
                break

            contador_frames += 1

            if contador_frames % CADA_CUANTOS_FRAMES_GUARDAR == 0:
                fifo_frames.append(preprocesar_frame(frame))

                if len(fifo_frames) == TAM_VENTANA:
                    input_tensor = (
                        torch.from_numpy(np.stack(fifo_frames, axis=0))
                        .unsqueeze(0)
                        .to(device)
                    )
                    with torch.no_grad():
                        output = modelo(input_tensor)
                        probabilidad = torch.softmax(output, dim=1)
                        confianza, pred = torch.max(probabilidad, 1)
                        clase_cruda = CLASES_NAVEGACION[pred.item()]
                        confianza_val = confianza.item()

                    # --- Vigilancia del CRUCE_T (estricta) ---
                    es_cruce = (clase_cruda == "CRUCE_T"
                                and confianza_val >= CONF_MIN_CRUCE)
                    hist_cruce.append(es_cruce)

                    # --- Debounce de la direccion (suave) ---
                    # El comando solo cambia si una clase distinta a la actual
                    # se repite CONFIRMACIONES_CURVA veces seguidas.
                    if clase_cruda != "CRUCE_T":
                        if clase_cruda == clase_manejo:
                            contador_candidata = 0          # ya estamos ahi
                        elif clase_cruda == clase_candidata:
                            contador_candidata += 1
                            if contador_candidata >= CONFIRMACIONES_CURVA:
                                clase_manejo = clase_candidata
                                contador_candidata = 0
                        else:
                            clase_candidata = clase_cruda
                            contador_candidata = 1

                    # --- Decision final ---
                    n_cruce = sum(hist_cruce)
                    if (len(hist_cruce) == VENTANA_CRUCE
                            and n_cruce >= MIN_FRAMES_CRUCE):
                        print(f"CRUCE en T CONFIRMADO ({n_cruce}/{VENTANA_CRUCE}).")
                        ejecutar_accion(bt, "CRUCE_T")
                        hist_cruce.clear()
                        fifo_frames.clear()
                        clase_manejo = "RECTA"
                        clase_candidata = "RECTA"
                        contador_candidata = 0
                    else:
                        ejecutar_accion(bt, clase_manejo)

            texto = (f"{clase_manejo}  conf:{confianza_val * 100:.0f}%  "
                     f"cruceT:{sum(hist_cruce)}/{VENTANA_CRUCE}")
            cv2.putText(frame, texto, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("CarritoIA - Control Autonomo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("Cerrando sistema...")
        if USAR_BLUETOOTH:
            bt.enviar(COMANDO_DETENER)
            bt.cerrar()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()