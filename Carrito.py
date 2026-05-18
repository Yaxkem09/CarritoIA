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


# Poner en True el dia de la demostracion (robot conectado).
USAR_BLUETOOTH = True
TAM_VENTANA = 3
# Predicciones recientes usadas en el voto por mayoria. Mas alto = mas
# estable pero reacciona mas lento. Bajalo a 3 si reacciona tarde.
TAM_VOTO = 5


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
    historial_pred = deque(maxlen=TAM_VOTO)   # para el voto por mayoria
    contador_frames = 0
    clase_estable = "..."
    confianza_val = 0.0

    print("Sistema iniciado. Presione 'q' para salir.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la camara.")
                break

            contador_frames += 1

            # 1 de cada CADA_CUANTOS_FRAMES_GUARDAR frames -> mismo espaciado
            # temporal que el dataset.
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

                    # Voto por mayoria: suaviza errores sueltos del modelo y
                    # evita que el robot oscile entre comandos contradictorios.
                    historial_pred.append(clase_cruda)
                    clase_estable = Counter(historial_pred).most_common(1)[0][0]

                    ejecutar_accion(bt, clase_estable)

                    # Tras resolver un cruce en T, limpiar los buffers para no
                    # volver a dispararlo con votos viejos.
                    if clase_estable == "CRUCE_T":
                        historial_pred.clear()
                        fifo_frames.clear()

            texto = f"IA: {clase_estable} ({confianza_val * 100:.1f}%)"
            cv2.putText(frame, texto, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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