import time

import cv2
import numpy as np
import torch

from Bluetooth import BluetoothCarrito
from Config import (
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 66))
    img_array = np.array(resized, dtype=np.float32) / 255.0
    return img_array


def ejecutar_accion(bt, clase_predicha):
    if clase_predicha == "CRUCE_T":
        print("Cruce en T detectado. Deteniendo 2 segundos...")
        bt.enviar(COMANDO_DETENER)
        time.sleep(2)

        decision = np.random.choice([COMANDO_GIRO_IZQUIERDA, COMANDO_GIRO_DERECHA])
        print(f"Decision tomada: {decision}")
        bt.enviar(decision)
        time.sleep(1.5)
        return

    comando = MAPEO_COMANDOS.get(clase_predicha, COMANDO_DETENER)
    bt.enviar(comando)


def main():
    bt = BluetoothCarrito()
    if not bt.conectar():
        print("Error: No se pudo establecer conexion Bluetooth. Abortando...")
        return

    cap = cv2.VideoCapture(INDICE_CAMARA)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la camara con indice {INDICE_CAMARA}.")
        bt.cerrar()
        return

    fifo_frames = []
    print("Sistema iniciado. Presione 'q' para salir.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la camara.")
                break

            img_processed = preprocesar_frame(frame)
            fifo_frames.append(img_processed)

            if len(fifo_frames) == 3:
                input_tensor = torch.from_numpy(np.array(fifo_frames)).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = modelo(input_tensor)
                    probabilidad = torch.softmax(output, dim=1)
                    confianza, pred = torch.max(probabilidad, 1)
                    clase_nombre = CLASES_NAVEGACION[pred.item()]

                ejecutar_accion(bt, clase_nombre)

                texto = f"IA: {clase_nombre} ({confianza.item() * 100:.1f}%)"
                cv2.putText(
                    frame,
                    texto,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                fifo_frames.pop(0)

            cv2.imshow("CarritoIA - Control Autonomo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("Cerrando sistema...")
        bt.enviar(COMANDO_DETENER)
        bt.cerrar()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
