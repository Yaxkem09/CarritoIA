import cv2
import numpy as np
import torch
import time
from modelo_navegacion import ModeloNavegacion
from bluetooth import BluetoothCarrito # Importamos tu clase personalizada
from Config import COMANDO_DETENER # Asumiendo que 'K' está en tu Config

# 1. Configuración de Dispositivos y Carga de Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ModeloNavegacion(num_clases=6).to(device)
modelo.load_state_dict(torch.load('modelos/modelo_final.pth', map_location=device))
modelo.eval()

# Inicializar y conectar Bluetooth
bt = BluetoothCarrito()
if not bt.conectar():
    print("Error: No se pudo establecer conexión Bluetooth. Abortando...")
    exit()

# Mapeo de Clases de la IA a los comandos que entiende tu Arduino (según tu bluetooth.py)
# Orden: RECTA, CURVA_I, CURVA_D, GIRO_90_I, GIRO_90_D, CRUCE_T
MAPEO_COMANDOS = {
    "RECTA": "I",
    "CURVA_I": "J",
    "CURVA_D": "L",
    "GIRO_90_I": "U",
    "GIRO_90_D": "O",
    "CRUCE_T": "CRUCE_T" # Caso especial para lógica
}

CLASES = ["RECTA", "CURVA_I", "CURVA_D", "GIRO_90_I", "GIRO_90_D", "CRUCE_T"]

# 2. Preprocesamiento Matricial Manual (REGLA ESTRICTA)
def preprocesar_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 66))

    # Normalización manual con NumPy
    img_array = np.array(resized, dtype=np.float32) / 255.0
    return img_array

# 3. Lógica de Decisión y Control (Cruce en T)
def ejecutar_accion(clase_predicha):
    if clase_predicha == "CRUCE_T":
        print("¡Cruce en T detectado! Deteniendo 2 seg...")
        bt.enviar(COMANDO_DETENER) # Envía 'K'
        time.sleep(2)

        # Decisión pseudoaleatoria requerida
        decision = np.random.choice(["U", "O"]) # Giro fuerte I o D
        print(f"Decisión tomada: {decision}")
        bt.enviar(decision)
        time.sleep(1.5) # Tiempo para completar maniobra
    else:
        comando = MAPEO_COMANDOS.get(clase_predicha)
        bt.enviar(comando)

# 4. Bucle Principal de Inferencia
cap = cv2.VideoCapture(0)
fifo_frames = []

print("Sistema iniciado. Presione 'q' para salir.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_processed = preprocesar_frame(frame)
        fifo_frames.append(img_processed)

        if len(fifo_frames) == 3:
            # Preparar tensor para el modelo (Batch, Canales/Frames, H, W)
            input_tensor = torch.from_numpy(np.array(fifo_frames)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = modelo(input_tensor)
                _, pred = torch.max(output, 1)
                clase_nombre = CLASES[pred.item()]

            ejecutar_accion(clase_nombre)

            # Feedback visual
            cv2.putText(frame, f"IA: {clase_nombre}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            fifo_frames.pop(0)

        cv2.imshow('CarritoIA - Control Autónomo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    print("Cerrando sistema...")
    bt.enviar(COMANDO_DETENER)
    bt.cerrar()
    cap.release()
    cv2.destroyAllWindows()