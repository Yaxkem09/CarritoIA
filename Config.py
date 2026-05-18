import os


# Clases para la navegacion base del carrito.
CLASES_NAVEGACION = [
    "RECTA",
    "CURVA_IZQUIERDA",
    "CURVA_DERECHA",
    "GIRO_90_IZQ",
    "GIRO_90_DER",
    "CRUCE_T",
]


# Clases para el reto bonus.
CLASES_BONUS = [
    "ALTO",
    "REDUCCION_VELOCIDAD",
    "PEATONES",
    "NINGUNA_SENAL",
]


# Rutas principales del proyecto.
RUTA_DATA_NAVEGACION = "data_navegacion"
RUTA_DATA_BONUS = "data_bonus"
RUTA_DATASET = "dataset"
RUTA_MODELOS = "modelos"
RUTA_RESULTADOS = "resultados"


# Parametros de camara para tomar datos.
INDICE_CAMARA = 1
ANCHO_CAMARA = 640
ALTO_CAMARA = 480
CADA_CUANTOS_FRAMES_GUARDAR = 5


# Parametros de Bluetooth para comunicarse con Arduino.
PUERTO_BLUETOOTH = "COM8"
BAUDRATE = 9600
TIMEOUT_BLUETOOTH = 1


# Comandos Bluetooth que Arduino interpretara para mover el carrito.
COMANDO_AVANZAR = "I"
COMANDO_IZQUIERDA = "J"
COMANDO_DERECHA = "L"
COMANDO_GIRO_IZQUIERDA = "U"
COMANDO_GIRO_DERECHA = "O"
COMANDO_DETENER = "K"
COMANDO_VELOCIDAD_BAJA = "1"
COMANDO_VELOCIDAD_NORMAL = "2"


def crear_carpetas():
    """Crea las carpetas del proyecto sin borrar datos existentes."""

    for clase in CLASES_NAVEGACION:
        os.makedirs(os.path.join(RUTA_DATA_NAVEGACION, clase), exist_ok=True)

    for clase in CLASES_BONUS:
        os.makedirs(os.path.join(RUTA_DATA_BONUS, clase), exist_ok=True)

    os.makedirs(RUTA_DATASET, exist_ok=True)
    os.makedirs(RUTA_MODELOS, exist_ok=True)
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
