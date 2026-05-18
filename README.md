# CarritoIA

Proyecto universitario de Inteligencia Artificial para un carrito autonomo con vision por computadora.

El sistema usa una camara conectada a la computadora para capturar imagenes, preprocesarlas, ejecutar modelos de IA entrenados desde cero y enviar comandos simples por Bluetooth a un Arduino. El Arduino interpreta esos comandos y controla los motores mediante un modulo L298N.

## Objetivo

Desarrollar un carrito capaz de:

- Reconocer patrones de navegacion en una pista.
- Clasificar el movimiento que debe realizar: avanzar, curva, giro o detenerse ante un cruce.
- Enviar comandos al Arduino por Bluetooth.
- Entrenar modelos propios sin usar datasets publicos ni modelos preentrenados.

## Restricciones del Proyecto

- No usar datasets publicos.
- No usar modelos preentrenados.
- No usar transferencia de aprendizaje.
- No usar MediaPipe, YOLO, ResNet, MobileNet ni arquitecturas ya entrenadas.
- OpenCV se usa solo para captura, visualizacion, guardado, lectura y redimensionamiento basico.
- El preprocesamiento importante se realiza manualmente con NumPy y operaciones propias.

## Tecnologias

- Python
- OpenCV
- NumPy
- PyTorch
- Matplotlib
- scikit-learn
- PySerial
- Arduino
- Bluetooth serial

## Estructura del Proyecto

```text
CarritoIA/
|-- Bluetooth.py              # Comunicacion serial Bluetooth con Arduino
|-- Carrito.py                # Ejecucion del carrito autonomo
|-- Config.py                 # Clases, rutas, camara, Bluetooth y comandos
|-- CrearDataset.py           # Preprocesamiento con filtro Sobel
|-- EntrenarNavegacion.py     # Entrenamiento del modelo principal
|-- EvaluarMetricas.py        # Reporte de metricas y matriz de confusion
|-- ModeloNavegacion.py       # CNN espacio-temporal creada desde cero
|-- TomarDatos.py             # Captura de imagenes propias con camara
|-- requirements.txt          # Dependencias del proyecto
|-- data_navegacion/          # Imagenes propias para navegacion
|-- dataset/                  # Datos procesados o auxiliares
|-- modelos/                  # Modelos entrenados
|-- resultados/               # Graficas y resultados de evaluacion
```

## Clases del Dataset

### Navegacion

- `RECTA`
- `CURVA_IZQUIERDA`
- `CURVA_DERECHA`
- `GIRO_90_IZQ`
- `GIRO_90_DER`
- `CRUCE_T`

## Instalacion

1. Crear un entorno virtual:

```bash
python -m venv venv
```

2. Activar el entorno virtual en Windows:

```bash
venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuracion

Los parametros principales se cambian en `Config.py`.

Valores importantes:

- `INDICE_CAMARA`: indice de la camara usada por OpenCV.
- `ANCHO_CAMARA` y `ALTO_CAMARA`: resolucion de captura.
- `CADA_CUANTOS_FRAMES_GUARDAR`: frecuencia de guardado durante la captura.
- `PUERTO_BLUETOOTH`: puerto COM del modulo Bluetooth.
- `BAUDRATE`: velocidad serial, por defecto `9600`.

Si la camara no abre, probar con `INDICE_CAMARA = 0`, `1` o `2`.

## Flujo de Uso

### 1. Capturar imagenes propias

Ejecutar:

```bash
python TomarDatos.py
```

El programa permite elegir entre:

- Capturar datos de navegacion.
- Capturar datos del bonus.

Controles durante la captura:

- `espacio`: pausar o reanudar.
- `q`: terminar captura.

Las imagenes se guardan automaticamente en la carpeta de la clase seleccionada.

### 2. Crear dataset procesado

Ejecutar:

```bash
python CrearDataset.py
```

Este script convierte las imagenes a escala de grises, las redimensiona y aplica un filtro Sobel normalizado. El filtro resalta bordes y cambios de direccion de la pista.

Nota: el entrenamiento de navegacion tambien aplica este mismo preprocesamiento internamente.

### 3. Entrenar modelo de navegacion

Ejecutar:

```bash
python EntrenarNavegacion.py
```

Este entrenamiento:

- Usa las imagenes de `data_navegacion/`.
- Forma ventanas temporales de 3 frames consecutivos.
- Divide datos en entrenamiento y validacion.
- Aplica aumentos solo al conjunto de entrenamiento.
- Guarda el mejor modelo en `modelos/modelo_final.pth`.
- Guarda las curvas de entrenamiento en `resultados/curvas_entrenamiento.png`.

El modelo principal es una CNN espacio-temporal creada desde cero en `ModeloNavegacion.py`.

### 4. Evaluar metricas

Ejecutar:

```bash
python EvaluarMetricas.py
```

El script muestra:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusion

La matriz de confusion se guarda en:

```text
resultados/matriz_confusion.png
```

### 5. Ejecutar el carrito

Ejecutar:

```bash
python Carrito.py
```

Antes de usar el carrito real, revisar en `Carrito.py`:

```python
USAR_BLUETOOTH = False
```

Para la demostracion con Arduino conectado, cambiarlo a:

```python
USAR_BLUETOOTH = True
```

El programa carga:

```text
modelos/modelo_final.pth
```

Luego lee la camara, predice la clase de navegacion y envia el comando correspondiente al Arduino.

## Comandos Bluetooth

Los comandos configurados son:

| Comando | Accion |
|---|---|
| `I` | Avanzar |
| `J` | Curva suave a la izquierda |
| `L` | Curva suave a la derecha |
| `U` | Giro fuerte / giro 90 a la izquierda |
| `O` | Giro fuerte / giro 90 a la derecha |
| `K` | Detener |
| `1` | Velocidad baja |
| `2` | Velocidad normal |

El archivo `Bluetooth.py` tambien permite probar manualmente la comunicacion:

```bash
python Bluetooth.py
```

## Funcionamiento del Modelo de Navegacion

El modelo recibe 3 frames consecutivos apilados como canales de entrada:

```text
Entrada: (Batch, 3, 66, 200)
Salida: 6 clases de navegacion
```

Cada frame se procesa con:

1. Conversion a escala de grises.
2. Redimensionamiento a `200x66`.
3. Filtro Sobel normalizado.
4. Normalizacion de pixeles entre `0` y `1`.

Durante la ejecucion, el carrito usa voto por mayoria sobre predicciones recientes para reducir cambios bruscos entre comandos.

## Recomendaciones para Captura de Datos

- Tomar imagenes propias para todas las clases.
- Mantener condiciones de iluminacion similares a la pista final.
- Capturar suficientes ejemplos por clase.
- Evitar que una clase tenga muchas mas imagenes que las demas.
- Grabar el carrito desde la misma posicion de camara que se usara en la demostracion.
- Verificar que las imagenes no esten borrosas ni demasiado oscuras.

## Archivos Generados

Durante el uso del proyecto se generan archivos como:

- `modelos/modelo_final.pth`
- `resultados/curvas_entrenamiento.png`
- `resultados/matriz_confusion.png`

Estos archivos pueden cambiar cada vez que se entrena nuevamente.