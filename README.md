# CarritoIA

Proyecto universitario de Inteligencia Artificial: **Navegacion Autonoma y Reconocimiento de Senales**.

## Descripcion

La computadora ejecutara Python, usara una camara, procesara imagenes, ejecutara modelos de IA entrenados desde cero y enviara comandos por Bluetooth al Arduino.

El Arduino solo recibira comandos simples y controlara los motores mediante el modulo L298N.

## Estructura inicial

- `data_navegacion/`: imagenes propias para las clases de navegacion.
- `data_senales/`: imagenes propias para las clases de senales.
- `dataset/`: datos procesados para entrenamiento.
- `modelos/`: modelos entrenados.
- `resultados/`: graficas, pruebas o resultados del entrenamiento.

## Restricciones

- No usar datasets publicos.
- No usar modelos preentrenados.
- No usar MediaPipe, YOLO, ResNet, MobileNet ni transferencia de aprendizaje.
- OpenCV solo para captura, visualizacion, guardado y redimensionamiento basico.
- El preprocesamiento importante se hara manualmente con NumPy.

