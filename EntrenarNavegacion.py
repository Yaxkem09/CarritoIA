import os
import collections

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from Config import (
    CLASES_NAVEGACION,
    RUTA_DATA_NAVEGACION,
    RUTA_MODELOS,
    RUTA_RESULTADOS,
)
from ModeloNavegacion import ModeloNavegacion
from CrearDataset import aplicar_filtro_sobel_normalizado


# ----------------------------------------------------------------------
# 1. Configuracion y parametros
# ----------------------------------------------------------------------
DATOS_DIR = RUTA_DATA_NAVEGACION
MODELOS_DIR = RUTA_MODELOS
EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg")

TAM_VENTANA = 3
FRACCION_TRAIN = 0.8

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

for carpeta in (MODELOS_DIR, RUTA_RESULTADOS):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)


# ----------------------------------------------------------------------
# 2. Ventanas temporales con split train/val (sin fuga de datos)
# ----------------------------------------------------------------------
def construir_listas_ventanas(root_dir, fraccion_train=FRACCION_TRAIN,
                              tam_ventana=TAM_VENTANA):
    """
    Ordena las imagenes de cada clase por nombre (== orden de captura) y
    arma ventanas de 'tam_ventana' frames consecutivos. El split se hace
    ANTES de armar ventanas, asi ninguna ventana de validacion comparte
    frames con una de entrenamiento.
    """
    ventanas_train, ventanas_val = [], []
    conteo = {clase: 0 for clase in CLASES_NAVEGACION}

    for idx, clase in enumerate(CLASES_NAVEGACION):
        path = os.path.join(root_dir, clase)
        if not os.path.exists(path):
            print(f"ALERTA: no existe la carpeta {path}, se omite la clase.")
            continue

        archivos = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(EXTENSIONES_IMAGEN)
        )
        conteo[clase] = len(archivos)

        corte = int(len(archivos) * fraccion_train)
        archivos_train = archivos[:corte]
        archivos_val = archivos[corte:]

        for i in range(len(archivos_train) - tam_ventana + 1):
            ventanas_train.append((archivos_train[i:i + tam_ventana], idx))
        for i in range(len(archivos_val) - tam_ventana + 1):
            ventanas_val.append((archivos_val[i:i + tam_ventana], idx))

    return ventanas_train, ventanas_val, conteo


# ----------------------------------------------------------------------
# 3. Data augmentation (solo entrenamiento)
# ----------------------------------------------------------------------
def _kernel_motion_blur(longitud):
    """Kernel de desenfoque de movimiento horizontal."""
    kernel = np.zeros((longitud, longitud), dtype=np.float32)
    kernel[longitud // 2, :] = 1.0 / longitud
    return kernel


def aplicar_augmentation(frames):
    """
    Aplica el MISMO aumento geometrico, de brillo y de desenfoque a los 3
    frames de la ventana (para no romper la coherencia temporal). El ruido
    gaussiano es independiente por frame. Cubre rotacion, zoom, brillo,
    ruido y desenfoque por movimiento.
    """
    h, w = frames[0].shape
    angulo = np.random.uniform(-10.0, 10.0)
    escala = np.random.uniform(0.85, 1.15)
    brillo = np.random.uniform(0.6, 1.4)
    matriz = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angulo, escala)

    usar_blur = np.random.random() < 0.4
    kernel_blur = (_kernel_motion_blur(int(np.random.choice([3, 5, 7])))
                   if usar_blur else None)

    aumentados = []
    for f in frames:
        g = cv2.warpAffine(f, matriz, (w, h), borderMode=cv2.BORDER_REFLECT)
        if kernel_blur is not None:
            g = cv2.filter2D(g, -1, kernel_blur)
        g = g.astype(np.float32) * brillo
        ruido = np.random.normal(0.0, 10.0, g.shape).astype(np.float32)
        g = np.clip(g + ruido, 0, 255).astype(np.uint8)
        aumentados.append(g)
    return aumentados


# ----------------------------------------------------------------------
# 4. Dataset
# ----------------------------------------------------------------------
class DatasetNavegacion(Dataset):
    def __init__(self, ventanas, aumentar=False):
        self.ventanas = ventanas
        self.aumentar = aumentar

    def __len__(self):
        return len(self.ventanas)

    def __getitem__(self, idx):
        rutas, label = self.ventanas[idx]

        frames = []
        for ruta in rutas:
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {ruta}")
            img = cv2.resize(img, (200, 66))
            frames.append(img)

        if self.aumentar:
            frames = aplicar_augmentation(frames)

        procesados = []
        for f in frames:
            sobel = aplicar_filtro_sobel_normalizado(f)
            procesados.append(sobel.astype(np.float32) / 255.0)

        img_stack = np.stack(procesados, axis=0)
        return torch.from_numpy(img_stack), label


# ----------------------------------------------------------------------
# 5. Entrenamiento con validacion
# ----------------------------------------------------------------------
def iniciar_entrenamiento():
    ventanas_train, ventanas_val, conteo = construir_listas_ventanas(DATOS_DIR)

    print(f"Dataset de navegacion: {os.path.abspath(DATOS_DIR)}")
    for clase, cantidad in conteo.items():
        print(f"  {clase}: {cantidad} imagenes")
    print(f"Ventanas de entrenamiento: {len(ventanas_train)}")
    print(f"Ventanas de validacion:    {len(ventanas_val)}")

    if len(ventanas_train) == 0 or len(ventanas_val) == 0:
        raise ValueError("No hay suficientes imagenes para armar ventanas.")

    ds_train = DatasetNavegacion(ventanas_train, aumentar=True)
    ds_val = DatasetNavegacion(ventanas_val, aumentar=False)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Pesos de clase: compensan el desbalance (CRUCE_T tiene el doble de
    # imagenes que RECTA). Una clase con menos muestras pesa mas en la perdida.
    conteo_train = collections.Counter(lbl for _, lbl in ventanas_train)
    total = sum(conteo_train.values())
    pesos = torch.tensor(
        [(total / (len(CLASES_NAVEGACION) * max(conteo_train[i], 1))) ** 0.5
         for i in range(len(CLASES_NAVEGACION))],
        dtype=torch.float32,
    ).to(DEVICE)
    print("Pesos de clase:", [f"{p:.3f}" for p in pesos.tolist()])

    modelo = ModeloNavegacion(num_clases=len(CLASES_NAVEGACION)).to(DEVICE)

    # label_smoothing reduce la sobreconfianza del modelo: evita que el
    # val_loss se dispare y mejora la generalizacion.
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    # Baja el learning rate cuando el val_acc deja de mejorar.
    scheduler = ReduceLROnPlateau(optimizador, mode="max", factor=0.5,
                                  patience=3)

    print(f"\nEntrenando en {DEVICE}...")

    hist_train_loss, hist_val_loss = [], []
    hist_train_acc, hist_val_acc = [], []
    mejor_val_acc = -1.0

    for epoch in range(EPOCHS):
        # ---------------- entrenamiento ----------------
        modelo.train()
        train_loss, train_correctos, train_total = 0.0, 0, 0
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
            optimizador.zero_grad()
            outputs = modelo(imagenes)
            loss = criterio(outputs, etiquetas)
            loss.backward()
            optimizador.step()
            train_loss += loss.item()
            train_correctos += (outputs.argmax(1) == etiquetas).sum().item()
            train_total += etiquetas.size(0)

        # ---------------- validacion ----------------
        modelo.eval()
        val_loss, val_correctos, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
                outputs = modelo(imagenes)
                loss = criterio(outputs, etiquetas)
                val_loss += loss.item()
                val_correctos += (outputs.argmax(1) == etiquetas).sum().item()
                val_total += etiquetas.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correctos / train_total
        val_acc = val_correctos / val_total

        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_train_acc.append(train_acc)
        hist_val_acc.append(val_acc)

        scheduler.step(val_acc)
        lr_actual = optimizador.param_groups[0]["lr"]

        print(
            f"Epoca [{epoch + 1}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | lr={lr_actual:.5f}"
        )

        # Guardar el MEJOR modelo por ACCURACY de validacion. NO por loss:
        # el loss se dispara con el sobreajuste aunque el acc siga subiendo.
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(modelo.state_dict(),
                       os.path.join(MODELOS_DIR, "modelo_final.pth"))
            print(f"  -> Nuevo mejor modelo (val_acc={val_acc:.3f}). Guardado.")

    # ---------------- curvas Loss / Accuracy ----------------
    epocas = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epocas, hist_train_loss, label="train")
    ax1.plot(epocas, hist_val_loss, label="val")
    ax1.set_title("Perdida (Loss)")
    ax1.set_xlabel("Epoca")
    ax1.legend()
    ax2.plot(epocas, hist_train_acc, label="train")
    ax2.plot(epocas, hist_val_acc, label="val")
    ax2.set_title("Exactitud (Accuracy)")
    ax2.set_xlabel("Epoca")
    ax2.legend()
    plt.tight_layout()
    ruta_curvas = os.path.join(RUTA_RESULTADOS, "curvas_entrenamiento.png")
    plt.savefig(ruta_curvas)
    print(f"\nCurvas guardadas en: {ruta_curvas}")
    print(f"Mejor val_acc: {mejor_val_acc:.3f}")
    print("Modelo (el de mejor val_acc) guardado en /modelos/modelo_final.pth")


if __name__ == "__main__":
    iniciar_entrenamiento()