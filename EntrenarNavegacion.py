import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Config import CLASES_NAVEGACION, RUTA_DATA_NAVEGACION, RUTA_MODELOS
from ModeloNavegacion import ModeloNavegacion


# 1. Configuracion de carpetas y parametros
DATOS_DIR = RUTA_DATA_NAVEGACION
MODELOS_DIR = RUTA_MODELOS
EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg")

if not os.path.exists(MODELOS_DIR):
    os.makedirs(MODELOS_DIR)


# Hiperparametros sugeridos para la semana 4
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Dataset personalizado para carga de imagenes propias
class DatasetNavegacion(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clases = CLASES_NAVEGACION
        self.data = []
        self.conteo_por_clase = {clase: 0 for clase in self.clases}

        # Recoleccion de rutas de imagenes (minimo 3,000 segun rubrica)
        for idx, clase in enumerate(self.clases):
            path = os.path.join(root_dir, clase)
            if not os.path.exists(path):
                continue

            for img_name in os.listdir(path):
                if img_name.lower().endswith(EXTENSIONES_IMAGEN):
                    self.data.append((os.path.join(path, img_name), idx))
                    self.conteo_por_clase[clase] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # Preprocesamiento Matricial Manual (REGLA ESTRICTA)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {img_path}")

        img = cv2.resize(img, (200, 66))
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Stacking de 3 frames (dimension temporal). Si no hay secuencia,
        # se triplica el mismo frame para mantener la entrada esperada.
        img_stack = np.stack([img_np, img_np, img_np], axis=0)

        return torch.from_numpy(img_stack), label


# 3. Funcion principal de entrenamiento
def iniciar_entrenamiento():
    dataset = DatasetNavegacion(DATOS_DIR)

    print(f"Dataset de navegacion: {os.path.abspath(DATOS_DIR)}")
    for clase, cantidad in dataset.conteo_por_clase.items():
        print(f"  {clase}: {cantidad} imagenes")

    if len(dataset) == 0:
        clases = ", ".join(CLASES_NAVEGACION)
        raise ValueError(
            "No se encontraron imagenes para entrenamiento. "
            f"Verifica que '{DATOS_DIR}' exista y contenga estas carpetas: {clases}"
        )

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciar modelo desde cero (sin pesos externos)
    modelo = ModeloNavegacion(num_clases=len(CLASES_NAVEGACION)).to(DEVICE)

    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    print(f"Entrenando en {DEVICE} con {len(dataset)} imagenes...")

    for epoch in range(EPOCHS):
        modelo.train()
        running_loss = 0.0

        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)

            optimizador.zero_grad()
            outputs = modelo(imagenes)
            loss = criterio(outputs, etiquetas)
            loss.backward()
            optimizador.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoca [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(modelo.state_dict(), os.path.join(MODELOS_DIR, "modelo_final.pth"))
    print("Entrenamiento finalizado. Modelo guardado en /modelos/modelo_final.pth")


if __name__ == "__main__":
    iniciar_entrenamiento()
