import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from ModeloNavegacion import ModeloNavegacion

# 1. Configuración de carpetas y parámetros
DATOS_DIR = 'dataset_procesado'
MODELOS_DIR = 'modelos'
if not os.path.exists(MODELOS_DIR):
    os.makedirs(MODELOS_DIR)

# Hiperparámetros sugeridos para la semana 4
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset personalizado para carga de imágenes propias
class DatasetNavegacion(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clases = ["RECTA", "CURVA_I", "CURVA_D", "GIRO_90_I", "GIRO_90_D", "CRUCE_T"]
        self.data = []

        # Recolección de rutas de imágenes (mínimo 3,000 según rúbrica)
        for idx, clase in enumerate(self.clases):
            path = os.path.join(root_dir, clase)
            if not os.path.exists(path): continue
            for img_name in os.listdir(path):
                self.data.append((os.path.join(path, img_name), idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # Preprocesamiento Matricial Manual (REGLA ESTRICTA)
        # 1. Carga en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 2. Redimensionamiento
        img = cv2.resize(img, (200, 66))
        # 3. Normalización manual con NumPy (píxeles de 0 a 1)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Stacking de 3 frames (Dimensión temporal)
        # Para el entrenamiento, si no tienen secuencias, triplicamos el canal
        img_stack = np.stack([img_np, img_np, img_np], axis=0)

        return torch.from_numpy(img_stack), label

# 3. Función principal de entrenamiento
def iniciar_entrenamiento():
    dataset = DatasetNavegacion(DATOS_DIR)
    # División para validación (opcional pero recomendado)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciar modelo desde cero (sin pesos externos)
    modelo = ModeloNavegacion(num_clases=6).to(DEVICE)

    # Función de pérdida y optimizador
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    print(f"Entrenando en {DEVICE} con {len(dataset)} imágenes...")

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
        print(f"Época [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # 4. Guardar el "cerebro" del robot
    torch.save(modelo.state_dict(), os.path.join(MODELOS_DIR, 'modelo_final.pth'))
    print("Entrenamiento finalizado. Modelo guardado en /modelos/modelo_final.pth")

if __name__ == "__main__":
    iniciar_entrenamiento()