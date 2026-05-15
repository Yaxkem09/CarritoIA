import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from ModeloBonus import ModeloBonus
from Config import RUTA_DATA_BONUS, CLASES_BONUS, RUTA_MODELOS

# Parametros
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetBonus(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for idx, clase in enumerate(CLASES_BONUS):
            path = os.path.join(root_dir, clase)
            if os.path.exists(path):
                for img_name in os.listdir(path):
                    self.data.append((os.path.join(path, img_name), idx))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0), label

def entrenar():
    dataset = DatasetBonus(RUTA_DATA_BONUS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    modelo = ModeloBonus(num_clases=len(CLASES_BONUS)).to(DEVICE)
    optimizer = optim.Adam(modelo.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Entrenando Bonus (ViT) en {DEVICE}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = modelo(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    if not os.path.exists(RUTA_MODELOS): os.makedirs(RUTA_MODELOS)
    torch.save(modelo.state_dict(), os.path.join(RUTA_MODELOS, 'modelo_bonus.pth'))
    print("Modelo Bonus guardado.")

if __name__ == "__main__":
    entrenar()