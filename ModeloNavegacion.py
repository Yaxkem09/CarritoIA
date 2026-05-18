import torch
import torch.nn as nn


class ModeloNavegacion(nn.Module):
    def __init__(self, num_clases=6):
        """
        CNN espacio-temporal disenada desde cero.
        Entrada: (Batch, 3, 66, 200) -> 3 frames CONSECUTIVOS apilados en
        los canales (dimension temporal). La red aprende el patron de ruta
        a partir del movimiento real entre los 3 frames.
        Salida: logits para 6 clases.
        """
        super(ModeloNavegacion, self).__init__()

        # Bloque convolucional: extraccion de caracteristicas espacio-temporales.
        self.red_convolucional = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )

        self.aplanado = nn.Flatten()

        # Clasificador con Dropout: apaga neuronas al azar SOLO en entrenamiento
        # para que la red no memorice el dataset. En model.eval() se desactiva.
        self.clasificador = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, num_clases),
        )

        self.aplicar_inicializacion()

    def forward(self, x):
        x = self.red_convolucional(x)
        x = self.aplanado(x)
        x = self.clasificador(x)
        return x

    def aplicar_inicializacion(self):
        """Inicializacion Kaiming para capas con activacion ReLU."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)