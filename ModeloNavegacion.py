import torch
import torch.nn as nn

class ModeloNavegacion(nn.Module):
    def __init__(self, num_clases=6):
        """
        Arquitectura CNN diseñada desde cero.
        Entrada: (Batch, 3, 66, 200) -> 3 frames apilados en escala de grises.
        Salida: Probabilidades para 6 clases (Recta, Curva I/D, Giro I/D, Cruce T).
        """
        super(ModeloNavegacion, self).__init__()
        
        # Bloque de Convolución: Extracción de características espaciales
        # No usamos modelos preentrenados, definimos cada capa manualmente.
        self.red_convolucional = nn.Sequential(
            # Capa 1: Detecta bordes y formas básicas
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            # Capa 2: Empieza a identificar la curvatura de la línea
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            # Capa 3: Características de alto nivel
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            # Capas de profundidad para mayor abstracción (como el Cruce en T)
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        
        self.aplanado = nn.Flatten()
        
        # Bloque Totalmente Conectado (Perceptrón Multicapa)
        # El tamaño 1152 depende del resize (66x200). Se debe ajustar si cambian la resolución.
        self.clasificador = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, num_clases)
        )
        
        # Inicialización de pesos manual (Obligatorio para evitar pesos por defecto)
        self.aplicar_inicializacion()

    def forward(self, x):
        x = self.red_convolucional(x)
        x = self.aplanado(x)
        x = self.clasificador(x)
        return x

    def aplicar_inicializacion(self):
        """Inicialización Kaiming para redes con activación ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Para usarlo en EntrenarNavegacion.py:
# modelo = ModeloNavegacion(num_clases=6)