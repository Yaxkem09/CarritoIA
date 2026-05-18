import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from Config import (
    CLASES_NAVEGACION,
    RUTA_MODELOS,
    RUTA_DATA_NAVEGACION,
    RUTA_RESULTADOS,
)
from ModeloNavegacion import ModeloNavegacion
# Reutilizamos el dataset y el constructor de ventanas del entrenamiento,
# para evaluar EXACTAMENTE el mismo split de validacion.
from EntrenarNavegacion import DatasetNavegacion, construir_listas_ventanas


def evaluar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando metricas en: {device}")

    # 1. SOLO el conjunto de validacion (no visto durante el entrenamiento).
    _, ventanas_val, _ = construir_listas_ventanas(RUTA_DATA_NAVEGACION)
    dataset = DatasetNavegacion(ventanas_val, aumentar=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Evaluando sobre {len(dataset)} ventanas de validacion.")

    # 2. Cargar el modelo entrenado.
    modelo = ModeloNavegacion(num_clases=len(CLASES_NAVEGACION)).to(device)
    modelo.load_state_dict(
        torch.load(f"{RUTA_MODELOS}/modelo_final.pth", map_location=device)
    )
    modelo.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            outputs = modelo(imagenes)
            predicciones = outputs.argmax(1)
            y_true.extend(etiquetas.cpu().numpy())
            y_pred.extend(predicciones.cpu().numpy())

    # 3. Metricas cuantitativas.
    print("\n" + "=" * 60)
    print(" REPORTE: Accuracy, Precision, Recall, F1-Score (VALIDACION)")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=CLASES_NAVEGACION,
                                zero_division=0))

    # 4. Matriz de confusion.
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(CLASES_NAVEGACION))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASES_NAVEGACION)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Matriz de Confusion - Navegacion Base (Validacion)")
    plt.tight_layout()
    ruta_grafica = f"{RUTA_RESULTADOS}/matriz_confusion.png"
    plt.savefig(ruta_grafica)
    print(f"\nGrafica guardada en: {ruta_grafica}")
    plt.show()


if __name__ == "__main__":
    evaluar_modelo()