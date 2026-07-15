import sys
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.config import DEFAULT_YOLO_MODEL, FOV_SIZE, ROBOFLOW_DATASET_YAML, RUNS_DETECT_DIR, V1_MODEL


def main() -> None:
    model = YOLO(str(DEFAULT_YOLO_MODEL))

    print("Démarrage de l'entraînement sur la RTX 4070...")

    model.train(
        data=str(ROBOFLOW_DATASET_YAML),
        epochs=100,
        imgsz=FOV_SIZE,
        device=0,
        batch=32,
        workers=4,
        patience=25,
        project=str(RUNS_DETECT_DIR),
        name="apex_model_v1",
    )

    print(f"Entraînement terminé ! Le modèle final est disponible dans {V1_MODEL}")


if __name__ == "__main__":
    main()
