import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import (
    APEX_V2_YAML,
    DEFAULT_YOLO_MODEL,
    FOV_SIZE,
    RUNS_DETECT_DIR,
    V1_MODEL,
    V2_MODEL,
    V3_MODEL,
)


def main() -> None:
    model_path = (
        V2_MODEL
        if V2_MODEL.exists()
        else V1_MODEL if V1_MODEL.exists() else DEFAULT_YOLO_MODEL
    )
    model = YOLO(str(model_path))

    print(f"Modèle de base : {model_path}")
    print("Démarrage de l'entraînement V3 sur la RTX 4070...")

    model.train(
        data=str(APEX_V2_YAML),
        epochs=50,
        imgsz=FOV_SIZE,
        batch=16,
        device=0,
        project=str(RUNS_DETECT_DIR),
        name="apex_model_v3",
    )

    print(f"Entraînement terminé ! Le modèle final est disponible dans {V3_MODEL}")


if __name__ == "__main__":
    main()
