import sys
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.config import FOV_SIZE, V2_MODEL


def main() -> None:
    if not V2_MODEL.exists():
        raise FileNotFoundError(f"Modèle introuvable : {V2_MODEL}")

    print(f"Chargement du modèle : {V2_MODEL}")
    print("Export TensorRT en cours (FP16, workspace=4 Go)...")

    model = YOLO(str(V2_MODEL))
    export_path = model.export(
        format="engine",
        imgsz=FOV_SIZE,
        device=0,
        workspace=4,
        half=True,
    )

    print(f"Export terminé ! Moteur TensorRT disponible : {export_path}")


if __name__ == "__main__":
    main()
