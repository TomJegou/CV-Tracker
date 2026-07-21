import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import FOV_SIZE
from core.model_paths import resolve_export_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorRT d'un modèle entraîné.")
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help=(
            "Chemin vers best.pt ou dossier models/apex_{NNN}. "
            "Défaut : dernier models/apex_*/weights/best.pt"
        ),
    )
    args = parser.parse_args()

    try:
        model_path = resolve_export_model(model=args.model)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    engine_path = model_path.with_suffix(".engine")

    print(f"Chargement : {model_path}")
    print("Export TensorRT (FP16, workspace=4 Go)...")

    model = YOLO(str(model_path))
    export_path = model.export(
        format="engine",
        imgsz=FOV_SIZE,
        device=0,
        workspace=4,
        half=True,
    )

    print(f"Export terminé : {export_path}")
    print(f"Chemin attendu : {engine_path}")


if __name__ == "__main__":
    main()
