import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import FOV_SIZE, TRAIN_PROFILES, TRAIN_TARGET_VERSION, get_train_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorRT d'un modèle entraîné.")
    parser.add_argument(
        "--version",
        "-v",
        default=TRAIN_TARGET_VERSION,
        help=f"Version à exporter (défaut : {TRAIN_TARGET_VERSION})",
    )
    args = parser.parse_args()

    profile = get_train_profile(args.version)
    model_path = profile.weights_out
    engine_path = model_path.with_suffix(".engine")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle {profile.version} introuvable : {model_path}\n"
            f"Lance d'abord : python scripts/train.py --version {profile.version}"
        )

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
