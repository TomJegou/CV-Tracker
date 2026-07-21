"""Entraînement YOLO — sortie automatique dans models/apex_{NNN}/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import (
    APEX_DATASET_YAML,
    FOV_SIZE,
    MODELS_DIR,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_WORKERS,
)
from core.model_paths import (
    create_next_apex_model_dir,
    list_apex_model_dirs,
    resolve_train_base,
    weights_engine_path,
    weights_pt_path,
)


def _print_models() -> None:
    dirs = list_apex_model_dirs()
    if not dirs:
        print("Aucun modèle models/apex_{NNN}/ trouvé.")
        return

    print("Modèles disponibles :\n")
    for model_dir in dirs:
        pt_path = weights_pt_path(model_dir)
        engine_path = weights_engine_path(model_dir)
        pt_status = "OK" if pt_path.exists() else "manquant"
        engine_status = "OK" if engine_path.exists() else "—"
        print(f"  {model_dir.name}/")
        print(f"    best.pt     : {pt_status}  ({pt_path})")
        print(f"    best.engine : {engine_status}")
        print()


def run_train(
    *,
    base: Path | None = None,
    dataset_yaml: Path = APEX_DATASET_YAML,
    epochs: int | None = None,
    batch: int | None = None,
    patience: int | None = None,
    workers: int | None = None,
    device: int = 0,
) -> Path:
    base_path = resolve_train_base(base=base)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_yaml}")

    run_dir = create_next_apex_model_dir()
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs if epochs is not None else TRAIN_EPOCHS,
        "imgsz": FOV_SIZE,
        "batch": batch if batch is not None else TRAIN_BATCH,
        "device": device,
        "project": str(MODELS_DIR),
        "name": run_dir.name,
        "workers": workers if workers is not None else TRAIN_WORKERS,
        "exist_ok": True,
    }
    patience_value = patience if patience is not None else TRAIN_PATIENCE
    if patience_value is not None:
        train_kwargs["patience"] = patience_value

    weights_out = weights_pt_path(run_dir)

    print(f"Run        : {run_dir.name}")
    print(f"Modèle base: {base_path}")
    print(f"Dataset    : {dataset_yaml}")
    print(f"Sortie     : {weights_out}")
    print(f"Hyperparams: epochs={train_kwargs['epochs']}, batch={train_kwargs['batch']}")
    print()

    model = YOLO(str(base_path))
    model.train(**train_kwargs)

    print(f"\nEntraînement terminé ! Modèle : {weights_out}")
    return weights_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle Apex (sortie models/apex_{NNN}/).",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Lister les modèles models/apex_*",
    )
    parser.add_argument(
        "--base",
        "-b",
        type=Path,
        default=None,
        help="Poids de départ (.pt). Défaut : dernier apex_*/best.pt ou yolov8n.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=APEX_DATASET_YAML,
        help=f"Dataset YAML (défaut : {APEX_DATASET_YAML})",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch")
    parser.add_argument("--patience", type=int, default=None, help="Override patience")
    parser.add_argument("--workers", type=int, default=None, help="Override workers")
    parser.add_argument("--device", type=int, default=0, help="GPU device (défaut 0)")
    args = parser.parse_args()

    if args.list:
        _print_models()
        return

    run_train(
        base=args.base,
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
