"""Entraînement YOLO unifié — profils définis dans core/config.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import (
    FOV_SIZE,
    RUNS_DETECT_DIR,
    TRAIN_PROFILES,
    TRAIN_TARGET_VERSION,
    TrainProfile,
    get_train_profile,
    resolve_train_base,
)


def _print_profiles() -> None:
    print("Profils disponibles :\n")
    for key, profile in sorted(TRAIN_PROFILES.items()):
        base = resolve_train_base(profile)
        default = " (défaut)" if key == TRAIN_TARGET_VERSION else ""
        print(f"  {key}{default}")
        print(f"    run      : {profile.run_name}")
        print(f"    base     : {base}")
        print(f"    dataset  : {profile.dataset_yaml}")
        print(f"    sortie   : {profile.weights_out}")
        print(f"    epochs   : {profile.epochs} | batch : {profile.batch}")
        print()


def _build_train_kwargs(profile: TrainProfile, overrides: argparse.Namespace) -> dict:
    kwargs: dict = {
        "data": str(profile.dataset_yaml),
        "epochs": overrides.epochs if overrides.epochs is not None else profile.epochs,
        "imgsz": FOV_SIZE,
        "batch": overrides.batch if overrides.batch is not None else profile.batch,
        "device": overrides.device,
        "project": str(RUNS_DETECT_DIR),
        "name": profile.run_name,
        "workers": overrides.workers if overrides.workers is not None else profile.workers,
        # Sans ça, Ultralytics crée apex_model_v4-2, -3, ... à chaque relance et
        # les chemins fixes (V4_MODEL, etc.) dans config.py restent bloqués sur le 1er run.
        "exist_ok": overrides.exist_ok,
    }
    patience = overrides.patience if overrides.patience is not None else profile.patience
    if patience is not None:
        kwargs["patience"] = patience
    return kwargs


def run_train(
    version: str | None = None,
    *,
    epochs: int | None = None,
    batch: int | None = None,
    patience: int | None = None,
    workers: int | None = None,
    device: int = 0,
    exist_ok: bool = True,
) -> Path:
    profile = get_train_profile(version)
    base_path = resolve_train_base(profile)

    if not profile.dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset introuvable : {profile.dataset_yaml}")

    overrides = argparse.Namespace(
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
        exist_ok=exist_ok,
    )
    train_kwargs = _build_train_kwargs(profile, overrides)

    print(f"Version    : {profile.version}")
    print(f"Modèle base: {base_path}")
    print(f"Dataset    : {profile.dataset_yaml}")
    print(f"Run        : {profile.run_name}" + ("" if exist_ok else " (nouveau dossier -N)"))
    print(f"Hyperparams: epochs={train_kwargs['epochs']}, batch={train_kwargs['batch']}")
    print()

    model = YOLO(str(base_path))
    model.train(**train_kwargs)

    print(f"\nEntraînement terminé ! Modèle : {profile.weights_out}")
    return profile.weights_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle Apex (profils dans core/config.py).",
    )
    parser.add_argument(
        "--version",
        "-v",
        default=TRAIN_TARGET_VERSION,
        help=f"Version à entraîner (défaut config : {TRAIN_TARGET_VERSION})",
    )
    parser.add_argument("--list", "-l", action="store_true", help="Lister les profils")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch")
    parser.add_argument("--patience", type=int, default=None, help="Override patience")
    parser.add_argument("--workers", type=int, default=None, help="Override workers")
    parser.add_argument("--device", type=int, default=0, help="GPU device (défaut 0)")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ne pas écraser le run existant : Ultralytics crée un nouveau dossier "
            "(apex_model_vX-2, -3, ...) au lieu de réutiliser le même."
        ),
    )
    args = parser.parse_args()

    if args.list:
        _print_profiles()
        return

    run_train(
        args.version,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=not args.fresh,
    )


if __name__ == "__main__":
    main()
