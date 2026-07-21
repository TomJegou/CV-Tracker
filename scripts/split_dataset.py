import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATASET_TRAIN_DIR, DATASET_VAL_DIR
from core.dataset_paths import list_dataset_source_dirs

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def copy_pair(image_path: Path, destination: Path) -> bool:
    shutil.copy2(image_path, destination / image_path.name)

    label_path = image_path.with_suffix(".txt")
    if label_path.exists():
        shutil.copy2(label_path, destination / label_path.name)

    return label_path.exists()


def collect_images(source_dirs: tuple[Path, ...]) -> list[Path]:
    """Fusionne toutes les sources ; en cas de doublon de nom, la source la plus récente gagne."""
    by_name: dict[str, Path] = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for image_path in source_dir.glob("*.jpg"):
            by_name[image_path.name] = image_path
    return sorted(by_name.values())


def resolve_source_dirs(dirs: list[Path] | None) -> tuple[Path, ...]:
    if dirs:
        source_dirs: list[Path] = []
        for path in dirs:
            resolved = path.expanduser().resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"Dossier introuvable : {resolved}")
            source_dirs.append(resolved)
        return tuple(source_dirs)

    source_dirs = list_dataset_source_dirs()
    if not source_dirs:
        raise FileNotFoundError(
            "Aucune source trouvée dans data/images_extraites/ "
            "(v* / data_mining_*). Ou passe --dir."
        )
    return source_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split train/val depuis les dossiers images_extraites.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        action="append",
        dest="dirs",
        help=(
            "Dossier source explicite (répétable). "
            "Remplace la découverte auto v* / data_mining_*."
        ),
    )
    args = parser.parse_args()

    try:
        source_dirs = resolve_source_dirs(args.dirs)
    except FileNotFoundError as exc:
        print(exc)
        return

    images = collect_images(source_dirs)
    total_images = len(images)

    print("Sources :")
    for source_dir in source_dirs:
        count = len(list(source_dir.glob("*.jpg"))) if source_dir.exists() else 0
        print(f"  {source_dir.name}/ : {count} image(s)")
    print(f"Total unique : {total_images} image(s)")

    if not images:
        print("Aucune image .jpg à traiter.")
        return

    random.seed(RANDOM_SEED)
    shuffled = images.copy()
    random.shuffle(shuffled)

    split_index = int(total_images * TRAIN_RATIO)
    train_images = shuffled[:split_index]
    val_images = shuffled[split_index:]

    DATASET_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_VAL_DIR.mkdir(parents=True, exist_ok=True)

    for dest_dir in (DATASET_TRAIN_DIR, DATASET_VAL_DIR):
        for old in dest_dir.glob("*"):
            if old.is_file():
                old.unlink()

    train_pairs = sum(copy_pair(path, DATASET_TRAIN_DIR) for path in train_images)
    val_pairs = sum(copy_pair(path, DATASET_VAL_DIR) for path in val_images)

    print("\nSéparation terminée (seed=42, ratio 80/20).")
    print(
        f"  {DATASET_TRAIN_DIR}/ : {len(train_images)} image(s), "
        f"{train_pairs} paire(s) image+txt"
    )
    print(
        f"  {DATASET_VAL_DIR}/ : {len(val_images)} image(s), "
        f"{val_pairs} paire(s) image+txt"
    )


if __name__ == "__main__":
    main()
