import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATASET_TRAIN_DIR, DATASET_VAL_DIR, IMAGES_EXTRAITES_DIR

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def copy_pair(image_path: Path, destination: Path) -> bool:
    shutil.copy2(image_path, destination / image_path.name)

    label_path = image_path.with_suffix(".txt")
    if label_path.exists():
        shutil.copy2(label_path, destination / label_path.name)

    return label_path.exists()


def main() -> None:
    images = sorted(IMAGES_EXTRAITES_DIR.glob("*.jpg"))
    total_images = len(images)

    print(f"{total_images} image(s) trouvée(s) dans {IMAGES_EXTRAITES_DIR}/")

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
    print(f"  Total copié    : {len(train_images) + len(val_images)} image(s)")


if __name__ == "__main__":
    main()
