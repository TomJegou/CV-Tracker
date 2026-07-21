"""Déplace les images minées de auto_collected/ vers images_extraites/{DATA_VERSION}/."""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_MINING_SAVE_DIR, IMAGES_EXTRAITES_DIR

REASONS = (
    "enemy_as_ally_suspect",
    "ally_fp_suspect",
    "fp_suspect",
    "fn_suspect",
)


def extract_reason(filename: str) -> str:
    for reason in REASONS:
        if filename.endswith(f"_{reason}.jpg"):
            return reason
    return "other"


def main() -> None:
    source_dir = DATA_MINING_SAVE_DIR
    dest_dir = IMAGES_EXTRAITES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(source_dir.glob("*.jpg"))
    if not images:
        print(f"Aucune image .jpg dans {source_dir}/")
        return

    counts: Counter[str] = Counter()
    moved = 0
    skipped = 0

    print(f"Source : {source_dir}/")
    print(f"Destination : {dest_dir}/\n")

    for image_path in images:
        dest_path = dest_dir / image_path.name
        reason = extract_reason(image_path.name)
        counts[reason] += 1

        if dest_path.exists():
            print(f"  Ignoré (existe déjà) : {image_path.name}")
            skipped += 1
            continue

        shutil.move(str(image_path), str(dest_path))
        moved += 1

    print(f"\nImport terminé : {moved} image(s) déplacée(s), {skipped} ignorée(s).")
    print("Répartition :")
    for reason in REASONS:
        print(f"  {reason} : {counts[reason]}")
    if counts["other"]:
        print(f"  other : {counts['other']}")


if __name__ == "__main__":
    main()
