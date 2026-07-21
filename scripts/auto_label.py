import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import (
    AUTO_LABEL_CONF,
    AUTO_LABEL_PRESERVE_CLASS_IDS,
    CLASS_NAMES,
)
from core.dataset_paths import list_auto_label_dirs
from core.model_paths import resolve_prelabel_model


def to_yolo_lines(result) -> list[str]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    lines: list[str] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x_center, y_center, width, height = box.xywhn[0].tolist()
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return lines


def label_has_preserved_class(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if int(parts[0]) in AUTO_LABEL_PRESERVE_CLASS_IDS:
            return True
    return False


def auto_label_image(model: YOLO, image_path: Path) -> tuple[int, dict[int, int]]:
    results = model.predict(
        source=str(image_path),
        conf=AUTO_LABEL_CONF,
        verbose=False,
        device=0,
    )

    lines = to_yolo_lines(results[0])
    label_path = image_path.with_suffix(".txt")
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )

    per_class: dict[int, int] = {}
    for line in lines:
        class_id = int(line.split()[0])
        per_class[class_id] = per_class.get(class_id, 0) + 1

    return len(lines), per_class


def process_directory(model: YOLO, source_dir: Path) -> dict[str, int]:
    all_images = sorted(source_dir.glob("*.jpg"))
    images = [img for img in all_images if not img.name.startswith("FAUX_POSITIF_")]
    total_images = len(images)

    print(f"\nDossier : {source_dir}/")
    print(f"{len(all_images)} image(s) trouvée(s)")
    if len(all_images) != total_images:
        print(f"{len(all_images) - total_images} faux positif(s) ignoré(s) (FAUX_POSITIF_*)")

    if not images:
        print("Aucune image .jpg à traiter.")
        return {
            "total": 0,
            "labeled": 0,
            "empty": 0,
            "preserved": 0,
        }

    labeled_count = 0
    empty_count = 0
    preserved_count = 0
    class_totals: dict[int, int] = {}

    for index, image_path in enumerate(images, start=1):
        label_path = image_path.with_suffix(".txt")
        if label_has_preserved_class(label_path):
            preserved_count += 1
            continue

        box_count, per_class = auto_label_image(model, image_path)

        if box_count > 0:
            labeled_count += 1
        else:
            empty_count += 1

        for class_id, count in per_class.items():
            class_totals[class_id] = class_totals.get(class_id, 0) + count

        if index % 50 == 0 or index == total_images:
            print(
                f"Progression : {index}/{total_images} "
                f"({labeled_count} avec cibles, {empty_count} vides, "
                f"{preserved_count} préservées)"
            )

    print("  Résultat :")
    print(f"    Images traitées : {total_images}")
    print(f"    Images avec cibles : {labeled_count}")
    print(f"    Images vides : {empty_count}")
    print(f"    Préservées (allié manuel) : {preserved_count}")
    for class_id, count in sorted(class_totals.items()):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls_{class_id}"
        print(f"    Boxes {name} : {count}")

    return {
        "total": total_images,
        "labeled": labeled_count,
        "empty": empty_count,
        "preserved": preserved_count,
    }


def resolve_source_dirs(*, dirs: list[Path] | None, latest_only: bool) -> list[Path]:
    if dirs:
        source_dirs: list[Path] = []
        for path in dirs:
            resolved = path.expanduser().resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"Dossier introuvable : {resolved}")
            source_dirs.append(resolved)
        return source_dirs

    source_dirs = list_auto_label_dirs(latest_only=latest_only)
    if not source_dirs:
        raise FileNotFoundError(
            "Aucune session data_mining_* trouvée dans data/images_extraites/. "
            "Lance la pipeline avec ENABLE_DATA_MINING=True, ou passe --dir."
        )
    return source_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-annotation YOLO des sessions data_mining_*.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        action="append",
        dest="dirs",
        help="Dossier source explicite (répétable). Remplace la découverte auto data_mining_*.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help="Modèle .pt pour pré-annoter. Défaut : dernier models/apex_*/best.pt",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Traiter uniquement la dernière session data_mining_{NNN} (ignoré si --dir)",
    )
    args = parser.parse_args()

    try:
        source_dirs = resolve_source_dirs(dirs=args.dirs, latest_only=args.latest)
    except FileNotFoundError as exc:
        print(exc)
        return

    try:
        model_path = resolve_prelabel_model(model=args.model)
    except FileNotFoundError as exc:
        print(exc)
        return
    model = YOLO(str(model_path))
    model_nc = len(model.names) if hasattr(model, "names") else 1

    print(f"Modèle chargé : {model_path}")
    print(f"Classes dataset : {', '.join(CLASS_NAMES)} (nc={len(CLASS_NAMES)})")
    print(f"Classes modèle  : nc={model_nc}")
    if model_nc < len(CLASS_NAMES):
        print(
            "  ⚠ Modèle mono-classe : auto-label ne produira que class_id=0 (ennemi). "
            "Annoter les alliés à la main (classe 1)."
        )
    print(f"Sessions à traiter : {len(source_dirs)}")

    totals = {"total": 0, "labeled": 0, "empty": 0, "preserved": 0}
    for source_dir in source_dirs:
        stats = process_directory(model, source_dir)
        for key in totals:
            totals[key] += stats[key]

    print("\nAuto-labeling terminé.")
    print(f"  Sessions : {len(source_dirs)}")
    print(f"  Images traitées : {totals['total']}")
    print(f"  Images avec cibles : {totals['labeled']}")
    print(f"  Images vides : {totals['empty']}")
    print(f"  Préservées (allié manuel) : {totals['preserved']}")


if __name__ == "__main__":
    main()
