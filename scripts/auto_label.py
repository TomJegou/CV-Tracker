import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import AUTO_LABEL_CONF, DEFAULT_YOLO_MODEL, IMAGES_EXTRAITES_DIR, V1_MODEL, V2_MODEL

MODEL_CANDIDATES = (
    V2_MODEL,
    V1_MODEL,
    DEFAULT_YOLO_MODEL,
)


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Aucun modèle trouvé. Chemins testés : {searched}")


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


def auto_label_image(model: YOLO, image_path: Path) -> int:
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
    return len(lines)


def main() -> None:
    model_path = resolve_model_path()
    model = YOLO(str(model_path))

    all_images = sorted(IMAGES_EXTRAITES_DIR.glob("*.jpg"))
    images = [img for img in all_images if not img.name.startswith("FAUX_POSITIF_")]
    skipped = len(all_images) - len(images)
    total_images = len(images)

    print(f"Modèle chargé : {model_path}")
    print(f"{len(all_images)} image(s) trouvée(s) dans {IMAGES_EXTRAITES_DIR}/")
    if skipped:
        print(f"{skipped} faux positif(s) ignoré(s) (FAUX_POSITIF_*)")

    if not images:
        print("Aucune image .jpg à traiter.")
        return

    labeled_count = 0
    empty_count = 0

    for index, image_path in enumerate(images, start=1):
        detections = auto_label_image(model, image_path)

        if detections > 0:
            labeled_count += 1
        else:
            empty_count += 1

        if index % 50 == 0 or index == total_images:
            print(
                f"Progression : {index}/{total_images} "
                f"({labeled_count} avec cibles, {empty_count} vides)"
            )

    print("\nAuto-labeling terminé.")
    print(f"  Images traitées : {total_images}")
    print(f"  Images avec cibles : {labeled_count}")
    print(f"  Images vides : {empty_count}")
    print(f"  Faux positifs préservés : {skipped}")
    print(f"  Labels sauvegardés dans {IMAGES_EXTRAITES_DIR}/")


if __name__ == "__main__":
    main()
