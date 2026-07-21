import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core.config import FOV_SIZE
from core.dataset_paths import create_manual_extract_dir, default_derush_dir


def crop_center(frame, fov_size: int = FOV_SIZE):
    height, width = frame.shape[:2]
    if width < fov_size or height < fov_size:
        return None

    center_x = width // 2
    center_y = height // 2
    half = fov_size // 2

    return frame[
        center_y - half : center_y + half,
        center_x - half : center_x + half,
    ]


def extract_from_video(video_path: Path, output_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Erreur : impossible d'ouvrir {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = max(1, round(fps))
    video_name = video_path.stem
    frame_index = 0
    saved_count = 0

    print(f"  Extraction de {video_path.name} ({fps:.1f} FPS, 1 image/seconde)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            cropped = crop_center(frame)
            if cropped is None:
                print(
                    f"  Avertissement : résolution insuffisante pour un crop "
                    f"{FOV_SIZE}x{FOV_SIZE} dans {video_path.name}"
                )
                break

            saved_count += 1
            output_path = output_dir / f"{video_name}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_path), cropped)

        frame_index += 1

    cap.release()
    print(f"  -> {saved_count} image(s) extraite(s)")
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction de frames depuis des vidéos derush.")
    parser.add_argument(
        "--derush",
        type=Path,
        default=None,
        help="Dossier source des .mp4 (défaut : dernier data/derush/vN/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dossier de sortie (défaut : prochain data/images_extraites/vN/)",
    )
    args = parser.parse_args()

    derush_dir = args.derush or default_derush_dir()
    output_dir = args.output or create_manual_extract_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(derush_dir.glob("*.mp4"))
    print(f"{len(videos)} vidéo(s) trouvée(s) dans {derush_dir}/")
    print(f"Sortie : {output_dir}/")

    if not videos:
        print("Aucune vidéo .mp4 à traiter.")
        return

    total_images = 0
    for video_path in videos:
        total_images += extract_from_video(video_path, output_dir)

    print(
        f"\nExtraction terminée : {total_images} image(s) "
        f"sauvegardée(s) dans {output_dir}/"
    )


if __name__ == "__main__":
    main()
