"""Ré-inférence des sessions data_mining_* — triage AUTO / NEGATIF / REVIEW."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from core.config import (
    CLASS_NAMES,
    REINFER_ALWAYS_REVIEW_REASONS,
    REINFER_HIGH_CONF,
    REINFER_LOW_CONF,
    REINFER_MANUAL_EDIT_MARGIN_S,
)
from core.dataset_paths import (
    ensure_classes_txt,
    list_dataset_source_dirs,
    list_review_dirs,
    parse_mining_reason,
    review_dir,
    skipped_dir,
)
from core.model_paths import resolve_prelabel_model

BUCKET_AUTO = "auto"
BUCKET_NEGATIF = "negatif"
BUCKET_REVIEW = "review"
BUCKET_PRESERVED = "preserved"

_NEGATIF_REASONS = frozenset({"fp_suspect", "ally_fp_suspect"})
_REPORT_NAME = "reinfer_report.csv"
_PREDICT_BATCH = 16


@contextmanager
def source_list_file(images: list[Path]) -> Iterator[str]:
    """Liste temporaire de chemins pour Ultralytics.

    Une list[Path] passée à predict() serait chargée en RAM d'un coup
    (LoadPilAndNumpy) et perdrait les chemins source ; un .txt garde le
    streaming batch par batch et des `result.path` exploitables.
    """
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        handle.write("\n".join(str(path.absolute()) for path in images))
        handle.close()
        yield handle.name
    finally:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)


def default_device() -> int | str:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


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


def box_confs(result) -> list[float]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    return [float(c) for c in result.boxes.conf.tolist()]


def is_manual_edit(
    image_path: Path,
    *,
    margin_s: float = REINFER_MANUAL_EDIT_MARGIN_S,
) -> bool:
    """True si le .txt a été modifié nettement après le .jpg (correction LabelImg)."""
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists() or not image_path.exists():
        return False
    return label_path.stat().st_mtime > image_path.stat().st_mtime + margin_s


def classify_bucket(
    *,
    reason: str | None,
    confs: list[float],
    high_conf: float,
    always_review: frozenset[str],
) -> str:
    if reason is not None and reason in always_review:
        return BUCKET_REVIEW

    if not confs:
        if reason in _NEGATIF_REASONS:
            return BUCKET_NEGATIF
        return BUCKET_REVIEW

    if all(c >= high_conf for c in confs):
        return BUCKET_AUTO
    return BUCKET_REVIEW


def resolve_source_dirs(*, dirs: list[Path] | None, latest_only: bool) -> list[Path]:
    if dirs:
        source_dirs: list[Path] = []
        for path in dirs:
            resolved = path.expanduser().resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"Dossier introuvable : {resolved}")
            source_dirs.append(resolved)
        return source_dirs

    source_dirs = list(list_dataset_source_dirs(latest_only=latest_only))
    if not source_dirs:
        raise FileNotFoundError(
            "Aucune session data_mining_* trouvée dans data/images_extraites/. "
            "Lance la pipeline avec ENABLE_DATA_MINING=True, ou passe --dir."
        )
    return source_dirs


def _move_pair(image_path: Path, dest_dir: Path) -> None:
    """Déplace .jpg + .txt vers dest_dir (crée le dossier + classes.txt)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ensure_classes_txt(dest_dir)

    dest_img = dest_dir / image_path.name
    if dest_img.exists():
        dest_img.unlink()
    src_label = image_path.parent / f"{image_path.stem}.txt"
    shutil.move(str(image_path), str(dest_img))

    dest_label = dest_dir / f"{image_path.stem}.txt"
    if src_label.exists():
        if dest_label.exists():
            dest_label.unlink()
        shutil.move(str(src_label), str(dest_label))
    elif not dest_label.exists():
        dest_label.write_text("", encoding="utf-8")


def _move_pair_to_review(image_path: Path, session_dir: Path) -> None:
    _move_pair(image_path, review_dir(session_dir))


def _write_label(image_path: Path, lines: list[str]) -> None:
    label_path = image_path.with_suffix(".txt")
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    # Aligner le mtime du label sur l'image : seules les corrections LabelImg
    # doivent apparaître comme postérieures (cf. is_manual_edit), sinon une
    # deuxième passe classerait tout en « preserved ».
    if image_path.exists():
        image_stat = image_path.stat()
        os.utime(label_path, (image_stat.st_atime, image_stat.st_mtime))


def _match_source_path(result, by_path: dict[str, Path]) -> Path | None:
    """Retrouve l'image source d'un résultat (l'ordre du flux n'est pas garanti)."""
    raw = getattr(result, "path", None)
    if not raw:
        return None
    # Ultralytics LoadImagesAndVideos utilise Path.absolute() (pas resolve).
    key = str(Path(raw).absolute())
    found = by_path.get(key)
    if found is not None:
        return found
    return by_path.get(str(Path(raw).resolve()))


def _write_report(session_dir: Path, rows: list[dict[str, str | int | float]]) -> Path:
    report_path = session_dir / _REPORT_NAME
    fieldnames = ["image", "reason", "bucket", "n_boxes", "min_conf", "max_conf"]
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def process_directory(
    model: YOLO,
    session_dir: Path,
    *,
    low_conf: float,
    high_conf: float,
    always_review: frozenset[str],
    apply: bool,
    force: bool,
    device: int | str = 0,
) -> dict[str, int]:
    images = sorted(session_dir.glob("*.jpg"))
    total = len(images)

    print(f"\nDossier : {session_dir}/")
    print(f"{total} image(s) au niveau session (hors review/)")
    if not apply:
        print("Mode dry-run : aucune écriture (sauf rapport CSV).")

    stats = {
        "total": total,
        BUCKET_AUTO: 0,
        BUCKET_NEGATIF: 0,
        BUCKET_REVIEW: 0,
        BUCKET_PRESERVED: 0,
    }
    if not images:
        print("Aucune image .jpg à traiter.")
        return stats

    report_rows: list[dict[str, str | int | float]] = []
    by_path = {str(path.absolute()): path for path in images}
    unmatched = 0

    with source_list_file(images) as source:
        results = model.predict(
            source=source,
            conf=low_conf,
            stream=True,
            verbose=False,
            device=device,
            batch=_PREDICT_BATCH,
        )

        for index, result in enumerate(results, start=1):
            image_path = _match_source_path(result, by_path)
            if image_path is None:
                unmatched += 1
                print(f"  ⚠ image source non identifiée (ignorée) : {result.path}")
                continue

            reason = parse_mining_reason(image_path)
            confs = box_confs(result)
            lines = to_yolo_lines(result)

            if not force and is_manual_edit(image_path):
                bucket = BUCKET_PRESERVED
            else:
                bucket = classify_bucket(
                    reason=reason,
                    confs=confs,
                    high_conf=high_conf,
                    always_review=always_review,
                )

            stats[bucket] += 1
            report_rows.append(
                {
                    "image": image_path.name,
                    "reason": reason or "",
                    "bucket": bucket,
                    "n_boxes": len(confs),
                    "min_conf": f"{min(confs):.4f}" if confs else "",
                    "max_conf": f"{max(confs):.4f}" if confs else "",
                }
            )

            if apply and bucket != BUCKET_PRESERVED:
                if bucket == BUCKET_AUTO:
                    _write_label(image_path, lines)
                elif bucket == BUCKET_NEGATIF:
                    _write_label(image_path, [])
                elif bucket == BUCKET_REVIEW:
                    # Pré-écrire le .txt (pré-anno) avant déplacement
                    _write_label(image_path, lines)
                    _move_pair_to_review(image_path, session_dir)

            if index % 50 == 0 or index == total:
                print(
                    f"Progression : {index}/{total} "
                    f"(auto={stats[BUCKET_AUTO]}, negatif={stats[BUCKET_NEGATIF]}, "
                    f"review={stats[BUCKET_REVIEW]}, preserved={stats[BUCKET_PRESERVED]})"
                )

    report_path = _write_report(session_dir, report_rows)
    print("  Résultat :")
    print(f"    auto      : {stats[BUCKET_AUTO]}")
    print(f"    negatif   : {stats[BUCKET_NEGATIF]}")
    print(f"    review    : {stats[BUCKET_REVIEW]}")
    print(f"    preserved : {stats[BUCKET_PRESERVED]}")
    if unmatched:
        print(f"    ignorées  : {unmatched} (source non identifiée)")
    print(f"    rapport   : {report_path}")
    return stats


def _cleanup_subdir(path: Path, *, label: str) -> None:
    if not path.is_dir():
        return
    leftover = [p for p in path.iterdir() if p.name != "classes.txt"]
    if not leftover:
        classes = path / "classes.txt"
        if classes.exists():
            classes.unlink()
        try:
            path.rmdir()
            print(f"  {label}/ supprimé")
        except OSError:
            pass
    else:
        jpg_left = sum(1 for p in leftover if p.suffix.lower() == ".jpg")
        print(f"  {label}/ non vide ({jpg_left} image(s) restantes)")


def restore_review(
    session_dirs: list[Path],
    *,
    only_manual: bool = False,
) -> dict[str, int]:
    """Remonte review/* vers la session.

    Si only_manual : ne remonte que les labels corrigés (mtime .txt >> .jpg),
    le reste part dans session/skipped/ (exclu du split).
    """
    totals = {
        "sessions": 0,
        "restored": 0,
        "parked": 0,
        "conflicts": 0,
    }
    for session_dir in session_dirs:
        dest_review = review_dir(session_dir)
        if not dest_review.is_dir():
            continue
        totals["sessions"] += 1
        images = sorted(dest_review.glob("*.jpg"))
        mode = "only-manual" if only_manual else "all"
        print(
            f"\nRestore ({mode}) : {session_dir.name}/review/ "
            f"({len(images)} image(s))"
        )

        for image_path in images:
            if only_manual and not is_manual_edit(image_path):
                _move_pair(image_path, skipped_dir(session_dir))
                totals["parked"] += 1
                continue

            target_img = session_dir / image_path.name
            if target_img.exists():
                print(f"  conflit : {image_path.name}")
                totals["conflicts"] += 1
                continue

            shutil.move(str(image_path), str(target_img))
            review_label = dest_review / f"{image_path.stem}.txt"
            if review_label.exists():
                target_label = session_dir / review_label.name
                if target_label.exists():
                    target_label.unlink()
                shutil.move(str(review_label), str(target_label))
            totals["restored"] += 1

        _cleanup_subdir(dest_review, label="review")
        parked = skipped_dir(session_dir)
        if parked.is_dir():
            n = len(list(parked.glob("*.jpg")))
            if n:
                print(f"  skipped/ : {n} image(s) non relues (hors dataset)")

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ré-inférence des sessions data_mining_* : "
            "AUTO (high-conf), NEGATIF (hard neg), REVIEW (→ review/)."
        ),
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        action="append",
        dest="dirs",
        help="Dossier session explicite (répétable). Remplace la découverte auto.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help="Modèle .pt. Défaut : dernier models/apex_*/best.pt",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Uniquement la dernière session data_mining_* (ignoré si --dir)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Écrire les .txt et déplacer les REVIEW vers review/ (sinon dry-run).",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Remonter review/ vers la session (après LabelImg). Pas d'inférence.",
    )
    parser.add_argument(
        "--only-manual",
        action="store_true",
        help=(
            "Avec --restore : ne remonte que les labels corrigés dans LabelImg "
            "(mtime .txt > .jpg). Le reste part dans skipped/ (hors dataset)."
        ),
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Écraser aussi les labels considérés comme corrections manuelles.",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=None,
        help=f"Plancher d'inférence (défaut : {REINFER_LOW_CONF})",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help=f"Seuil auto-accept (défaut : {REINFER_HIGH_CONF})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device Ultralytics (ex. 0, cpu). Défaut : CUDA 0 si dispo, sinon cpu.",
    )
    args = parser.parse_args()

    try:
        source_dirs = resolve_source_dirs(dirs=args.dirs, latest_only=args.latest)
    except FileNotFoundError as exc:
        print(exc)
        return

    if args.restore:
        totals = restore_review(source_dirs, only_manual=args.only_manual)
        print("\nRestore terminé.")
        print(f"  Sessions : {totals['sessions']}")
        print(f"  Images remontées : {totals['restored']}")
        print(f"  Non relues → skipped/ : {totals['parked']}")
        print(f"  Conflits : {totals['conflicts']}")
        return

    if args.only_manual:
        print("--only-manual n'a d'effet qu'avec --restore.")
        return

    low_conf = args.low if args.low is not None else REINFER_LOW_CONF
    high_conf = args.high if args.high is not None else REINFER_HIGH_CONF
    if not (0.0 < low_conf <= high_conf <= 1.0):
        print(f"Seuils invalides : low={low_conf}, high={high_conf}")
        return

    always_review = frozenset(REINFER_ALWAYS_REVIEW_REASONS)

    if args.device is None:
        device: int | str = default_device()
    else:
        device = int(args.device) if str(args.device).isdigit() else args.device

    try:
        model_path = resolve_prelabel_model(model=args.model)
    except FileNotFoundError as exc:
        print(exc)
        return

    model = YOLO(str(model_path))
    print(f"Modèle chargé : {model_path}")
    print(f"Classes : {', '.join(CLASS_NAMES)}")
    print(f"Device : {device}")
    print(f"Seuils : low={low_conf:.2f}, high={high_conf:.2f}")
    print(f"Always review : {', '.join(sorted(always_review)) or '(aucun)'}")
    print(f"Sessions : {len(source_dirs)}")
    if args.apply:
        print("Mode --apply : écriture active.")
    else:
        print("Dry-run (passe --apply pour écrire).")
    if args.force:
        print("⚠ --force : corrections manuelles seront écrasées.")

    totals = {
        "total": 0,
        BUCKET_AUTO: 0,
        BUCKET_NEGATIF: 0,
        BUCKET_REVIEW: 0,
        BUCKET_PRESERVED: 0,
    }
    for session_dir in source_dirs:
        stats = process_directory(
            model,
            session_dir,
            low_conf=low_conf,
            high_conf=high_conf,
            always_review=always_review,
            apply=args.apply,
            force=args.force,
            device=device,
        )
        for key in totals:
            totals[key] += stats[key]

    print("\nRé-inférence terminée.")
    print(f"  Sessions : {len(source_dirs)}")
    print(f"  Images   : {totals['total']}")
    print(f"  auto     : {totals[BUCKET_AUTO]}")
    print(f"  negatif  : {totals[BUCKET_NEGATIF]}")
    print(f"  review   : {totals[BUCKET_REVIEW]}")
    print(f"  preserved: {totals[BUCKET_PRESERVED]}")
    pending = list_review_dirs()
    if args.apply and pending:
        print("\nÀ corriger dans LabelImg :")
        for path in pending:
            print(f"  {path}")
        print(
            "Puis : python scripts/reinfer_mining.py --latest --restore --only-manual"
        )


if __name__ == "__main__":
    main()
