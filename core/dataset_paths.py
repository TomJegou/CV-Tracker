"""Découverte dynamique des dossiers dataset sous data/images_extraites/."""
from pathlib import Path

from core.config import CLASS_NAMES, DATA_DIR

IMAGES_EXTRAITES_ROOT = DATA_DIR / "images_extraites"
DATA_MINING_DIR_PREFIX = "data_mining_"
DATA_MINING_INDEX_WIDTH = 3  # data_mining_001, data_mining_002, …


def format_data_mining_dir_name(index: int) -> str:
    return f"{DATA_MINING_DIR_PREFIX}{index:0{DATA_MINING_INDEX_WIDTH}d}"


def _parse_indexed_dir(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def ensure_classes_txt(directory: Path) -> None:
    classes_path = directory / "classes.txt"
    if not classes_path.exists():
        classes_path.write_text(
            "\n".join(CLASS_NAMES) + "\n",
            encoding="utf-8",
        )


def list_data_mining_dirs() -> list[Path]:
    if not IMAGES_EXTRAITES_ROOT.exists():
        return []

    indexed: list[tuple[int, Path]] = []
    for path in IMAGES_EXTRAITES_ROOT.iterdir():
        if not path.is_dir():
            continue
        index = _parse_indexed_dir(path.name, DATA_MINING_DIR_PREFIX)
        if index is not None:
            indexed.append((index, path))

    return [path for _, path in sorted(indexed)]


def next_data_mining_index() -> int:
    dirs = list_data_mining_dirs()
    if not dirs:
        return 1
    last = dirs[-1].name
    index = _parse_indexed_dir(last, DATA_MINING_DIR_PREFIX)
    return (index or 0) + 1


def create_data_mining_session_dir() -> Path:
    """Crée data/images_extraites/data_mining_{NNN} pour une nouvelle session de collecte."""
    IMAGES_EXTRAITES_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = IMAGES_EXTRAITES_ROOT / format_data_mining_dir_name(next_data_mining_index())
    session_dir.mkdir(parents=True, exist_ok=True)
    ensure_classes_txt(session_dir)
    return session_dir


def list_dataset_source_dirs() -> tuple[Path, ...]:
    """Toutes les sources pour split_dataset (sessions data mining)."""
    return tuple(list_data_mining_dirs())


def list_auto_label_dirs(*, latest_only: bool = False) -> list[Path]:
    """Dossiers data_mining_* à pré-annoter."""
    mining_dirs = list_data_mining_dirs()
    if not mining_dirs:
        return []
    if latest_only:
        return [mining_dirs[-1]]
    return mining_dirs
