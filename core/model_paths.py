"""Découverte dynamique des modèles sous models/apex_{NNN}/."""
from pathlib import Path

from core.config import DEFAULT_YOLO_MODEL, MODELS_DIR

APEX_MODEL_DIR_PREFIX = "apex_"
APEX_MODEL_INDEX_WIDTH = 3  # apex_001, apex_002, …


def format_apex_dir_name(index: int) -> str:
    return f"{APEX_MODEL_DIR_PREFIX}{index:0{APEX_MODEL_INDEX_WIDTH}d}"


def _parse_indexed_dir(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def list_apex_model_dirs() -> list[Path]:
    if not MODELS_DIR.exists():
        return []

    indexed: list[tuple[int, Path]] = []
    for path in MODELS_DIR.iterdir():
        if not path.is_dir():
            continue
        index = _parse_indexed_dir(path.name, APEX_MODEL_DIR_PREFIX)
        if index is not None:
            indexed.append((index, path))

    return [path for _, path in sorted(indexed)]


def next_apex_model_index() -> int:
    dirs = list_apex_model_dirs()
    if not dirs:
        return 1
    index = _parse_indexed_dir(dirs[-1].name, APEX_MODEL_DIR_PREFIX)
    return (index or 0) + 1


def create_next_apex_model_dir() -> Path:
    """Retourne le chemin models/apex_{NNN}/ pour le prochain run (créé par Ultralytics)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / format_apex_dir_name(next_apex_model_index())


def weights_pt_path(model_dir: Path) -> Path:
    return model_dir / "weights" / "best.pt"


def weights_engine_path(model_dir: Path) -> Path:
    return model_dir / "weights" / "best.engine"


def resolve_train_base(*, base: Path | None = None) -> Path:
    """Poids de départ pour un nouvel entraînement (dernier apex_*/best.pt ou YOLO de base)."""
    if base is not None:
        resolved = base.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Modèle de base introuvable : {resolved}")
        return resolved

    for model_dir in reversed(list_apex_model_dirs()):
        pt_path = weights_pt_path(model_dir)
        if pt_path.exists():
            return pt_path

    return DEFAULT_YOLO_MODEL


def resolve_active_model(*, model: Path | None = None) -> Path:
    """Modèle runtime : override CLI, sinon dernier apex_* (.engine prioritaire)."""
    if model is not None:
        resolved = model.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Modèle introuvable : {resolved}")
        return resolved

    for model_dir in reversed(list_apex_model_dirs()):
        engine_path = weights_engine_path(model_dir)
        if engine_path.exists():
            return engine_path
        pt_path = weights_pt_path(model_dir)
        if pt_path.exists():
            return pt_path

    return DEFAULT_YOLO_MODEL


def resolve_prelabel_model(*, model: Path | None = None) -> Path:
    """Meilleur .pt pour auto-label (pas d'engine)."""
    if model is not None:
        resolved = model.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Modèle introuvable : {resolved}")
        return resolved

    for model_dir in reversed(list_apex_model_dirs()):
        pt_path = weights_pt_path(model_dir)
        if pt_path.exists():
            return pt_path

    return DEFAULT_YOLO_MODEL


def resolve_export_model(*, model: Path | None = None) -> Path:
    """best.pt à exporter en TensorRT (dernier apex_* par défaut)."""
    if model is not None:
        resolved = model.expanduser().resolve()
        if resolved.is_dir():
            pt_path = weights_pt_path(resolved)
            if pt_path.exists():
                return pt_path
            raise FileNotFoundError(f"best.pt introuvable dans : {resolved}")
        if not resolved.exists():
            raise FileNotFoundError(f"Modèle introuvable : {resolved}")
        return resolved

    for model_dir in reversed(list_apex_model_dirs()):
        pt_path = weights_pt_path(model_dir)
        if pt_path.exists():
            return pt_path

    raise FileNotFoundError(
        "Aucun models/apex_*/weights/best.pt trouvé. "
        "Lance d'abord : python scripts/train.py"
    )
