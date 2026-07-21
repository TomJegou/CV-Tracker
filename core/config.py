from dataclasses import dataclass
from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DETECT_DIR = ROOT_DIR / "runs" / "detect"

# --- Pipeline runtime ---
FOV_SIZE = 416
DEBUG = True
AIM_ASSIST = False
AIM_ASSIST_REQUIRE_LMB = False

# --- Détection ---
CONF_THRESHOLD = 0.65
AUTO_LABEL_CONF = 0.65
# Multiclass (aligné sur apex.yaml) — id 0 = ennemi, id 1 = allie
CLASS_NAMES = ("ennemi", "allie")
TARGET_CLASS_ID = 0
# Auto-label : ne pas écraser les .txt contenant déjà une box allié (classe 1)
AUTO_LABEL_PRESERVE_CLASS_IDS = (1,)

# --- Aim ---
# "lock" = snap direct (banc de test) | "assist" = friction magnétique
AIM_MODE = "lock"
LOCK_SCALE = 1.0
AIM_DEBUG_MOVES = True
MAX_SMOOTHING = 0.9
MAGNETIC_RADIUS = 150.0

# --- Data mining (FP / FN suspects) ---
ENABLE_DATA_MINING = True
DATA_MINING_SAVE_DIR = DATA_DIR / "auto_collected"
DATA_MINING_UNCERTAIN_MIN = 0.65
DATA_MINING_UNCERTAIN_MAX = 0.85
# FN suspect : LMB + RMB maintenus et meilleure conf < ce seuil
DATA_MINING_FN_MAX_CONF = 0.75
DATA_MINING_COOLDOWN_FP = 0.5
DATA_MINING_COOLDOWN_FN = 0.3

# --- Dataset / entraînement ---
# v4 = images minées avec le modèle V4 en jeu ; v2/v3 = historique (inclus au split)
DATA_VERSION = "v4"
DERUSH_DIR = DATA_DIR / "derush" / DATA_VERSION
IMAGES_EXTRAITES_DIR = DATA_DIR / "images_extraites" / DATA_VERSION
IMAGES_EXTRAITES_DIRS = (
    DATA_DIR / "images_extraites" / "v2",
    DATA_DIR / "images_extraites" / "v3",
    DATA_DIR / "images_extraites" / "v4",
)
DATASET_TRAIN_DIR = DATA_DIR / "dataset" / "train"
DATASET_VAL_DIR = DATA_DIR / "dataset" / "val"
APEX_DATASET_YAML = ROOT_DIR / "apex.yaml"
ROBOFLOW_DATASET_YAML = DATA_DIR / "datasets_roboflow" / "apex-dataset" / "data.yaml"

# --- Modèles (V4 = prod actuelle, V5 = prochain entraînement) ---
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
V1_MODEL = RUNS_DETECT_DIR / "apex_model_v1" / "weights" / "best.pt"
V2_MODEL = RUNS_DETECT_DIR / "apex_model_v2" / "weights" / "best.pt"
V3_MODEL = RUNS_DETECT_DIR / "apex_model_v3" / "weights" / "best.pt"
V4_MODEL = RUNS_DETECT_DIR / "apex_model_v4" / "weights" / "best.pt"
V5_MODEL = RUNS_DETECT_DIR / "apex_model_v5" / "weights" / "best.pt"
V2_ENGINE = V2_MODEL.with_suffix(".engine")
V3_ENGINE = V3_MODEL.with_suffix(".engine")
V4_ENGINE = V4_MODEL.with_suffix(".engine")
V5_ENGINE = V5_MODEL.with_suffix(".engine")

# Version cible par défaut : python scripts/train.py (sans argument)
TRAIN_TARGET_VERSION = "v5"


@dataclass(frozen=True)
class TrainProfile:
    version: str
    run_name: str
    weights_out: Path
    dataset_yaml: Path
    base_chain: tuple[Path, ...]
    epochs: int
    batch: int
    patience: int | None = None
    workers: int = 4


TRAIN_PROFILES: dict[str, TrainProfile] = {
    "v1": TrainProfile(
        version="v1",
        run_name="apex_model_v1",
        weights_out=V1_MODEL,
        dataset_yaml=ROBOFLOW_DATASET_YAML,
        base_chain=(DEFAULT_YOLO_MODEL,),
        epochs=100,
        batch=32,
        patience=25,
    ),
    "v2": TrainProfile(
        version="v2",
        run_name="apex_model_v2",
        weights_out=V2_MODEL,
        dataset_yaml=APEX_DATASET_YAML,
        base_chain=(V1_MODEL, DEFAULT_YOLO_MODEL),
        epochs=50,
        batch=16,
    ),
    "v3": TrainProfile(
        version="v3",
        run_name="apex_model_v3",
        weights_out=V3_MODEL,
        dataset_yaml=APEX_DATASET_YAML,
        base_chain=(V2_MODEL, V1_MODEL, DEFAULT_YOLO_MODEL),
        epochs=50,
        batch=16,
    ),
    "v4": TrainProfile(
        version="v4",
        run_name="apex_model_v4",
        weights_out=V4_MODEL,
        dataset_yaml=APEX_DATASET_YAML,
        base_chain=(V3_MODEL, V2_MODEL, V1_MODEL, DEFAULT_YOLO_MODEL),
        epochs=50,
        batch=16,
    ),
    "v5": TrainProfile(
        version="v5",
        run_name="apex_model_v5",
        weights_out=V5_MODEL,
        dataset_yaml=APEX_DATASET_YAML,
        base_chain=(V4_MODEL, V3_MODEL, V2_MODEL, V1_MODEL, DEFAULT_YOLO_MODEL),
        epochs=50,
        batch=16,
    ),
}


def get_train_profile(version: str | None = None) -> TrainProfile:
    key = (version or TRAIN_TARGET_VERSION).lower()
    if key not in TRAIN_PROFILES:
        available = ", ".join(sorted(TRAIN_PROFILES))
        raise ValueError(f"Version inconnue : {key!r}. Disponibles : {available}")
    return TRAIN_PROFILES[key]


def resolve_train_base(profile: TrainProfile) -> Path:
    """Premier poids disponible dans la chaîne de fine-tune du profil."""
    for path in profile.base_chain:
        if path.exists():
            return path
    return profile.base_chain[-1]


def resolve_active_model() -> Path:
    """Modèle utilisé par la pipeline runtime (V5 prioritaire si entraîné)."""
    for path in (
        V5_ENGINE,
        V5_MODEL,
        V4_ENGINE,
        V4_MODEL,
        V3_ENGINE,
        V3_MODEL,
        V2_ENGINE,
        V2_MODEL,
        V1_MODEL,
    ):
        if path.exists():
            return path
    return DEFAULT_YOLO_MODEL


def resolve_prelabel_model() -> Path:
    """Meilleur .pt disponible pour pré-annoter de nouvelles images."""
    for path in (V4_MODEL, V3_MODEL, V2_MODEL, V1_MODEL, DEFAULT_YOLO_MODEL):
        if path.exists():
            return path
    return DEFAULT_YOLO_MODEL


def resolve_train_base_model(version: str | None = None) -> Path:
    """Poids de départ pour entraîner la version demandée (défaut : TRAIN_TARGET_VERSION)."""
    return resolve_train_base(get_train_profile(version))

