from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

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
DATA_MINING_UNCERTAIN_MIN = 0.65
DATA_MINING_UNCERTAIN_MAX = 0.85
# FN suspect : LMB + RMB maintenus et meilleure conf < ce seuil
DATA_MINING_FN_MAX_CONF = 0.75
DATA_MINING_COOLDOWN_FP = 0.5
DATA_MINING_COOLDOWN_FN = 0.3

# --- Dataset / entraînement ---
# Dossiers images : core/dataset_paths.py | modèles : core/model_paths.py (models/apex_{NNN}/)
DATASET_TRAIN_DIR = DATA_DIR / "dataset" / "train"
DATASET_VAL_DIR = DATA_DIR / "dataset" / "val"
APEX_DATASET_YAML = ROOT_DIR / "apex.yaml"

DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
TRAIN_EPOCHS = 50
TRAIN_BATCH = 16
TRAIN_WORKERS = 4
TRAIN_PATIENCE: int | None = None
