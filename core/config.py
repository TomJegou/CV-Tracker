from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Pipeline runtime ---
FOV_SIZE = 416
DEBUG = True
AIM_ASSIST = True
AIM_ASSIST_REQUIRE_LMB = False
# Pause capture si dxcam.grab() renvoie None (évite busy-loop CPU). 0 = désactivé.
CAPTURE_IDLE_SLEEP_S = 0.001

# --- Détection ---
# Seuil aim / targeting (boxes « propres »)
CONF_THRESHOLD = 0.65
AUTO_LABEL_CONF = 0.65
# Multiclass (aligné sur apex.yaml) — id 0 = ennemi, id 1 = allie
CLASS_NAMES = ("ennemi", "allie")
TARGET_CLASS_ID = 0

# --- Aim ---
# "lock" = snap direct (banc de test) | "assist" = friction magnétique
AIM_MODE = "lock"
LOCK_SCALE = 1.0
AIM_DEBUG_MOVES = True
MAX_SMOOTHING = 0.9
MAGNETIC_RADIUS = 150.0

# --- Arduino Leonardo (injection HID via Serial) ---
ARDUINO_PORT = "COM5"
ARDUINO_BAUD = 115200
# Pause à l'ouverture du port (reset CDC Leonardo). 0 pour skip.
ARDUINO_SETTLE_S = 2.0
# Windows garde souvent le COM verrouillé ~1–3 s après Ctrl+C / fermeture.
ARDUINO_OPEN_RETRIES = 8
ARDUINO_OPEN_RETRY_S = 0.4

# --- Data mining (FP / FN suspects) ---
ENABLE_DATA_MINING = False
# Plancher YOLO quand mining ON (une passe ; aim filtre ensuite à CONF_THRESHOLD)
DATA_MINING_CONF = 0.40
# Bande FP suspect (doit être ≥ DATA_MINING_CONF pour être visible)
DATA_MINING_UNCERTAIN_MIN = 0.40
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
