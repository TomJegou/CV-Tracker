from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Pipeline runtime ---
FOV_SIZE = 416
DEBUG = False
AIM_ASSIST = True
# Triggers aim (lock/assist). Si LMB et RMB à True → OU.
# REQUIRE_BOTH à True → LMB et RMB obligatoires (prioritaire sur le OU).
AIM_ASSIST_REQUIRE_LMB = True
AIM_ASSIST_REQUIRE_RMB = True
AIM_ASSIST_REQUIRE_BOTH = False
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
AIM_MODE = "assist"
LOCK_SCALE = 1.0
AIM_DEBUG_MOVES = False
MAX_SMOOTHING = 1.0
MAGNETIC_RADIUS = 100.0
# Point visé dans la box YOLO (0=bord gauche/haut, 0.5=centre, 1=bord droit/bas).
# Ex. Y=0.30 ≈ haut du corps / tête ; X=0.50 = centre horizontal.
AIM_POINT_X = 0.50
AIM_POINT_Y = 0.35
# Pull-down pendant ADS+tir (LMB+RMB), en px/s (positif = vers le bas). 0 = désactivé.
# Courbe : PEAK pendant PEAK_DURATION, transition sur DECAY, puis plateau DY_PER_S.
AIM_FIRE_PULL_PEAK_DY_PER_S = 350.0
AIM_FIRE_PULL_PEAK_DURATION_S = 0.20
AIM_FIRE_PULL_DECAY_S = 0.15
AIM_FIRE_PULL_DY_PER_S = 100.0

# --- No-recoil (jitter aim Apex) ---
# LMB+RMB → aller/retour sec X+Y (amplitude min–max px), via Arduino.
NO_RECOIL = True
NO_RECOIL_JITTER_MIN = 5     # amplitude px (axes X et Y)
NO_RECOIL_JITTER_MAX = 5
# Période du thread mouse en mode fusionné (jitter et/ou fire-pull).
NO_RECOIL_TICK_S = 0.001     # 1000 Hz
NO_RECOIL_DEBUG = False

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
# Bande FP suspect : [MIN, MAX) — MAX = CONF_THRESHOLD exclut les boxes aim-valides (conf >= seuil)
DATA_MINING_UNCERTAIN_MIN = 0.40
DATA_MINING_UNCERTAIN_MAX = 0.65
# FN suspect : LMB+RMB et meilleure conf ennemi < ce seuil (≤ CONF_THRESHOLD)
DATA_MINING_FN_MAX_CONF = 0.65
DATA_MINING_COOLDOWN_FP = 1
DATA_MINING_COOLDOWN_FN = 1

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
