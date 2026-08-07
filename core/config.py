from pathlib import Path

# =============================================================================
# Chemins de base
# =============================================================================

# Racine du projet (dossier qui contient main.py, core/, data/, models/, …).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Données (images_extraites, dataset train/val, …).
DATA_DIR = ROOT_DIR / "data"

# Poids YOLO et runs d'entraînement (models/apex_{NNN}/).
MODELS_DIR = ROOT_DIR / "models"

# =============================================================================
# Affichage / debug runtime
# =============================================================================

# Fenêtre OpenCV séparée avec le crop FOV annoté (pas un overlay in-game).
DEBUG = False

# Overlay transparent click-through au-dessus du jeu (cadre FOV + boxes).
# Fonctionne mieux en borderless ; F8 = show/hide.
OVERLAY = False

# Affiche une croix au centre du FOV sur l'overlay.
OVERLAY_SHOW_CROSSHAIR = False

# Affiche le cercle de rayon MAGNETIC_RADIUS sur l'overlay.
OVERLAY_SHOW_MAGNETIC_RADIUS = True

# =============================================================================
# Capture
# =============================================================================

# Taille du crop carré centré à l'écran (px). Doit matcher imgsz d'entraînement.
FOV_SIZE = 32*13

# Pause (s) si dxcam.grab() renvoie None, pour éviter un busy-loop CPU. 0 = off.
CAPTURE_IDLE_SLEEP_S = 0.0001

# =============================================================================
# Détection YOLO
# =============================================================================

# Seuil de confiance pour l'aim / le targeting (boxes « propres »).
CONF_THRESHOLD = 0.62

# Seuil de confiance pour le pré-labeling (scripts/auto_label.py).
AUTO_LABEL_CONF = 0.60

# Noms des classes, alignés sur apex.yaml (index = class_id).
CLASS_NAMES = ("ennemi", "allie")

# Classe ciblée par l'aim assist (0 = ennemi).
TARGET_CLASS_ID = 0

# =============================================================================
# Aim assist
# =============================================================================

# Active le thread mouse + injection des deltas aim via Arduino.
AIM_ASSIST = True

# Triggers aim. Si LMB et RMB à True → OU logique.
# REQUIRE_BOTH=True → LMB et RMB obligatoires (prioritaire sur le OU).
AIM_ASSIST_REQUIRE_LMB = True
AIM_ASSIST_REQUIRE_RMB = True
AIM_ASSIST_REQUIRE_BOTH = False

# "lock" = snap direct (banc de test) | "assist" = friction magnétique.
AIM_MODE = "assist"

# Facteur d'échelle des deltas en mode "lock" (1.0 = 1 px écran → 1 px HID).
LOCK_SCALE = 1.0

# Log stdout de chaque SNAP aim ([aim] dx/dy → <move>).
AIM_DEBUG_MOVES = False

# Intensité max du lissage en mode "assist" (plus haut = plus agressif près du centre).
MAX_SMOOTHING = 1.30

# Rayon (px) autour du réticule dans lequel l'aim assist attire la souris.
MAGNETIC_RADIUS = 120.0

# Point visé dans la box YOLO (0 = bord gauche/haut, 0.5 = centre, 1 = bord droit/bas).
AIM_POINT_X = 0.50
AIM_POINT_Y = 0.33

# =============================================================================
# Compensation spray (jitter + pull-down)
# =============================================================================

# Active le tremblement sec X/Y pendant LMB+RMB (via Arduino).
ACTIVE_JITTER = True

# Active le pull-down vertical pendant LMB+RMB (via Arduino).
ACTIVE_PULL_DOWN = True

# Pull-down : courbe PEAK → DECAY → plateau (px/s). Ignoré si ACTIVE_PULL_DOWN=False.
AIM_FIRE_PULL_PEAK_DY_PER_S = 350.0
AIM_FIRE_PULL_PEAK_DURATION_S = 0.20
AIM_FIRE_PULL_DECAY_S = 0.15
AIM_FIRE_PULL_DY_PER_S = 90.0

# Amplitude min/max du jitter (px) sur les axes X et Y.
NO_RECOIL_JITTER_MIN = 5
NO_RECOIL_JITTER_MAX = 5

# Période du thread mouse fusionné (jitter et/ou pull), en secondes.
NO_RECOIL_TICK_S = 0.005

# Logs périodiques [mouse] (LMB/RMB, aim, jitter, pull).
NO_RECOIL_DEBUG = False

# =============================================================================
# Arduino Leonardo (injection HID via Serial)
# =============================================================================

# Port COM : None / "auto" = détection auto ; sinon override (ex. "COM5").
ARDUINO_PORT: str | None = None

# Baud rate Serial (doit matcher le sketch Arduino).
ARDUINO_BAUD = 115200

# Pause (s) à l'ouverture du port (reset CDC Leonardo). 0 = skip.
ARDUINO_SETTLE_S = 2.0

# Nombre de tentatives d'ouverture du COM (souvent verrouillé 1–3 s après Ctrl+C).
ARDUINO_OPEN_RETRIES = 8

# Délai (s) entre deux tentatives d'ouverture du COM.
ARDUINO_OPEN_RETRY_S = 0.4

# =============================================================================
# Data mining (FP / FN suspects pendant le jeu)
# =============================================================================

# Active la collecte async des frames ambiguës vers data_mining_{NNN}/.
ENABLE_DATA_MINING = False

# Plancher YOLO quand mining ON (une passe ; l'aim filtre ensuite à CONF_THRESHOLD).
DATA_MINING_CONF = 0.50

# Bande FP suspect : conf ∈ [MIN, MAX). MAX devrait ≈ CONF_THRESHOLD.
DATA_MINING_UNCERTAIN_MIN = 0.50
DATA_MINING_UNCERTAIN_MAX = 0.65

# FN suspect : LMB+RMB et meilleure conf ennemi < ce seuil.
DATA_MINING_FN_MAX_CONF = 0.65

# Cooldown (s) anti-spam entre deux captures FP / FN.
DATA_MINING_COOLDOWN_FP = 1
DATA_MINING_COOLDOWN_FN = 1

# =============================================================================
# Ré-inférence mining (triage auto / review)
# =============================================================================

# Plancher de confiance YOLO pour la ré-inférence (= DATA_MINING_CONF en pratique).
REINFER_LOW_CONF = 0.40

# Au-dessus de ce seuil : auto-accepte sans passer en review/.
REINFER_HIGH_CONF = 0.85

# Raisons de mining toujours envoyées en review/ (même si conf haute).
REINFER_ALWAYS_REVIEW_REASONS = ("enemy_as_ally_suspect",)

# Si mtime(.txt) > mtime(.jpg) + marge (s) → label considéré comme relu à la main.
REINFER_MANUAL_EDIT_MARGIN_S = 60.0

# =============================================================================
# Dataset / entraînement
# =============================================================================
# Sources images : core/dataset_paths.py | modèles : core/model_paths.py

# Sortie du split train (scripts/split_dataset.py).
DATASET_TRAIN_DIR = DATA_DIR / "dataset" / "train"

# Sortie du split val (scripts/split_dataset.py).
DATASET_VAL_DIR = DATA_DIR / "dataset" / "val"

# YAML Ultralytics (nc, names, chemins train/val).
APEX_DATASET_YAML = ROOT_DIR / "apex.yaml"

# Poids YOLO de fallback si aucun models/apex_*/best.pt n'existe.
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"

# Nombre d'epochs d'entraînement (scripts/train.py).
TRAIN_EPOCHS = 75

# Taille de batch d'entraînement.
TRAIN_BATCH = 16

# Nombre de workers DataLoader.
TRAIN_WORKERS = 4

# Early stopping : epochs sans amélioration avant arrêt. None = désactivé.
TRAIN_PATIENCE: int | None = None
