from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DETECT_DIR = RUNS_DIR / "detect"

FOV_SIZE = 416
DEBUG = True
AIM_ASSIST = True
AIM_ASSIST_REQUIRE_LMB = True  # Si True, l'aim assist ne s'active que pendant le clic gauche

ENABLE_DATA_MINING = True
DATA_MINING_COOLDOWN = 0.5
DATA_MINING_SAVE_DIR = DATA_DIR / "auto_collected"

# Version active pour l'extraction / labeling / split (v1 = archive, v2 = nouveau dataset)
DATA_VERSION = "v2"

DERUSH_DIR = DATA_DIR / "derush" / DATA_VERSION
IMAGES_EXTRAITES_DIR = DATA_DIR / "images_extraites" / DATA_VERSION
DERUSH_V1_DIR = DATA_DIR / "derush" / "v1"
IMAGES_EXTRAITES_V1_DIR = DATA_DIR / "images_extraites" / "v1"
DATASET_TRAIN_DIR = DATA_DIR / "dataset" / "train"
DATASET_VAL_DIR = DATA_DIR / "dataset" / "val"
APEX_V2_YAML = ROOT_DIR / "apex_v2.yaml"
ROBOFLOW_DATASET_YAML = DATA_DIR / "datasets_roboflow" / "apex-dataset" / "data.yaml"
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
V1_MODEL = RUNS_DETECT_DIR / "apex_model_v1" / "weights" / "best.pt"
V2_MODEL = RUNS_DETECT_DIR / "apex_model_v2" / "weights" / "best.pt"
V3_MODEL = RUNS_DETECT_DIR / "apex_model_v3" / "weights" / "best.pt"
V2_ENGINE = V2_MODEL.with_suffix(".engine")
V3_ENGINE = V3_MODEL.with_suffix(".engine")


def resolve_active_model() -> Path:
    if V3_ENGINE.exists():
        return V3_ENGINE
    if V3_MODEL.exists():
        return V3_MODEL
    if V2_ENGINE.exists():
        return V2_ENGINE
    if V2_MODEL.exists():
        return V2_MODEL
    if V1_MODEL.exists():
        return V1_MODEL
    return DEFAULT_YOLO_MODEL


ACTIVE_MODEL = resolve_active_model()
