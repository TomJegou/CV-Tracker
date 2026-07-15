from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DETECT_DIR = RUNS_DIR / "detect"

FOV_SIZE = 416
DEBUG = True
AIM_ASSIST = False

DERUSH_DIR = DATA_DIR / "derush"
IMAGES_EXTRAITES_DIR = DATA_DIR / "images_extraites"
DATASET_TRAIN_DIR = DATA_DIR / "dataset" / "train"
DATASET_VAL_DIR = DATA_DIR / "dataset" / "val"
APEX_V2_YAML = ROOT_DIR / "apex_v2.yaml"
ROBOFLOW_DATASET_YAML = DATA_DIR / "datasets_roboflow" / "apex-dataset" / "data.yaml"
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
V1_MODEL = RUNS_DETECT_DIR / "apex_model_v1" / "weights" / "best.pt"
V2_MODEL = RUNS_DETECT_DIR / "apex_model_v2" / "weights" / "best.pt"
V2_ENGINE = V2_MODEL.with_suffix(".engine")


def resolve_active_model() -> Path:
    if V2_ENGINE.exists():
        return V2_ENGINE
    if V2_MODEL.exists():
        return V2_MODEL
    if V1_MODEL.exists():
        return V1_MODEL
    return DEFAULT_YOLO_MODEL


ACTIVE_MODEL = resolve_active_model()
