from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "yolov11n-face.pt"

OUTPUT_DIR = PROJECT_DIR / "outputs"

# Face detection configuration
CONFIDENCE_THRESHOLD = 0.25

IMAGE_SIZE = 1280

MINIMUM_FACE_AREA = 200
