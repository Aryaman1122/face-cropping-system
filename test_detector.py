from pathlib import Path

from src.detector import FaceDetector
from src.cropper import crop_faces


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "yolov11n-face.pt"
IMAGE_PATH = PROJECT_DIR / "data" / "00000930.jpg"

TEST_OUTPUT_DIR = PROJECT_DIR / "outputs" / "test_crops"


# ------------------------------------------------------------
# Create detector
# ------------------------------------------------------------

detector = FaceDetector(
    model_path=str(MODEL_PATH),
    confidence=0.25,
    image_size=1280,
    minimum_face_area=200,
)


# ------------------------------------------------------------
# Detect faces
# ------------------------------------------------------------

detections = detector.detect(
    image_path=str(IMAGE_PATH)
)


# ------------------------------------------------------------
# Crop faces
# ------------------------------------------------------------

crop_results = crop_faces(
    image_path=str(IMAGE_PATH),
    detections=detections,
    output_dir=str(TEST_OUTPUT_DIR),
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("=" * 60)
print("FACE DETECTION + CROPPING TEST")
print("=" * 60)

print("Input image:", IMAGE_PATH)
print("Device:", detector.device)

print("\nFaces detected:", len(detections))
print("Crops created:", len(crop_results))

for result in crop_results:

    print(
        f"Face {result['face_id']}: "
        f"confidence={result['confidence']:.4f}, "
        f"size={result['crop_width']}x"
        f"{result['crop_height']}, "
        f"crop={result['crop_path']}"
    )

print("\nOutput directory:")
print(TEST_OUTPUT_DIR)

print("\nTest completed successfully.")