from pathlib import Path

import torch
from ultralytics import YOLO


class FaceDetector:
    """
    YOLO11n-face based face detector.

    Uses the same configuration established during
    the Colab experimentation phase.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        image_size: int = 1280,
        minimum_face_area: int = 200,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.minimum_face_area = minimum_face_area

        # Use Apple Metal Performance Shaders when available.
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image_path: str):
        """
        Detect faces in an image.

        Returns:
            list of dictionaries containing:
            - bounding box coordinates
            - confidence
            - face area
        """

        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box)

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            face_area = width * height

            # Same minimum-area rule finalized in Colab.
            if face_area < self.minimum_face_area:
                continue

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(confidence),
                    "face_area": face_area,
                    "width": width,
                    "height": height,
                }
            )

        return detections