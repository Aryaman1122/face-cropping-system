from pathlib import Path

from PIL import Image


def crop_faces(
    image_path: str,
    detections: list,
    output_dir: str,
):
    """
    Crop detected faces from an image.

    Args:
        image_path: Path to the original image.
        detections: Detection dictionaries returned by FaceDetector.
        output_dir: Directory where face crops will be saved.

    Returns:
        List of dictionaries containing crop information.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = Image.open(image_path).convert("RGB")

    image_width, image_height = image.size

    crop_results = []

    for face_id, detection in enumerate(
        detections,
        start=1
    ):

        x1 = max(0, detection["x1"])
        y1 = max(0, detection["y1"])

        x2 = min(image_width, detection["x2"])
        y2 = min(image_height, detection["y2"])

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = image.crop(
            (x1, y1, x2, y2)
        )

        crop_filename = (
            f"{image_path.stem}_face_{face_id:03d}.jpg"
        )

        crop_path = output_dir / crop_filename

        face_crop.save(
            crop_path,
            format="JPEG",
            quality=95
        )

        crop_results.append(
            {
                "image_name": image_path.name,
                "face_id": face_id,
                "confidence": detection["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "face_area": detection["face_area"],
                "crop_path": str(crop_path),
            }
        )

    return crop_results