import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MINIMUM_FACE_AREA,
)
from src.detector import FaceDetector
from src.cropper import crop_faces


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Face Cropping System",
    page_icon="👤",
    layout="wide",
)


# --------------------------------------------------
# Load model only once
# --------------------------------------------------

@st.cache_resource
def load_detector():
    return FaceDetector(
        model_path=str(MODEL_PATH),
        confidence=CONFIDENCE_THRESHOLD,
        image_size=IMAGE_SIZE,
        minimum_face_area=MINIMUM_FACE_AREA,
    )


detector = load_detector()


# --------------------------------------------------
# Helper: draw bounding boxes
# --------------------------------------------------

def draw_detections(image, detections):
    """
    Draw detected face bounding boxes on a copy
    of the original image.
    """

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for face_id, detection in enumerate(
        detections,
        start=1
    ):
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        confidence = detection["confidence"]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3,
        )

        label = (
            f"Face {face_id} "
            f"{confidence:.2f}"
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            label,
            fill="red",
        )

    return annotated_image


# --------------------------------------------------
# Helper: create ZIP
# --------------------------------------------------

def create_zip(crop_results):
    """
    Create an in-memory ZIP file containing
    all generated face crops.
    """

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for crop in crop_results:

            crop_path = Path(
                crop["crop_path"]
            )

            if crop_path.exists():
                zip_file.write(
                    crop_path,
                    arcname=crop_path.name,
                )

    zip_buffer.seek(0)

    return zip_buffer


# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("👤 Face Cropping System")

st.write(
    "Upload an image to detect and crop all faces "
    "using YOLO11n-face."
)

st.divider()


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    # Load uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    # --------------------------------------------------
    # Temporary processing directory
    # --------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        input_path = (
            temp_dir / uploaded_file.name
        )

        output_dir = (
            temp_dir / "face_crops"
        )

        # Save uploaded image temporarily
        image.save(input_path)

        # --------------------------------------------------
        # Detect faces
        # --------------------------------------------------

        with st.spinner(
            "Detecting faces..."
        ):

            detections = detector.detect(
                str(input_path)
            )

        # --------------------------------------------------
        # Detection result
        # --------------------------------------------------

        st.divider()

        st.subheader(
            f"Detected Faces: {len(detections)}"
        )

        if len(detections) == 0:

            st.warning(
                "No faces were detected in this image."
            )

        else:

            # --------------------------------------------------
            # Annotated image
            # --------------------------------------------------

            annotated_image = draw_detections(
                image,
                detections,
            )

            st.image(
                annotated_image,
                caption="Detected Faces",
                use_container_width=True,
            )

            # --------------------------------------------------
            # Crop faces
            # --------------------------------------------------

            with st.spinner(
                "Creating face crops..."
            ):

                crop_results = crop_faces(
                    str(input_path),
                    detections,
                    str(output_dir),
                )

            st.divider()

            st.subheader(
                "Face Crops"
            )

            # --------------------------------------------------
            # Display crops
            # --------------------------------------------------

            columns = st.columns(
                min(4, len(crop_results))
            )

            for index, crop in enumerate(
                crop_results
            ):

                crop_path = Path(
                    crop["crop_path"]
                )

                crop_image = Image.open(
                    crop_path
                )

                with columns[
                    index % len(columns)
                ]:

                    st.image(
                        crop_image,
                        caption=(
                            f"Face {crop['face_id']} "
                            f"| Confidence: "
                            f"{crop['confidence']:.2f}"
                        ),
                        use_container_width=True,
                    )

                    with open(
                        crop_path,
                        "rb",
                    ) as file:

                        st.download_button(
                            label="Download",
                            data=file.read(),
                            file_name=crop_path.name,
                            mime="image/jpeg",
                            key=f"download_{index}",
                        )

            # --------------------------------------------------
            # Download all crops
            # --------------------------------------------------

            st.divider()

            zip_buffer = create_zip(
                crop_results
            )

            st.download_button(
                label="⬇️ Download All Face Crops",
                data=zip_buffer,
                file_name="face_crops.zip",
                mime="application/zip",
            )

            # --------------------------------------------------
            # Detection summary
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "Detection Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Faces Detected",
                    len(detections),
                )

            with col2:
                average_confidence = sum(
                    d["confidence"]
                    for d in detections
                ) / len(detections)

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.3f}",
                )

            with col3:
                st.metric(
                    "Minimum Face Area",
                    f"{min(d['face_area'] for d in detections):,} px²",
                )