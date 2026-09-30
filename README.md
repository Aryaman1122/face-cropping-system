# Face Cropping System

A YOLO11n-face based face detection and cropping system that detects multiple faces in human images, extracts individual face crops, validates the generated dataset, and provides an interactive Streamlit application for real-time image processing.

The project was developed through two major stages:

1. **Experimental Development in Google Colab** — dataset preparation, YOLO11n-face evaluation, confidence-threshold experiments, false-positive analysis, crop generation, and final dataset validation.
2. **Production Implementation in VS Code** — modular Python implementation with a Streamlit interface, Apple MPS acceleration, downloadable face crops, and a GitHub-ready project structure.

---

## 📌 Project Overview

The objective of this project is to build a practical face-cropping pipeline that takes an image containing one or more people and automatically:

* Detects faces using a pretrained YOLO-based face detector
* Handles both single-face and multi-face images
* Filters extremely small detections
* Extracts individual face crops
* Preserves detection confidence and bounding-box metadata
* Validates the generated face-crop dataset
* Provides an interactive Streamlit application for end users

The project uses **YOLO11n-face**, a YOLO-based face detection model trained specifically for face detection rather than a general-purpose COCO YOLO model.

---

## ✨ Key Features

* 👤 Face detection using YOLO11n-face
* 🖼️ Single-face and multi-face detection
* ✂️ Automatic face cropping
* 📦 Batch face-crop generation
* 🔍 Confidence-threshold experimentation
* 🔬 Small-face and false-positive analysis
* 📊 Dataset and detection statistics
* ✅ Final crop integrity validation
* 🖥️ Streamlit web interface
* ⬇️ Individual face-crop downloads
* 📦 Download all detected faces as a ZIP file
* ⚡ Apple MPS acceleration when available
* 🧪 Complete Google Colab experimental notebooks

---

# 🏗️ Project Workflow

```text
                         Input Dataset
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Dataset Preparation  │
                 │                        │
                 │ • CSV validation       │
                 │ • Image validation     │
                 │ • Bounding boxes       │
                 │ • GT visualization     │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    YOLO11n-face        │
                 │    Experimentation     │
                 │                        │
                 │ • Single images        │
                 │ • Multi-face images    │
                 │ • Threshold tests      │
                 │ • Small faces          │
                 │ • FP analysis          │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Final Configuration  │
                 │                        │
                 │ Confidence = 0.25      │
                 │ Image size = 1280      │
                 │ Min area = 200 px²     │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Face Detection       │
                 │   + Filtering          │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │     Face Cropping      │
                 │                        │
                 │ Individual crops       │
                 │ + metadata             │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    Final Validation    │
                 │                        │
                 │ • File integrity       │
                 │ • Statistics           │
                 │ • Duplicates           │
                 │ • Missing crops        │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │     Streamlit App      │
                 │                        │
                 │ Upload → Detect        │
                 │ → Crop → Download      │
                 └────────────────────────┘
```

---

# 📂 Dataset

The input dataset consists of human images accompanied by CSV-based face bounding-box annotations.

The annotation CSV contains approximately **3,350 annotated face instances** and uses the following fields:

| Column       | Description                    |
| ------------ | ------------------------------ |
| `image_name` | Source image filename          |
| `width`      | Image width                    |
| `height`     | Image height                   |
| `x0`         | Bounding-box left coordinate   |
| `y0`         | Bounding-box top coordinate    |
| `x1`         | Bounding-box right coordinate  |
| `y1`         | Bounding-box bottom coordinate |

A single image can contain multiple annotation rows because an image may contain multiple faces.

## Dataset Preparation

The dataset preparation stage performs:

1. CSV loading
2. Schema validation
3. Missing-value checks
4. Bounding-box validation
5. Image existence verification
6. Ground-truth bounding-box visualization
7. Ground-truth face cropping
8. Image-level dataset splitting

### Image-Level Dataset Splitting

The train/validation/test split is performed at the **image level** rather than at the annotation-row level.

This prevents different annotations belonging to the same source image from being distributed across multiple dataset splits.

---

# 🔬 Experimental Development

The complete experimental development is preserved in the `notebooks/` directory.

## Notebook 1 — Dataset Preparation

**File:**

```text
01_Dataset_Preparation.ipynb
```

This notebook documents the initial dataset preparation process, including:

* Google Drive setup
* Dataset path configuration
* CSV loading
* Annotation inspection
* Dataset statistics
* Bounding-box validation
* Image existence checks
* Ground-truth visualization
* Ground-truth face crop generation
* Image-level train/validation/test splitting

---

# 🤖 YOLO11n-face Experiments

## Model Selection

The project uses **YOLO11n-face**, a pretrained YOLO-based face detector.

The model was selected specifically for the face-detection stage rather than using a general-purpose object-detection model.

### Selected Model

```text
YOLO11n-face
```

The downloaded model file used by the local application is:

```text
models/yolov11n-face.pt
```

---

## Initial Detection Experiments

The model was evaluated on:

* Individual single-face images
* Multiple single-face images
* Multi-face images
* Small-face images
* Difficult images
* The complete image collection

The experiments were used to understand:

* Detection confidence
* Number of detections
* Multi-face behavior
* Small-face detection
* Low-confidence detections
* False positives
* Bounding-box behavior

---

# 📊 Confidence Threshold Experiments

Several confidence thresholds were evaluated:

| Confidence Threshold |
| -------------------: |
|                 0.25 |
|                 0.40 |
|                 0.50 |
|                 0.60 |
|                 0.70 |

The experiments showed that increasing the confidence threshold could remove some low-confidence detections, but it also caused some genuine detections to disappear.

In particular, several images that produced genuine detections at lower confidence produced zero detections at a confidence threshold of **0.60**.

Visual inspection of difficult low-confidence detections confirmed that some of those detections represented genuine faces.

Therefore, the final pipeline uses:

```text
Confidence threshold = 0.25
```

This prioritizes retaining genuine face detections, followed by a separate minimum-area filtering stage to remove pathological tiny detections.

---

# 🔍 Small-Face Analysis

The project also evaluated images containing small or distant faces.

Examples included faces with bounding-box dimensions around:

| Width × Height |
| -------------: |
|        30 × 44 |
|        40 × 34 |
|        30 × 46 |
|        33 × 42 |
|        32 × 45 |
|        29 × 51 |
|        38 × 39 |
|        31 × 48 |
|        30 × 50 |

The experiments showed that some genuine small faces could have relatively low detection confidence.

Therefore, confidence alone was not used as the final filtering mechanism.

---

# 🚨 False-Positive Analysis

A full-dataset experiment using a low confidence threshold revealed several extreme detection outliers.

For example, one image produced:

```text
300 detections
```

Visual inspection showed that most of these detections corresponded to tiny background regions rather than faces.

The detections were therefore further analyzed using bounding-box area.

This led to the introduction of a **minimum face-area filter**.

---

# 📐 Minimum Face-Area Filtering

Several minimum-area thresholds were evaluated:

| Minimum Area | Remaining Detections | Removed | Images Lost |
| -----------: | -------------------: | ------: | ----------: |
|       50 px² |                4,958 |      20 |           0 |
|      100 px² |                4,652 |     326 |           0 |
|      200 px² |                4,601 |     377 |           0 |
|      300 px² |                4,589 |     389 |           0 |
|      500 px² |                4,560 |     418 |           0 |
|      800 px² |                4,469 |     509 |           5 |
|    1,000 px² |                4,394 |     584 |          12 |
|    1,500 px² |                4,175 |     803 |          44 |
|    2,000 px² |                4,018 |     960 |          80 |

A minimum face area of:

```text
200 px²
```

was selected for the final pipeline.

This removed extremely small pathological detections while keeping all **2,204 input images** represented.

---

# ⚙️ Final Detection Configuration

The final configuration established through the experiments is:

| Parameter            | Value        |
| -------------------- | ------------ |
| Detector             | YOLO11n-face |
| Confidence threshold | 0.25         |
| Inference image size | 1280         |
| Minimum face area    | 200 px²      |
| NMS                  | YOLO default |

These settings are used consistently by the production implementation.

---

# 📦 Final Dataset Generation

Using the final configuration:

```text
Confidence = 0.25
Image size = 1280
Minimum face area = 200 px²
```

the pipeline initially produced:

```text
4,978 detections
```

The minimum-area filter removed:

```text
377 detections
```

resulting in:

```text
4,601 final face crops
```

---

# 📊 Final Dataset Statistics

The final dataset contains:

| Metric                     |         Value |
| -------------------------- | ------------: |
| Input images               |         2,204 |
| Final face crops           |         4,601 |
| Images with detected faces |         2,204 |
| Images without final crops |             0 |
| Average faces/image        |        2.0876 |
| Median faces/image         |             1 |
| Maximum faces/image        |            40 |
| Mean confidence            |        0.8281 |
| Median confidence          |        0.8624 |
| Minimum final face area    |       208 px² |
| Maximum face area          | 3,284,859 px² |
| Mean crop width            |     131.11 px |
| Mean crop height           |     169.52 px |

## Confidence Distribution

| Confidence Range | Detections | Percentage |
| ---------------- | ---------: | ---------: |
| 0.25–0.40        |        120 |      2.61% |
| 0.40–0.60        |        141 |      3.06% |
| 0.60–0.80        |        402 |      8.74% |
| 0.80–0.90        |      3,687 |     80.13% |
| 0.90–1.00        |        251 |      5.46% |

---

# ✅ Final Dataset Validation

The final dataset was independently validated.

| Validation Metric    | Result |
| -------------------- | -----: |
| Metadata records     |  4,601 |
| Existing crop files  |  4,601 |
| Missing crop files   |      0 |
| Unique source images |  2,204 |
| Input images         |  2,204 |
| Images without crops |      0 |
| Duplicate records    |      0 |
| Physical crop files  |  4,601 |

### Final Dataset Integrity

```text
FINAL DATASET INTEGRITY: PASSED
```

This validation is documented in:

```text
03_Final_Dataset_Validation.ipynb
```

---

# 💻 Production Implementation

After completing the Colab experimentation, the finalized pipeline was implemented locally using **Python and Streamlit**.

The production implementation separates the main responsibilities into modular components.

## `src/detector.py`

Responsible for:

* Loading YOLO11n-face
* Selecting the available device
* Running inference
* Extracting bounding boxes
* Extracting confidence scores
* Calculating face area
* Applying minimum-area filtering

## `src/cropper.py`

Responsible for:

* Loading the source image
* Validating/clipping bounding-box coordinates
* Extracting face regions
* Saving face crops
* Generating crop metadata

## `src/config.py`

Centralizes:

* Model path
* Output path
* Confidence threshold
* Inference image size
* Minimum face area

## `app.py`

Provides the Streamlit user interface.

---

# 🖥️ Streamlit Application

The application provides the following workflow:

```text
Upload Image
     │
     ▼
YOLO11n-face Detection
     │
     ▼
Minimum Area Filtering
     │
     ▼
Bounding Boxes
     │
     ▼
Face Cropping
     │
     ├───────────────┐
     ▼               ▼
Individual       Download
Face Crops        All ZIP
```

The interface displays:

* Original uploaded image
* Number of detected faces
* Bounding boxes
* Detection confidence
* Individual face crops
* Individual download buttons
* Download-all ZIP
* Detection summary

The model is cached using Streamlit's resource caching so that it does not need to be loaded again for every interaction.

---

# ⚡ Hardware Acceleration

The local implementation checks for Apple's **Metal Performance Shaders (MPS)** backend.

When available:

```text
device = mps
```

Otherwise:

```text
device = cpu
```

## Tested Environment

| Component   | Version / Status |
| ----------- | ---------------- |
| Python      | 3.13.5           |
| PyTorch     | 2.14.0           |
| Ultralytics | 8.4.165          |
| OpenCV      | 5.0.0            |
| NumPy       | 2.5.3            |
| Pandas      | 3.0.6            |
| Streamlit   | 1.64.0           |
| MPS         | Available        |

---

# 📁 Project Structure

```text
face-cropping-system/
│
├── app.py
├── requirements.txt
├── .gitignore
├── test_detector.py
│
├── models/
│   └── yolov11n-face.pt
│
├── src/
│   ├── config.py
│   ├── detector.py
│   └── cropper.py
│
└── notebooks/
    ├── 01_Dataset_Preparation.ipynb
    ├── 02_YOLO_Experiments.ipynb
    └── 03_Final_Dataset_Validation.ipynb
```

> The original dataset and generated face-crop collections are intentionally not included in the repository.

---

# 🛠️ Technologies Used

| Category                        | Technologies                                       |
| ------------------------------- | -------------------------------------------------- |
| Programming                     | Python                                             |
| Deep Learning / Computer Vision | YOLO11n-face, PyTorch, Ultralytics, OpenCV, Pillow |
| Data Processing                 | Pandas, NumPy, Scikit-learn                        |
| Visualization                   | Matplotlib                                         |
| Application                     | Streamlit                                          |
| Development                     | VS Code, Google Colab, Git, GitHub                 |
| Hardware Acceleration           | Apple MPS                                          |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Aryaman1122/face-cropping-system.git
```

## 2. Move into the Project Directory

```bash
cd face-cropping-system
```

## 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

## 4. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

Upload an image through the interface to detect and crop faces.

---

# 🧪 Running the Detector Test

The repository also contains:

```text
test_detector.py
```

Run it using:

```bash
python test_detector.py
```

This can be used to verify that the local detector and cropping pipeline are functioning correctly.

---

# 📓 Experimental Notebooks

The complete experimental workflow is available under:

```text
notebooks/
```

## 01 — Dataset Preparation

```text
01_Dataset_Preparation.ipynb
```

Covers:

* Preparation and validation of the original annotated dataset
* CSV inspection
* Bounding-box validation
* Image verification
* Ground-truth visualization
* Dataset splitting
* Ground-truth face crop generation

## 02 — YOLO Experiments

```text
02_YOLO_Experiments.ipynb
```

Contains the complete model experimentation process, including:

* Threshold testing
* Multi-face testing
* Small-face analysis
* False-positive investigation
* Full-dataset processing
* Final parameter selection

## 03 — Final Dataset Validation

```text
03_Final_Dataset_Validation.ipynb
```

Contains:

* Final integrity checks
* Dataset statistics
* Crop verification
* Duplicate checks
* Generated dataset summary

The notebooks intentionally preserve the experimental development process, including intermediate tests and investigative steps.

---

# 📈 Results Summary

The final pipeline processed:

```text
2,204 input images
```

and generated:

```text
4,601 final face crops
```

with:

| Result                     | Count |
| -------------------------- | ----: |
| Input images               | 2,204 |
| Final face crops           | 4,601 |
| Failed images              |     0 |
| Missing crop files         |     0 |
| Duplicate metadata records |     0 |

All **2,204 input images** had at least one final detected face after the final filtering stage.

---

# 🔎 Important Design Decisions

## Why Image-Level Splitting?

One image can contain multiple face annotations. Splitting individual CSV rows could therefore place annotations from the same source image into different subsets.

The split is therefore performed at the **image level**.

---

## Why Confidence = 0.25?

Experiments showed that higher thresholds could remove genuine low-confidence detections, including difficult and small faces.

A confidence threshold of **0.25** was therefore selected to prioritize retaining genuine detections.

---

## Why Minimum Face Area = 200 px²?

Very small detections were observed in background regions during full-dataset inference.

A minimum-area filter removed pathological tiny detections while retaining all source images.

---

## Why Not Use Only Confidence Filtering?

Some genuine small faces produced relatively low confidence scores.

Combining confidence-based detection with geometric area filtering provided a more suitable filtering strategy for this dataset.

---

# ⚠️ Limitations

The current system is a **face detection and cropping system**, not a face recognition system.

It does not:

* Identify a person's identity
* Perform facial recognition
* Track faces across video frames
* Classify demographic attributes
* Perform face alignment beyond the detector's bounding boxes
* Generate synthetic faces

The quality of a crop depends on the detector's predicted bounding box.

The final dataset statistics are specific to the evaluated image collection and should not be interpreted as a universal benchmark for YOLO11n-face.

---

# 🔮 Future Improvements

Potential extensions include:

* Video-based face detection
* Real-time webcam face cropping
* Face alignment
* Configurable confidence and area thresholds in the UI
* Batch-folder processing
* Detection performance benchmarking against additional face detectors
* Precision/recall evaluation using the available annotations
* Automated experiment tracking
* Docker-based deployment
* Cloud deployment of the Streamlit application

---

# 👨‍💻 Development Approach

The project followed an **experimental-to-production workflow**:

```text
Dataset Understanding
        ↓
Data Validation
        ↓
Ground-Truth Analysis
        ↓
Model Selection
        ↓
Single-Image Experiments
        ↓
Multi-Face Experiments
        ↓
Confidence Experiments
        ↓
Small-Face Analysis
        ↓
False-Positive Investigation
        ↓
Minimum-Area Selection
        ↓
Full Dataset Processing
        ↓
Final Dataset Validation
        ↓
Local Production Implementation
        ↓
Streamlit Application
        ↓
GitHub Repository
```

This repository therefore contains both the **final application** and the **experimental journey used to develop it**.

---

# 📜 License

Add a license here if you decide to distribute the project under a specific open-source license.

Also review the licensing and usage terms of the pretrained **YOLO11n-face weights** and the original dataset before redistributing them independently.

---

# 🙌 Acknowledgements

This project uses a pretrained YOLO-based face detection model and builds an application-specific face detection and cropping pipeline around it.

The project also uses the associated dataset and annotation structure for experimentation and validation.

---

## 📌 Summary

This project demonstrates an end-to-end **face detection and cropping pipeline**, beginning with dataset analysis and model experimentation in Google Colab and progressing to a modular local production implementation with Streamlit.

The final system combines:

**YOLO11n-face + confidence filtering + minimum-area filtering + face cropping + metadata generation + dataset validation + Streamlit deployment**

to provide a practical and reproducible face-cropping workflow.
