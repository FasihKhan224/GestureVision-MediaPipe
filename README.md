# GestureVision-MediaPipe

A real-time computer vision pipeline for hand gesture recognition and motion trajectory classification powered by Google MediaPipe and TensorFlow Lite.

---

## Overview

**GestureVision-MediaPipe** delivers low-latency, accurate hand pose estimation and gesture interpretation directly from a webcam feed. By extracting normalized 21-point 3D hand landmarks via MediaPipe and processing them through lightweight neural network architectures, the system achieves real-time inference without requiring dedicated GPU hardware.

The pipeline operates on two distinct classification layers:
1. **Keypoint Classifier (Static Signs)**: Recognizes discrete hand poses (e.g., Open, Closed, Pointing, OK, etc.) based on instantaneous landmark geometry.
2. **Point History Classifier (Dynamic Gestures)**: Analyzes fingertip coordinate sequences over time to identify continuous motion patterns (e.g., Clockwise, Counter-Clockwise, Moving, Stationary).

---

## Key Features

- **Real-Time Landmark Detection**: 21 3D hand keypoints tracked at 30+ FPS on standard CPUs.
- **Dual-Mode Classification**: Simultaneous support for static hand signs and temporal movement gestures.
- **Interactive Dataset Collection**: Built-in interactive logging mode to capture, label, and append custom gestures on the fly.
- **Optimized for Edge Deployment**: Trained models exported to compact TensorFlow Lite (`.tflite`) format for fast inference.
- **End-to-End Retraining Workflows**: Jupyter notebooks included for training, tuning, and exporting updated models.

---

## Repository Structure

```
GestureVision-MediaPipe/
├── app.py                               # Main real-time webcam inference application
├── keypoint_classification.ipynb        # Model training pipeline for static hand signs
├── point_history_classification.ipynb   # Model training pipeline for dynamic finger gestures
├── model/
│   ├── keypoint_classifier/             # Static pose model, labels, dataset, and inference module
│   │   ├── keypoint.csv
│   │   ├── keypoint_classifier.hdf5
│   │   ├── keypoint_classifier.py
│   │   ├── keypoint_classifier.tflite
│   │   └── keypoint_classifier_label.csv
│   └── point_history_classifier/        # Dynamic gesture model, labels, dataset, and inference module
│       ├── point_history.csv
│       ├── point_history_classifier.hdf5
│       ├── point_history_classifier.py
│       ├── point_history_classifier.tflite
│       └── point_history_classifier_label.csv
└── utils/
    └── cvfpscalc.py                     # FPS calculation utility
```

---

## Requirements & Setup

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install mediapipe opencv-python tensorflow scikit-learn matplotlib
```

---

## Quick Start

Launch the real-time inference application with default webcam settings:

```bash
python app.py
```

### Command-Line Arguments

Customize camera parameters and inference thresholds:

```bash
python app.py --device 0 --width 960 --height 540 --min_detection_confidence 0.7 --min_tracking_confidence 0.5
```

| Parameter | Default | Description |
|---|---|---|
| `--device` | `0` | Camera device index |
| `--width` | `960` | Camera capture width |
| `--height` | `540` | Camera capture height |
| `--use_static_image_mode` | `False` | Static image mode flag for MediaPipe |
| `--min_detection_confidence` | `0.5` | Landmark detector confidence threshold |
| `--min_tracking_confidence` | `0.5` | Landmark tracking confidence threshold |

---

## Custom Gesture Training

### 1. Data Collection Mode

You can log custom samples directly within the running application:

- **Static Hand Signs**: Press `k` to enter keypoint logging mode. Press `0`–`9` to assign classes and record landmark vectors to `model/keypoint_classifier/keypoint.csv`.
- **Dynamic Motion Gestures**: Press `h` to enter trajectory logging mode. Move your fingertip and press `0`–`9` to record sequential coordinate histories into `model/point_history_classifier/point_history.csv`.
- Press `Esc` or `q` to exit logging mode.

### 2. Model Retraining

1. Open `keypoint_classification.ipynb` or `point_history_classification.ipynb` in Jupyter Notebook.
2. Adjust `NUM_CLASSES` to match your label set in the corresponding `_label.csv` file.
3. Run all cells to train the architecture and export updated `.tflite` model weights.

---

## Tech Stack

- **Framework**: MediaPipe Hands
- **Deep Learning**: TensorFlow / Keras, TensorFlow Lite
- **Vision Processing**: OpenCV (Python)
- **Data & Evaluation**: NumPy, Scikit-learn, Matplotlib
