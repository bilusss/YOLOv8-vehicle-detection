# 🚦 YOLOv8 Traffic Object Detection

This repository contains the implementation and training process of a **YOLOv8-Large** model for detecting traffic-related objects (cars, pedestrians, signs, etc.) using the [Traffic Road Object Detection Dataset](https://www.kaggle.com/datasets/boukraailyesali/traffic-road-object-detection-dataset-using-yolo).

---

## 📊 Dataset Overview

### 📦 Source
Dataset: [Traffic Road Object Detection Dataset Using YOLO](https://www.kaggle.com/datasets/boukraailyesali/traffic-road-object-detection-dataset-using-yolo)

This dataset consists of real-world road scenes containing:
- 🚗 **Cars**

It is designed for **vehicle detection** in computer vision systems like autonomous driving or traffic monitoring.

### 🧩 Format
The dataset is pre-formatted for **YOLO**, with image-label pairs organized into `train/` and `val/` folders:

```

dataset/
├── train/
│   ├── images/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   └── labels/
│       ├── frame_0001.txt
│       ├── frame_0002.txt
├── val/
│   ├── images/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   └── labels/
│       ├── frame_0001.txt
│       ├── frame_0002.txt
└── classes.txt

```

Each label file (`.txt`) contains bounding box annotations in normalized YOLO format:

```

<class_id> <x_center> <y_center> <width> <height>

```

Example:
```

0 0.512 0.622 0.143 0.218

```
This means the object of class `0` (e.g., *car*) is centered at `(0.512, 0.622)` of the image, with width `0.143` and height `0.218` (all values normalized to image dimensions).

---

## 🧠 About YOLO

**YOLO (You Only Look Once)** is a family of deep learning models for **real-time object detection**.  
Unlike traditional region-based detectors (like R-CNN), YOLO predicts bounding boxes and class probabilities **in a single forward pass**, making it extremely fast and efficient.

### Key advantages:
- ⚡ **Real-time inference**  
- 🎯 **High detection accuracy**  
- 🔄 **End-to-end trainable**  
- 🧠 **Supports multiple object classes simultaneously**

In this project, the **YOLOv8-Large** variant from [Ultralytics](https://github.com/ultralytics/ultralytics) was used, providing a strong balance between **accuracy** and **speed**.

---

## 🧪 Model Training

### ⚙️ Configuration
| Parameter | Value |
|------------|--------|
| Model | YOLOv8-Large (`yolov8l.pt`) |
| Epochs | 100 |
| Batch Size | 6 |
| Image Size | 640×640 |
| Optimizer | SGD + Cosine LR |
| Early Stopping | Yes (patience = 20) |
| Augmentations | Mosaic, MixUp, HSV, Flip, Scale, Rotate |

### 💻 Environment
- GPU: **NVIDIA RTX 5070 Ti**
- VRAM: **16 GB**
- Framework: **Ultralytics YOLOv8**
- Python: **3.10+**

---

## 📈 Expected Metrics

To evaluate detection performance, we monitor several key metrics:

| Metric | Description | Ideal Range |
|---------|-------------|-------------|
| **Precision (P)** | Percentage of correct detections among all detections | ↑ High (→ 1.0) |
| **Recall (R)** | Percentage of correctly detected objects among all ground truth objects | ↑ High (→ 1.0) |
| **F1 Score** | Harmonic mean of Precision and Recall | ≥ 0.8 |
| **mAP@50** | Mean Average Precision at IoU = 0.5 | > 0.8 |
| **mAP@50–95** | Average Precision over IoU thresholds (0.5–0.95) | > 0.6 |

Additionally, **Confusion Matrices** and **PR Curves** are used to visualize model performance across classes.

---

## 📊 Results & Visualizations

All generated plots are available in  
[`/yolo_final/model_final/`](https://github.com/bilusss/traffic-road-object-detection/tree/main/yolo_final/model_final)

### 📈 Metric Curves

| Metric | Visualization |
|--------|----------------|
| Precision | ![Precision Curve](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/BoxP_curve.png) |
| Recall | ![Recall Curve](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/BoxR_curve.png) |
| F1 Score | ![F1 Curve](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/BoxF1_curve.png) |
| Precision–Recall | ![PR Curve](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/BoxPR_curve.png) |

---

### 🔢 Confusion Matrices

| Type | Image |
|------|--------|
| Standard | ![Confusion Matrix](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/confusion_matrix.png) |
| Normalized | ![Confusion Matrix Normalized](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/confusion_matrix_normalized.png) |

---

### 🧩 Validation Examples

| Labels | Predictions |
|--------|-------------|
| ![val_batch0_labels](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch0_labels.jpg) | ![val_batch0_pred](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch0_pred.jpg) |
| ![val_batch1_labels](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch1_labels.jpg) | ![val_batch1_pred](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch1_pred.jpg) |
| ![val_batch2_labels](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch2_labels.jpg) | ![val_batch2_pred](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/val_batch2_pred.jpg) |

---

### 🧭 Training Progress

![Training Results](https://github.com/bilusss/traffic-road-object-detection/blob/main/yolo_final/model_final/results.png)

---

## 📦 Model Export

After training, the best model checkpoint was exported into multiple formats for deployment (which are not uploaded):

Supported formats:
- **ONNX** – cross-platform inference
- **TorchScript** – optimized for PyTorch

---

## 🧾 Evaluation Results (TODO)

| Metric | Value |
|---------|--------|
| Precision | 0.XX |
| Recall | 0.XX |
| mAP@50 | 0.XX |
| mAP@50–95 | 0.XX |

🟢 The model achieves strong detection accuracy and generalizes well across diverse traffic scenes.

---

## 💡 Future Improvements

- Experiment with **YOLOv8x** or **YOLOv9** for higher accuracy  
- Apply advanced augmentation techniques (CutMix, Mosaic 9x)  
- Fine-tune learning rate and anchor box strategies  
- Integrate real-time inference in a dashboard or edge device

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Kaggle Dataset: Traffic Road Object Detection](https://www.kaggle.com/datasets/boukraailyesali/traffic-road-object-detection-dataset-using-yolo)

---

### ✍️ Author
Developed by **[bilusss](https://github.com/bilusss)**  
as part of an experiment with **Ultralytics YOLOv8** and real-world traffic datasets.

---

> _“You Only Look Once — but you detect everything.”_
