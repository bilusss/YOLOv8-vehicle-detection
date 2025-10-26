import os
import sys
import yaml
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch
import kagglehub

print("")

# ============================================================================
# 1. GPU CHECK
# ============================================================================
print("\n[1/5] Checking GPU...")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. DATASET DOWNLOAD (if not exists)
# ============================================================================
print("\n[2/5] Checking dataset...")

dataset_name = "boukraailyesali/traffic-road-object-detection-dataset-using-yolo"

# Check if dataset already exists
if os.path.exists("dataset_cache"):
    dataset_dir = "dataset_cache"
    print(f"  [OK] Dataset already downloaded: {dataset_dir}")
else:
    print(f"  [DOWNLOADING] Dataset download (may take a few minutes)...")
    try:
        dataset_dir = kagglehub.dataset_download(dataset_name)
        # Copy to local directory
        import shutil
        shutil.copytree(dataset_dir, "dataset_cache", dirs_exist_ok=True)
        dataset_dir = "dataset_cache"
        print(f"  [OK] Dataset downloaded and cached")
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit("Failed to download dataset")

# ============================================================================
# 3. PREPARE data.yaml
# ============================================================================
print("\n[3/5] Preparing configuration...")

# Load classes
classes_file = os.path.join(dataset_dir, "classes.txt")
with open(classes_file, 'r') as f:
    classes_list = [line.strip() for line in f.readlines()]

num_classes = len(classes_list)
print(f"  [OK] Classes: {classes_list}")

# Create data.yaml
work_dir = "yolo_final"
os.makedirs(work_dir, exist_ok=True)

data_yaml = f"""path: {os.path.abspath(dataset_dir)}
train: train/images
val: val/images

nc: {num_classes}
names:
"""

for i, cls in enumerate(classes_list):
    data_yaml += f"  {i}: {cls}\n"

data_yaml_path = os.path.join(work_dir, "data.yaml")
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"  [OK] data.yaml: {data_yaml_path}")

# ============================================================================
# 4. TRAINING - YOLOv8 LARGE (balance between quality and speed)
# ============================================================================
print("\n[4/5] Training YOLOv8 Large model...")
print("  " + "="*60)
print("  PARAMETERS:")
print("  - Model: YOLOv8 Large (yolov8l)")
print("  - Epochs: 100 (well balanced)")
print("  - Batch: 6 (optimal for RTX 5070 Ti)")
print("  - Image size: 640x640 (optimal balance)")
print("  - Augmentation: FULL")
print("  - Optimizer: SGD + Cosine LR")
print("  - Early stopping: Yes (patience=20)")
print("  ")
print("  Estimated time: 3-4 hours")
print("  " + "="*60 + "\n")

# Load model
model = YOLO('yolov8l.pt')

# Training
results = model.train(
    data=data_yaml_path,
    
    # === PARAMETERS TUNED FOR RTX 5070 Ti ===
    epochs=100,                    # Full training
    batch=6,                       # Optimal for 16GB VRAM
    imgsz=640,                     # Optimal size
    device=0,
    
    # === AUGMENTATION ===
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    
    # === OPTIMIZER ===
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    
    # === EARLY STOPPING ===
    patience=20,                   # Stop after 20 epochs without improvement
    
    # === SAVING ===
    save=True,
    save_period=5,
    project=work_dir,
    name='model_final',
    
    # === VALIDATION ===
    val=True,
    
    # === PERFORMANCE ===
    workers=8,
    seed=42,
    
    # === LOSS WEIGHTS ===
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    close_mosaic=10,
    verbose=True,
    plots=True,
)

# ============================================================================
# 5. VALIDATION AND SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL VALIDATION")
print("="*70)

best_model_path = os.path.join(work_dir, 'model_final/weights/best.pt')

if not os.path.exists(best_model_path):
    print(f"\n[ERROR] Model not found: {best_model_path}")
    sys.exit("Training failed")

print(f"[OK] Model found: {best_model_path}\n")

best_model = YOLO(best_model_path)

# Validation
metrics = best_model.val(
    data=data_yaml_path,
    imgsz=640,
    batch=6,
    device=0,
    conf=0.25,
    iou=0.6
)

print("\nVALIDATION METRICS:")
print(f"  - Precision: {metrics.box.mp:.4f}")
print(f"  - Recall: {metrics.box.mr:.4f}")
print(f"  - mAP50: {metrics.box.map50:.4f}")
print(f"  - mAP50-95: {metrics.box.map:.4f}")

# ============================================================================
# 6. INFERENCE TEST
# ============================================================================
print("\n" + "="*70)
print("INFERENCE TEST ON VALIDATION DATA")
print("="*70)

val_images = os.path.join(dataset_dir, 'val/images')
test_results = best_model.predict(
    source=val_images,
    conf=0.25,
    iou=0.6,
    imgsz=640,
    device=0,
    save=True,
    line_width=2,
    max_det=300,
    project=work_dir,
    name='inference_results'
)

print(f"[OK] Inference results: {work_dir}/inference_results")

# ============================================================================
# 7. MODEL EXPORT
# ============================================================================
print("\n" + "="*70)
print("MODEL EXPORT")
print("="*70)

formats = [
    ('onnx', 'ONNX'),
    ('torchscript', 'TorchScript'),
]

for fmt, name in formats:
    try:
        print(f"\n  [EXPORTING] Exporting to {name}...")
        best_model.export(format=fmt, imgsz=640, device=0)
        print(f"  [OK] {name} - Success")
    except Exception as e:
        print(f"  [WARNING] {name} - skipped")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*70)

print(f"""
FILE PATHS:
  - Best model: {best_model_path}
  - Training directory: {work_dir}/model_final/
  - Inference results: {work_dir}/inference_results/
  - Training plots: {work_dir}/model_final/

FINAL METRICS:
  - Precision: {metrics.box.mp:.4f}
  - Recall: {metrics.box.mr:.4f}
  - mAP50: {metrics.box.map50:.4f}
  - mAP50-95: {metrics.box.map:.4f}

MODEL USAGE:
  from ultralytics import YOLO
  
  model = YOLO('{best_model_path}')
  results = model.predict('your_image.jpg', conf=0.25)
  
  # On video:
  results = model.predict('video.mp4', conf=0.25, save=True)

LIVE WEBCAM DEMO:
  from ultralytics import YOLO
  
  model = YOLO('{best_model_path}')
  results = model.predict(source=0, conf=0.25)  # source=0 = webcam

TRAINING PLOTS:
  Open: {work_dir}/model_final/results.png

AVAILABLE MODELS IN: {work_dir}/model_final/weights/
  - best.pt (best model)
  - last.pt (last checkpoint)
""")

print("="*70)
print("Model ready to use!")
print("="*70)