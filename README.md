# Object Detection - Local Deployment

## Overview
This project implements a CPU-based object detection using YOLOv8n exported to ONNX format.  
The system runs entirely locally without any cloud-based processing and runs within 8GB RAM constraint.

## Dataset
This project uses the pretrained YOLOv8n model, which was originally trained on the COCO dataset.
The dataset contains 80 object categories, relevant object classes include:
- person
- chair
- cellphone
- keyboard
- dining table
- bottle

## Training
The YOLOv8n pretrained model was exported to ONNX format for optimized inference:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx")
```

## Model Architecture
The Architecture is structured as follows:

Training
- Export to ONNX
- OpenCV-based video capture and visualization
 
Key Deployment Characteristics:

- CPU-only execution (device="cpu")
- No GPU dependency during runtime
- No cloud-based processing
- Real-time webcam detection
- Fully local execution

## Performance Metrics
Tested on an Intel i5 CPU system:

- 11 FPS at 640×640 resolution
- Around 400 MB RAM usage during inference
- The system runs in real time while staying well within the 8GB RAM constraint.

## Optimization Techniques
- YOLOv8n was chosen to reduce computational load
- Exported to ONNX format for efficient CPU inference
- OpenCV multi-threading enabled (cv2.setNumThreads(16))

## How to Run
Step 1: Create Virtual Environment
```python
python3 -m venv cv_env
source cv_env/bin/activate
```
Step 2: Install Dependencies
```python
pip install ultralytics opencv-python
```
Step 3: Run
```python
python3 yolo.py
```
