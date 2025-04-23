import torch
from ultralytics import YOLO  # Use Ultralytics official package
from config import MODEL_PATH

def load_yolov11_model():
    model = YOLO(MODEL_PATH)
    return model