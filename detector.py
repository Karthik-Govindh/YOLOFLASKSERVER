import cv2
import datetime
import os
import logging
import pandas as pd
import torch
from config import OUTPUT_DIR

class FireDetector:
    def __init__(self, model, conf_threshold=0.85, classes=['fire', 'smoke']):
        self.model = model
        self.conf_threshold = conf_threshold
        self.classes = classes
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Initialized detector with confidence threshold: {self.conf_threshold}")

    def detect(self, tensor_frame):
        """Run detection with enhanced preprocessing"""
        if tensor_frame.dtype == torch.float16:
            tensor_frame = tensor_frame.float()
        if tensor_frame.max() > 1.0:
            tensor_frame /= 255.0
        with torch.no_grad():
            results = self.model(tensor_frame, augment=False, conf=self.conf_threshold, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        class_names = [results[0].names[int(cls)] for cls in detections[:, 5]]
        df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class_idx'])
        df['class_name'] = class_names
        return df[df['class_name'].isin(self.classes) & (df['confidence'] > self.conf_threshold * 0.9)]

    def annotate_frame(self, frame, detections):
        """Draw bounding boxes with class-specific colors"""
        annotated = frame.copy()
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, det[['xmin', 'ymin', 'xmax', 'ymax']])
            color = (0, 0, 255) if det['class_name'] == 'fire' else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated

    def save_result(self, frame):
        """Save original frame without annotations"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fire_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, frame)
        return filename