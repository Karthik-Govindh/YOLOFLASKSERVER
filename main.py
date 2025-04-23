import cv2
import logging
import torch
import numpy as np
from model_loader import load_yolov11_model
from video_processor import VideoProcessor
from detector import FireDetector
from calculator import FireCalculator
from helpers import send_to_backend
from config import VIDEO_STREAM_URL
import time

def main():
    # Initialize components
    logging.basicConfig(level=logging.INFO)
    
    # Load and optimize model
    model = load_yolov11_model()
    model.fuse()
    if torch.cuda.is_available():
        model = model.cuda().half()

    video = VideoProcessor(VIDEO_STREAM_URL)
    detector = FireDetector(model, conf_threshold=0.7, classes=['fire'])
    calculator = FireCalculator()

    # Cooldown system variables
    last_detection_time = 0
    COOLDOWN_DURATION = 30  # seconds
    frame_counter = 0
    PROCESS_EVERY = 5  # Process 1 in 5 frames

    try:
        video.start_capture()
        
        # Warmup GPU
        dummy_input = torch.zeros(1, 3, 640, 640)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda().half()
        _ = model(dummy_input)

        while True:
            current_time = time.time()
            frame = video.get_latest_frame()
            
            # Skip processing during cooldown
            if current_time - last_detection_time < COOLDOWN_DURATION:
                if frame is not None:
                    cooldown_left = int(COOLDOWN_DURATION - (current_time - last_detection_time))
                    status_frame = frame.copy()
                    cv2.putText(status_frame, f"COOLDOWN: {cooldown_left}s", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow('Fire Detection', status_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            if frame is None:
                continue

            # Frame skipping for stability
            frame_counter += 1
            if frame_counter % PROCESS_EVERY != 0:
                cv2.imshow('Fire Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Preprocessing
            resized_frame = cv2.resize(frame, (640, 640))
            normalized_frame = resized_frame[:, :, ::-1].transpose(2, 0, 1)
            normalized_frame = np.ascontiguousarray(normalized_frame) / 255.0

            # Convert to tensor
            tensor_frame = torch.from_numpy(normalized_frame)
            if torch.cuda.is_available():
                tensor_frame = tensor_frame.cuda().half()
            tensor_frame = tensor_frame.unsqueeze(0)  # Add batch dimension

            # Inference
            with torch.no_grad():
                detections = detector.detect(tensor_frame)

            annotated_frame = frame.copy()
            if not detections.empty:
                det = detections.iloc[0]
                fire_width = det['xmax'] - det['xmin']
                current_distance = calculator.calculate_distance(fire_width)
                
                # Temporary: Force initial calibration (remove after testing)
                if calculator.k is None and fire_width > 0:
                    calculator.update_calibration(fire_width)
                
                # Automatic calibration when in target range
                if 0.20 <= current_distance <= 0.25 and fire_width > 0:
                    calculator.update_calibration(fire_width)
                
                # Save and send data only if calibrated
                if calculator.k:
                    filename = detector.save_result(frame)
                    data = {
                        "detectionType": det['class_name'],
                        "confidence": round(float(det['confidence']), 2),
                        "imagePath": filename,
                        "deviceId": "camera-01",
                        "distance": round(float(current_distance), 2),  # Convert numpy.float32 to Python float
                        "direction": calculator.calculate_direction((det['xmin'] + det['xmax']) // 2)
                    }                    
                    send_to_backend(data)

                # Create annotated version for display only
                annotated_frame = detector.annotate_frame(frame, detections)
                cv2.putText(annotated_frame, "FIRE DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                last_detection_time = time.time()

            # Display annotated frame
            cv2.imshow('Fire Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()