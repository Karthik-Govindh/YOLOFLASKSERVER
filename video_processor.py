import cv2
import threading
from collections import deque

class VideoProcessor:
    def __init__(self, stream_url, buffer_size=2):
        self.stream_url = stream_url
        self.cap = None
        self.frame_queue = deque(maxlen=buffer_size)
        self.running = False
        self.thread = None

    def start_capture(self):
        """Start background thread for continuous frame capture"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        return True

    def _capture_frames(self):
        """Continuous frame grabbing thread"""
        self.cap = cv2.VideoCapture(self.stream_url)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.append(frame)
            else:
                self.running = False

    def get_latest_frame(self):
        """Get most recent frame from buffer"""
        return self.frame_queue[-1] if self.frame_queue else None

    def release(self):
        """Safely release resources"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()