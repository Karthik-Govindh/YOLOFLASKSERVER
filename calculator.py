import logging

class FireCalculator:
    def __init__(self):
        self.calibration_distance = 0.23  # 23cm target
        self.calibration_samples = []
        self.k = None  # Will be calculated dynamically
        self.min_calibration_samples = 3
        logging.basicConfig(level=logging.INFO)

    def update_calibration(self, w_pixels):
        """Add new calibration data point"""
        if w_pixels <= 0:
            return
            
        self.calibration_samples.append(w_pixels)
        logging.info(f"Calibration sample added: {w_pixels}px")

        if len(self.calibration_samples) >= self.min_calibration_samples:
            avg_width = sum(self.calibration_samples) / len(self.calibration_samples)
            self.k = self.calibration_distance * avg_width
            self.calibration_samples = []
            logging.info(f"New calibration: k={self.k:.2f}")

    def calculate_distance(self, w_pixels):
        if not self.k:
            logging.warning("System not calibrated yet!")
            return 0.0
            
        if w_pixels <= 0:
            return 0.0
            
        return self.k / w_pixels

    def calculate_direction(self, x_center):
        if x_center < 213:
            return "NW"
        elif 213 <= x_center <= 426:
            return "N"
        else:
            return "NE"