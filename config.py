# Video & Camera Settings
VIDEO_STREAM_URL = 0
FLASK_SERVER_URL = "http://localhost:5000/api/alerts/"
ADDITIONAL_SERVER_URL = "http://192.168.56.221/fire-alert"
OUTPUT_DIR = "E:/Website/flaskserver/static/detections"

# YOLO Model Config
MODEL_PATH = "E:/Website/flaskserver/best.pt"

# Distance Calculation Constants
REF_TREE_HEIGHT = 25  # meters
CAMERA_FOCAL_LENGTH = 0.0036  # 3.6mm
SENSOR_HEIGHT = 0.00276  # OV2640 sensor
FRAME_HEIGHT = 480
TREE_DISTANCE = 60
FRAME_WIDTH = 640
CAMERA_FOV = 65  # degrees