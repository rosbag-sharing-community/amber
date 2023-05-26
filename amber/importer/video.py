import cv2
from amber.dataset import sensor_msgs


class VideoImporter:
    def __init__(self, video_path: str, topic_name: str):
        self.capture = cv2.VideoCapture(video_path)
