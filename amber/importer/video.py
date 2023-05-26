import cv2
from amber.dataset.schema import ImageMessageSchema
from mcap_ros2.writer import Writer as McapWriter


class VideoImporter:
    def __init__(self, video_path: str, topic_name: str, fps: float = 30.0) -> None:
        self.capture = cv2.VideoCapture(video_path)

    def write(self, rosbag_path: str) -> None:
        with open(rosbag_path, "wb") as f:
            writer = McapWriter(f)
            schema = writer.register_msgdef(
                ImageMessageSchema.name, ImageMessageSchema.schema_text
            )
            i = 0
            while True:
                print("Frame: " + str(i))
                ret, img = self.capture.read()
                if ret == False:
                    break
                i = i + 1
