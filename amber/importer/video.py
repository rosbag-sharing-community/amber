import cv2
from amber.dataset.schema import ImageMessageSchema
from amber.unit.time import Time, TimeUnit
from mcap_ros2.writer import Writer as McapWriter
from tqdm import tqdm


class VideoImporter:
    def __init__(self, video_path: str, topic_name: str) -> None:
        self.capture = cv2.VideoCapture(video_path)

    def write(self, rosbag_path: str) -> None:
        with open(rosbag_path, "wb") as f:
            writer = McapWriter(f)
            schema = writer.register_msgdef(
                ImageMessageSchema.name, ImageMessageSchema.schema_text
            )
            for _ in tqdm(range(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, img = self.capture.read()
                timestamp = Time(
                    self.capture.get(cv2.CAP_PROP_POS_MSEC), TimeUnit.MILLISECOND
                )
