import cv2
from amber.dataset.schema import ImageMessageSchema
from amber.unit.time import Time, TimeUnit
from amber.dataset.conversion import build_message_from_image
from mcap_ros2.writer import Writer as McapWriter
from tqdm import tqdm
import math


class VideoImporter:
    def __init__(self, video_path: str, topic_name: str) -> None:
        self.topic_name = topic_name
        self.capture = cv2.VideoCapture(video_path)

    def write(self, rosbag_path: str) -> None:
        with open(rosbag_path, "wb") as f:
            writer = McapWriter(f)
            schema = writer.register_msgdef(
                ImageMessageSchema.name, ImageMessageSchema.schema_text
            )
            nanoseconds_timestamp = math.floor(
                Time(self.capture.get(cv2.CAP_PROP_POS_MSEC), TimeUnit.MILLISECOND).get(
                    TimeUnit.NANOSECOND
                )
            )
            for i in tqdm(range(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, cv_image = self.capture.read()
                writer.write_message(
                    topic=self.topic_name,
                    schema=schema,
                    message=build_message_from_image(
                        cv_image,
                        "camera",
                        Time(
                            self.capture.get(cv2.CAP_PROP_POS_MSEC),
                            TimeUnit.MILLISECOND,
                        ),
                        "bgr8",
                    ),
                    log_time=nanoseconds_timestamp,
                    publish_time=nanoseconds_timestamp,
                    sequence=i,
                )
            writer.finish()
