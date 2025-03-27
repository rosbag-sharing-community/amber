import cv2
from amber_mcap.dataset.schema import ImageMessageSchema
from amber_mcap.unit.time import Time, TimeUnit
from amber_mcap.dataset.conversion import build_message_from_image
from mcap_ros2.writer import Writer as McapWriter
from tqdm import tqdm
import math
from typing import Callable

from dataclasses import dataclass
from dataclass_wizard import YAMLWizard


@dataclass
class VideoImporterConfig(YAMLWizard):  # type: ignore
    topic_name: str = "/camera/image_raw"
    rosbag_path: str = "output.mcap"


class VideoImporter:
    def __init__(self, video_path: str, config: VideoImporterConfig) -> None:
        self.config: VideoImporterConfig = config
        self.capture = cv2.VideoCapture(video_path)

    def write(self) -> None:
        with open(self.config.rosbag_path, "wb") as f:
            writer = McapWriter(f)
            schema = writer.register_msgdef(
                ImageMessageSchema.name, ImageMessageSchema.schema_text
            )
            get_nanoseconds_timestamp: Callable[[], int] = lambda: math.floor(
                Time(self.capture.get(cv2.CAP_PROP_POS_MSEC), TimeUnit.MILLISECOND).get(
                    TimeUnit.NANOSECOND
                )
            )
            for i in tqdm(range(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, cv_image = self.capture.read()
                writer.write_message(
                    topic=self.config.topic_name,
                    schema=schema,
                    message=build_message_from_image(
                        cv_image,
                        "camera",
                        Time(
                            self.capture.get(cv2.CAP_PROP_POS_MSEC),
                            TimeUnit.MILLISECOND,
                        ),
                        "rgb8",
                    ),
                    log_time=get_nanoseconds_timestamp(),
                    publish_time=get_nanoseconds_timestamp(),
                    sequence=i,
                )
            writer.finish()
