from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber_mcap.automation.annotation import BoundingBoxAnnotation, ImageAnnotation
from amber_mcap.dataset.topic_config import ImageTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber_mcap.unit.time import Time, TimeUnit
from amber_mcap.dataset.conversion import decode_image_message, decode_message
from typing import Any, List, Dict, Tuple
from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader
import json
import datetime


@dataclass
class ReadImagesAndAnnotationsConfig(YAMLWizard):  # type: ignore
    image_topics: List[ImageTopicConfig] = field(default_factory=list)
    annotation_topic: str = ""
    compressed: bool = True

    def get_image_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.image_topics:
            topics.append(topic.topic_name)
        return topics


class ImagesAndAnnotationsDataset(Rosbag2Dataset):  # type: ignore
    num_images = 0
    annotations: Dict[int, ImageAnnotation] = {}
    config = ReadImagesAndAnnotationsConfig()

    def __init__(
        self,
        rosbag_path: str,
        config: ReadImagesAndAnnotationsConfig,
        transform: Any = None,
        target_transform: Any = None,
    ):
        self.config = config
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
            transform,
            target_transform,
        )
        self.count_images()
        self.read_annotations()

    def count_images(self) -> None:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    self.num_images = self.num_images + 1
                    self.message_metadata.append(
                        MessageMetaData.from_dict(
                            {
                                "publish_time": datetime.datetime.fromtimestamp(
                                    Time(message.publish_time, TimeUnit.NANOSECOND).get(
                                        TimeUnit.SECOND
                                    ),
                                    tz=datetime.timezone.utc,
                                ),
                                "topic": channel.topic,
                                "rosbag_path": rosbag_file,
                            }
                        )
                    )

    def read_annotations(self) -> None:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.annotation_topic:
                    annotation_json = decode_message(
                        message, schema, self.config.compressed
                    )
                    for annotation in json.loads(annotation_json.data):
                        annotation = ImageAnnotation.from_json(annotation)
                        self.annotations[annotation.image_index] = annotation

    def __len__(self) -> int:
        return self.num_images

    def __iter__(self):
        current_index = 0
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    current_index = current_index + 1
                    yield (
                        decode_image_message(message, schema, self.config.compressed),
                        self.annotations[current_index - 1],
                    )
