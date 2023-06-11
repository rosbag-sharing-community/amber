from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.automation.annotation import BoundingBoxAnnotation, ImageAnnotation
from amber.dataset.topic_config import ImageTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber.exception import TaskDescriptionError
from amber.dataset.conversion import decode_image_message
from typing import Any, List
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader


@dataclass
class ReadImagesAndBoundingBoxConfig(YAMLWizard):  # type: ignore
    image_topics: List[ImageTopicConfig] = field(default_factory=list)
    annotations: List[ImageAnnotation] = field(default_factory=list)

    def get_image_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.image_topics:
            topics.append(topic.topic_name)
        return topics

    def compressed(self, topic_name: str) -> bool:
        for topic in self.image_topics:
            if topic.topic_name == topic_name:
                return bool(topic.compressed)
        raise TaskDescriptionError(
            "Topic : " + topic_name + " does not exist in rosbag."
        )
