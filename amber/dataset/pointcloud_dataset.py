from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import PointcloudTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber.exception import TaskDescriptionError
from amber.dataset.conversion import decode_message
from typing import Any, List
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader


@dataclass
class ReadImagesConfig(YAMLWizard):  # type: ignore
    pointcloud_topics: List[PointcloudTopicConfig] = field(default_factory=list)
    compressed: bool = True

    def get_pointcloud_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.pointcloud_topics:
            topics.append(topic.topic_name)
        return topics


# class PointcloudDataset(Rosbag2Dataset):
