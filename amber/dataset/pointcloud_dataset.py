from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import PointcloudTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber.exception import TaskDescriptionError
from amber.dataset.conversion import decode_pointcloud_message
from typing import Any, List, Dict
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader
import open3d


@dataclass
class ReadPointCloudConfig(YAMLWizard):  # type: ignore
    pointcloud_topics: List[PointcloudTopicConfig] = field(default_factory=list)
    compressed: bool = True

    def get_pointcloud_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.pointcloud_topics:
            topics.append(topic.topic_name)
        return topics


class PointcloudDataset(Rosbag2Dataset):  # type: ignore
    pointclouds: List[torch.Tensor] = []
    config: ReadPointCloudConfig = ReadPointCloudConfig()

    def __init__(
        self,
        rosbag_path: str,
        task_description_yaml_path: str,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.config = ReadPointCloudConfig.from_yaml_file(task_description_yaml_path)
        print(self.config)
        super().__init__(
            rosbag_path,
            task_description_yaml_path,
            self.config.compressed,
            transform,
            target_transform,
        )
        self.read_pointclouds()

    def read_pointclouds(self) -> None:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_pointcloud_topics():
                    self.pointclouds.append(
                        decode_pointcloud_message(
                            message, schema, self.config.compressed
                        )
                    )
                    self.message_metadata.append(
                        MessageMetaData.from_dict(
                            {
                                "topic": channel.topic,
                                "rosbag_path": rosbag_file,
                            }
                        )
                    )
        assert len(self.pointclouds) == len(self.message_metadata)

    def __len__(self) -> int:
        return len(self.pointclouds)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.pointclouds[index]
