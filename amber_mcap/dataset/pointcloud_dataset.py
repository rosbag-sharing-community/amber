from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber_mcap.dataset.topic_config import PointcloudTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber_mcap.unit.time import Time, TimeUnit
from amber_mcap.dataset.conversion import decode_pointcloud_message
from typing import Any, List, Dict
from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader
import open3d
import datetime


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
    num_pointclouds = 0
    config: ReadPointCloudConfig = ReadPointCloudConfig()

    def __init__(
        self,
        rosbag_path: str,
        config: ReadPointCloudConfig,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.config = config
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
            transform,
            target_transform,
        )
        self.count_pointclouds()

    def count_pointclouds(self) -> None:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_pointcloud_topics():
                    self.num_pointclouds = self.num_pointclouds + 1
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

    def __len__(self) -> int:
        return self.num_pointclouds

    def __iter__(self) -> torch.Tensor:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_pointcloud_topics():
                    yield decode_pointcloud_message(
                        message, schema, self.config.compressed
                    )
