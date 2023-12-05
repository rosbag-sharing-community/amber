from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import ImageTopicConfig
import torch
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber.dataset.conversion import decode_image_message
from typing import Any, List
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from mcap.reader import NonSeekingReader
from amber.unit.time import Time, TimeUnit
import datetime


@dataclass
class ReadImagesConfig(YAMLWizard):  # type: ignore
    image_topics: List[ImageTopicConfig] = field(default_factory=list)
    compressed: bool = True

    def get_image_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.image_topics:
            topics.append(topic.topic_name)
        return topics


class ImagesDataset(Rosbag2Dataset):  # type: ignore
    images: List[torch.Tensor] = []
    config: ReadImagesConfig = ReadImagesConfig()

    def __init__(
        self,
        rosbag_path: str,
        config: ReadImagesConfig,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.images.clear()
        self.config = config
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
            transform,
            target_transform,
        )
        self.read_images()

    def read_images(self) -> None:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    self.images.append(
                        decode_image_message(message, schema, self.config.compressed)
                    )
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
        assert len(self.images) == len(self.message_metadata)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]
