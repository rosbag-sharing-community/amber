from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber_mcap.dataset.topic_config import TfTopicConfig
from amber_mcap.unit.time import Time, TimeUnit
from amber_mcap.util.geometry import build_tf_buffer
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
from mcap.reader import NonSeekingReader
from amber_mcap.tf2_amber import (
    BufferCore,
    timeFromSec,
    durationFromSec,
    TransformStamped,
)
import sys
import torch
from typing import List
import datetime


@dataclass
class ReadTfTopicConfig(YAMLWizard):  # type: ignore
    tf_topic: TfTopicConfig = TfTopicConfig()
    compressed: bool = True
    sampling_duration: float = 0.01666666666  # 60Hz
    target_frame: str = ""
    source_frame: str = ""

    def get_tf_topic(self) -> str:
        return self.tf_topic.topic_name

    def get_static_tf_topic(self) -> str:
        return self.tf_topic.static_tf_topic_name


class TfDataset(Rosbag2Dataset):  # type: ignore
    def __init__(
        self,
        rosbag_path: str,
        config: ReadTfTopicConfig,
    ) -> None:
        self.config = config
        self.transforms: List[TransformStamped] = []
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
        )
        tf_buffer, first_timestamp, last_timestamp = build_tf_buffer(
            [rosbag_path], self.config.tf_topic, self.config.compressed
        )
        sampled_timestamps = self.get_sampled_timestamps(
            Time(float(first_timestamp), TimeUnit.NANOSECOND),
            Time(float(last_timestamp), TimeUnit.NANOSECOND),
            Time(self.config.sampling_duration, TimeUnit.SECOND),
        )
        for timestamp in sampled_timestamps:
            timestamp_nanosec = Time(float(timestamp), TimeUnit.NANOSECOND)
            try:
                transform = tf_buffer.lookupTransform(
                    self.config.source_frame,
                    self.config.target_frame,
                    timeFromSec(timestamp_nanosec.get(TimeUnit.SECOND)),
                )
                self.transforms.append(transform)
            except:
                pass
            self.message_metadata.append(
                MessageMetaData.from_dict(
                    {
                        "publish_time": datetime.datetime.fromtimestamp(
                            timestamp_nanosec.get(TimeUnit.SECOND),
                            tz=datetime.timezone.utc,
                        ),
                        "topic": "N/A",
                        "rosbag_path": "N/A",
                    }
                )
            )

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> torch.Tensor:
        current_index = 0
        for transfrom in self.transforms:
            current_index = current_index + 1
            yield torch.Tensor(
                [
                    transfrom.transform.translation.x,
                    transfrom.transform.translation.y,
                    transfrom.transform.translation.z,
                    transfrom.transform.rotation.x,
                    transfrom.transform.rotation.y,
                    transfrom.transform.rotation.z,
                    transfrom.transform.rotation.w,
                ]
            )

    def get_sampled_timestamps(
        self, first_timestamp: Time, last_timestamp: Time, sampling_duration: Time
    ) -> List[int]:
        return list(
            range(
                int(first_timestamp.get(TimeUnit.NANOSECOND)),
                int(last_timestamp.get(TimeUnit.NANOSECOND)),
                int(sampling_duration.get(TimeUnit.NANOSECOND)),
            )
        )
