from amber.dataset.conversion import build_transform_stamped_message
from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import TfTopicConfig
from amber.unit.time import Time, TimeUnit
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
from mcap.reader import NonSeekingReader
from tf2_amber import BufferCore, timeFromSec, durationFromSec, TransformStamped
import sys
import torch
from typing import List


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
        first_timestamp = None
        last_timestamp = None
        tf_buffer = BufferCore(durationFromSec(sys.float_info.max))
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                timestamp = message.log_time

                if first_timestamp is None or timestamp < first_timestamp:
                    first_timestamp = timestamp
                if last_timestamp is None or timestamp > last_timestamp:
                    last_timestamp = timestamp

                if channel.topic in [self.config.get_tf_topic()]:
                    for tf_amber_message in build_transform_stamped_message(
                        message, schema, self.config.compressed
                    ):
                        tf_buffer.setTransform(
                            tf_amber_message, "Authority undetectable", False
                        )
                if channel.topic in [self.config.get_static_tf_topic()]:
                    for tf_amber_message in build_transform_stamped_message(
                        message, schema, self.config.compressed
                    ):
                        tf_buffer.setTransform(
                            tf_amber_message, "Authority undetectable", True
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
                    self.config.target_frame,
                    self.config.source_frame,
                    timeFromSec(timestamp_nanosec.get(TimeUnit.SECOND)),
                )
                self.transforms.append(transform)
            except:
                pass

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> torch.Tensor:
        current_index = 0
        return torch.zeros(0)

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
