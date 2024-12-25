from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import TfTopicConfig
from dataclass_wizard import YAMLWizard
from tf2_amber import BufferCore, durationFromSec, TransformStamped


@dataclass
class ReadTfTopicConfig(YAMLWizard):  # type: ignore
    tf_topics: TfTopicConfig = TfTopicConfig()
    compressed: bool = True
    sampling_duration: double = 0.01666666666  # 60Hz

    def get_tf_topic(self) -> str:
        return self.tf_topics.topic_name

    def get_static_tf_topic(self) -> str:
        return self.tf_topics.static_tf_topic_name


class TfDataset(Rosbag2Dataset):  # type: ignore
    def __init__(
        self,
        rosbag_path: str,
        config: ReadTfTopicConfig,
    ) -> None:
        self.config = config
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
        )
        self.tf_buffer = BufferCore(
            durationFromSec(
                (self.get_last_timestamp - self.get_first_timestamp()).total_seconds()
            )
        )
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in [self.config.get_tf_topic()]:
                    # self.tf_buffer.setTransform()
                    pass
                if channel.topic in [self.config.get_static_tf_topic()]:
                    pass
