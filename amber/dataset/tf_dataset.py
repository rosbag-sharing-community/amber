from amber.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber.dataset.topic_config import TfTopicConfig
from dataclass_wizard import YAMLWizard


@dataclass
class ReadTfTopicConfig(YAMLWizard):  # type: ignore
    tf_topic_config: TfTopicConfig = TfTopicConfig()
    compressed: bool = True

    def get_tf_topic(self) -> str:
        return self.tf_topic_config.topic_name

    def get_static_tf_topic(self) -> str:
        return self.tf_topic_config.static_tf_topic_name


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
