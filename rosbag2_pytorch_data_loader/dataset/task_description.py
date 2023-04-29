from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from rosbag2_pytorch_data_loader.exception import TaskDescriptionError


@dataclass
class ImageTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = ""
    compressed: bool = True


@dataclass
class ImageOnlyConfig(YAMLWizard):  # type: ignore
    dataset_type: str = "image_only"
    image_topics: list[ImageTopicConfig] = field(default_factory=list)

    def get_image_topics(self) -> list[str]:
        topics: list[str] = []
        for topic in self.image_topics:
            topics.append(topic.topic_name)
        return topics

    def compressed(self, name: str) -> bool:
        for topic in self.image_topics:
            if topic.topic_name == name:
                return topic.compressed
        raise TaskDescriptionError(
            "Topic : " + name + " does not exist in task description"
        )
