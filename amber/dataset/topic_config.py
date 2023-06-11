from dataclasses import dataclass
from dataclass_wizard import YAMLWizard


@dataclass
class ImageTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = ""
    compressed: bool = True
