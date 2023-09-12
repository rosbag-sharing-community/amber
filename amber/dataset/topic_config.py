from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from enum import Enum


@dataclass
class ImageTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = ""
    compressed_image: bool = False


class PointCloudType(Enum):
    XYZ = "XYZ"


@dataclass
class PointcloudTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = ""
    pointcloud_type: PointCloudType = PointCloudType.XYZ
