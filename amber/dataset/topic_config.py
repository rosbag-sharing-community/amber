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


@dataclass
class TfTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = "/tf"
    static_tf_topic_name: str = "/tf_static"
