from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from enum import Enum
from typing import Optional


@dataclass
class ImageTopicConfig(YAMLWizard):  # type: ignore
    topic_name: str = ""
    camera_info_topic_name: Optional[str] = None
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
    robot_description_topic: Optional[str] = None
    urdf_path: Optional[str] = None
