import os
from torch.utils.data import Dataset
from typing import Any, Dict, List
from yaml import safe_load  # type: ignore
from amber.exception import TaskDescriptionError
from dataclasses import dataclass
from dataclass_wizard import JSONWizard
import glob


@dataclass
class MessageMetaData(JSONWizard):  # type: ignore
    sequence: int = 0
    topic: str = ""
    rosbag_path: str = ""


class Rosbag2Dataset(Dataset):  # type: ignore
    message_metadata: List[MessageMetaData] = []

    def __init__(
        self,
        rosbag_path: str,
        task_description_yaml_path: str,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        if os.path.isfile(rosbag_path):
            self.rosbag_files = [rosbag_path]
        else:
            self.rosbag_files = glob.glob(rosbag_path + "/**/*.mcap", recursive=True)
        self.transform = transform
        self.target_transform = target_transform
        self.task_description_yaml_path = task_description_yaml_path

    def get_metadata(self, index: int) -> MessageMetaData:
        return self.message_metadata[index]
