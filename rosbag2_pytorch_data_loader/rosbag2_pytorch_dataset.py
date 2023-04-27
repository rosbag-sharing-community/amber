import os
from torch.utils.data import Dataset
from typing import Any
from mcap_ros2.reader import read_ros2_messages


class Rosbag2Dataset(Dataset):  # type: ignore
    def __init__(
        self,
        rosbag_path: str,
        task_description_yaml_path: str,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        for msg in read_ros2_messages(rosbag_path):
            print(msg)
            # print(f"{msg.topic}: f{msg.ros_msg}")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return []
