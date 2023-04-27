import os
from torch.utils.data import Dataset
from typing import Any


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

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return []
