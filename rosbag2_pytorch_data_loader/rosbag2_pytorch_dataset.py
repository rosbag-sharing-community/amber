import os
from torch.utils.data import Dataset
from typing import Any
from mcap.reader import NonSeekingReader


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
        with open("rosbag/vrx.mcap", "rb") as f:
            self.reader = NonSeekingReader(rosbag_path)
        for schema, channel, message in self.reader.iter_messages():
            print(schema)
            print(channel.topic)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return []
