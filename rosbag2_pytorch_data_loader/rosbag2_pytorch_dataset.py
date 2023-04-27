import os
from torch.utils.data import Dataset
from typing import Any
from mcap.reader import NonSeekingReader
import yaml  # type: ignore
from rosbag2_pytorch_data_loader.exception import DatasetTypeError


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
        self.reader = NonSeekingReader(rosbag_path)
        with open(task_description_yaml_path, "rb") as file:
            obj = yaml.safe_load(file)
            print(obj)
            if obj["dataset_type"] == "image_only":
                self.read_images(obj["image_topics"])
            else:
                raise DatasetTypeError(
                    "Dataset type should be image_only, please check the "
                    + task_description_yaml_path
                )

    def read_images(self, image_topics: list[str]) -> None:
        for schema, channel, message in self.reader.iter_messages():
            if channel.topic in image_topics:
                print(message)
        pass

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return []
