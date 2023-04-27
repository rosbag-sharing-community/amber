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
        self.task_description_yaml_path = task_description_yaml_path
        self.dispatch(self.read_images)

    def dispatch(self, image_only_function: Any) -> None:
        with open(self.task_description_yaml_path, "rb") as file:
            obj = yaml.safe_load(file)
            if obj["dataset_type"] == "image_only":
                image_only_function(obj)
            else:
                raise DatasetTypeError(
                    "Dataset type should be image_only, please check the "
                    + self.task_description_yaml_path
                )

    def read_images(self, obj: Any) -> None:
        image_topics = obj["image_topics"]
        self.image_messages = []
        for schema, channel, message in self.reader.iter_messages():
            if channel.topic in image_topics:
                self.image_messages.append(message)
        pass

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return []
