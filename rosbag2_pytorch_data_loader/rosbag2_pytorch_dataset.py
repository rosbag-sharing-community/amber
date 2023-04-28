import os
from torch.utils.data import Dataset
from typing import Any
from mcap.reader import NonSeekingReader
from yaml import safe_load  # type: ignore
from rosbag2_pytorch_data_loader.exception import DatasetTypeError
from rosbag2_pytorch_data_loader.conversion import decode_image_message


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
        self.dispatch(lambda obj: self.read_images(obj))

    def dispatch(self, image_only_function: Any) -> Any:
        with open(self.task_description_yaml_path, "rb") as file:
            obj = safe_load(file)
            match obj["dataset_type"]:
                case "image_only":
                    return image_only_function(obj)
                case _:
                    raise DatasetTypeError(
                        "Dataset type should be image_only, please check the "
                        + self.task_description_yaml_path
                    )

    def read_images(self, obj: Any) -> None:
        image_topics = obj["image_topics"]
        self.images = []
        for schema, channel, message in self.reader.iter_messages():
            if channel.topic in image_topics:
                self.images.append(decode_image_message(message, schema))

    def __len__(self) -> int:
        return self.dispatch(lambda obj: len(self.images))  # type: ignore

    def __getitem__(self, index: int) -> Any:
        return self.dispatch(lambda obj: self.images[index])
