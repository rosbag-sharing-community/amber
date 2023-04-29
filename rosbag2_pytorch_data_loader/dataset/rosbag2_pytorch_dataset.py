import os
from torch.utils.data import Dataset
from typing import Any
from mcap.reader import NonSeekingReader
from yaml import safe_load  # type: ignore
from rosbag2_pytorch_data_loader.exception import TaskDescriptionError
from rosbag2_pytorch_data_loader.dataset.conversion import decode_image_message
from rosbag2_pytorch_data_loader.dataset.task_description import ImageOnlyConfig


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
                    return image_only_function(self.task_description_yaml_path)
                case _:
                    raise TaskDescriptionError(
                        "Dataset type should be image_only, please check the "
                        + self.task_description_yaml_path
                    )

    def read_images(self, yaml_path: str) -> None:
        config = ImageOnlyConfig.from_yaml_file(yaml_path)
        self.images = []
        for schema, channel, message in self.reader.iter_messages():
            if channel.topic in config.get_image_topics():
                self.images.append(
                    decode_image_message(
                        message, schema, config.compressed(channel.topic)
                    )
                )

    def __len__(self) -> int:
        return self.dispatch(lambda yaml_path: len(self.images))  # type: ignore

    def __getitem__(self, index: int) -> Any:
        return self.dispatch(lambda yaml_path: self.images[index])
