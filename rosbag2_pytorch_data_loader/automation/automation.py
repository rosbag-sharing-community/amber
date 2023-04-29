from typing import Any
from abc import ABC, abstractmethod
from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset


class Automation(ABC):
    @abstractmethod
    def inference(self, dataset: Rosbag2Dataset) -> Any:
        pass
