import torch
import clip
from PIL import Image
from rosbag2_pytorch_data_loader.automation.automation import Automation
from typing import Any
from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset

import torchvision.transforms as transforms


class ClipImageFilter(Automation):  # type: ignore
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.transform = transforms.ToPILImage()

    def inference(self, dataset: Rosbag2Dataset) -> Any:
        for image in dataset:
            image = self.preprocess(self.transform(image)).unsqueeze(0).to(self.device)
