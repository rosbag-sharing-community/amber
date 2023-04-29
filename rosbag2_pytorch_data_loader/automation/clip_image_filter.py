import torch
import clip
from PIL import Image
from rosbag2_pytorch_data_loader.automation.automation import Automation
from typing import Any


class ClipImageFilter(Automation):  # type: ignore
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

    def inference(self) -> Any:
        print("👶")
