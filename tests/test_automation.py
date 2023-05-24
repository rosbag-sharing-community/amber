from rosbag2_pytorch_data_loader.automation.detic_image_labeler import DeticImageLabeler
from pathlib import Path
import os
from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset


def test_detic_auto_labeler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    labeler = DeticImageLabeler(str(current_path / "detic_image_labeler.yaml"))
    dataset = Rosbag2Dataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        str(current_path / "read_image_ford.yaml"),
    )
    labeler.inference(dataset)
