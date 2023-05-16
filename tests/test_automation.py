from rosbag2_pytorch_data_loader.automation.clip_image_filter import ClipImageFilter
from pathlib import Path
import os
from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset


def test_clip_image_filter() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    filter = ClipImageFilter(str(current_path / "clip_image_filter.yaml"))
    dataset = Rosbag2Dataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        str(current_path / "read_image_ford.yaml"),
    )
    annotation = filter.inference(dataset)
    filter.write(dataset, "clip_image_filter", annotation, "output.mcap")
    os.remove("output.mcap")
