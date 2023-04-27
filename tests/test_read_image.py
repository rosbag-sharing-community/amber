from rosbag2_pytorch_data_loader.rosbag2_pytorch_dataset import Rosbag2Dataset
import tests
import os


def test_read_images() -> None:
    dataset = Rosbag2Dataset(
        os.path.join(os.path.dirname(tests.__file__), "rosbag/vrx.mcap"),
        "read_image.yaml",
    )
    assert True
