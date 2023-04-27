from rosbag2_pytorch_data_loader.rosbag2_pytorch_dataset import Rosbag2Dataset


def test_read_images() -> None:
    dataset = Rosbag2Dataset("rosbag", "read_image.yaml")
    assert True
