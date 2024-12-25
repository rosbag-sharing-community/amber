from amber.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from amber.dataset.images_and_annotations_dataset import (
    ImagesAndAnnotationsDataset,
    ReadImagesAndAnnotationsConfig,
)
from amber.dataset.pointcloud_dataset import PointcloudDataset, ReadPointCloudConfig
from amber.dataset.tf_dataset import TfDataset, ReadTfTopicConfig
from amber.dataset.rosbag2_dataset import download_rosbag
from amber.dataset.conversion import image_to_tensor
from torch.utils.data import DataLoader
import torch
import tests
import os
from PIL import Image
from pathlib import Path
import pytest
import datetime


def test_read_image_vrx() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "vrx" / "vrx.mcap"),
        ReadImagesConfig.from_yaml_file(
            str(current_path / "rosbag" / "vrx" / "read_image.yaml")
        ),
    )
    assert len(dataset) == 2
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            image = str(count) + ".png"
            assert torch.equal(
                sample,
                image_to_tensor(
                    Image.open(
                        str(current_path / "images" / "vrx" / image),
                    )
                ),
            )
            count = count + 1
    assert dataset.get_first_timestamp() == datetime.datetime(
        1970, 1, 1, 0, 0, 3, 604000, tzinfo=datetime.timezone.utc
    )
    assert dataset.get_last_timestamp() == datetime.datetime(
        1970, 1, 1, 0, 0, 3, 604000, tzinfo=datetime.timezone.utc
    )


def test_read_image_ford() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        ReadImagesConfig.from_yaml_file(
            str(current_path / "rosbag" / "ford" / "read_image.yaml")
        ),
    )
    assert len(dataset) == 39
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            image = str(count) + ".png"
            assert torch.equal(
                sample,
                image_to_tensor(
                    Image.open(
                        str(current_path / "images" / "ford" / image),
                    )
                ),
            )
            assert dataset.get_metadata(count).topic == "/image_front_left"
            count = count + 1
    assert dataset.get_first_timestamp() == datetime.datetime(
        2017, 8, 4, 4, 48, 43, 820895, tzinfo=datetime.timezone.utc
    )
    assert dataset.get_last_timestamp() == datetime.datetime(
        2017, 8, 4, 4, 49, 10, 155844, tzinfo=datetime.timezone.utc
    )


def test_read_tf() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = TfDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        ReadTfTopicConfig.from_yaml_file(
            str(current_path / "rosbag" / "ford" / "read_tf.yaml")
        ),
    )


@pytest.mark.skipif(
    (not os.getenv("AWS_ACCESS_KEY_ID")) or (not os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Do you have access rights to the test data?",
)  # type: ignore
def test_download_rosbag() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    download_rosbag(
        bucket_name="amber-test-rosbag",
        remote_rosbag_directory="ford_with_annotation",
        remote_rosbag_filename="bounding_box.mcap",
        endpoint_url="https://s3.us-west-1.wasabisys.com",
        is_public=False,
        download_dir=str(current_path / "rosbag"),
    )


@pytest.mark.skipif(
    (not os.getenv("AWS_ACCESS_KEY_ID")) or (not os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Do you have access rights to the test data?",
)  # type: ignore
def test_read_images_with_bounding_box_ford() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesAndAnnotationsDataset(
        download_rosbag(
            bucket_name="amber-test-rosbag",
            remote_rosbag_directory="ford_with_annotation",
            remote_rosbag_filename="bounding_box.mcap",
            endpoint_url="https://s3.us-west-1.wasabisys.com",
            is_public=True,
            download_dir=str(current_path / "rosbag"),
        ),
        ReadImagesAndAnnotationsConfig.from_yaml_file(
            str(
                current_path
                / "rosbag"
                / "ford_with_annotation"
                / "read_images_and_bounding_box.yaml"
            )
        ),
    )


@pytest.mark.skipif(
    (not os.getenv("AWS_ACCESS_KEY_ID")) or (not os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Do you have access rights to the test data?",
)  # type: ignore
def test_read_pointcloud() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = PointcloudDataset(
        download_rosbag(
            bucket_name="amber-test-rosbag",
            remote_rosbag_directory="vrx",
            remote_rosbag_filename="vrx_teleop.mcap",
            endpoint_url="https://s3.us-west-1.wasabisys.com",
            is_public=True,
            download_dir=str(current_path / "rosbag"),
        ),
        ReadPointCloudConfig.from_yaml_file(
            str(current_path / "rosbag" / "vrx" / "read_pointcloud.yaml")
        ),
    )
    assert len(dataset) == 46
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            assert (
                dataset.get_metadata(count).topic
                == "/wamv/sensors/lidars/lidar_wamv_sensor/points"
            )
            count = count + 1
    assert count == 46
