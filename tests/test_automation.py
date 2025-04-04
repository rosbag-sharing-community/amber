from amber_mcap.automation.blip2_encoder import Blip2Encoder
from amber_mcap.automation.clip_image_annotation_filter import ClipImageAnnotationFilter
from amber_mcap.automation.detic_image_labeler import DeticImageLabeler
from amber_mcap.automation.nerf_3d_reconstruction import Nerf3DReconstruction
from pathlib import Path
import os
from amber_mcap.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from amber_mcap.dataset.images_and_annotations_dataset import (
    ImagesAndAnnotationsDataset,
    ReadImagesAndAnnotationsConfig,
)
from amber_mcap.dataset.rosbag2_dataset import download_rosbag
import torch
from torchvision import transforms
import pytest
from PIL import Image


def test_detic_image_labeler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    labeler = DeticImageLabeler(
        str(current_path / "automation" / "detic_image_labeler.yaml")
    )
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        ReadImagesConfig.from_yaml_file(
            str(current_path / "rosbag" / "ford" / "read_image.yaml")
        ),
    )
    labeler.write(
        dataset,
        "/detic_image_labeler/annotation",
        labeler.inference(dataset),
        str(current_path / "rosbag" / "ford" / "output.mcap"),
    )


@pytest.mark.skipif(
    (not os.getenv("AWS_ACCESS_KEY_ID")) or (not os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Do you have access rights to the test data?",
)  # type: ignore
def test_clip_image_annotation_filter() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    filter = ClipImageAnnotationFilter(
        str(current_path / "automation" / "clip_image_annotation_filter.yaml")
    )
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
    assert len(dataset) == 39
    annotations = filter.inference(dataset)


@pytest.mark.skipif(
    (not os.getenv("AWS_ACCESS_KEY_ID")) or (not os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="Do you have access rights to the test data?",
)  # type: ignore
def test_clip_image_annotation_filter_with_lvis() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    filter = ClipImageAnnotationFilter(
        str(current_path / "automation" / "clip_image_annotation_filter_with_lvis.yaml")
    )
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
    assert len(dataset) == 39
    annotations = filter.inference(dataset)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NeRF is too heavy for CPU machine."
)  # type: ignore
def test_nerf_3d_reconstruction() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    labeler = Nerf3DReconstruction(
        str(current_path / "automation" / "nerf_3d_reconstruction.yaml")
    )
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "soccer_goal"),
        ReadImagesConfig.from_yaml_file(
            str(current_path / "rosbag" / "soccer_goal" / "read_image.yaml")
        ),
    )
    labeler.inference(dataset)


def test_blip2_encoder() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    enc = Blip2Encoder()
    itm_score_negative = enc.get_itm_score(
        transforms.ToTensor()(
            Image.open(str(current_path / "images" / "ford" / "28.png"))
        ),
        "Hoge",
    )
    itm_score_positive = enc.get_itm_score(
        transforms.ToTensor()(
            Image.open(str(current_path / "images" / "ford" / "28.png"))
        ),
        "A white car is on the left lane.",
    )
    assert itm_score_positive > itm_score_negative
