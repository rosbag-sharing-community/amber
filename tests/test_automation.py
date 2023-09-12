from amber.automation.detic_image_labeler import DeticImageLabeler
from amber.automation.clip_image_annotation_filter import ClipImageAnnotationFilter
from amber.automation.nerf_3d_reconstruction import Nerf3DReconstruction
from pathlib import Path
import os
from amber.dataset.images_dataset import ImagesDataset
from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
import torch
import pytest


def test_detic_auto_labeler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    labeler = DeticImageLabeler(
        str(current_path / "automation" / "detic_image_labeler.yaml")
    )
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        str(current_path / "rosbag" / "ford" / "read_image.yaml"),
    )
    labeler.inference(dataset)


def test_clip_image_annotation_filter() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    filter = ClipImageAnnotationFilter(
        str(current_path / "automation" / "clip_image_annotation_filter.yaml")
    )
    dataset = ImagesAndAnnotationsDataset(
        str(current_path / "rosbag" / "ford_with_annotation" / "bounding_box.mcap"),
        str(
            current_path
            / "rosbag"
            / "ford_with_annotation"
            / "read_images_and_bounding_box.yaml"
        ),
    )
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
        str(current_path / "rosbag" / "soccer_goal" / "read_image.yaml"),
    )
    labeler.inference(dataset)
