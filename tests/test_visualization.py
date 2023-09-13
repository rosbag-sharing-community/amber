from amber.visualization.clip_embeddings_visualization import (
    ClipEmbeddingsVisualization,
)
import pytest
from pathlib import Path
import os
from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset


def test_visualization() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    visualization = ClipEmbeddingsVisualization(
        str(current_path / "visualization" / "clip_image_visualization.yaml")
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
    visualization.visualize(dataset)
