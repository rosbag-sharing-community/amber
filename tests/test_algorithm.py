import pytest
from amber.automation.blip2_encoder import Blip2Encoder
from pathlib import Path
import os
import torch


def test_blip2_encoder() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    encoder = Blip2Encoder()
    image_of_green_traffic_light: Path = Path(
        current_path / "images" / "ford" / "26.png"
    )
    assert encoder.get_cosine_similarity_from_image_file_and_text(
        image_of_green_traffic_light, "The color of traffic light is green."
    ) > encoder.get_cosine_similarity_from_image_file_and_text(
        image_of_green_traffic_light, "The color of traffic light is red."
    )
