import pytest
from amber.automation.blip2_encoder import Blip2Encoder
from pathlib import Path
import os
import torch


def test_blip2_encoder() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    encoder = Blip2Encoder()
    image_embedding = encoder.encode_image_from_file(
        Path(current_path / "images" / "ford" / "26.png")
    )
    text_embedding_green = encoder.encode_text("The color of traffic light is green.")
    text_embedding_red = encoder.encode_text("The color of traffic light is red.")
    # sims = torch.bmm(image_embedding, text_embedding.unsqueeze(-1))
    sims_green = torch.bmm(image_embedding, text_embedding_green.unsqueeze(-1))
    sim_green, _ = torch.max(sims_green, dim=1)
    sims_red = torch.bmm(image_embedding, text_embedding_red.unsqueeze(-1))
    sim_red, _ = torch.max(sims_red, dim=1)
    assert sim_green > sim_red
