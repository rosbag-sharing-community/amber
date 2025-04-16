import pytest
from amber_mcap.automation.blip2_encoder import Blip2Encoder
from pathlib import Path
import os
import torch
from torchvision import transforms
from PIL import Image


# def test_blip2_encoder() -> None:
#     current_path = Path(os.path.dirname(os.path.realpath(__file__)))
#     encoder = Blip2Encoder()
#     image_of_green_traffic_light: Path = Path(
#         current_path / "images" / "ford" / "26.png"
#     )
#     assert encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_green_traffic_light, "The color of traffic light is green."
#     ) > encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_green_traffic_light, "The color of traffic light is red."
#     )
#     image_of_speed_limit: Path = Path(current_path / "images" / "ford" / "35.png")
#     assert encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 35."
#     ) > encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 20."
#     )
#     assert encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 35."
#     ) > encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 40."
#     )
#     assert encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 35."
#     ) > encoder.get_cosine_similarity_from_image_file_and_text(
#         image_of_speed_limit, "The speed limit is 50."
#     )
#     encoder.encode_image_from_file(image_of_speed_limit)


def test_blip2_encoder_itm() -> None:
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
