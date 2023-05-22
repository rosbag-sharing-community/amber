# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys

from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset
from rosbag2_pytorch_data_loader.automation.automation import Automation

from gradio_client import Client

import rosbag2_pytorch_data_loader
from rosbag2_pytorch_data_loader.automation.task_description import (
    DeticImageLabalerConfig,
)

import os
import docker
import socket
from torchvision import transforms


class DemoArguments:
    def __init__(self, config: DeticImageLabalerConfig) -> None:
        self.vocabulary = config.vocabulary.value
        self.custom_vocabulary = ",".join(config.custom_vocabulary)


class DeticImageLabeler(Automation):  # type: ignore
    models = {
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size": {
            "filename": "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
            "url": "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        }
    }

    def __init__(self, yaml_path: str) -> None:
        self.temporary_image_filepath = "/tmp/input.jpg"
        self.to_pil_image = transforms.ToPILImage()
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.docker_client = docker.from_env()
        self.container = self.docker_client.containers.run(
            "detic", detach=True, network_mode="host"
        )
        # waiting for until the docker container is ready
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while sock.connect_ex(("0.0.0.0", 8000)) != 0:
            continue
        self.client = Client("http://0.0.0.0:8000")

    def __del__(self) -> None:
        self.container.stop()

    def inference(self, dataset: Rosbag2Dataset) -> None:
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(self.temporary_image_filepath)
            print(self.client.predict(self.temporary_image_filepath))
            # Image.open()
