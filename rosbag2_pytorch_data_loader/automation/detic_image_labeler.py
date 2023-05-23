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
import cv2
import time
from tqdm import tqdm
from typing import Any


class DemoArguments:
    def __init__(self, config: DeticImageLabalerConfig) -> None:
        self.vocabulary = config.vocabulary.value
        self.custom_vocabulary = ",".join(config.custom_vocabulary)


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.temporary_image_filepath = "/tmp/input.jpg"
        self.to_pil_image = transforms.ToPILImage()
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.docker_client = docker.from_env()
        self.container = self.docker_client.containers.run(
            "wamvtan/detic", detach=True, network_mode="host"
        )
        # waiting for until the docker container is ready
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while sock.connect_ex(("0.0.0.0", 8000)) != 0:
            print("Try connecting to container...")
            time.sleep(1)
            continue
        self.client = Client("http://0.0.0.0:8000")

    def __del__(self) -> None:
        self.container.stop()

    def inference(self, dataset: Rosbag2Dataset) -> None:
        images = []
        video: Any = None
        bar = tqdm(total=len(dataset))
        bar.set_description("Annotation progress")
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(self.temporary_image_filepath)
            images.append(self.client.predict(self.temporary_image_filepath))
            bar.update()
        for image in images:
            opencv_image = cv2.imread(image)
            if video == None:
                video = cv2.VideoWriter(
                    "output.mp4",
                    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                    30.0,  # FPS
                    (opencv_image.shape[1], opencv_image.shape[0]),
                )
            video.write(opencv_image)
