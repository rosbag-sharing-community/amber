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
from typing import Any, List
import shutil
import subprocess


class DemoArguments:
    def __init__(self, config: DeticImageLabalerConfig) -> None:
        self.vocabulary = config.vocabulary.value
        self.custom_vocabulary = ",".join(config.custom_vocabulary)


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.temporary_image_directory = "/tmp/detic_image_labaler"
        if not os.path.exists(self.temporary_image_directory):
            os.makedirs(self.temporary_image_directory)
        else:
            shutil.rmtree(self.temporary_image_directory)
            os.makedirs(self.temporary_image_directory)
        self.to_pil_image = transforms.ToPILImage()
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.config.validate()
        self.docker_client = docker.from_env()
        self.docker_client.images.pull("wamvtan/detic")
        self.container = self.docker_client.containers.run(
            image="wamvtan/detic",
            volumes={
                self.temporary_image_directory: {
                    "bind": "/workspace/Detic/outputs",
                    "mode": "rw",
                },
            },
            device_requests=self.build_device_requests(),
            command=["/bin/sh"],
            detach=True,
            tty=True,
            runtime=None,
        )

    def build_device_requests(self) -> list[docker.types.DeviceRequest]:
        requests = []
        if self.config.docker_config.use_gpu:
            requests.append(
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            )
        return requests

    def cpu_or_gpu(self):
        if self.config.docker_config.use_gpu:
            return "cuda"
        else:
            return "cpu"

    def build_command(self, index: int) -> List[str]:
        return [
            "/bin/bash",
            "-c",
            "python demo.py \
        --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --input "
            + "/workspace/Detic/outputs/input"
            + str(index)
            + ".jpeg"
            + " --output outputs/output"
            + str(index)
            + ".jpeg \
        --json_output outputs/detection"
            + str(index)
            + ".json \
        --vocabulary lvis \
        --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth MODEL.DEVICE "
            + self.cpu_or_gpu(),
        ]

    def __del__(self) -> None:
        self.container.stop()
        self.container.remove()

    def run_command(self, index: int) -> None:
        self.container.exec_run(self.build_command(index))

    def inference(self, dataset: Rosbag2Dataset) -> None:
        video: Any = None
        bar = tqdm(total=len(dataset))
        bar.set_description("Annotation progress")
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(
                os.path.join(
                    self.temporary_image_directory, "input" + str(index) + ".jpeg"
                )
            )
            self.run_command(index)
            bar.update()

        for index in range(len(dataset)):
            if self.config.video_output_path != "":
                opencv_image = cv2.imread(
                    os.path.join(
                        self.temporary_image_directory,
                        "output" + str(index) + ".jpeg",
                    )
                )
                if video == None:
                    video = cv2.VideoWriter(
                        self.config.video_output_path,
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        30.0,  # FPS
                        (opencv_image.shape[1], opencv_image.shape[0]),
                    )
                video.write(opencv_image)
