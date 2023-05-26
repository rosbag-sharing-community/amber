# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys

from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.automation.automation import Automation

from gradio_client import Client

import amber
from amber.automation.task_description import (
    DeticImageLabalerConfig,
)
from amber.automation.annotation import ImageAnnotations, BoundingBoxAnnotation

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
import json


class DemoArguments:
    def __init__(self, config: DeticImageLabalerConfig) -> None:
        self.vocabulary = config.vocabulary.value
        self.custom_vocabulary = ",".join(config.custom_vocabulary)


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.temporary_image_directory = "/tmp/detic_image_labaler"
        # self.setup_directory(self.temporary_image_directory)
        self.to_pil_image = transforms.ToPILImage()
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.config.validate()
        self.docker_client = docker.from_env()
        # self.docker_client.images.pull("wamvtan/detic")
        # self.container = self.docker_client.containers.run(
        #     image="wamvtan/detic",
        #     volumes={
        #         os.path.join(self.temporary_image_directory, "inputs"): {
        #             "bind": "/workspace/Detic/inputs",
        #             "mode": "rw",
        #         },
        #         os.path.join(self.temporary_image_directory, "outputs"): {
        #             "bind": "/workspace/Detic/outputs",
        #             "mode": "rw",
        #         },
        #     },
        #     device_requests=self.build_device_requests(),
        #     command=["/bin/sh"],
        #     detach=True,
        #     tty=True,
        #     runtime=None,
        # )

    def setup_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        os.makedirs(os.path.join(self.temporary_image_directory, "inputs"))
        os.makedirs(os.path.join(self.temporary_image_directory, "outputs"))

    def build_device_requests(self) -> list[docker.types.DeviceRequest]:
        requests = []
        if self.config.docker_config.use_gpu:
            requests.append(
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            )
        return requests

    def cpu_or_gpu(self) -> str:
        if self.config.docker_config.use_gpu:
            return "cuda"
        else:
            return "cpu"

    def build_command(self) -> List[str]:
        return [
            "/bin/bash",
            "-c",
            "python demo.py \
        --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --input /workspace/Detic/inputs/*.jpeg --output outputs \
        --vocabulary lvis \
        --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth MODEL.DEVICE "
            + self.cpu_or_gpu(),
        ]

    # def __del__(self) -> None:
    #     self.container.stop()
    #     self.container.remove()

    def run_command(self) -> None:
        _, stream = self.container.exec_run(self.build_command(), stream=True)
        for data in stream:
            print(data.decode())

    def get_image_annotations(self, index: int) -> ImageAnnotations:
        image_annotation = ImageAnnotations()
        image_annotation.image_index = index
        for annotation in json.load(
            open(
                os.path.join(
                    self.temporary_image_directory,
                    "outputs",
                    "input" + str(index) + ".json",
                ),
                "r",
            )
        ):
            image_annotation.bounding_boxes.append(
                BoundingBoxAnnotation.from_dict(annotation)
            )
        return image_annotation

    def inference(self, dataset: Rosbag2Dataset) -> None:
        video: Any = None
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(
                os.path.join(
                    self.temporary_image_directory,
                    "inputs",
                    "input" + str(index) + ".jpeg",
                )
            )
        # self.run_command()
        for index in range(len(dataset)):
            if self.config.video_output_path != "":
                opencv_image = cv2.imread(
                    os.path.join(
                        self.temporary_image_directory,
                        "outputs",
                        "input" + str(index) + ".jpeg",
                    )
                )
                self.get_image_annotations(index)
                if video == None:
                    video = cv2.VideoWriter(
                        self.config.video_output_path,
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        30.0,  # FPS
                        (opencv_image.shape[1], opencv_image.shape[0]),
                    )
                video.write(opencv_image)
