import argparse
import os
import sys

from amber.dataset.images_dataset import ImagesDataset
from amber.automation.automation import Automation

from gradio_client import Client

import amber
from amber.automation.task_description import (
    DeticImageLabalerConfig,
)
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation

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


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.temporary_image_directory = "/tmp/detic_image_labaler"
        self.setup_directory(self.temporary_image_directory)
        self.to_pil_image = transforms.ToPILImage()
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.config.validate()
        self.docker_client = docker.from_env()
        self.docker_image_name = "wamvtan/detic"
        self.docker_client.images.pull(self.docker_image_name)
        self.container = self.docker_client.containers.run(
            image=self.docker_image_name,
            volumes={
                os.path.join(self.temporary_image_directory, "inputs"): {
                    "bind": "/workspace/Detic/inputs",
                    "mode": "rw",
                },
                os.path.join(self.temporary_image_directory, "outputs"): {
                    "bind": "/workspace/Detic/outputs",
                    "mode": "rw",
                },
            },
            device_requests=self.build_device_requests(),
            command=["/bin/bash"],
            detach=True,
            tty=True,
            runtime=None,
        )

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

    def __del__(self) -> None:
        self.container.stop()
        self.container.remove()
        if self.config.docker_config.claenup_image_on_shutdown:
            self.docker_client.images.remove(
                image=self.docker_image_name, force=False, noprune=False
            )

    def run_command(self) -> None:
        _, stream = self.container.exec_run(self.build_command(), stream=True)
        for data in stream:
            print(data.decode())

    def get_image_annotations(self, index: int) -> ImageAnnotation:
        image_annotation = ImageAnnotation()
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
        )["detections"]:
            image_annotation.bounding_boxes.append(
                BoundingBoxAnnotation.from_dict(annotation)
            )
        return image_annotation

    def inference(self, dataset: ImagesDataset) -> List[ImageAnnotation]:
        video: Any = None
        image_annotations: List[ImageAnnotation] = []
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(
                os.path.join(
                    self.temporary_image_directory,
                    "inputs",
                    "input" + str(index) + ".jpeg",
                )
            )
        self.run_command()
        for index in range(len(dataset)):
            if self.config.video_output_path != "":
                opencv_image = cv2.imread(
                    os.path.join(
                        self.temporary_image_directory,
                        "outputs",
                        "input" + str(index) + ".jpeg",
                    )
                )
                image_annotations.append(self.get_image_annotations(index))
                if video == None:
                    video = cv2.VideoWriter(
                        self.config.video_output_path,
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        30.0,  # FPS
                        (opencv_image.shape[1], opencv_image.shape[0]),
                    )
                video.write(opencv_image)
        return image_annotations
