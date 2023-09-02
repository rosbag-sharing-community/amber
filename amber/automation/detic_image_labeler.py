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
from typing import Any, List, Tuple
import shutil
import subprocess
import json
from download import download


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.download_weight_and_model("Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis")
        self.temporary_image_directory = "/tmp/detic_image_labaler"
        self.setup_directory(self.temporary_image_directory)
        # self.to_pil_image = transforms.ToPILImage()
        # self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        # self.config.validate()
        # self.docker_client = docker.from_env()
        # self.docker_image_name = "wamvtan/detic"
        # self.docker_client.images.pull(self.docker_image_name)
        # self.container = self.docker_client.containers.run(
        #     image=self.docker_image_name,
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
        #     command=["/bin/bash"],
        #     detach=True,
        #     tty=True,
        #     runtime=None,
        # )

    def download_weight_and_model(
        self,
        model: str,
        base_url: str = "https://storage.googleapis.com/ailia-models/detic/",
    ) -> Tuple[str, str]:
        download_directory = os.path.join(amber.__path__[0], "automation", "onnx")
        weight_path = os.path.join(download_directory, model + ".onnx")
        model_path = os.path.join(download_directory, model + ".onnx.prototxt")
        if not os.path.exists(weight_path):
            download(base_url + model + ".onnx", weight_path)
        if not os.path.exists(model_path):
            download(base_url + model + ".onnx.prototxt", model_path)
        return (weight_path, model_path)

    def setup_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        os.makedirs(os.path.join(self.temporary_image_directory, "inputs"))
        os.makedirs(os.path.join(self.temporary_image_directory, "outputs"))

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
        return []
        # video: Any = None
        # image_annotations: List[ImageAnnotation] = []
        # for index, image in enumerate(dataset):
        #     self.to_pil_image(image).save(
        #         os.path.join(
        #             self.temporary_image_directory,
        #             "inputs",
        #             "input" + str(index) + ".jpeg",
        #         )
        #     )
        # self.run_command()
        # for index in range(len(dataset)):
        #     if self.config.video_output_path != "":
        #         opencv_image = cv2.imread(
        #             os.path.join(
        #                 self.temporary_image_directory,
        #                 "outputs",
        #                 "input" + str(index) + ".jpeg",
        #             )
        #         )
        #         image_annotations.append(self.get_image_annotations(index))
        #         if video == None:
        #             video = cv2.VideoWriter(
        #                 self.config.video_output_path,
        #                 cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        #                 30.0,  # FPS
        #                 (opencv_image.shape[1], opencv_image.shape[0]),
        #             )
        #         video.write(opencv_image)
        # return image_annotations
