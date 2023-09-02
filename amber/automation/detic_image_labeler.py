import os

from amber.dataset.images_dataset import ImagesDataset
from amber.automation.automation import Automation

import amber
from amber.automation.task_description import (
    DeticImageLabalerConfig,
)
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation

import os
from torchvision import transforms
from tqdm import tqdm
from typing import Any, List
import shutil
from download import download
import onnxruntime
from PIL import Image
from torch.nn.functional import grid_sample
import numpy as np
import cv2
import json


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.weight_and_model = self.download_onnx(
            "Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16"
        )
        self.session = onnxruntime.InferenceSession(
            self.weight_and_model,
            providers=["CPUExecutionProvider"],  # "CUDAExecutionProvider"],
        )
        self.temporary_image_directory = "/tmp/detic_image_labaler"
        self.setup_directory(self.temporary_image_directory)
        self.to_pil_image = transforms.ToPILImage()
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

    def download_onnx(
        self,
        model: str,
        base_url: str = "https://storage.googleapis.com/ailia-models/detic/",
    ) -> str:
        download_directory = os.path.join(amber.__path__[0], "automation", "onnx")
        weight_path = os.path.join(download_directory, model + ".onnx")
        if not os.path.exists(weight_path):
            download(base_url + model + ".onnx", weight_path)
        return weight_path

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

    # This code was comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/detic.py#L276-L301
    def preprocess(self, image: np.ndarray, detection_width: int = 800) -> np.ndarray:
        im_h, im_w, _ = image.shape
        image = image[:, :, ::-1]  # BGR -> RGB
        size = detection_width
        max_size = detection_width
        scale = size / min(im_h, im_w)
        if im_h < im_w:
            oh, ow = size, scale * im_w
        else:
            oh, ow = scale * im_h, size
        if max(oh, ow) > max_size:
            scale = max_size / max(oh, ow)
            oh = oh * scale
            ow = ow * scale
        ow = int(ow + 0.5)
        oh = int(oh + 0.5)
        image = np.asarray(Image.fromarray(image).resize((ow, oh), Image.BILINEAR))
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def inference(self, dataset: ImagesDataset) -> List[ImageAnnotation]:
        image_annotations: List[ImageAnnotation] = []
        for index, image in enumerate(dataset):
            input_width, input_height = image.shape[:2]
            input_image = self.preprocess(
                cv2.cvtColor(np.asarray(self.to_pil_image(image)), cv2.COLOR_BGRA2BGR)
            )
            output_width, output_height = input_image.shape[:2]
            output = self.session.run(
                None,
                {
                    "img": input_image,
                    "im_hw": np.array([input_height, input_width]).astype(np.int64),
                },
            )
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
