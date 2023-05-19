# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset
from rosbag2_pytorch_data_loader.automation.automation import Automation

import urllib.request

import rosbag2_pytorch_data_loader
from rosbag2_pytorch_data_loader.automation.task_description import (
    DeticImageLabalerConfig,
)

import os


class DeticImageLabeler(Automation):  # type: ignore
    models = {
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size": {
            "filename": "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
            "url": "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        }
    }

    def __init__(self, yaml_path: str) -> None:
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.download_model(self.config.model.value)

    def inference(self, dataset: Rosbag2Dataset) -> None:
        pass

    def get_model_url(
        self, weight: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> str:
        if weight in self.models:
            return self.models[weight]["url"]
        else:
            raise Exception(
                "weight name "
                + weight
                + " does not existing on rosbag2_pytorch_data_loader."
            )

    def get_model_filename(
        self, weight: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> str:
        if weight in self.models:
            return self.models[weight]["filename"]
        else:
            raise Exception(
                "weight name "
                + weight
                + " does not existing on rosbag2_pytorch_data_loader."
            )

    def download_model(
        self,
        weight: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size",
        base_directory: str = os.path.join(
            rosbag2_pytorch_data_loader.__path__[0], "automation", "models", "detic"
        ),
    ) -> None:
        path = os.path.join(base_directory, self.get_model_filename(weight))
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                self.get_model_url(weight),
                os.path.join(base_directory, self.get_model_filename(weight)),
            )
