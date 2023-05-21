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

from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset
from rosbag2_pytorch_data_loader.automation.automation import Automation

from gradio_client import Client

import rosbag2_pytorch_data_loader
from rosbag2_pytorch_data_loader.automation.task_description import (
    DeticImageLabalerConfig,
)

import os

from typing import Any


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
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)

    def inference(self, dataset: Rosbag2Dataset) -> None:
        pass
