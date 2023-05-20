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

from fvcore.common.config import CfgNode
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
    download_directory = os.path.join(
        rosbag2_pytorch_data_loader.__path__[0], "automation", "models", "detic"
    )
    config_directory = os.path.join(
        rosbag2_pytorch_data_loader.__path__[0], "automation", "config", "detic"
    )
    metadata_directory = os.path.join(
        rosbag2_pytorch_data_loader.__path__[0],
        "automation",
        "datasets",
        "detic",
        "metadata",
    )

    def __init__(self, yaml_path: str) -> None:
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.download_model(self.config.model.value)
        print(self.setup_detectron2_cfg())
        self.demo = VisualizationDemo(
            self.setup_detectron2_cfg(),
            self.setup_demo_arguments(),
        )

    def inference(self, dataset: Rosbag2Dataset) -> None:
        pass

    def get_model_url(
        self, model: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> str:
        if model in self.models:
            return self.models[model]["url"]
        else:
            raise Exception(
                "model name "
                + model
                + " does not existing on rosbag2_pytorch_data_loader."
            )

    def get_model_filename(
        self, model: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> str:
        if model in self.models:
            return self.models[model]["filename"]
        else:
            raise Exception(
                "model name "
                + model
                + " does not existing on rosbag2_pytorch_data_loader."
            )

    def get_model_path(
        self, model: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> str:
        return os.path.join(self.download_directory, self.get_model_filename(model))

    def download_model(
        self, model: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    ) -> None:
        if not os.path.exists(self.get_model_path(model)):
            urllib.request.urlretrieve(
                self.get_model_url(model), self.get_model_path(model)
            )

    def setup_detectron2_cfg(self) -> CfgNode:
        detectron2_config = get_cfg()
        add_centernet_config(detectron2_config)
        add_detic_config(detectron2_config)
        detectron2_config.merge_from_file(
            os.path.join(self.config_directory, self.config.config_file)
        )
        # Set score_threshold for builtin models
        detectron2_config.MODEL.RETINANET.SCORE_THRESH_TEST = (
            self.config.confidence_threshold
        )
        detectron2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.config.confidence_threshold
        )
        detectron2_config.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            self.config.confidence_threshold
        )
        detectron2_config.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(
            self.metadata_directory, "lvis_v1_train_cat_info.json"
        )
        detectron2_config.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = os.path.join(
            self.metadata_directory, "lvis_v1_clip_a+cname.npy"
        )
        detectron2_config.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        detectron2_config.MODEL.WEIGHTS = self.get_model_path(self.config.model.value)
        detectron2_config.freeze()
        return detectron2_config

    def setup_demo_arguments(self) -> Any:
        return DemoArguments(self.config)
