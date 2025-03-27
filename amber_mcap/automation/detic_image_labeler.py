import os

from amber_mcap.dataset.images_dataset import ImagesDataset
from amber_mcap.automation.automation import Automation
from amber_mcap.automation.clip_encoder import ClipEncoder

import amber_mcap
from amber_mcap.automation.task_description import (
    DeticImageLabalerConfig,
)
from amber_mcap.automation.annotation import ImageAnnotation, BoundingBoxAnnotation


from amber_mcap.util.lvis.lvis_v1_categories import (
    LVIS_CATEGORIES as LVIS_V1_CATEGORIES,
)
from amber_mcap.util.imagenet_21k.in21k_categories import IN21K_CATEGORIES
from amber_mcap.util.color import color_brightness, random_color

import os
from torchvision import transforms
from tqdm import tqdm
from typing import Any, List, Dict, Tuple
import shutil
from download import download
import onnxruntime
from PIL import Image

# from torch.nn.functional import grid_sample
import numpy as np
import cv2
import json
import copy


class DeticImageLabeler(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = DeticImageLabalerConfig.from_yaml_file(yaml_path)
        self.weight_and_model = self.download_onnx(self.config.get_onnx_filename())
        self.session = onnxruntime.InferenceSession(
            self.weight_and_model,
            providers=["CPUExecutionProvider"],  # "CUDAExecutionProvider"],
        )
        self.to_pil_image = transforms.ToPILImage()

    def download_onnx(
        self,
        model: str,
        base_url: str = "https://storage.googleapis.com/ailia-models/detic/",
    ) -> str:
        download_directory = os.path.join(amber_mcap.__path__[0], "automation", "onnx")
        weight_path = os.path.join(download_directory, model)
        if not os.path.exists(weight_path):
            download(base_url + model, weight_path)
        return weight_path

    # This code comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/detic.py#L276-L301
    def preprocess(self, image: np.ndarray, detection_width: int = 800) -> np.ndarray:
        height, width, _ = image.shape
        image = image[:, :, ::-1]  # BGR -> RGB
        size = detection_width
        max_size = detection_width
        scale = size / min(height, width)
        if height < width:
            oh, ow = size, scale * width
        else:
            oh, ow = scale * height, size
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

    # This function comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/detic.py#L153-L174
    def mask_to_polygons(self, mask: np.ndarray) -> List[Any]:
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return []
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        return [x + 0.5 for x in res if len(x) >= 6]

    # This function comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/dataset_utils.py#L5-L11
    def get_lvis_meta_v1(self) -> Dict[str, List[str]]:
        thing_classes = [k["synonyms"][0] for k in LVIS_V1_CATEGORIES]
        meta = {"thing_classes": thing_classes}
        return meta

    # This function comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/dataset_utils.py#L14-L18
    def get_in21k_meta_v1(self) -> Dict[str, List[str]]:
        thing_classes = IN21K_CATEGORIES
        meta = {"thing_classes": thing_classes}
        return meta

    # This function comes from https://github.com/axinc-ai/ailia-models/blob/da1c277b602606586cd83943ef6b23eb705ec604/object_detection/detic/detic.py#L177-L269
    def draw_predictions(
        self, image: np.ndarray, detection_results: Any, vocabulary: str
    ) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = detection_results["boxes"].astype(np.int64)
        scores = detection_results["scores"]
        classes = detection_results["classes"].tolist()
        masks = detection_results["masks"].astype(np.uint8)

        class_names = (
            self.get_lvis_meta_v1()
            if vocabulary == "lvis"
            else self.get_in21k_meta_v1()
        )["thing_classes"]
        labels = [class_names[i] for i in classes]
        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        num_instances = len(boxes)

        np.random.seed()
        assigned_colors = [random_color(maximum=255) for _ in range(num_instances)]

        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs]
            labels = [labels[k] for k in sorted_idxs]
            masks = [masks[idx] for idx in sorted_idxs]
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        default_font_size = int(max(np.sqrt(height * width) // 90, 10))

        for i in range(num_instances):
            color = assigned_colors[i]
            color = (int(color[0]), int(color[1]), int(color[2]))
            image_b = image.copy()

            # draw box
            x0, y0, x1, y1 = boxes[i]
            cv2.rectangle(
                image_b,
                (x0, y0),
                (x1, y1),
                color=color,
                thickness=default_font_size // 4,
            )

            # draw segment
            polygons = self.mask_to_polygons(masks[i])
            for points in polygons:
                points = np.array(points).reshape((1, -1, 2)).astype(np.int32)
                cv2.fillPoly(image_b, pts=[points], color=color)

            image = cv2.addWeighted(image, 0.5, image_b, 0.5, 0)

        for i in range(num_instances):
            color = assigned_colors[i]
            color_text = color_brightness(color, brightness_factor=0.7)

            color = (int(color[0]), int(color[1]), int(color[2]))
            color_text = (int(color_text[0]), int(color_text[1]), int(color_text[2]))

            x0, y0, x1, y1 = boxes[i]

            SMALL_OBJECT_AREA_THRESH = 1000
            instance_area = (y1 - y0) * (x1 - x0)

            # for small objects, draw text at the side to avoid occlusion
            text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
            if instance_area < SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40:
                if y1 >= height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            # draw label
            x, y = text_pos
            text = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            height_ratio = (y1 - y0) / np.sqrt(height * width)
            font_scale = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(
                image, text_pos, (int(x + text_w * 0.6), y + text_h), (0, 0, 0), -1
            )
            cv2.putText(
                image,
                text,
                (x, y + text_h - 5),
                fontFace=font,
                fontScale=font_scale * 0.6,
                color=color_text,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

        return image

    def inference(self, dataset: ImagesDataset) -> List[ImageAnnotation]:
        clip_encoder = ClipEncoder()
        image_annotations: List[ImageAnnotation] = []
        vocabulary = "lvis"
        class_names = (
            self.get_lvis_meta_v1()
            if vocabulary == "lvis"
            else self.get_in21k_meta_v1()
        )["thing_classes"]

        video: Any = None
        for index, image in enumerate(dataset):
            print("Labeling image " + str(index) + "/" + str(len(dataset)))
            input_width = image.shape[2]
            input_height = image.shape[1]
            input_image = self.preprocess(np.asarray(self.to_pil_image(image)), 800)
            boxes, scores, classes, masks = self.session.run(
                None,
                {
                    "img": input_image,
                    "im_hw": np.array([input_height, input_width]).astype(np.int64),
                },
            )
            image_annotation = ImageAnnotation()
            image_annotation.image_index = index
            detection_results = {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "masks": masks,
            }
            for bbox_id in range(len(boxes)):
                bounding_box = BoundingBoxAnnotation()
                bounding_box.box.x1 = boxes[bbox_id][0]
                bounding_box.box.y1 = boxes[bbox_id][1]
                bounding_box.box.x2 = boxes[bbox_id][2]
                bounding_box.box.y2 = boxes[bbox_id][3]
                bounding_box.score = scores[bbox_id]
                bounding_box.object_class = class_names[classes[bbox_id]]
                if (
                    abs(bounding_box.box.x2 - bounding_box.box.x1)
                    >= self.config.min_width
                    or abs(bounding_box.box.y2 - bounding_box.box.y1)
                    >= self.config.min_height
                ) and abs(bounding_box.box.x2 - bounding_box.box.x1) * abs(
                    bounding_box.box.y2 - bounding_box.box.y1
                ) >= self.config.min_area:
                    image_annotation.bounding_boxes.append(copy.deepcopy(bounding_box))
            image_annotations.append(
                clip_encoder.get_single_image_embeddings_for_objects(
                    self.to_pil_image(image), image_annotation
                )
            )
            visualization = self.draw_predictions(
                np.asarray(self.to_pil_image(image)), detection_results, "lvis"
            )
            if video == None:
                video = cv2.VideoWriter(
                    self.config.video_output_path,
                    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                    30.0,  # FPS
                    (visualization.shape[1], visualization.shape[0]),
                )
            video.write(visualization)
        video.release()
        return image_annotations
