from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
from amber.automation.automation import Automation
from amber.automation.annotation import ImageAnnotation
from amber.automation.clip_encoder import ClipEncoder
from amber.automation.task_description import (
    ClipImageAnnotationFilterConfig,
)
from torchvision import transforms
from typing import List, Dict
from torch.nn.functional import cosine_similarity
import torch


class ClipImageAnnotationFilter(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipImageAnnotationFilterConfig.from_yaml_file(yaml_path)
        self.clip_encoder = ClipEncoder()
        self.text_embeddings: Dict[str, torch.Tensor] = {}
        for target_object in self.config.target_objects:
            self.text_embeddings[
                target_object
            ] = self.clip_encoder.get_text_embeddings_for_positive_negative_prompts(
                target_object
            )
        self.to_pil_image = transforms.ToPILImage()
        return

    def inference(self, dataset: ImagesAndAnnotationsDataset) -> List[ImageAnnotation]:
        filtered_annotations: List[ImageAnnotation] = []

        for index, image_and_annotation in enumerate(dataset):
            for bounding_box_index, bounding_box in enumerate(
                image_and_annotation[1].bounding_boxes
            ):
                width = bounding_box.box.x2 - bounding_box.box.x1
                height = bounding_box.box.y2 - bounding_box.box.y1
                if width * height >= self.config.min_area and (
                    width >= self.config.min_width or height >= self.config.min_height
                ):
                    for target_object in self.config.target_objects:
                        image_features = torch.tensor(
                            bounding_box.clip_embeddings, dtype=float
                        )
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features = self.text_embeddings[target_object]
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = (image_features.float() @ text_features.T).softmax(
                            dim=-1
                        )
                        values, indices = similarity[0].topk(1)
                        print("index : => " + str(indices.item()))
                        print("value : => " + str(values.item()))
        return filtered_annotations
