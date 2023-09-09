from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
from amber.automation.automation import Automation
from amber.automation.annotation import ImageAnnotation
from amber.automation.clip_encoder import ClipEncoder
from amber.automation.task_description import (
    ClipImageAnnotationFilterConfig,
)
from torchvision import transforms
from typing import List, Dict, Tuple
from torch.nn.functional import cosine_similarity
import torch
import uuid


class ClipImageAnnotationFilter(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipImageAnnotationFilterConfig.from_yaml_file(yaml_path)
        self.clip_encoder = ClipEncoder()
        self.text_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
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
                        positive = cosine_similarity(
                            torch.tensor(bounding_box.clip_embeddings, dtype=float),
                            self.text_embeddings[target_object][0],
                        )
                        negative = cosine_similarity(
                            torch.tensor(bounding_box.clip_embeddings, dtype=float),
                            self.text_embeddings[target_object][1],
                        )
                        if (
                            bounding_box.object_class == "car_(automobile)"
                            or bounding_box.object_class == "bus_(vehicle)"
                        ):
                            # if (positive.item() / negative.item()) > 0.98:
                            self.to_pil_image(image_and_annotation[0]).crop(
                                (
                                    int(bounding_box.box.x1),
                                    int(bounding_box.box.y1),
                                    int(bounding_box.box.x2),
                                    int(bounding_box.box.y2),
                                )
                            ).save("data/" + str(uuid.uuid4()) + ".jpeg")
                            print(positive.item() / negative.item())
                            print(str(positive.item()) + "," + str(negative.item()))
        return filtered_annotations
