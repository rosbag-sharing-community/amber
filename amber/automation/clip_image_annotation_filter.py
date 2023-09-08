from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
from amber.automation.automation import Automation
from amber.automation.annotation import ImageAnnotation
from amber.automation.clip_encoder import ClipEncoder
from amber.automation.task_description import (
    ClipImageAnnotationFilterConfig,
)
from torchvision import transforms
from typing import List


class ClipImageAnnotationFilter(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipImageAnnotationFilterConfig.from_yaml_file(yaml_path)
        self.clip_encoder = ClipEncoder()
        self.to_pil_image = transforms.ToPILImage()
        return

    def inference(self, dataset: ImagesAndAnnotationsDataset) -> List[ImageAnnotation]:
        filtered_annotations: List[ImageAnnotation] = []
        for index, image_and_annotation in enumerate(dataset):
            self.to_pil_image(image_and_annotation[0]).save(
                "data/" + str(index) + ".jpeg", quality=95
            )
            for bounding_box_index, bounding_box in enumerate(
                image_and_annotation[1].bounding_boxes
            ):
                self.to_pil_image(image_and_annotation[0]).crop(
                    (
                        (
                            int(bounding_box.box.x1),
                            int(bounding_box.box.y1),
                            int(bounding_box.box.x2),
                            int(bounding_box.box.y2),
                        )
                    )
                ).save(
                    "data/" + str(index) + "-" + str(bounding_box_index) + ".jpeg",
                    quality=95,
                )
        return filtered_annotations
