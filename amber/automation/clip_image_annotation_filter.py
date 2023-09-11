from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
from amber.automation.automation import Automation
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from amber.automation.clip_encoder import ClipEncoder
from amber.automation.sentence_transformer import TextEncoder
from amber.automation.task_description import (
    ClipImageAnnotationFilterConfig,
    ClipClassifyMethod,
)
from torchvision import transforms
from typing import List, Dict, Tuple
from torch.nn.functional import cosine_similarity
import torch
from amber.exception import RuntimeError


class ClipImageAnnotationFilter(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipImageAnnotationFilterConfig.from_yaml_file(yaml_path)
        self.clip_encoder = ClipEncoder()
        self.text_encoder = TextEncoder()
        self.text_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for target_object in self.config.target_objects:
            self.text_embeddings[
                target_object
            ] = self.clip_encoder.get_text_embeddings_for_positive_negative_prompts(
                target_object
            )
        self.to_pil_image = transforms.ToPILImage()
        return

    def inference_with_lvis_vocabulary(
        self,
        image: torch.Tensor,
        clip_embeddings: torch.Tensor,
        bounding_box: BoundingBoxAnnotation,
        image_number: int,
    ) -> None:
        result = self.clip_encoder.classify_with_custom_vocabulary(
            clip_embeddings, self.config.target_objects
        )
        if result != None:
            self.to_pil_image(image).crop(
                (
                    int(bounding_box.box.x1),
                    int(bounding_box.box.y1),
                    int(bounding_box.box.x2),
                    int(bounding_box.box.y2),
                )
            ).save("data/" + str(image_number) + ".jpeg")
            image_number = image_number + 1

    def inference_with_bert(
        self,
        image: torch.Tensor,
        clip_embeddings: torch.Tensor,
        bounding_box: BoundingBoxAnnotation,
        image_number: int,
    ) -> None:
        annotation_text_embeddings = self.clip_encoder.get_text_embeddings(
            "A photo of a " + bounding_box.object_class
        )
        for target_object in self.config.target_objects:
            clip_similarity = cosine_similarity(
                clip_embeddings / torch.sum(clip_embeddings),
                self.text_embeddings[target_object][0],
            )
            positive = cosine_similarity(
                clip_embeddings / torch.sum(clip_embeddings)
                + annotation_text_embeddings
                / torch.sum(annotation_text_embeddings)
                * self.text_encoder.cosine_similarity(
                    bounding_box.object_class, target_object
                ),
                self.text_embeddings[target_object][0],
            )
            negative = cosine_similarity(
                clip_embeddings / torch.sum(clip_embeddings),
                # - annotation_text_embeddings
                # / torch.sum(annotation_text_embeddings)
                # * bounding_box.score
                # * self.text_encoder.cosine_similarity(
                #     bounding_box.object_class, target_object
                # ),
                self.text_embeddings[target_object][1],
            )
            if (
                positive.item() > 2.0 * negative.item()
                and positive.item() >= 0.5
                and clip_similarity.item() >= 0.25
            ):
                self.to_pil_image(image).crop(
                    (
                        int(bounding_box.box.x1),
                        int(bounding_box.box.y1),
                        int(bounding_box.box.x2),
                        int(bounding_box.box.y2),
                    )
                ).save("data/" + str(image_number) + ".jpeg")
                print("Image Number : " + str(image_number))
                print("P/N ratio : " + str(positive.item() / negative.item()))
                print("Score : " + str(bounding_box.score))
                print("Class : " + str(bounding_box.object_class))
                print(clip_similarity)
                print(cosine_similarity(clip_embeddings, annotation_text_embeddings))
                print(
                    "Prompt Similarity : "
                    + str(
                        self.text_encoder.cosine_similarity(
                            bounding_box.object_class, target_object
                        )
                    )
                )
                print("Positive : " + str(positive.item()))
                print(str(positive.item()) + "," + str(negative.item()))
                print("")

    def inference(self, dataset: ImagesAndAnnotationsDataset) -> List[ImageAnnotation]:
        print(self.config)
        filtered_annotations: List[ImageAnnotation] = []
        image_number = 0
        for index, image_and_annotation in enumerate(dataset):
            for bounding_box_index, bounding_box in enumerate(
                image_and_annotation[1].bounding_boxes
            ):
                width = bounding_box.box.x2 - bounding_box.box.x1
                height = bounding_box.box.y2 - bounding_box.box.y1
                if width * height >= self.config.min_area and (
                    width >= self.config.min_width or height >= self.config.min_height
                ):
                    clip_embeddings = torch.tensor(
                        bounding_box.clip_embeddings, dtype=float
                    )
                    if (
                        self.config.classify_method
                        == ClipClassifyMethod.CLIP_WITH_LVIS_AND_CUSTOM_VOCABULARY
                    ):
                        self.inference_with_lvis_vocabulary(
                            image_and_annotation[0],
                            clip_embeddings,
                            bounding_box,
                            image_number,
                        )
                    elif (
                        self.config.classify_method
                        == ClipClassifyMethod.CONSIDER_ANNOTATION_WITH_BERT
                    ):
                        self.inference_with_bert(
                            image_and_annotation[0],
                            clip_embeddings,
                            bounding_box,
                            image_number,
                        )
                    else:
                        raise RuntimeError(
                            "Classify method "
                            + self.config.classify_method.value
                            + " does not support."
                        )
                    image_number = image_number + 1
        return filtered_annotations
