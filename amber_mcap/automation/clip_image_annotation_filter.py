from amber_mcap.dataset.images_and_annotations_dataset import (
    ImagesAndAnnotationsDataset,
)
from amber_mcap.automation.automation import Automation
from amber_mcap.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from amber_mcap.automation.clip_encoder import ClipEncoder
from amber_mcap.automation.sentence_transformer import TextEncoder
from amber_mcap.automation.task_description import (
    ClipImageAnnotationFilterConfig,
    ClipClassifyMethod,
)
from torchvision import transforms
from typing import List, Dict, Tuple
from torch.nn.functional import cosine_similarity
import torch
from amber_mcap.exception import RuntimeError
import copy


class ClipImageAnnotationFilter(Automation):  # type: ignore
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipImageAnnotationFilterConfig.from_yaml_file(yaml_path)
        self.clip_encoder = ClipEncoder()
        self.text_encoder = TextEncoder()
        self.text_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for target_object in self.config.target_objects:
            self.text_embeddings[
                target_object
            ] = self.clip_encoder.get_single_text_embeddings_for_positive_negative_prompts(
                target_object
            )
        self.to_pil_image = transforms.ToPILImage()
        return

    def inference_with_lvis_vocabulary(
        self,
        bounding_box: BoundingBoxAnnotation,
    ) -> bool:
        clip_embeddings: torch.Tensor = torch.tensor(
            bounding_box.clip_embeddings, dtype=float
        )
        result = self.clip_encoder.classify_with_custom_vocabulary(
            clip_embeddings, self.config.target_objects
        )
        if result != None:
            return True
        else:
            return False

    def inference_with_bert(
        self,
        bounding_box: BoundingBoxAnnotation,
    ) -> bool:
        annotation_text_embeddings = self.clip_encoder.get_single_text_embeddings(
            "A photo of a " + bounding_box.object_class
        )
        clip_embeddings: torch.Tensor = torch.tensor(
            bounding_box.clip_embeddings, dtype=float
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
            # print("P/N ratio : " + str(positive.item() / negative.item()))
            # print("Score : " + str(bounding_box.score))
            # print("Class : " + str(bounding_box.object_class))
            # print("Clip Cosine Similarity with bert: " + str(positive.item()))
            # print("Clip Cosine Similarity without bert: " + str(clip_similarity.item()))
            # print("")
            if (
                positive.item()
                > self.config.consider_annotation_with_bert_config.positive_nagative_ratio
                * negative.item()
                and positive.item()
                >= self.config.consider_annotation_with_bert_config.min_clip_cosine_similarity_with_bert
                and clip_similarity.item()
                >= self.config.consider_annotation_with_bert_config.min_clip_cosine_similarity
            ):
                return True
        return False

    def inference(self, dataset: ImagesAndAnnotationsDataset) -> List[ImageAnnotation]:
        print(self.config)
        filtered_annotations: List[ImageAnnotation] = []
        image_number = 0
        for index, image_and_annotation in enumerate(dataset):
            annotation = ImageAnnotation()
            annotation.image_index = index
            for bounding_box_index, bounding_box in enumerate(
                image_and_annotation[1].bounding_boxes
            ):
                width = bounding_box.box.x2 - bounding_box.box.x1
                height = bounding_box.box.y2 - bounding_box.box.y1
                if width * height >= self.config.min_area and (
                    width >= self.config.min_width or height >= self.config.min_height
                ):
                    is_detected: bool = False
                    if (
                        self.config.classify_method
                        == ClipClassifyMethod.CLIP_WITH_LVIS_AND_CUSTOM_VOCABULARY
                    ):
                        is_detected = self.inference_with_lvis_vocabulary(bounding_box)
                    elif (
                        self.config.classify_method
                        == ClipClassifyMethod.CONSIDER_ANNOTATION_WITH_BERT
                    ):
                        is_detected = self.inference_with_bert(bounding_box)
                    else:
                        raise RuntimeError(
                            "Classify method "
                            + self.config.classify_method.value
                            + " does not support."
                        )
                    if is_detected:
                        # print("Image Number : " + str(image_number))
                        # print(
                        #     (
                        #         int(bounding_box.box.x1),
                        #         int(bounding_box.box.y1),
                        #         int(bounding_box.box.x2),
                        #         int(bounding_box.box.y2),
                        #     )
                        # )
                        # self.to_pil_image(copy.deepcopy(image_and_annotation[0])).crop(
                        #     (
                        #         int(bounding_box.box.x1),
                        #         int(bounding_box.box.y1),
                        #         int(bounding_box.box.x2),
                        #         int(bounding_box.box.y2),
                        #     )
                        # ).save("data/" + str(image_number) + ".jpeg")
                        # print("")
                        annotation.bounding_boxes.append(copy.deepcopy(bounding_box))
                        image_number = image_number + 1
        return filtered_annotations
