from amber_mcap.automation.clip_encoder import ClipEncoder
from PIL import Image
from amber_mcap.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from typing import List
from amber_mcap.dataset.images_and_annotations_dataset import (
    ImagesAndAnnotationsDataset,
)
import torch
from torchvision import transforms
import tensorboardX

from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from enum import Enum
import random


class VisualizationTarget(Enum):
    IMAGE_EMBEDDINGS = "image_embeddings"


@dataclass
class ClipEmbeddingVisualizationConfig(YAMLWizard):  # type: ignore
    label_image_size: int = 50
    max_visualization_items: int = 1000
    target: VisualizationTarget = VisualizationTarget.IMAGE_EMBEDDINGS
    custom_vocabulary: List[str] = field(default_factory=list)


class ClipEmbeddingsVisualization:
    def __init__(self, yaml_path: str) -> None:
        self.config = ClipEmbeddingVisualizationConfig.from_yaml_file(yaml_path)
        self.encoder = ClipEncoder()
        self.image_embeddings = torch.zeros(0)
        self.text_embeddings = torch.zeros(0)
        self.label_images = torch.zeros(0)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.label_image_size, self.config.label_image_size)
                ),
                transforms.ToTensor(),
            ]
        )
        self.current_object_index: int = 0
        self.to_pil_image = transforms.ToPILImage()

    def add_image_embedding(
        self, image: torch.Tensor, annotation: ImageAnnotation
    ) -> None:
        with torch.no_grad():
            pil_image = self.to_pil_image(image)
            for bounding_box in annotation.bounding_boxes:
                if self.current_object_index in self.target_object_index:
                    cropped_image = pil_image.crop(
                        (
                            int(bounding_box.box.x1),
                            int(bounding_box.box.y1),
                            int(bounding_box.box.x2),
                            int(bounding_box.box.y2),
                        )
                    )
                    self.image_embeddings = torch.cat(
                        (
                            self.image_embeddings,
                            self.encoder.get_single_image_embeddings(cropped_image),
                        )
                    )
                    self.label_images = torch.cat(
                        (self.label_images, self.transform(cropped_image))
                    )
                self.current_object_index = self.current_object_index + 1

    def add_text_embedding(self, object_name: str) -> None:
        self.text_embeddings = torch.cat(
            (
                self.text_embeddings,
                self.encoder.get_single_text_embeddings("A photo of a " + object_name),
            )
        )

    def visualize(self, dataset: ImagesAndAnnotationsDataset) -> None:
        self.num_objects = 0
        for index, image_and_annotation in enumerate(dataset):
            for bounding_box in image_and_annotation[1].bounding_boxes:
                self.num_objects = self.num_objects + 1
        if self.num_objects >= self.config.max_visualization_items:
            self.target_object_index = random.sample(
                list(range(self.num_objects)), self.config.max_visualization_items
            )
        else:
            self.target_object_index = list(range(self.num_objects))
        self.current_object_index = 0
        for index, image_and_annotation in enumerate(dataset):
            self.add_image_embedding(image_and_annotation[0], image_and_annotation[1])
        for object_name in self.config.custom_vocabulary:
            self.add_text_embedding(object_name)
        # Append Black images for text embeddings
        self.label_images = torch.cat(
            (
                self.label_images,
                torch.zeros(
                    (
                        3 * len(self.config.custom_vocabulary),
                        self.config.label_image_size,
                        self.config.label_image_size,
                    )
                ),
            )
        )
        image_metadata = [
            "image_in_rosbag" for i in range(len(self.target_object_index))
        ]
        embeddings = torch.cat((self.image_embeddings, self.text_embeddings))
        writer = tensorboardX.SummaryWriter()
        writer.add_embedding(
            embeddings.view(
                len(self.target_object_index) + len(self.config.custom_vocabulary), 512
            ),
            label_img=self.label_images.view(
                len(self.target_object_index) + len(self.config.custom_vocabulary),
                3,
                self.config.label_image_size,
                self.config.label_image_size,
            ),
            metadata=image_metadata + self.config.custom_vocabulary,
        )
        writer.close()
