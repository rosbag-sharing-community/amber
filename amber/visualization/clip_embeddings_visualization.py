from amber.automation.clip_encoder import ClipEncoder
from PIL import Image
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from typing import List
from amber.dataset.images_and_annotations_dataset import ImagesAndAnnotationsDataset
import torch
from torchvision import transforms
import tensorboardX

from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from enum import Enum


class VisualizationTarget(Enum):
    IMAGE_EMBEDDINGS = "image_embeddings"


@dataclass
class ClipEmbeddingVisualizationConfig(YAMLWizard):  # type: ignore
    label_image_size: int = 50
    max_visualization_items: int = 1000
    target: VisualizationTarget = VisualizationTarget.IMAGE_EMBEDDINGS


class ClipEmbeddingsVisualization:
    def __init__(self, config: ClipEmbeddingVisualizationConfig) -> None:
        self.config = config
        self.encoder = ClipEncoder()
        self.image_embeddings = torch.zeros(0)
        self.label_images = torch.zeros(0)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.label_image_size, self.config.label_image_size)
                ),
                transforms.ToTensor(),
            ]
        )
        self.num_objects = 0
        self.to_pil_image = transforms.ToPILImage()

    def add_image_embedding(
        self, image: torch.Tensor, annotation: ImageAnnotation
    ) -> None:
        with torch.no_grad():
            pil_image = self.to_pil_image(image)
            for bounding_box in annotation.bounding_boxes:
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
                        self.encoder.model.encode_image(
                            self.encoder.preprocess(cropped_image)
                            .unsqueeze(0)
                            .to(self.encoder.device)
                        ),
                    )
                )
                self.label_images = torch.cat(
                    (self.label_images, self.transform(cropped_image))
                )
                self.num_objects = self.num_objects + 1

    def visualize(self, dataset: ImagesAndAnnotationsDataset) -> None:
        for index, image_and_annotation in enumerate(dataset):
            self.add_image_embedding(image_and_annotation[0], image_and_annotation[1])
        writer = tensorboardX.SummaryWriter()
        writer.add_embedding(
            self.image_embeddings.view(self.num_objects, 512),
            label_img=self.label_images.view(
                self.num_objects,
                3,
                self.config.label_image_size,
                self.config.label_image_size,
            ),
        )
        writer.close()
