import torch
from clip.clip import load, tokenize
from torchvision import transforms
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from typing import Tuple


class ClipEncoder:
    def __init__(self, model: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load("ViT-B/32", device=self.device)
        self.to_pil_image = transforms.ToPILImage()

    def get_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_image(
                self.preprocess(self.to_pil_image(image)).unsqueeze(0).to(self.device)
            )

    def get_image_embeddings_for_objects(
        self, image: torch.Tensor, annotation: ImageAnnotation
    ) -> ImageAnnotation:
        with torch.no_grad():
            pil_image = self.to_pil_image(image)
            for bounding_box in annotation.bounding_boxes:
                bounding_box.clip_embeddings = self.model.encode_image(
                    self.preprocess(
                        pil_image.crop(
                            (
                                int(bounding_box.box.x1),
                                int(bounding_box.box.y1),
                                int(bounding_box.box.x2),
                                int(bounding_box.box.y2),
                            )
                        )
                    )
                    .unsqueeze(0)
                    .to(self.device)
                ).tolist()[0]
                assert len(bounding_box.clip_embeddings) == 512
        return annotation

    def get_text_embeddings(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(tokenize([text]).to(self.device))

    def get_text_embeddings_for_positive_negative_prompts(
        self, object_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get_text_embeddings("A photo of a " + object_name),
            self.get_text_embeddings("Not a photo of a " + object_name),
        )
