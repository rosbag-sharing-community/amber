import torch
from clip.clip import load, tokenize
from torchvision import transforms
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation
from typing import Tuple, List, Optional
from amber.util.lvis.lvis_v1_categories import (
    LVIS_CATEGORIES as LVIS_V1_CATEGORIES,
)
import tensorboardX


class ClipEncoder:
    def __init__(self, model: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load("ViT-B/32", device=self.device)
        self.lvis_classes: List[str] = [k["synonyms"][0] for k in LVIS_V1_CATEGORIES]
        self.lvis_prompts: List[str] = []
        for lvis_class in self.lvis_classes:
            self.lvis_prompts.append("A photo of a " + lvis_class)
        self.lvis_text_embeddings: Optional[torch.Tensor] = None
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

    def classify_with_custom_vocabulary(
        self, image_embeddings: torch.Tensor, texts: List[str]
    ) -> Optional[Tuple[str, float]]:
        if self.lvis_text_embeddings == None:
            with torch.no_grad():
                self.lvis_text_embeddings = self.model.encode_text(
                    tokenize(self.lvis_prompts).to(self.device)
                )
        prompts: List[str] = []
        for text in texts:
            prompts.append("A photo of a " + text)
        with torch.no_grad():
            text_embeddings = torch.cat(
                [
                    self.lvis_text_embeddings,
                    self.model.encode_text(tokenize(prompts).to(self.device)),
                ],
                dim=0,
            )
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            similarity = (
                image_embeddings.to(torch.float32) @ text_embeddings.to(torch.float32).T
            ).softmax(dim=-1)
            values, indices = similarity.topk(1)
            for value, index in zip(values, indices):
                if index < len(self.lvis_classes):
                    return None
                else:
                    return (texts[index - len(self.lvis_classes)], value.item())
        return None

    def get_text_embeddings_for_positive_negative_prompts(
        self, object_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get_text_embeddings("A photo of a " + object_name),
            self.get_text_embeddings("Not a photo of a " + object_name),
        )


if __name__ == "__main__":
    pass
