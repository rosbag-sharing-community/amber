import torch
from transformers import CLIPProcessor, CLIPModel
from amber_mcap.automation.annotation import ImageAnnotation
from typing import Tuple, List, Optional
from amber_mcap.util.lvis.lvis_v1_categories import (
    LVIS_CATEGORIES as LVIS_V1_CATEGORIES,
)
from PIL.Image import Image


class ClipEncoder:
    def __init__(self, model: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model, self.preprocess = load("ViT-B/32", device=self.device)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lvis_classes: List[str] = [k["synonyms"][0] for k in LVIS_V1_CATEGORIES]
        self.lvis_prompts: List[str] = []
        for lvis_class in self.lvis_classes:
            self.lvis_prompts.append("A photo of a " + lvis_class)
        self.lvis_text_embeddings: Optional[torch.Tensor] = None

    def get_single_image_embeddings(self, image: Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(
                text=None, images=[image], return_tensors="pt", padding=True
            )
            return self.model.get_image_features(pixel_values=inputs["pixel_values"])

    def get_single_image_embeddings_for_objects(
        self, image: Image, annotation: ImageAnnotation
    ) -> ImageAnnotation:
        with torch.no_grad():
            for bounding_box in annotation.bounding_boxes:
                bounding_box.clip_embeddings = self.get_single_image_embeddings(
                    image.crop(
                        (
                            int(bounding_box.box.x1),
                            int(bounding_box.box.y1),
                            int(bounding_box.box.x2),
                            int(bounding_box.box.y2),
                        )
                    )
                ).tolist()[0]
                assert len(bounding_box.clip_embeddings) == 512
        return annotation

    def get_single_text_embeddings(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(
                text=[text], images=None, return_tensors="pt", padding=True
            )
            return self.model.get_text_features(
                inputs["input_ids"], inputs["attention_mask"]
            )

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(
                text=texts, images=None, return_tensors="pt", padding=True
            )
            return self.model.get_text_features(
                inputs["input_ids"], inputs["attention_mask"]
            )

    def classify_with_custom_vocabulary(
        self, image_embeddings: torch.Tensor, texts: List[str]
    ) -> Optional[Tuple[str, float]]:
        if self.lvis_text_embeddings == None:
            with torch.no_grad():
                inputs = self.processor(
                    text=[self.lvis_prompts],
                    images=None,
                    return_tensors="pt",
                    padding=True,
                )
                self.lvis_text_embeddings = self.model.get_text_features(
                    inputs["input_ids"], inputs["attention_mask"]
                )
        prompts: List[str] = []
        for text in texts:
            prompts.append("A photo of a " + text)
        with torch.no_grad():
            text_embeddings = torch.cat(
                [self.lvis_text_embeddings, self.get_text_embeddings(prompts)],
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

    def get_single_text_embeddings_for_positive_negative_prompts(
        self, object_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get_single_text_embeddings("A photo of a " + object_name),
            self.get_single_text_embeddings("Not a photo of a " + object_name),
        )


if __name__ == "__main__":
    pass
