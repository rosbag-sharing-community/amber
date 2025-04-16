import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from typing import Optional, Tupl
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from torchvision import transforms


class Blip2Encoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

    def get_itm_score(self, image: torch.Tensor, text: str) -> float:
        image_fp16 = image.new_tensor(image, dtype=torch.float16, device=self.device)
        inputs = self.processor(images=image_fp16, text=text, return_tensors="pt").to(
            self.device, torch.float16
        )
        itm_out = self.model(**inputs, use_image_text_matching_head=True)
        return (
            torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
            .softmax(dim=1)[0][1]
            .item()
        )

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = processor(text=texts, return_tensors="pt").to(
            self.device, torch.float16
        )
        itc_out = self.model(**inputs, use_image_text_matching_head=False)
        if not itc_out.text_embeds:
            raise Exception("Failed to get text embeddings")
        return itc_out.text_embeds

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_fp16 = image.new_tensor(image, dtype=torch.float16, device=self.device)
        inputs = processor(images=image_fp16, return_tensors="pt").to(
            self.device, torch.float16
        )
        itc_out = self.model(**inputs, use_image_text_matching_head=False)
        if not itc_out.image_embeds:
            raise Exception("Failed to get image embeddings")
        return itc_out.image_embeds

    def encode_image_from_file(self, image_path: Path) -> torch.Tensor:
        return self.encode_image(transforms.ToTensor()(Image.open(image_path)))

    def get_cosine_similarity_from_image_and_text(
        self, image: torch.Tensor, text: str
    ) -> float:
        text_embedding = self.encode_text(text)
        image_embedding = self.encode_image(image)
        sim: float = 0.0
        sim, _ = torch.max(
            torch.bmm(image_embedding, text_embedding.unsqueeze(-1)), dim=1
        )
        return sim

    def get_cosine_similarity_from_image_file_and_text(
        self, image_path: Path, text: str
    ) -> float:
        return self.get_cosine_similarity_from_image_and_text(
            transforms.ToTensor()(Image.open(image_path)), text
        )


if __name__ == "__main__":
    pass
