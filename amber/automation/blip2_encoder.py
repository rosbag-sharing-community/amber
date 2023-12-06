from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM
from lavis.processors import load_processor
import torch
from pathlib import Path
from torchvision.io import read_image
from lavis.processors.blip_processors import BlipImageBaseProcessor
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from typing import Optional


class Blip2ImageProcessor(BlipImageBaseProcessor):  # type: ignore
    def __init__(
        self,
        image_size: int = 384,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> None:
        super().__init__(mean=mean, std=std)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                self.normalize,
            ]
        )

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        return self.transform(item)


class Blip2Encoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, _, self.text_processors = load_model_and_preprocess(
            "blip2_image_text_matching", "pretrain", device=self.device, is_eval=True
        )
        self.image_processor = Blip2ImageProcessor(image_size=224)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.image_processor(image)
        assert image.dim() == 3
        image = image.unsqueeze(0).to(self.device)
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return F.normalize(
            self.model.vision_proj(query_output.last_hidden_state), dim=-1
        )

    def encode_text(self, text: str) -> torch.Tensor:
        text_tensor: torch.Tensor = self.model.tokenizer(
            text,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text_output = self.model.Qformer.bert(
            text_tensor.input_ids,
            attention_mask=text_tensor.attention_mask,
            return_dict=True,
        )
        return F.normalize(
            self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

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
            self.image_processor(transforms.ToTensor()(Image.open(image_path))), text
        )


if __name__ == "__main__":
    pass
