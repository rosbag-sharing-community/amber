import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.tokenization_utils_base import BatchEncoding

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval


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
        itm_out = model(**inputs, use_image_text_matching_head=True)
        return (
            torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
            .softmax(dim=1)[0][1]
            .item()
        )

    # def preprocess_image_embeddings(
    #     self, image: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     image = self.image_processor(image)
    #     assert image.dim() == 3
    #     image = image.unsqueeze(0).to(self.device)
    #     with self.model.maybe_autocast():
    #         image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
    #     image_embeds = image_embeds.float()
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
    #         image.device
    #     )
    #     return (image_embeds, image_atts)

    # https://github.com/salesforce/LAVIS/blob/506965b9c4a18c1e565bd32acaccabe0198433f7/lavis/models/blip2_models/blip2_image_text_matching.py#L71-L89
    # def get_itm_score(self, image: torch.Tensor, text: str) -> float:
    #     image_embeds, image_atts = self.preprocess_image_embeddings(image)
    #     text_tensor: torch.Tensor = self.get_text_tensor(text)
    #     query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
    #         text_tensor["input_ids"].device
    #     )
    #     attention_mask = torch.cat([query_atts, text_tensor.attention_mask], dim=1)
    #     output_itm = self.model.Qformer.bert(
    #         text_tensor.input_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=attention_mask,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )
    #     itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    #     itm_logit = self.model.itm_head(itm_embeddings)
    #     itm_logit = itm_logit.mean(dim=1)
    #     return float(itm_logit[:, 1].item())

    # def get_text_tensor(self, text: str) -> BatchEncoding:
    #     text_tensor: BatchEncoding = self.model.tokenizer(
    #         text,
    #         truncation=True,
    #         max_length=self.model.max_txt_len,
    #         return_tensors="pt",
    #     ).to(self.device)
    #     return text_tensor

    # def encode_image(self, image: torch.Tensor) -> torch.Tensor:
    #     image_embeds, image_atts = self.preprocess_image_embeddings(image)
    #     query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_output = self.model.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )
    #     return F.normalize(
    #         self.model.vision_proj(query_output.last_hidden_state), dim=-1
    #     )

    # def encode_text(self, text: str) -> torch.Tensor:
    #     text_tensor: torch.Tensor = self.get_text_tensor(text)
    #     text_output = self.model.Qformer.bert(
    #         text_tensor.input_ids,
    #         attention_mask=text_tensor.attention_mask,
    #         return_dict=True,
    #     )
    #     return F.normalize(
    #         self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    #     )

    # def encode_image_from_file(self, image_path: Path) -> torch.Tensor:
    #     return self.encode_image(transforms.ToTensor()(Image.open(image_path)))

    # def get_cosine_similarity_from_image_and_text(
    #     self, image: torch.Tensor, text: str
    # ) -> float:
    #     text_embedding = self.encode_text(text)
    #     image_embedding = self.encode_image(image)
    #     sim: float = 0.0
    #     sim, _ = torch.max(
    #         torch.bmm(image_embedding, text_embedding.unsqueeze(-1)), dim=1
    #     )
    #     return sim

    # def get_cosine_similarity_from_image_file_and_text(
    #     self, image_path: Path, text: str
    # ) -> float:
    #     return self.get_cosine_similarity_from_image_and_text(
    #         self.image_processor(transforms.ToTensor()(Image.open(image_path))), text
    #     )


if __name__ == "__main__":
    pass
