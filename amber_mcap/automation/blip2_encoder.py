import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from torchvision import transforms


# See also https://github.com/huggingface/transformers/blob/a91020aed0b15794d0842e5799ec9d360e939f4e/src/transformers/models/blip_2/modeling_blip_2.py#L2590C13-L2618
class Blip2Encoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

    def get_itm_score(self, image: torch.Tensor, text: str) -> float:
        image_fp32 = image.new_tensor(image, dtype=torch.float32, device=self.device)
        inputs = self.processor(images=image_fp32, text=text, return_tensors="pt").to(
            self.device, torch.float32
        )
        itm_out = self.model(**inputs, use_image_text_matching_head=True)
        return float(
            torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
            .softmax(dim=1)[0][1]
            .item()
        )

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt").to(
            self.device, torch.float32
        )

        output_attentions = False
        if "output_attentions" in inputs.keys():
            output_attentions = True
        else:
            output_attentions = self.model.config.output_attentions

        output_hidden_states = False
        if "output_hidden_states" in inputs.keys():
            output_hidden_states = True
        else:
            output_hidden_states = self.model.config.output_hidden_states

        return_dict = False
        if "return_dict" in inputs.keys():
            return_dict = True
        else:
            return_dict = self.model.config.output_hidden_states

        query_embeds = self.model.embeddings(
            input_ids=inputs["input_ids"],
        )
        text_outputs = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=inputs["attention_mask"],
            return_dict=return_dict,
        )
        question_embeds = (
            text_outputs[0] if not return_dict else text_outputs.last_hidden_state
        )
        return torch.nn.functional.normalize(
            self.model.text_projection(question_embeds[:, 0, :]).squeeze(0), dim=-1
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_fp32 = image.new_tensor(image, dtype=torch.float32, device=self.device)
        inputs = self.processor(images=image_fp32, return_tensors="pt").to(
            self.device, torch.float32
        )

        output_attentions = False
        if "output_attentions" in inputs.keys():
            output_attentions = True
        else:
            output_attentions = self.model.config.output_attentions

        output_hidden_states = False
        if "output_hidden_states" in inputs.keys():
            output_hidden_states = True
        else:
            output_hidden_states = self.model.config.output_hidden_states

        return_dict = False
        if "return_dict" in inputs.keys():
            return_dict = True
        else:
            return_dict = self.model.config.output_hidden_states

        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        image_embeds = (
            query_outputs[0] if not return_dict else query_outputs.last_hidden_state
        )
        return torch.nn.functional.normalize(
            self.model.vision_projection(image_embeds).squeeze(0), dim=-1
        )

    def encode_image_from_file(self, image_path: Path) -> torch.Tensor:
        return self.encode_image(transforms.ToTensor()(Image.open(image_path)))

    def get_cosine_similarity_from_image_and_text(
        self, image: torch.Tensor, text: str
    ) -> float:
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(text)
        cosine_similarity = torch.matmul(image_embeds, text_embeds.T)
        return float(cosine_similarity.max().item())

    def get_cosine_similarity_from_image_file_and_text(
        self, image_path: Path, text: str
    ) -> float:
        return self.get_cosine_similarity_from_image_and_text(
            transforms.ToTensor()(Image.open(image_path)), text
        )


if __name__ == "__main__":
    pass
