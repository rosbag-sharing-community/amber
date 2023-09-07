import torch
from clip.clip import load, tokenize
from torchvision import transforms


class ClipEncoder:
    def __init__(self, model: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load("ViT-B/32", device=self.device)
        self.to_pil_image = transforms.ToPILImage()

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_image(
                self.preprocess(self.to_pil_image(image)).unsqueeze(0).to(self.device)
            )

    def get_text_features(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(tokenize([text]).to(self.device))
