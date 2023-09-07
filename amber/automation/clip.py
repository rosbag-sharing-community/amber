import torch
from clip.clip import load, tokenize
from torchvision import transforms
from amber.automation.annotation import ImageAnnotation, BoundingBoxAnnotation


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
                            int(bounding_box.x1),
                            int(bounding_box.y1),
                            int(bounding_box.x2),
                            int(bounding_box.y2),
                        )
                    )
                    .unsqueeze(0)
                    .to(self.device)
                ).tolist()[0]
        return annotation

    def get_text_embeddings(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(tokenize([text]).to(self.device))
