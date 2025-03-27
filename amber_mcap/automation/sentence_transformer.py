from sentence_transformers import util, SentenceTransformer
from typing import Dict
import torch


class TextEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.cache: Dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor:
        if text in self.cache:
            return self.cache[text]
        self.cache[text] = self.model.encode(text)
        return self.cache[text]

    def cosine_similarity(self, text0: str, text1: str) -> float:
        return float(
            util.pytorch_cos_sim(self.encode(text0), self.encode(text1)).item()
        )
