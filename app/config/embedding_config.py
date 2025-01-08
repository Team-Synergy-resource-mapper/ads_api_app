from dataclasses import dataclass
import torch

@dataclass
class EmbeddingConfig:
    model_name: str = 'FacebookAI/xlm-roberta-base'
    pooling: str = 'mean'
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"