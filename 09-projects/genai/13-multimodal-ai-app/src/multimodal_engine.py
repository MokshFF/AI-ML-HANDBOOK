"""Vision-Language Dual Encoder Multimodal Engine."""
import torch
import torch.nn as nn
from typing import List, Tuple

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 4, embed_dim)
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        out = self.conv(imgs)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Mean pooling across tokens
        emb = self.embed(token_ids).mean(dim=1)
        out = self.proj(emb)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

class MultimodalEngine:
    def __init__(self):
        self.v_enc = VisionEncoder()
        self.t_enc = TextEncoder()
        self.v_enc.eval()
        self.t_enc.eval()

    def compute_similarity(self, imgs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_feats = self.v_enc(imgs)
            text_feats = self.t_enc(tokens)
            sim = img_feats @ text_feats.T
        return sim

if __name__ == "__main__":
    engine = MultimodalEngine()
    dummy_imgs = torch.randn(3, 3, 32, 32)
    dummy_tokens = torch.randint(0, 100, (3, 5))
    sim_matrix = engine.compute_similarity(dummy_imgs, dummy_tokens)
    print("Cross-Modal Similarity Matrix (3x3):\n", sim_matrix)
