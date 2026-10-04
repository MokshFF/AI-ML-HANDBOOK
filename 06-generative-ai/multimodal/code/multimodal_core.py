"""
Multimodal Generative AI Core:
1. Contrastive Vision-Language Pretraining (CLIP-style dual encoder and InfoNCE loss).
2. Vision-Language Model (VLM) projection adapter (LLaVA-style MLP projector + prefix injection).
3. Classifier-Free Guidance (CFG) and diffusion noise scheduling mechanics.
4. Audio-Language frame projector.
5. Multimodal RAG indexing and hybrid retrieval.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Dual-Encoder Contrastive Learning (CLIP style)
# --------------------------------------------------------------------------- #
class ToyPatchVisionEncoder(nn.Module):
    """Educational vision encoder: splits image (C, H, W) into patches and maps to d_model."""
    def __init__(self, in_channels: int = 3, patch_size: int = 4, d_model: int = 32):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        # Unfold into patches: (B, C, H//p, p, W//p, p) -> (B, num_patches, patch_dim)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C * p * p)
        tokens = self.proj(patches)
        # Global pooled vector (mean across patches)
        pooled = tokens.mean(dim=1)
        return pooled, tokens


class ToyTextEncoder(nn.Module):
    """Educational text encoder: token embedding + mean pooling."""
    def __init__(self, vocab_size: int = 1000, d_model: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, seq_len)
        h = self.emb(token_ids)
        h = self.fc(h)
        return h.mean(dim=1)


class CLIPModel(nn.Module):
    """Dual encoder with normalized projections and learnable logit scale."""
    def __init__(self, d_vision: int = 32, d_text: int = 32, embed_dim: int = 16):
        super().__init__()
        self.visual = ToyPatchVisionEncoder(d_model=d_vision)
        self.textual = ToyTextEncoder(d_model=d_text)
        self.v_proj = nn.Linear(d_vision, embed_dim, bias=False)
        self.t_proj = nn.Linear(d_text, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.visual(images)
        z_v = self.v_proj(pooled)
        return F.normalize(z_v, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.textual(text_ids)
        z_t = self.t_proj(pooled)
        return F.normalize(z_t, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_v = self.encode_image(images)
        z_t = self.encode_text(text_ids)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        # Cosine similarity matrix: (B_v, B_t)
        logits_per_image = logit_scale * (z_v @ z_t.T)
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text


def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE cross-entropy loss along image and text axes."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


def zero_shot_classifier(
    clip_model: CLIPModel,
    images: torch.Tensor,
    candidate_tokens: torch.Tensor
) -> torch.Tensor:
    """Computes zero-shot class probabilities over candidate prompt token sequences."""
    clip_model.eval()
    with torch.no_grad():
        z_v = clip_model.encode_image(images)
        z_t = clip_model.encode_text(candidate_tokens)
        sims = z_v @ z_t.T
        probs = F.softmax(sims, dim=-1)
    return probs


# --------------------------------------------------------------------------- #
# 2. Vision-Language Model (VLM) Projector (LLaVA style)
# --------------------------------------------------------------------------- #
class VLMProjector(nn.Module):
    """Two-layer MLP projecting visual patch tokens into LLM embedding space."""
    def __init__(self, d_vision: int = 32, d_llm: int = 48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_vision, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, num_patches, d_vision) -> (B, num_patches, d_llm)
        return self.mlp(patch_tokens)


def build_multimodal_input(
    projected_patches: torch.Tensor,
    text_embeddings: torch.Tensor
) -> torch.Tensor:
    """Prefix injection: prepends projected visual tokens to text token embeddings."""
    # projected_patches: (B, N_v, D), text_embeddings: (B, N_t, D)
    return torch.cat([projected_patches, text_embeddings], dim=1)


# --------------------------------------------------------------------------- #
# 3. Audio-Language Projection
# --------------------------------------------------------------------------- #
class AudioFrameProjector(nn.Module):
    """Projects Mel-spectrogram temporal frames into LLM token space."""
    def __init__(self, n_mels: int = 80, d_llm: int = 48, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(n_mels, d_llm, kernel_size=3, stride=stride, padding=1)
        self.proj = nn.Linear(d_llm, d_llm)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        # mel_spec: (B, n_mels, time_steps)
        h = F.gelu(self.conv(mel_spec))
        h = h.permute(0, 2, 1)  # (B, downsampled_time, d_llm)
        return self.proj(h)


# --------------------------------------------------------------------------- #
# 4. Diffusion & Classifier-Free Guidance (CFG) Concepts
# --------------------------------------------------------------------------- #
class DiffusionNoiseSchedule:
    """Linear beta noise schedule for discrete-time diffusion processes."""
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion step: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t]).view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, *([1] * (x_start.ndim - 1)))
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise


def classifier_free_guidance(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, guidance_scale: float = 7.5) -> torch.Tensor:
    """
    Applies Classifier-Free Guidance (CFG):
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    """
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


# --------------------------------------------------------------------------- #
# 5. Multimodal RAG Indexing & Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class MultimodalItem:
    id: str
    modality: str  # "text" | "image"
    content: str   # text string or image file/caption
    embedding: torch.Tensor


class MultimodalStore:
    """In-memory joint vector store for text passages and image embeddings."""
    def __init__(self):
        self.items: List[MultimodalItem] = []

    def add(self, item_id: str, modality: str, content: str, embedding: torch.Tensor) -> None:
        norm_emb = F.normalize(embedding.view(1, -1), dim=-1).squeeze(0)
        self.items.append(MultimodalItem(id=item_id, modality=modality, content=content, embedding=norm_emb))

    def search(self, query_embedding: torch.Tensor, top_k: int = 3, filter_modality: Optional[str] = None) -> List[Tuple[MultimodalItem, float]]:
        if not self.items:
            return []
        q_norm = F.normalize(query_embedding.view(1, -1), dim=-1).squeeze(0)
        candidates = self.items if filter_modality is None else [it for it in self.items if it.modality == filter_modality]
        if not candidates:
            return []
        scored = []
        for it in candidates:
            sim = float(torch.dot(q_norm, it.embedding).item())
            scored.append((it, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
