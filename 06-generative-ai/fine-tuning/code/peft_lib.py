"""
Parameter-efficient fine-tuning building blocks in plain PyTorch:
LoRA, bottleneck adapters, a QLoRA-style quantized base layer, SFT label masking,
and the DPO / reward-model / RLHF objective formulas.

These are *teaching implementations* (small, readable, tested). For real training use
maintained libraries (e.g. Hugging Face PEFT / TRL) and read their docs.
"""

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# LoRA: W' = W + (alpha / r) * B @ A, with B initialised to zero
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain Linear (zero inference overhead)."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def apply_lora(model: nn.Module, targets: Iterable[str], r: int = 4, alpha: float = 8.0) -> int:
    """Freeze everything, then wrap Linear layers whose attribute name is in `targets`.

    Returns the number of wrapped layers.
    """
    targets = set(targets)
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, r, alpha))
                n += 1
    return n


def merge_lora(model: nn.Module) -> nn.Module:
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return model


def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Bottleneck adapter (Houlsby-style): x + up(act(down(x))), up zero-initialised
# --------------------------------------------------------------------------- #
class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 8):
        super().__init__()
        self.down, self.up = nn.Linear(dim, bottleneck), nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


# --------------------------------------------------------------------------- #
# QLoRA-style: 4-bit blockwise-quantised frozen base weight + trainable LoRA
# --------------------------------------------------------------------------- #
def nf_style_levels(bits: int = 4) -> torch.Tensor:
    """16 levels at evenly spaced quantiles of N(0,1), scaled to [-1, 1].

    Inspired by NormalFloat (Dettmers et al., 2023) but NOT bit-identical to NF4.
    """
    k = 2 ** bits
    probs = (torch.arange(k, dtype=torch.float64) + 0.5) / k
    lv = torch.tensor(norm.ppf(probs.numpy()), dtype=torch.float32)
    return lv / lv.abs().max()


def uniform_levels(bits: int = 4) -> torch.Tensor:
    k = 2 ** bits
    return torch.linspace(-1, 1, k)


def blockwise_quantize(w: torch.Tensor, levels: torch.Tensor, block: int = 64):
    """Absmax-scale each block to [-1, 1], then snap to the nearest level. Returns (codes, scales, shape)."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normed = blocks / scales
    codes = (normed.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(dim=-1).to(torch.uint8)
    return codes, scales.squeeze(1), w.shape


def blockwise_dequantize(codes, scales, shape, levels: torch.Tensor) -> torch.Tensor:
    vals = levels[codes.long()] * scales.unsqueeze(1)
    n = 1
    for s in shape:
        n *= s
    return vals.reshape(-1)[:n].view(shape)


class QLoRALinear(nn.Module):
    """Frozen 4-bit base weight (dequantised on the fly) + trainable LoRA adapters in full precision."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, block: int = 64, nf: bool = True):
        super().__init__()
        self.levels = nf_style_levels() if nf else uniform_levels()
        codes, scales, shape = blockwise_quantize(base.weight.data, self.levels, block)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.shape = shape
        self.register_buffer("bias", None if base.bias is None else base.bias.data.clone())
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = blockwise_dequantize(self.codes, self.scales, self.shape, self.levels)
        return F.linear(x, w, self.bias) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def stored_bytes(self) -> Tuple[float, int]:
        """(approx. bytes for 4-bit codes + fp16 scales, bytes of the original fp16 weight)."""
        n = self.codes.numel()
        return n * 0.5 + self.scales.numel() * 2, int(torch.Size(self.shape).numel()) * 2


# --------------------------------------------------------------------------- #
# Supervised fine-tuning (SFT): train on response tokens only
# --------------------------------------------------------------------------- #
IGNORE = -100


def build_sft_example(prompt_ids: Sequence[int], response_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    ids = list(prompt_ids) + list(response_ids)
    labels = [IGNORE] * len(prompt_ids) + list(response_ids)
    return ids, labels


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy; positions with label -100 are ignored."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=IGNORE)


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log p(label_t | prefix) over non-ignored positions, per sequence."""
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != IGNORE
    picked = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Preference optimisation and RLHF objectives
# --------------------------------------------------------------------------- #
def dpo_loss(pol_chosen: torch.Tensor, pol_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """-log sigma( beta * [ (pi_c - pi_r) - (ref_c - ref_r) ] ), inputs are sequence log-probs."""
    margin = (pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def reward_model_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise loss used to train RLHF reward models."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def kl_shaped_reward(reward: torch.Tensor, logp_policy: torch.Tensor, logp_ref: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """RLHF reward with a per-sample KL penalty to the reference model: r - beta * (log pi - log pi_ref)."""
    return reward - beta * (logp_policy - logp_ref)


def ppo_clip_objective(logp_new: torch.Tensor, logp_old: torch.Tensor, advantage: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """PPO clipped surrogate (to be maximised)."""
    ratio = torch.exp(logp_new - logp_old)
    return torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage).mean()
