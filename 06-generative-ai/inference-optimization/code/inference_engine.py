"""
LLM Inference Optimization Engine:
1. KV cache sizing & memory bandwidth roofline modeling (MHA, GQA, MQA).
2. PagedAttention & Continuous Batching virtual block allocator.
3. Speculative decoding simulation with rejection sampling acceptance.
4. Uniform & blockwise INT8/INT4 weight quantization mechanics (AWQ/GPTQ concepts).
5. Latency metrics: TTFT (Time to First Token), ITL (Inter-Token Latency), throughput.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. KV Cache Sizing & Memory Bandwidth Roofline
# --------------------------------------------------------------------------- #
def kv_cache_size_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    precision_bytes: int = 2  # 2 for FP16 / BF16
) -> int:
    """
    Computes total bytes required for Key-Value cache:
    Total = 2 (Key + Value) * batch_size * seq_len * num_layers * num_kv_heads * head_dim * precision_bytes
    """
    return 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * precision_bytes


def memory_bandwidth_throughput_roofline(
    model_param_count: int,
    memory_bandwidth_gb_per_sec: float,
    precision_bytes: int = 2
) -> float:
    """
    Token generation during decoding is memory-bandwidth bound at batch_size=1:
    Every single token generated requires streaming all model weights through memory.
    Max tokens/sec = (Memory Bandwidth in bytes/sec) / (Model parameter bytes).
    """
    model_bytes = model_param_count * precision_bytes
    bandwidth_bytes_per_sec = memory_bandwidth_gb_per_sec * (1024 ** 3)
    return bandwidth_bytes_per_sec / model_bytes


@dataclass
class GenerationProfile:
    ttft_ms: float               # Time To First Token (Prefill latency)
    mean_itl_ms: float           # Mean Inter-Token Latency (Time per decode step)
    total_tokens_generated: int
    total_time_ms: float
    tokens_per_second: float


# --------------------------------------------------------------------------- #
# 2. PagedAttention & Continuous Virtual Memory Block Allocator
# --------------------------------------------------------------------------- #
@dataclass
class PhysicalBlock:
    block_id: int
    data: Optional[torch.Tensor] = None
    ref_count: int = 0


class PagedBlockAllocator:
    """
    Simulates vLLM PagedAttention virtual memory block manager:
    Splits continuous KV cache into fixed-size physical blocks (e.g., 16 tokens/block).
    Eliminates internal & external memory fragmentation.
    """
    def __init__(self, num_blocks: int, block_size: int = 16, d_kv: int = 64):
        self.block_size = block_size
        self.d_kv = d_kv
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_table: Dict[str, List[int]] = {}  # req_id -> list of physical block ids

    def allocate(self, req_id: str, num_tokens: int) -> List[int]:
        needed_blocks = math.ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < needed_blocks:
            raise MemoryError("Out of physical KV blocks (GPU memory full)")
        allocated = [self.free_blocks.pop() for _ in range(needed_blocks)]
        self.block_table[req_id] = allocated
        return allocated

    def append_token(self, req_id: str, current_token_count: int) -> int:
        """Appends one token to request; allocates new block if boundary crossed."""
        if current_token_count % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("Out of memory during decoding")
            new_block = self.free_blocks.pop()
            self.block_table[req_id].append(new_block)
            return new_block
        return self.block_table[req_id][-1]

    def free(self, req_id: str) -> None:
        if req_id in self.block_table:
            for b in self.block_table[req_id]:
                self.free_blocks.append(b)
            del self.block_table[req_id]

    def memory_utilization(self, total_blocks: int) -> float:
        used = total_blocks - len(self.free_blocks)
        return used / total_blocks


# --------------------------------------------------------------------------- #
# 3. Speculative Decoding Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SpeculativeResult:
    accepted_tokens: List[int]
    acceptance_rate: float
    target_calls: int
    draft_calls: int
    effective_speedup: float


def speculative_decoding_step(
    draft_logits_fn: Callable[[List[int]], torch.Tensor],
    target_logits_fn: Callable[[List[int]], torch.Tensor],
    prefix: List[int],
    gamma: int = 4
) -> Tuple[List[int], int]:
    """
    Standard speculative decoding step (Leviathan et al., Chen et al.):
    1. Draft model proposes gamma tokens autoregressively.
    2. Target model evaluates all proposed tokens in a single parallel verification pass.
    3. Accept draft token x with probability min(1, p_target(x) / p_draft(x)).
    4. If rejected, sample from residual distribution max(0, p_target - p_draft).
    """
    draft_tokens: List[int] = []
    draft_probs_list: List[torch.Tensor] = []

    # 1. Draft model proposes gamma tokens
    curr_context = list(prefix)
    for _ in range(gamma):
        logits = draft_logits_fn(curr_context)
        probs = F.softmax(logits, dim=-1)
        next_tok = int(torch.argmax(probs).item())  # greedy for deterministic teaching
        draft_tokens.append(next_tok)
        draft_probs_list.append(probs)
        curr_context.append(next_tok)

    # 2. Target model evaluates parallel verification
    accepted: List[int] = []
    verify_context = list(prefix)

    for i in range(gamma):
        t_logits = target_logits_fn(verify_context)
        t_probs = F.softmax(t_logits, dim=-1)
        tok = draft_tokens[i]

        p_t = t_probs[tok].item()
        p_d = draft_probs_list[i][tok].item()

        # Acceptance probability
        r = p_t / max(1e-9, p_d)
        if r >= 1.0 or (p_t > 0.5):  # acceptance condition
            accepted.append(tok)
            verify_context.append(tok)
        else:
            # Rejection: sample recovery token from target distribution
            recovery_tok = int(torch.argmax(t_probs).item())
            accepted.append(recovery_tok)
            break
    else:
        # If all gamma accepted, bonus target token
        bonus_logits = target_logits_fn(verify_context)
        bonus_tok = int(torch.argmax(F.softmax(bonus_logits, dim=-1)).item())
        accepted.append(bonus_tok)

    return accepted, len(accepted)


# --------------------------------------------------------------------------- #
# 4. INT8 & INT4 Quantization Mechanics
# --------------------------------------------------------------------------- #
def quantize_int8_symmetric(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-tensor / per-channel INT8 quantization:
    scale = max(|weights|) / 127
    q_weights = clamp(round(weights / scale), -128, 127)
    """
    abs_max = weights.abs().max()
    scale = abs_max / 127.0 if abs_max > 0 else torch.tensor(1.0)
    q = torch.clamp(torch.round(weights / scale), -128, 127).to(torch.int8)
    return q, scale


def dequantize_int8_symmetric(q_weights: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q_weights.to(torch.float32) * scale


def quantize_int4_blockwise(weights: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Blockwise (grouped) symmetric INT4 quantization (AWQ/GPTQ style):
    Each group of weights gets an individual scale factor.
    """
    orig_shape = weights.shape
    w_flat = weights.flatten()
    pad_len = (group_size - (w_flat.numel() % group_size)) % group_size
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len))
    groups = w_flat.view(-1, group_size)
    scales = groups.abs().max(dim=-1, keepdim=True).values / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q_groups = torch.clamp(torch.round(groups / scales), -8, 7).to(torch.int8)
    return q_groups, scales


def dequantize_int4_blockwise(
    q_groups: torch.Tensor,
    scales: torch.Tensor,
    orig_numel: int,
    orig_shape: torch.Size
) -> torch.Tensor:
    w_deq = (q_groups.to(torch.float32) * scales).view(-1)[:orig_numel]
    return w_deq.view(orig_shape)
