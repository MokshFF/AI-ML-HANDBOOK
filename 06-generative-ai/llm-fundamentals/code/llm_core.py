"""
LLM fundamentals: BPE tokenization, positional encoding, a tiny decoder-only
Transformer with a KV cache, decoding strategies, and scaling-law arithmetic.

Everything here is educational and CPU-friendly; no pretrained weights or APIs.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Byte-Pair Encoding (character level, with an end-of-word marker)
# --------------------------------------------------------------------------- #
EOW = "</w>"


def _word_symbols(word: str) -> Tuple[str, ...]:
    return tuple(word) + (EOW,)


def train_bpe(corpus: List[str], num_merges: int) -> List[Tuple[str, str]]:
    """Learn an ordered list of merges from whitespace-split text."""
    vocab: Counter = Counter()
    for line in corpus:
        for w in line.split():
            vocab[_word_symbols(w)] += 1

    merges: List[Tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        # deterministic tie-break: highest count, then lexicographic
        best = max(pairs.items(), key=lambda kv: (kv[1], -ord(kv[0][0][0]), kv[0]))[0]
        merges.append(best)
        new_vocab: Counter = Counter()
        for symbols, freq in vocab.items():
            new_vocab[_apply_merge(symbols, best)] += freq
        vocab = new_vocab
    return merges


def _apply_merge(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    out: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def bpe_encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Apply learned merges in the order they were learned."""
    symbols = _word_symbols(word)
    for pair in merges:
        symbols = _apply_merge(symbols, pair)
    return list(symbols)


def bpe_decode(tokens: List[str]) -> str:
    return "".join(tokens).replace(EOW, " ").strip()


# --------------------------------------------------------------------------- #
# 2. Sinusoidal positional encoding
# --------------------------------------------------------------------------- #
def sinusoidal_positions(seq_len: int, dim: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------- #
# 3. Tiny decoder-only Transformer with KV cache
# --------------------------------------------------------------------------- #
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class CachedAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.h, self.d).transpose(1, 2) for z in (q, k, v)]
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        total = k.size(2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        # query i (absolute position total-t+i) may attend to keys <= its position
        q_pos = torch.arange(total - t, total).unsqueeze(1)
        k_pos = torch.arange(total).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        y = F.softmax(scores, dim=-1) @ v
        y = y.transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.out(y), (k, v)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CachedAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, past=None):
        a, kv = self.attn(self.n1(x), past)
        x = x + a
        return x + self.mlp(self.n2(x)), kv


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 32, heads: int = 4, layers: int = 2, max_len: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.register_buffer("pe", sinusoidal_positions(max_len, dim), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, ids: torch.Tensor, cache: Optional[KVCache] = None):
        start = 0 if cache is None else cache[0][0].size(2)
        x = self.emb(ids) + self.pe[start:start + ids.size(1)]
        new_cache: KVCache = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.head(self.norm(x)), new_cache


# --------------------------------------------------------------------------- #
# 4. Decoding
# --------------------------------------------------------------------------- #
def filter_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Return a probability vector after temperature, top-k and nucleus filtering."""
    if temperature <= 0:
        raise ValueError("use argmax for temperature == 0")
    z = logits / temperature
    if top_k and top_k < z.numel():
        kth = torch.topk(z, top_k).values[-1]
        z = z.masked_fill(z < kth, float("-inf"))
    probs = F.softmax(z, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = (cum - sp) >= top_p  # keep the smallest set whose mass >= top_p
        sp = sp.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(0, si, sp)
        probs = probs / probs.sum()
    return probs


@torch.no_grad()
def generate(model: TinyGPT, prompt: List[int], max_new: int, use_cache: bool = True,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
             generator: Optional[torch.Generator] = None) -> List[int]:
    model.eval()
    ids = list(prompt)
    cache: Optional[KVCache] = None
    for step in range(max_new):
        if use_cache:
            inp = torch.tensor([ids if cache is None else ids[-1:]])
            logits, cache = model(inp, cache)
        else:
            logits, _ = model(torch.tensor([ids]))
        last = logits[0, -1]
        if temperature == 0:
            nxt = int(last.argmax())
        else:
            p = filter_logits(last, temperature, top_k, top_p)
            nxt = int(torch.multinomial(p, 1, generator=generator))
        ids.append(nxt)
    return ids


def kv_cache_matmul_savings(prompt_len: int, new_tokens: int) -> Tuple[int, int]:
    """Count token-positions pushed through the network with vs. without a cache."""
    without = sum(prompt_len + i for i in range(new_tokens))
    with_cache = prompt_len + (new_tokens - 1)
    return without, with_cache


# --------------------------------------------------------------------------- #
# 5. Context windows and scaling laws
# --------------------------------------------------------------------------- #
def fit_to_context(tokens: List[int], max_len: int, keep_prefix: int = 0) -> List[int]:
    """Keep a protected prefix (e.g. system prompt) plus the most recent tokens."""
    if len(tokens) <= max_len:
        return tokens
    if keep_prefix >= max_len:
        raise ValueError("keep_prefix must be smaller than max_len")
    return tokens[:keep_prefix] + tokens[-(max_len - keep_prefix):]


# Parametric loss L(N, D) = E + A / N^alpha + B / D^beta. Constants below are the
# published fit from Hoffmann et al. (2022) and are ILLUSTRATIVE: they depend on
# the data, tokenizer and architecture used in that study.
CHINCHILLA = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def parametric_loss(n_params: float, n_tokens: float, c: Dict[str, float] = CHINCHILLA) -> float:
    return c["E"] + c["A"] / n_params ** c["alpha"] + c["B"] / n_tokens ** c["beta"]


def compute_optimal(flops: float, c: Dict[str, float] = CHINCHILLA) -> Tuple[float, float]:
    """Minimise L(N, D) subject to C = 6 N D. Returns (N_opt, D_opt)."""
    a, b = c["alpha"], c["beta"]
    g = (a * c["A"] / (b * c["B"])) ** (1.0 / (a + b))
    n_opt = g * (flops / 6.0) ** (b / (a + b))
    return n_opt, flops / (6.0 * n_opt)
