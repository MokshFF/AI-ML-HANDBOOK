"""
Autoregressive Language Modeling Engine: N-Gram Smoothing, Causal Transformers, and Decoding Strategies.

Implements:
- NgramLanguageModel: Statistical n-gram LM with add-alpha Laplace smoothing and perplexity.
- MiniCausalLM: PyTorch autoregressive transformer with lower-triangular causal attention masking.
- compute_perplexity: Exponential cross-entropy loss converter.
- sample_next_token: Generation sampler supporting greedy, temperature, top-k, and nucleus (top-p) sampling.
- generate_sequence: Autoregressive decoding loop.
"""

from collections import Counter, defaultdict
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NgramLanguageModel:
    """
    Statistical n-gram language model with add-alpha (Laplace / Lidstone) smoothing.
    """
    def __init__(self, n: int = 2, alpha: float = 1.0):
        self.n = n
        self.alpha = alpha
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def fit(self, tokenized_corpus: List[List[str]]):
        for sentence in tokenized_corpus:
            padded = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for token in padded:
                self.vocab.add(token)

            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def score(self, word: str, context: Tuple[str, ...]) -> float:
        context_tuple = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        ngram = context_tuple + (word,)
        count_ngram = self.ngram_counts[ngram]
        count_ctx = self.context_counts[context_tuple]
        v_size = max(1, len(self.vocab))

        # Add-alpha smoothing
        prob = (count_ngram + self.alpha) / (count_ctx + self.alpha * v_size)
        return prob

    def compute_perplexity(self, sentence: List[str]) -> float:
        padded = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        log_prob_sum = 0.0
        m = len(sentence) + 1  # tokens evaluated including </s>

        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i]) if self.n > 1 else ()
            word = padded[i]
            p = self.score(word, context)
            log_prob_sum += math.log(max(p, 1e-12))

        cross_entropy = -log_prob_sum / m
        return math.exp(cross_entropy)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with lower-triangular causal mask.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: only attend to past and current positions (j <= i)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(context)


class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture (modern GPT standard)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniCausalLM(nn.Module):
    """
    Autoregressive GPT-style causal language model.
    """
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_dim: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch_size, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Shift targets so that tokens at step t predict target at t+1
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)

        return {"logits": logits, "loss": loss}


def compute_perplexity(cross_entropy_loss: float) -> float:
    """
    PPL = exp(CrossEntropyLoss)
    """
    return math.exp(cross_entropy_loss)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """
    Sample next token ID given raw unnormalized 1D logits.
    Supports Greedy (temp=0), Temperature scaling, Top-k, and Top-p (Nucleus) sampling.
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-5)

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        kth_val, _ = torch.topk(scaled_logits, top_k)
        min_top_k = kth_val[-1]
        scaled_logits = torch.where(scaled_logits < min_top_k, torch.tensor(float('-inf'), device=logits.device), scaled_logits)

    # Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least the first token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[indices_to_remove] = float('-inf')

    probs = F.softmax(scaled_logits, dim=-1)
    # Categorical sampling
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate_sequence(
    model: MiniCausalLM,
    prompt_ids: List[int],
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> List[int]:
    """
    Autoregressively generates new token IDs given an initial prompt.
    """
    model.eval()
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long)
            out = model(input_tensor)
            next_logits = out["logits"][0, -1, :]  # Logits for last token
            next_token = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated.append(next_token)

    return generated
