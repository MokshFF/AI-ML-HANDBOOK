"""
Sequence-to-Sequence (Seq2Seq) with Attention in PyTorch.
Implements:
1. Bahdanau Additive Attention and Luong Multiplicative Attention.
2. GRU-based Seq2Seq with Attention and scheduled Teacher Forcing.
3. Sentence-level BLEU Score computation from scratch.
"""

from __future__ import annotations
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


# ============================================================================
# 1. Attention Mechanisms: Bahdanau vs. Luong
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau et al. (ICLR 2015) Additive Attention:
    score(s_t, h_i) = v_a^T * tanh(W_s * s_t + W_h * h_i)
    """
    def __init__(self, dec_hidden_dim: int, enc_hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.w_s = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.w_h = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dec_hidden: (B, dec_hidden_dim)
        enc_outputs: (B, src_len, enc_hidden_dim)
        Returns: (context (B, enc_hidden_dim), attn_weights (B, src_len))
        """
        # (B, 1, attn_dim)
        s_proj = self.w_s(dec_hidden).unsqueeze(1)
        # (B, src_len, attn_dim)
        h_proj = self.w_h(enc_outputs)
        
        # Energy scores: (B, src_len, 1) -> (B, src_len)
        energy = self.v_a(torch.tanh(s_proj + h_proj)).squeeze(-1)
        attn_weights = F.softmax(energy, dim=-1)

        # Context vector: (B, 1, src_len) @ (B, src_len, enc_hidden_dim) -> (B, enc_hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


class LuongAttention(nn.Module):
    """
    Luong et al. (EMNLP 2015) Multiplicative (General) Attention:
    score(s_t, h_i) = s_t^T * W_a * h_i
    """
    def __init__(self, dec_hidden_dim: int, enc_hidden_dim: int):
        super().__init__()
        self.w_a = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dec_hidden: (B, dec_hidden_dim)
        enc_outputs: (B, src_len, enc_hidden_dim)
        """
        # enc_outputs @ W_a.T: (B, src_len, dec_hidden_dim)
        h_proj = self.w_a(enc_outputs)
        # (B, src_len, dec_hidden_dim) @ (B, dec_hidden_dim, 1) -> (B, src_len)
        scores = torch.bmm(h_proj, dec_hidden.unsqueeze(-1)).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


# ============================================================================
# 2. Seq2Seq Architecture with Attention
# ============================================================================

class Seq2SeqEncoder(nn.Module):
    def __init__(self, src_vocab_size: int, embed_dim: int, enc_hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, enc_hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int
    ):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.attention = BahdanauAttention(dec_hidden_dim, enc_hidden_dim)
        # Input to GRU is concatenated target embedding + context vector
        self.rnn = nn.GRU(embed_dim + enc_hidden_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim + embed_dim, tgt_vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        dec_hidden: torch.Tensor,
        enc_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_token: (B,) single target token ID
        dec_hidden: (1, B, dec_hidden_dim)
        enc_outputs: (B, src_len, enc_hidden_dim)
        """
        # (B, 1, embed_dim)
        embedded = self.embedding(input_token.unsqueeze(1))
        # Context from attention
        context, attn_weights = self.attention(dec_hidden.squeeze(0), enc_outputs)

        # Concatenate embedded token and context vector for GRU input
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, new_dec_hidden = self.rnn(rnn_input, dec_hidden)

        # Output projection combines decoder hidden, context, and embedding
        concat_out = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        logits = self.fc_out(concat_out)
        return logits, new_dec_hidden, attn_weights


class Seq2SeqModel(nn.Module):
    """
    Complete Seq2Seq Model with Teacher Forcing Support.
    """
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqAttentionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        src: (B, src_len)
        tgt: (B, tgt_len)
        Returns: outputs (B, tgt_len, tgt_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.tgt_vocab_size

        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = enc_hidden

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)
        input_token = tgt[:, 0]  # First token is <BOS> / Start-of-Sequence

        for t in range(1, tgt_len):
            logits, dec_hidden, _ = self.decoder.forward_step(input_token, dec_hidden, enc_outputs)
            outputs[:, t, :] = logits
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs


# ============================================================================
# 3. BLEU Score from Scratch
# ============================================================================

def compute_bleu_score(
    reference: List[str],
    candidate: List[str],
    max_n: int = 4
) -> float:
    """
    Sentence-level BLEU (Papineni et al., 2002) with modified n-gram precision
    and brevity penalty (BP).
    """
    c = len(candidate)
    r = len(reference)
    if c == 0:
        return 0.0

    # 1. Brevity Penalty
    bp = 1.0 if c > r else math.exp(1.0 - float(r) / float(c))

    # 2. Modified n-gram precision
    precisions: List[float] = []
    weights = [1.0 / max_n] * max_n

    for n in range(1, max_n + 1):
        if len(candidate) < n or len(reference) < n:
            precisions.append(1e-10)
            continue

        cand_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(candidate) - n + 1):
            ng = tuple(candidate[i:i + n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

        ref_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(reference) - n + 1):
            ng = tuple(reference[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        # Clipped counts
        clipped_count = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
        total_count = max(1, len(candidate) - n + 1)
        precisions.append((clipped_count + 1e-10) / total_count)

    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    return float(bp * math.exp(log_sum))
