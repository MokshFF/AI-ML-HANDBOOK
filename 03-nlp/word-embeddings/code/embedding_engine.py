"""
Word Embeddings from Scratch in PyTorch and NumPy.
Implements:
1. Word2Vec: Skip-Gram with Negative Sampling (SGNS).
2. Continuous Bag of Words (CBOW).
3. GloVe Co-occurrence Objective & Weighted Least Squares Loss.
4. Vector Analogy and Cosine Similarity Retrieval.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


# ============================================================================
# 1. Word2Vec: Skip-Gram with Negative Sampling (SGNS)
# ============================================================================

class SkipGramNegativeSampling(nn.Module):
    """
    Mikolov et al. (2013) Skip-Gram with Negative Sampling (SGNS):
    Loss = - log(sigmoid(v'_pos . v_target)) - sum_k log(sigmoid(-v'_neg_k . v_target))
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Center target embeddings and context embeddings
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.5 / self.embed_dim
        nn.init.uniform_(self.target_embed.weight, -bound, bound)
        nn.init.constant_(self.context_embed.weight, 0.0)

    def forward(
        self,
        center_words: torch.Tensor,
        context_words: torch.Tensor,
        neg_words: torch.Tensor
    ) -> torch.Tensor:
        """
        center_words: (B,)
        context_words: (B,) positive context words
        neg_words: (B, num_neg) negative sampled words
        """
        # (B, embed_dim)
        v_center = self.target_embed(center_words)
        # (B, embed_dim)
        v_context = self.context_embed(context_words)
        # (B, num_neg, embed_dim)
        v_neg = self.context_embed(neg_words)

        # 1. Positive pair loss: log(sigmoid(v_center . v_context))
        pos_scores = torch.sum(v_center * v_context, dim=-1)  # (B,)
        pos_loss = F.logsigmoid(pos_scores)

        # 2. Negative pairs loss: sum_k log(sigmoid(- v_center . v_neg_k))
        # (B, 1, embed_dim) @ (B, embed_dim, num_neg) -> (B, num_neg)
        neg_scores = torch.bmm(v_neg, v_center.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=-1)  # (B,)

        total_loss = -(pos_loss + neg_loss).mean()
        return total_loss

    def get_embeddings(self) -> np.ndarray:
        """Returns the final word embeddings as target + context weights."""
        W_target = self.target_embed.weight.detach().cpu().numpy()
        W_context = self.context_embed.weight.detach().cpu().numpy()
        return W_target + W_context


# ============================================================================
# 2. Continuous Bag of Words (CBOW)
# ============================================================================

class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW):
    Averages context word embeddings to predict the center target word.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        """
        context_words: (B, 2*window_size)
        Returns: logits (B, vocab_size)
        """
        # (B, 2*C, embed_dim) -> (B, embed_dim)
        context_mean = self.embeddings(context_words).mean(dim=1)
        logits = self.fc(context_mean)
        return logits


# ============================================================================
# 3. GloVe Weighted Least Squares Objective
# ============================================================================

class GloVeLoss(nn.Module):
    """
    Pennington et al. (EMNLP 2014) GloVe Objective:
    J = sum_{i, j} f(X_{ij}) * (w_i . w_tilde_j + b_i + b_tilde_j - log(X_{ij}))^2
    weighting function f(x) = (x / x_max)^alpha if x < x_max else 1.0
    """
    def __init__(self, x_max: float = 100.0, alpha: float = 0.75):
        super().__init__()
        self.x_max = x_max
        self.alpha = alpha

    def forward(
        self,
        w_i: torch.Tensor,
        w_j: torch.Tensor,
        b_i: torch.Tensor,
        b_j: torch.Tensor,
        cooccur: torch.Tensor
    ) -> torch.Tensor:
        """
        w_i, w_j: (B, embed_dim)
        b_i, b_j: (B, 1)
        cooccur: (B, 1) co-occurrence counts X_{ij}
        """
        dot_product = torch.sum(w_i * w_j, dim=-1, keepdim=True)
        pred = dot_product + b_i + b_j
        diff = pred - torch.log(cooccur + 1e-12)

        # Weighting function f(X_ij)
        weights = torch.where(
            cooccur < self.x_max,
            torch.pow(cooccur / self.x_max, self.alpha),
            torch.ones_like(cooccur)
        )
        return torch.mean(weights * torch.pow(diff, 2))


# ============================================================================
# 4. Vector Analogy & Cosine Similarity Utilities
# ============================================================================

def find_most_similar(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    idx2word: Dict[int, str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Computes cosine similarity between query_vector and all rows in embeddings.
    """
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embed_norm = embeddings / norms

    q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    scores = np.dot(embed_norm, q_norm)

    top_indices = np.argsort(-scores)[:top_k]
    return [(idx2word[i], float(scores[i])) for i in top_indices]


def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    embeddings: np.ndarray,
    top_k: int = 1
) -> List[Tuple[str, float]]:
    """
    Solves word analogy: A is to B as C is to ?
    Vector formula: target = v_B - v_A + v_C
    """
    v_a = embeddings[word2idx[word_a]]
    v_b = embeddings[word2idx[word_b]]
    v_c = embeddings[word2idx[word_c]]
    target_vec = v_b - v_a + v_c

    results = find_most_similar(target_vec, embeddings, idx2word, top_k=top_k + 3)
    # Exclude source words from candidates
    filtered = [r for r in results if r[0] not in {word_a, word_b, word_c}]
    return filtered[:top_k]
