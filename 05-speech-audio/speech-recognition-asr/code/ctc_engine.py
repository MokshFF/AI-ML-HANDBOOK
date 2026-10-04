"""
Speech Recognition (ASR) Engine: Acoustic Encoder, CTC Loss, Greedy Decoding, and WER/CER.

Implements from scratch using pure PyTorch:
- MiniAcousticEncoder: Temporal convolutional downsampler + recurrent/transformer sequence encoder.
- ctc_greedy_decode: Collapses repeated tokens and removes blank labels to reconstruct character transcriptions.
- compute_edit_distance: Dynamic programming Levenshtein distance for insertions, deletions, and substitutions.
- compute_wer & compute_cer: Word Error Rate and Character Error Rate metrics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniAcousticEncoder(nn.Module):
    """
    Lightweight Acoustic Model predicting CTC character logits from log-Mel spectrograms.
    Architecture: 1D temporal conv downsampling + 2-layer Bidirectional GRU + projection head.
    """
    def __init__(self, in_features: int = 40, hidden_dim: int = 64, num_classes: int = 28):
        super().__init__()
        # Downsample time dimension by factor of 2: Conv1d (stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_mels, time_frames)
        Returns:
            log_probs: (T_out, B, num_classes) formatted for PyTorch CTCLoss.
        """
        feats = self.conv(x)  # (B, hidden_dim, T_out)
        feats = feats.transpose(1, 2)  # (B, T_out, hidden_dim)

        rnn_out, _ = self.rnn(feats)  # (B, T_out, hidden_dim * 2)
        logits = self.classifier(rnn_out)  # (B, T_out, num_classes)

        # PyTorch CTCLoss expects log-probabilities of shape (T, B, C)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        return log_probs


def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """
    Greedy Best-Path CTC Decoding.
    Takes log-probabilities (T, B, C) and returns collapsed token sequences per batch item.
    Rules:
      1. Argmax at each time step t.
      2. Collapse adjacent repeated identical tokens.
      3. Drop all blank tokens.
    """
    # (T, B, C) -> (B, T)
    best_paths = torch.argmax(log_probs, dim=-1).permute(1, 0).tolist()
    batch_transcripts = []

    for path in best_paths:
        collapsed = []
        prev_token = None
        for tok in path:
            if tok != prev_token:
                if tok != blank_idx:
                    collapsed.append(tok)
                prev_token = tok
        batch_transcripts.append(collapsed)

    return batch_transcripts


def compute_edit_distance(reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int, int]:
    """
    Dynamic programming Levenshtein distance.
    Returns:
        (substitutions, deletions, insertions, total_ref_tokens)
    """
    r_len = len(reference)
    h_len = len(hypothesis)

    # DP table: (r_len + 1) x (h_len + 1)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                dp[i][j] = min(sub, dele, ins)

    total_dist = dp[r_len][h_len]
    return total_dist, r_len


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """
    Computes Word Error Rate: WER = Distance(ref_words, hyp_words) / len(ref_words).
    """
    ref_words = reference_text.strip().split()
    hyp_words = hypothesis_text.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist, n = compute_edit_distance(ref_words, hyp_words)
    return dist / float(n)


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    """
    Computes Character Error Rate: CER = Distance(ref_chars, hyp_chars) / len(ref_chars).
    """
    ref_chars = list(reference_text.replace(" ", "|"))
    hyp_chars = list(hypothesis_text.replace(" ", "|"))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist, n = compute_edit_distance(ref_chars, hyp_chars)
    return dist / float(n)
