"""
Unit tests for Audio Spectrogram Classification and Speaker Verification.
"""

import pytest
import torch
from audio_classifier import (
    SpectrogramConvBlock,
    SpectrogramCNNClassifier,
    SpeakerEmbeddingModel,
    compute_triplet_loss,
    evaluate_verification,
)


def test_spectrogram_conv_block():
    block = SpectrogramConvBlock(in_channels=1, out_channels=8, pool=True)
    x = torch.randn(2, 1, 40, 50)
    out = block(x)
    assert out.shape == (2, 8, 20, 25)


def test_spectrogram_classifier_forward_and_loss():
    torch.manual_seed(42)
    model = SpectrogramCNNClassifier(num_classes=4, base_dim=8, embedding_dim=16)
    x = torch.randn(3, 1, 40, 50)
    targets = torch.tensor([0, 2, 3])

    out = model(x, targets=targets)
    assert out["logits"].shape == (3, 4)
    assert out["embedding"].shape == (3, 16)
    assert out["loss"] is not None
    assert out["loss"].item() > 0


def test_speaker_embedding_and_triplet_loss():
    torch.manual_seed(42)
    backbone = SpectrogramCNNClassifier(num_classes=2, base_dim=8, embedding_dim=16)
    speaker_model = SpeakerEmbeddingModel(backbone)

    audio_a = torch.randn(2, 1, 40, 50)
    audio_p = audio_a + 0.05 * torch.randn(2, 1, 40, 50)
    audio_n = torch.randn(2, 1, 40, 50)

    emb_a = speaker_model(audio_a)
    emb_p = speaker_model(audio_p)
    emb_n = speaker_model(audio_n)

    # Unit norm check
    assert pytest.approx(torch.norm(emb_a, p=2, dim=-1).mean().item(), 1e-4) == 1.0

    loss = compute_triplet_loss(emb_a, emb_p, emb_n, margin=0.2)
    assert loss.item() >= 0.0


def test_evaluate_verification():
    # 4 pairs: [Same, Same, Diff, Diff]
    scores = torch.tensor([0.85, 0.40, 0.70, 0.15])
    labels = torch.tensor([1, 1, 0, 0])

    metrics = evaluate_verification(scores, labels, threshold=0.5)
    # Pair 0 (1, score 0.85): Correct (same)
    # Pair 1 (1, score 0.40): False reject (score < 0.5) -> FRR = 1/2 = 0.5
    # Pair 2 (0, score 0.70): False accept (score >= 0.5) -> FAR = 1/2 = 0.5
    # Pair 3 (0, score 0.15): Correct (diff)
    assert pytest.approx(metrics["FAR"], 1e-4) == 0.5
    assert pytest.approx(metrics["FRR"], 1e-4) == 0.5
    assert pytest.approx(metrics["accuracy"], 1e-4) == 0.5
