"""
Unit tests for Speech Synthesis (TTS): Length Regulation, Acoustic Model, Griffin-Lim, and MCD.
"""

import numpy as np
import pytest
import torch
from tts_engine import (
    LengthRegulator,
    MiniAcousticTTS,
    griffin_lim_reconstruction,
    compute_mcd,
)


def test_length_regulator():
    regulator = LengthRegulator()
    # Batch of 1 sequence, 3 tokens, hidden dim 4
    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]])
    # Token 0 repeated 2 times, token 1 repeated 3 times, token 2 repeated 1 time -> total len = 6
    durations = torch.tensor([[2, 3, 1]])

    out = regulator(x, durations)
    assert out.shape == (1, 6, 4)
    assert torch.allclose(out[0, 0], out[0, 1])
    assert torch.allclose(out[0, 2], out[0, 3])
    assert torch.allclose(out[0, 3], out[0, 4])


def test_mini_acoustic_tts_forward():
    torch.manual_seed(42)
    model = MiniAcousticTTS(vocab_size=20, embed_dim=16, num_mels=40)
    tokens = torch.tensor([[1, 5, 8, 2]])
    durations = torch.tensor([[2, 3, 2, 4]])  # Total frames = 11

    out = model(tokens, target_durations=durations)
    assert out["mel_out"].shape == (1, 11, 40)
    assert out["pred_durations"].shape == (1, 4)

    # Test with target_mel loss computation
    target_mel = torch.randn(1, 11, 40)
    out_with_loss = model(tokens, target_durations=durations, target_mel=target_mel)
    assert out_with_loss["loss"] is not None
    assert out_with_loss["loss"].item() > 0


def test_griffin_lim_reconstruction():
    np.random.seed(42)
    # Synthetic magnitude spectrogram: 257 bins x 15 frames
    mag = np.abs(np.random.randn(257, 15)).astype(np.float32)
    waveform = griffin_lim_reconstruction(mag, n_fft=512, hop_length=160, win_length=400, n_iter=4)

    assert len(waveform) > 0
    assert not np.isnan(waveform).any()


def test_compute_mcd():
    mel1 = np.ones((40, 20), dtype=np.float32)
    # Identical mels -> MCD = 0.0
    mcd_zero = compute_mcd(mel1, mel1)
    assert pytest.approx(mcd_zero, 1e-4) == 0.0

    # Perturbed mel
    mel2 = mel1 + 0.5
    mcd_val = compute_mcd(mel1, mel2)
    assert mcd_val > 0.0
