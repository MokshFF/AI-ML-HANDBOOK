"""
Unit tests for Speech Recognition (ASR): Acoustic Model, CTC Greedy Decoding, and WER/CER.
"""

import pytest
import torch
import torch.nn as nn
from ctc_engine import (
    MiniAcousticEncoder,
    ctc_greedy_decode,
    compute_edit_distance,
    compute_wer,
    compute_cer,
)


def test_mini_acoustic_encoder_and_ctc_loss():
    torch.manual_seed(42)
    model = MiniAcousticEncoder(in_features=40, hidden_dim=32, num_classes=5)
    # Input: (B=2, num_mels=40, time_frames=20)
    x = torch.randn(2, 40, 20)
    log_probs = model(x)

    # Time dimension downsampled by 2 -> T_out = 10
    assert log_probs.shape == (10, 2, 5)

    # CTCLoss verification
    targets = torch.tensor([[1, 2, 3], [2, 3, 1]])  # Target sequences
    input_lengths = torch.tensor([10, 10])
    target_lengths = torch.tensor([3, 3])

    ctc_loss_fn = nn.CTCLoss(blank=0)
    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_ctc_greedy_decode():
    # Construct synthetic log_probs for 1 batch item, length 7, 4 classes (0: blank, 1: 'c', 2: 'a', 3: 't')
    # Path: [1, 1, 0, 2, 2, 0, 3] -> 'c', 'c', '_', 'a', 'a', '_', 't' -> 'cat' -> [1, 2, 3]
    t_steps = 7
    log_probs = torch.zeros(t_steps, 1, 4)
    # Assign high values to target path
    path = [1, 1, 0, 2, 2, 0, 3]
    for t, p in enumerate(path):
        log_probs[t, 0, p] = 10.0

    decoded = ctc_greedy_decode(log_probs, blank_idx=0)
    assert decoded == [[1, 2, 3]]


def test_ctc_greedy_decode_double_letters():
    # Testing word with double letter: 'hello' -> [h, e, l, _, l, o]
    # Path: [1, 2, 3, 0, 3, 4] -> tokens [1, 2, 3, 3, 4]
    t_steps = 6
    log_probs = torch.zeros(t_steps, 1, 5)
    path = [1, 2, 3, 0, 3, 4]
    for t, p in enumerate(path):
        log_probs[t, 0, p] = 10.0

    decoded = ctc_greedy_decode(log_probs, blank_idx=0)
    assert decoded == [[1, 2, 3, 3, 4]]


def test_edit_distance_and_wer_cer():
    ref = "the cat sat on the mat"
    hyp = "the cat slept on the mat"

    # 1 word substitution: "sat" -> "slept", 6 words total -> WER = 1/6
    wer = compute_wer(ref, hyp)
    assert pytest.approx(wer, 0.01) == 1.0 / 6.0

    # CER test
    cer = compute_cer("cat", "cot")
    assert pytest.approx(cer, 0.01) == 1.0 / 3.0
