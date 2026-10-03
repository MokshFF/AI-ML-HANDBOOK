import pytest
import torch
from rnn_engine import (
    ScratchSimpleRNNCell,
    ScratchLSTMCell,
    ScratchGRUCell,
    BidirectionalLSTMClassifier,
    measure_bptt_gradient_norms
)


def test_scratch_simple_rnn_cell():
    cell = ScratchSimpleRNNCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 4)
    h_prev = torch.zeros(2, 8)
    h_next = cell.forward(x, h_prev)
    assert h_next.shape == (2, 8)
    assert (h_next >= -1.0).all() and (h_next <= 1.0).all()


def test_scratch_lstm_cell():
    cell = ScratchLSTMCell(input_size=4, hidden_size=8)
    x = torch.randn(3, 4)
    h_prev = torch.zeros(3, 8)
    c_prev = torch.zeros(3, 8)
    h_next, c_next = cell.forward(x, h_prev, c_prev)
    assert h_next.shape == (3, 8)
    assert c_next.shape == (3, 8)


def test_scratch_gru_cell():
    cell = ScratchGRUCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 4)
    h_prev = torch.zeros(2, 8)
    h_next = cell.forward(x, h_prev)
    assert h_next.shape == (2, 8)


def test_bidirectional_lstm_classifier():
    model = BidirectionalLSTMClassifier(
        vocab_size=100,
        embed_dim=16,
        hidden_dim=32,
        num_classes=4,
        num_layers=1,
        dropout=0.1
    )
    # Batch=4, SeqLen=10
    tokens = torch.randint(0, 100, (4, 10))
    logits = model(tokens)
    assert logits.shape == (4, 4)


def test_bptt_gradient_flow():
    rnn_grads, lstm_grads = measure_bptt_gradient_norms(seq_len=15, hidden_dim=32)
    assert len(rnn_grads) == 15
    assert len(lstm_grads) == 15
    # The gradient at the earliest timestep (index 0) compared to final timestep (index 14)
    # LSTM maintains significantly better gradient flow to the distant past than plain RNN
    lstm_retention = lstm_grads[0] / (lstm_grads[-1] + 1e-8)
    rnn_retention = rnn_grads[0] / (rnn_grads[-1] + 1e-8)
    assert lstm_retention > rnn_retention
