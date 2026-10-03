"""
Recurrent Neural Networks (RNN), LSTMs, and GRUs from scratch and in PyTorch.
Implements:
1. Analytical SimpleRNNCell, LSTMCell, and GRUCell forward transitions.
2. Bidirectional LSTM Sequence Classifier.
3. Gradient vanishing diagnostic demonstrating BPTT gradient norm decay.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional


# ============================================================================
# 1. Recurrent Cells from Scratch (PyTorch Tensor Math)
# ============================================================================

class ScratchSimpleRNNCell:
    """
    Standard Elman RNN Cell:
    h_t = tanh(x_t @ W_ih.T + h_{t-1} @ W_hh.T + b)
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        std = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_ih = torch.randn(hidden_size, input_size) * std
        self.W_hh = torch.randn(hidden_size, hidden_size) * std
        self.bias = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        Returns: h_t (batch_size, hidden_size)
        """
        linear = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.bias
        return torch.tanh(linear)


class ScratchLSTMCell:
    """
    Long Short-Term Memory (LSTM) Cell (Hochreiter & Schmidhuber, 1997):
    f_t = sigmoid(W_f x + U_f h + b_f)  [Forget Gate]
    i_t = sigmoid(W_i x + U_i h + b_i)  [Input Gate]
    c_tilde = tanh(W_c x + U_c h + b_c) [Candidate State]
    c_t = f_t * c_{t-1} + i_t * c_tilde [Cell State Update - Linear Additive Highway]
    o_t = sigmoid(W_o x + U_o h + b_o)  [Output Gate]
    h_t = o_t * tanh(c_t)               [Hidden State]
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for [f, i, c, o] gates: shape (4 * hidden_size, input_size)
        std = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_ih = torch.randn(4 * hidden_size, input_size) * std
        self.W_hh = torch.randn(4 * hidden_size, hidden_size) * std
        # Bias initialized with forget gate bias = 1.0 (Jozefowicz et al., 2015 trick)
        self.bias = torch.zeros(4 * hidden_size)
        self.bias[:hidden_size] = 1.0  # Forget gate bias

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (h_t, c_t)
        """
        gates = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.bias
        f, i, c_tilde, o = gates.chunk(4, dim=-1)
        
        f_t = torch.sigmoid(f)
        i_t = torch.sigmoid(i)
        c_tilde_t = torch.tanh(c_tilde)
        o_t = torch.sigmoid(o)
        
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ScratchGRUCell:
    """
    Gated Recurrent Unit (GRU) Cell (Cho et al., 2014):
    r_t = sigmoid(W_r x + U_r h + b_r)       [Reset Gate]
    z_t = sigmoid(W_z x + U_z h + b_z)       [Update Gate]
    h_tilde = tanh(W_h x + U_h (r_t * h) + b_h) [Candidate Hidden State]
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  [Hidden State Interpolation]
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        std = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_irz = torch.randn(2 * hidden_size, input_size) * std
        self.W_hrz = torch.randn(2 * hidden_size, hidden_size) * std
        self.b_rz = torch.zeros(2 * hidden_size)
        
        self.W_ih = torch.randn(hidden_size, input_size) * std
        self.W_hh = torch.randn(hidden_size, hidden_size) * std
        self.b_h = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        rz_gates = x @ self.W_irz.T + h_prev @ self.W_hrz.T + self.b_rz
        r, z = rz_gates.chunk(2, dim=-1)
        r_t = torch.sigmoid(r)
        z_t = torch.sigmoid(z)
        
        h_tilde = torch.tanh(x @ self.W_ih.T + (r_t * h_prev) @ self.W_hh.T + self.b_h)
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t


# ============================================================================
# 2. PyTorch Bidirectional Sequence Classifier
# ============================================================================

class BidirectionalLSTMClassifier(nn.Module):
    """
    Production PyTorch Bidirectional LSTM for sequence classification.
    Processes sequence in both forward and backward temporal directions.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Bidirectional concatenates forward and backward hidden states (2 * hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) token IDs
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Concatenate final forward (h_n[-2]) and backward (h_n[-1]) hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        
        return self.fc(self.dropout(h_cat))


# ============================================================================
# 3. Gradient Flow & Vanishing Gradients Diagnostic
# ============================================================================

def measure_bptt_gradient_norms(seq_len: int = 20, hidden_dim: int = 32) -> Tuple[List[float], List[float]]:
    """
    Compares gradient norm decay over unrolled timesteps between plain RNN and LSTM.
    Returns: (rnn_grad_norms, lstm_grad_norms) for each timestep back in time.
    """
    torch.manual_seed(42)
    # Plain RNN
    rnn = nn.RNN(input_size=1, hidden_size=hidden_dim, nonlinearity='tanh', batch_first=True)
    # LSTM
    lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
    
    # Track gradient magnitudes of input tokens at different temporal distances
    inputs = torch.randn(1, seq_len, 1, requires_grad=True)
    
    # 1. Plain RNN pass
    out_rnn, _ = rnn(inputs)
    loss_rnn = out_rnn[:, -1, :].sum()  # Loss on last timestep
    loss_rnn.backward(retain_graph=True)
    assert inputs.grad is not None
    rnn_grads = [float(inputs.grad[0, t].abs().sum()) for t in range(seq_len)]
    
    # 2. LSTM pass
    inputs.grad.zero_()
    out_lstm, _ = lstm(inputs)
    loss_lstm = out_lstm[:, -1, :].sum()
    loss_lstm.backward()
    lstm_grads = [float(inputs.grad[0, t].abs().sum()) for t in range(seq_len)]
    
    return rnn_grads, lstm_grads
