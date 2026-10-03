import pytest
import numpy as np
import torch
from dl_fundamentals import (
    DenseLayer,
    ReLU,
    Sigmoid,
    SoftmaxCrossEntropyLoss,
    SGDMomentum,
    RMSprop,
    AdamW,
    ReferencePyTorchMLP
)


def test_dense_layer_forward_backward():
    np.random.seed(42)
    layer = DenseLayer(in_features=4, out_features=2, init_method="he")
    X = np.random.randn(5, 4)
    out = layer.forward(X)
    assert out.shape == (5, 2)

    grad_output = np.ones((5, 2))
    grad_input = layer.backward(grad_output)
    assert grad_input.shape == (5, 4)
    assert layer.grad_W.shape == (4, 2)
    assert layer.grad_b.shape == (1, 2)

    # Numerical gradient check for W
    eps = 1e-5
    orig_w = layer.W[0, 0]
    layer.W[0, 0] = orig_w + eps
    out_pos = layer.forward(X)
    loss_pos = np.sum(out_pos * grad_output)

    layer.W[0, 0] = orig_w - eps
    out_neg = layer.forward(X)
    loss_neg = np.sum(out_neg * grad_output)

    num_grad = (loss_pos - loss_neg) / (2 * eps)
    layer.W[0, 0] = orig_w
    layer.backward(grad_output)
    assert np.isclose(layer.grad_W[0, 0], num_grad, rtol=1e-3, atol=1e-3)


def test_activations():
    x = np.array([-2.0, 0.0, 3.0])
    relu = ReLU()
    assert np.allclose(relu.forward(x), [0.0, 0.0, 3.0])
    assert np.allclose(relu.backward(np.ones_like(x)), [0.0, 0.0, 1.0])

    sigmoid = Sigmoid()
    sig_out = sigmoid.forward(x)
    assert 0.0 < sig_out[0] < 0.5
    assert np.isclose(sig_out[1], 0.5)
    grad_sig = sigmoid.backward(np.ones_like(x))
    assert np.allclose(grad_sig, sig_out * (1 - sig_out))


def test_softmax_cross_entropy():
    criterion = SoftmaxCrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
    y = np.array([0, 1])

    loss = criterion.forward(logits, y)
    assert loss > 0.0
    grad = criterion.backward()
    assert grad.shape == logits.shape
    # Sum of gradients along class dimension is 0 for softmax-CE
    assert np.allclose(np.sum(grad, axis=1), 0.0, atol=1e-7)


def test_optimizers_step():
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 3, size=20)
    
    # Train mini 1-layer network with AdamW
    layer = DenseLayer(5, 3, init_method="xavier")
    loss_fn = SoftmaxCrossEntropyLoss()
    opt = AdamW([layer], lr=0.05, weight_decay=0.01)

    initial_loss = loss_fn.forward(layer.forward(X), y)
    for _ in range(15):
        opt.zero_grad()
        out = layer.forward(X)
        loss = loss_fn.forward(out, y)
        d_out = loss_fn.backward()
        layer.backward(d_out)
        opt.step()

    final_loss = loss_fn.forward(layer.forward(X), y)
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_pytorch_mlp_reference():
    torch.manual_seed(42)
    model = ReferencePyTorchMLP(in_features=8, hidden_dim=16, out_features=2, dropout_rate=0.0)
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == (10, 2)

    criterion = torch.nn.CrossEntropyLoss()
    target = torch.randint(0, 2, (10,))
    loss = criterion(out, target)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
