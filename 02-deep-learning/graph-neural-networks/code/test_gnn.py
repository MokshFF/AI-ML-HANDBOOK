import pytest
import torch
from gnn_engine import GCNLayer, GraphSAGELayer, GATLayer, GCNNodeClassifier


def test_gcn_adjacency_normalization():
    # 3-node line graph: 0-1-2
    adj = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    adj_norm = GCNLayer.normalize_adjacency(adj)
    assert adj_norm.shape == (3, 3)
    # Symmetry
    assert torch.allclose(adj_norm, adj_norm.T)
    # Diagonal should be non-zero because of self-loops
    assert (torch.diag(adj_norm) > 0).all()


def test_gcn_layer_forward():
    layer = GCNLayer(in_features=8, out_features=4)
    x = torch.randn(5, 8)
    adj = torch.eye(5) + torch.ones(5, 5) * 0.1
    adj_norm = GCNLayer.normalize_adjacency(adj)
    out = layer(x, adj_norm)
    assert out.shape == (5, 4)


def test_graphsage_layer():
    sage = GraphSAGELayer(in_features=8, out_features=4)
    x = torch.randn(4, 8)
    adj = torch.eye(4)
    out = sage(x, adj)
    assert out.shape == (4, 4)


def test_gat_layer():
    gat = GATLayer(in_features=8, out_features=4)
    x = torch.randn(3, 8)
    # Fully connected graph with self-loops
    adj = torch.ones(3, 3)
    out, attn = gat(x, adj)
    assert out.shape == (3, 4)
    assert attn.shape == (3, 3)
    # Attention weights over connected neighbors should sum to 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_gcn_node_classifier():
    model = GCNNodeClassifier(in_features=6, hidden_dim=12, num_classes=3, dropout=0.0)
    x = torch.randn(6, 6)
    adj = torch.eye(6)
    adj_norm = GCNLayer.normalize_adjacency(adj)
    logits = model(x, adj_norm)
    assert logits.shape == (6, 3)

    labels = torch.randint(0, 3, (6,))
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None
