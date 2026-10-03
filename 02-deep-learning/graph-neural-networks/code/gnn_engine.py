"""
Graph Neural Networks (GNN) from scratch in PyTorch.
Implements:
1. Graph Convolutional Networks (GCN) with symmetric Laplacian normalization.
2. GraphSAGE with neighborhood mean aggregation.
3. Graph Attention Networks (GAT) with edge-level self-attention coefficients.
4. End-to-end Node Classification GNN model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 1. Graph Convolutional Network (GCN) Layer
# ============================================================================

class GCNLayer(nn.Module):
    """
    Kipf & Welling (ICLR 2017) GCN Layer:
    H^{(l+1)} = sigma( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{(l)} W )
    where A_tilde = A + I_N (self-loops added).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """
        Adds self-loops and computes symmetric normalization:
        A_norm = D^{-1/2} (A + I) D^{-1/2}
        """
        N = adj.size(0)
        adj_tilde = adj + torch.eye(N, device=adj.device)
        deg = torch.sum(adj_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return torch.matmul(torch.matmul(D_inv_sqrt, adj_tilde), D_inv_sqrt)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        adj_norm: (N, N) symmetrically normalized adjacency matrix
        """
        support = torch.matmul(x, self.weight)  # (N, out_features)
        output = torch.matmul(adj_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


# ============================================================================
# 2. GraphSAGE Layer
# ============================================================================

class GraphSAGELayer(nn.Module):
    """
    Hamilton et al. (NeurIPS 2017) GraphSAGE Layer with Mean Aggregator:
    h_{N(v)} = Mean_{u in N(v)} (h_u)
    h_v^{(l+1)} = sigma( W_self h_v + W_neigh h_{N(v)} )
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features, bias=False)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        adj: (N, N) binary or weighted adjacency matrix
        """
        # Compute row degrees for mean aggregation: D^{-1} A X
        deg = torch.sum(adj, dim=1, keepdim=True).clamp(min=1.0)
        neigh_mean = torch.matmul(adj, x) / deg

        h_self = self.linear_self(x)
        h_neigh = self.linear_neigh(neigh_mean)
        return F.relu(h_self + h_neigh)


# ============================================================================
# 3. Graph Attention Network (GAT) Layer
# ============================================================================

class GATLayer(nn.Module):
    """
    Veličković et al. (ICLR 2018) Graph Attention Network Layer:
    alpha_{ij} = Softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
    h_i^{(l+1)} = sigma( sum_{j in N(i)} alpha_{ij} W h_j )
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, in_features)
        adj: (N, N) binary adjacency matrix (should include self-loops)
        Returns: (output (N, out_features), attention_weights (N, N))
        """
        N = x.size(0)
        h = torch.matmul(x, self.W)  # (N, out_features)

        # Broadcast concatenation of all pairs (h_i || h_j): (N, N, 2 * out_features)
        h_i = h.unsqueeze(1).repeat(1, N, 1)
        h_j = h.unsqueeze(0).repeat(N, 1, 1)
        all_pairs = torch.cat([h_i, h_j], dim=-1)

        # Attention coefficients: e_{ij} = LeakyReLU( a^T [h_i || h_j] )
        e = self.leakyrelu(torch.matmul(all_pairs, self.a).squeeze(-1))  # (N, N)

        # Mask out non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # Softmax across neighborhood j

        h_prime = torch.matmul(attention, h)  # (N, out_features)
        return F.elu(h_prime), attention


# ============================================================================
# 4. 2-Layer GCN Node Classifier
# ============================================================================

class GCNNodeClassifier(nn.Module):
    """
    Standard 2-Layer GCN for semi-supervised node classification.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gcn1(x, adj_norm))
        x = self.dropout(x)
        logits = self.gcn2(x, adj_norm)
        return logits
