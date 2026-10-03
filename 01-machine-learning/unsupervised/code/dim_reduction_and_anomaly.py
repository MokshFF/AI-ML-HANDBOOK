"""
Unsupervised Learning - Dimensionality Reduction & Anomaly Detection
Implements PCA, t-SNE gradient descent, and Isolation Forest from scratch.
"""

from __future__ import annotations
import numpy as np


class PCAScratch:
    """
    Principal Component Analysis via SVD of centered data.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]

        eigenvals = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = (eigenvals[: self.n_components] / np.sum(eigenvals))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T  # type: ignore

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TSNEScratch:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) from scratch.
    High-dim Gaussian affinities P_ij, Low-dim Student-t affinities Q_ij.
    Loss: KL(P || Q) = sum_i sum_j p_ij * log(p_ij / q_ij)
    """
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, lr: float = 200.0, n_iter: int = 300):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iter = n_iter
        self.embedding_: np.ndarray | None = None

    @staticmethod
    def _compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
        sum_X = np.sum(X ** 2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2.0 * (X @ X.T)
        return np.maximum(D, 0.0)

    def _compute_p_matrix(self, D: np.ndarray) -> np.ndarray:
        # Approximate Gaussian affinity with fixed sigma for clarity
        sigma_sq = 2.0 * (np.median(D) + 1e-5)
        P = np.exp(-D / sigma_sq)
        np.fill_diagonal(P, 0.0)
        P = P / (np.sum(P) + 1e-12)
        # Symmetrize
        P = (P + P.T) / (2.0 * len(D))
        return np.maximum(P, 1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(42)
        n_samples = len(X)
        D = self._compute_pairwise_distances(X.astype(np.float64))
        P = self._compute_p_matrix(D)

        # Initialize low-dimensional map randomly
        Y = np.random.normal(0, 1e-4, size=(n_samples, self.n_components))
        velocity = np.zeros_like(Y)
        momentum = 0.8

        for step in range(self.n_iter):
            # Compute Q matrix using Student-t distribution (1 degree of freedom)
            sum_Y = np.sum(Y ** 2, axis=1)
            dist_Y = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2.0 * (Y @ Y.T)
            num = 1.0 / (1.0 + np.maximum(dist_Y, 0.0))
            np.fill_diagonal(num, 0.0)
            Q = num / (np.sum(num) + 1e-12)
            Q = np.maximum(Q, 1e-12)

            # Gradient computation: dC / dy_i = 4 * sum_j (p_ij - q_ij) * q_num_ij * (y_i - y_j)
            PQ_diff = (P - Q) * num
            grad = np.zeros_like(Y)
            for i in range(n_samples):
                grad[i] = 4.0 * np.sum((PQ_diff[i, :, np.newaxis]) * (Y[i] - Y), axis=0)

            velocity = momentum * velocity - self.lr * grad
            Y += velocity

        self.embedding_ = Y
        return self.embedding_


class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feature: int = -1, split_value: float = 0.0, size: int = 0):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class IsolationForestScratch:
    """
    Isolation Forest for unsupervised Anomaly Detection.
    Anomalies have shorter average path lengths in random partitioning trees.
    """
    def __init__(self, n_estimators: int = 50, max_samples: int = 128, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.trees: list[IsolationTreeNode] = []

    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # Euler-Mascheroni constant ~ 0.5772156649
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

    def _build_tree(self, X: np.ndarray, current_height: int, max_height: int) -> IsolationTreeNode:
        n_samples, n_features = X.shape
        if current_height >= max_height or n_samples <= 1:
            return IsolationTreeNode(size=n_samples)

        feat = np.random.randint(0, n_features)
        feat_min, feat_max = np.min(X[:, feat]), np.max(X[:, feat])
        if np.isclose(feat_min, feat_max):
            return IsolationTreeNode(size=n_samples)

        split_val = np.random.uniform(feat_min, feat_max)
        left_mask = X[:, feat] < split_val
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], current_height + 1, max_height)
        right = self._build_tree(X[right_mask], current_height + 1, max_height)
        return IsolationTreeNode(left=left, right=right, split_feature=feat, split_value=split_val, size=n_samples)

    def fit(self, X: np.ndarray) -> "IsolationForestScratch":
        np.random.seed(self.seed)
        n_samples = len(X)
        subsample_size = min(self.max_samples, n_samples)
        max_height = int(np.ceil(np.log2(max(subsample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            tree_root = self._build_tree(X[indices], 0, max_height)
            self.trees.append(tree_root)

        return self

    def _path_length(self, x: np.ndarray, node: IsolationTreeNode, current_depth: int) -> float:
        if node.is_leaf:
            return current_depth + self._c(node.size)
        if x[node.split_feature] < node.split_value:
            return self._path_length(x, node.left, current_depth + 1)
        return self._path_length(x, node.right, current_depth + 1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score s in [0, 1].
        s(x, n) = 2^(- E(h(x)) / c(n))
        Scores close to 1 are anomalies; scores < 0.5 are normal.
        """
        n_samples = len(X)
        scores = np.zeros(n_samples)
        c_factor = self._c(self.max_samples)

        for i in range(n_samples):
            paths = [self._path_length(X[i], tree, 0) for tree in self.trees]
            avg_path = np.mean(paths)
            scores[i] = 2.0 ** (-avg_path / (c_factor + 1e-12))

        return scores
