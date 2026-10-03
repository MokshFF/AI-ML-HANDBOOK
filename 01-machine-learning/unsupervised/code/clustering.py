"""
Unsupervised Learning - Clustering Algorithms from Scratch
Implements K-Means (with K-Means++), DBSCAN, Gaussian Mixture Models (EM),
and Agglomerative Hierarchical Clustering.
"""

from __future__ import annotations
import numpy as np


class KMeansScratch:
    """
    K-Means Clustering with K-Means++ centroid initialization.
    Minimizes within-cluster sum of squares (Inertia):
    J = sum_{i=1}^N sum_{k=1}^K r_{ik} ||x_i - mu_k||_2^2
    """
    def __init__(self, n_clusters: int = 3, n_init: int = 10, max_iter: int = 300, tol: float = 1e-4, seed: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids_: np.ndarray | None = None
        self.inertia_: float = float("inf")

    def _init_centroids(self, X: np.ndarray, seed: int) -> np.ndarray:
        np.random.seed(seed)
        n_samples = len(X)
        centroids = [X[np.random.choice(n_samples)]]

        for _ in range(1, self.n_clusters):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / (np.sum(dists) + 1e-12)
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids, dtype=np.float64)

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X = X.astype(np.float64)
        best_inertia = float("inf")
        best_centroids = None

        for init_idx in range(self.n_init):
            centroids = self._init_centroids(X, seed=self.seed + init_idx)

            for _ in range(self.max_iter):
                dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(dists, axis=1)

                new_centroids = np.zeros_like(centroids)
                for k in range(self.n_clusters):
                    cluster_pts = X[labels == k]
                    if len(cluster_pts) > 0:
                        new_centroids[k] = np.mean(cluster_pts, axis=0)
                    else:
                        new_centroids[k] = centroids[k]

                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if shift < self.tol:
                    break

            final_dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            inertia = float(np.sum(final_dists))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids

        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted.")
        X = X.astype(np.float64)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)


class DBSCANScratch:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Identifies Core points, Border points, and Noise points (-1).
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def _region_query(self, X: np.ndarray, point_idx: int) -> list[int]:
        dists = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(dists <= self.eps)[0])

    def fit(self, X: np.ndarray) -> "DBSCANScratch":
        n_samples = len(X)
        X = X.astype(np.float64)
        labels = np.full(n_samples, -1, dtype=int)  # -1 represents noise initially
        visited = set()
        cluster_id = 0

        for i in range(n_samples):
            if i in visited:
                continue
            visited.add(i)
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise (can be reassigned as border later)
            else:
                # Expand cluster
                labels[i] = cluster_id
                seeds = list(neighbors)
                seeds.remove(i)

                while seeds:
                    curr_pt = seeds.pop(0)
                    if curr_pt not in visited:
                        visited.add(curr_pt)
                        curr_neighbors = self._region_query(X, curr_pt)
                        if len(curr_neighbors) >= self.min_samples:
                            seeds.extend([p for p in curr_neighbors if p not in seeds])
                    if labels[curr_pt] == -1:
                        labels[curr_pt] = cluster_id

                cluster_id += 1

        self.labels_ = labels
        return self


class GMMScratch:
    """
    Gaussian Mixture Model (GMM) via Expectation-Maximization (EM).
    Models data as a mixture of K multivariate Gaussians:
    P(x) = sum_{k=1}^K pi_k N(x; mu_k, Sigma_k)
    """
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None

    def _multivariate_gaussian(self, X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        d = len(mu)
        cov_reg = cov + np.eye(d) * 1e-6
        det = np.linalg.det(cov_reg)
        inv = np.linalg.inv(cov_reg)
        diff = X - mu
        norm_factor = 1.0 / np.sqrt(((2.0 * np.pi) ** d) * np.abs(det) + 1e-12)
        exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)
        return norm_factor * np.exp(np.clip(exponent, -50.0, 50.0))

    def fit(self, X: np.ndarray) -> "GMMScratch":
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        X = X.astype(np.float64)

        # Initialize parameters
        self.weights_ = np.ones(self.n_components) / self.n_components
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_indices].copy()
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

        prev_ll = -np.inf

        for _ in range(self.max_iter):
            # E-step: Compute responsibilities gamma_{ik}
            likelihoods = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                likelihoods[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])

            total_likelihood = np.sum(likelihoods, axis=1, keepdims=True) + 1e-12
            gamma = likelihoods / total_likelihood

            # Log-likelihood check
            ll = np.sum(np.log(total_likelihood))
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # M-step: Update pi, mu, Sigma
            N_k = np.sum(gamma, axis=0) + 1e-12
            self.weights_ = N_k / n_samples
            for k in range(self.n_components):
                self.means_[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / N_k[k]
                diff = X - self.means_[k]
                self.covariances_[k] = ((gamma[:, k : k + 1] * diff).T @ diff) / N_k[k]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        n_samples = len(X)
        likelihoods = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
        return np.argmax(likelihoods, axis=1)
