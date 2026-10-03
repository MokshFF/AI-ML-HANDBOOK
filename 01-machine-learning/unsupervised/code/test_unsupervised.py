"""
Tests for Unsupervised Learning algorithms from scratch.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score

from clustering import KMeansScratch, DBSCANScratch, GMMScratch
from dim_reduction_and_anomaly import PCAScratch, TSNEScratch, IsolationForestScratch


def test_kmeans_scratch():
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
    kmeans = KMeansScratch(n_clusters=3, seed=42).fit(X)
    labels = kmeans.predict(X)

    ari = adjusted_rand_score(y, labels)
    assert ari > 0.85
    assert kmeans.inertia_ > 0


def test_dbscan_scratch():
    X, y = make_moons(n_samples=150, noise=0.08, random_state=42)
    dbscan = DBSCANScratch(eps=0.25, min_samples=5).fit(X)
    labels = dbscan.labels_

    # Should detect 2 clusters and minimal noise
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 2
    ari = adjusted_rand_score(y, labels)
    assert ari > 0.85


def test_gmm_scratch():
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=0.8, random_state=42)
    gmm = GMMScratch(n_components=3, max_iter=100, seed=42).fit(X)
    labels = gmm.predict(X)

    ari = adjusted_rand_score(y, labels)
    assert ari > 0.85
    assert np.allclose(np.sum(gmm.weights_), 1.0)


def test_pca_scratch():
    X, _ = make_blobs(n_samples=100, n_features=5, random_state=42)
    pca = PCAScratch(n_components=2).fit(X)
    X_proj = pca.transform(X)

    assert X_proj.shape == (100, 2)
    assert np.sum(pca.explained_variance_ratio_) <= 1.0


def test_tsne_scratch():
    X, _ = make_blobs(n_samples=60, centers=2, n_features=4, random_state=42)
    tsne = TSNEScratch(n_components=2, n_iter=100)
    embedding = tsne.fit_transform(X)

    assert embedding.shape == (60, 2)
    assert not np.isnan(embedding).any()


def test_isolation_forest_scratch():
    np.random.seed(42)
    # 100 inliers around origin
    X_inliers = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    # 5 extreme outliers far away
    X_outliers = np.random.uniform(low=8.0, high=12.0, size=(5, 2))
    X_all = np.vstack([X_inliers, X_outliers])

    iso = IsolationForestScratch(n_estimators=40, seed=42).fit(X_all)
    scores = iso.anomaly_score(X_all)

    # Outliers should have higher anomaly scores than inliers
    inlier_scores = scores[:100]
    outlier_scores = scores[100:]
    assert np.mean(outlier_scores) > np.mean(inlier_scores)
