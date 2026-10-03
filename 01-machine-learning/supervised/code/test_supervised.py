"""
Tests for Supervised Learning algorithms from scratch.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from linear_models import (
    LinearRegressionScratch,
    RidgeRegressionScratch,
    LassoRegressionScratch,
    LogisticRegressionScratch,
)
from trees_and_neighbors import (
    KNNClassifierScratch,
    GaussianNaiveBayesScratch,
    DecisionTreeClassifierScratch,
    LinearSVMScratch,
)


def test_linear_regression_scratch():
    X, y = make_regression(n_samples=100, n_features=3, noise=5.0, random_state=42)

    # Normal equation
    model_normal = LinearRegressionScratch(solver="normal").fit(X, y)
    y_pred_normal = model_normal.predict(X)
    r2_normal = r2_score(y, y_pred_normal)
    assert r2_normal > 0.90

    # Gradient descent
    model_gd = LinearRegressionScratch(solver="gd", lr=0.01, epochs=800).fit(X, y)
    y_pred_gd = model_gd.predict(X)
    r2_gd = r2_score(y, y_pred_gd)
    assert r2_gd > 0.85

    # Compare with sklearn
    sk_model = LinearRegression().fit(X, y)
    assert np.allclose(model_normal.weights_, sk_model.coef_, atol=1e-2)


def test_ridge_regression_scratch():
    X, y = make_regression(n_samples=80, n_features=5, noise=10.0, random_state=42)
    ridge_scratch = RidgeRegressionScratch(alpha=10.0).fit(X, y)
    sk_ridge = Ridge(alpha=10.0).fit(X, y)

    # Weights should be close
    assert np.allclose(ridge_scratch.weights_, sk_ridge.coef_, atol=1e-2)
    assert np.isclose(ridge_scratch.bias_, sk_ridge.intercept_, atol=1e-2)


def test_lasso_regression_scratch():
    X, y = make_regression(n_samples=100, n_features=6, n_informative=2, noise=2.0, random_state=42)
    lasso_scratch = LassoRegressionScratch(alpha=0.5, max_iter=1500).fit(X, y)
    y_pred = lasso_scratch.predict(X)
    assert r2_score(y, y_pred) > 0.85
    # Should induce sparsity on uninformative features
    zero_weights = np.sum(np.isclose(lasso_scratch.weights_, 0.0, atol=1e-2))
    assert zero_weights >= 1


def test_logistic_regression_scratch():
    X, y = make_classification(n_samples=150, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    clf = LogisticRegressionScratch(lr=0.2, epochs=800).fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.85


def test_knn_classifier_scratch():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    knn = KNNClassifierScratch(k=5).fit(X, y)
    preds = knn.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.85


def test_gaussian_naive_bayes_scratch():
    X, y = make_classification(n_samples=150, n_features=4, n_classes=2, random_state=42)
    gnb = GaussianNaiveBayesScratch().fit(X, y)
    preds = gnb.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.80


def test_decision_tree_classifier_scratch():
    X, y = make_classification(n_samples=120, n_features=4, n_classes=2, random_state=42)
    tree = DecisionTreeClassifierScratch(max_depth=4).fit(X, y)
    preds = tree.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.85


def test_linear_svm_scratch():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    svm = LinearSVMScratch(C=1.0, lr=0.001, epochs=500).fit(X, y)
    preds = svm.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.75
