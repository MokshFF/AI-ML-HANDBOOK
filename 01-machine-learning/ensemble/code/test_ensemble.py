"""
Tests for Ensemble Methods from scratch.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ensemble_models import (
    BaggingClassifierScratch,
    AdaBoostClassifierScratch,
    GradientBoostingRegressorScratch,
    StackingClassifierScratch,
)


def test_bagging_scratch():
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=42)
    bag = BaggingClassifierScratch(n_estimators=15, seed=42).fit(X, y)
    preds = bag.predict(X)

    acc = accuracy_score(y, preds)
    assert acc > 0.85
    assert bag.oob_score_ > 0.70


def test_adaboost_scratch():
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=42)
    ada = AdaBoostClassifierScratch(n_estimators=30, lr=0.5).fit(X, y)
    preds = ada.predict(X)

    acc = accuracy_score(y, preds)
    assert acc > 0.85
    assert len(ada.stumps) > 0


def test_gradient_boosting_regressor_scratch():
    X, y = make_regression(n_samples=120, n_features=4, noise=5.0, random_state=42)
    gbr = GradientBoostingRegressorScratch(n_estimators=40, lr=0.1, max_depth=3).fit(X, y)
    preds = gbr.predict(X)

    r2 = r2_score(y, preds)
    assert r2 > 0.85


def test_stacking_classifier_scratch():
    X, y = make_classification(n_samples=150, n_features=6, random_state=42)
    base_models = [
        DecisionTreeClassifier(max_depth=3),
        KNeighborsClassifier(n_neighbors=5)
    ]
    meta_model = LogisticRegression()

    stack = StackingClassifierScratch(base_models=base_models, meta_model=meta_model, n_splits=3)
    stack.fit(X, y)
    preds = stack.predict(X)

    acc = accuracy_score(y, preds)
    assert acc > 0.85
