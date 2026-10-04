"""Tests for Customer Churn Prediction Engine."""
import pytest
import numpy as np
from churn_data_loader import generate_churn_data
from churn_model import CustomerChurnModel

def test_churn_data_generator():
    df = generate_churn_data(100)
    assert len(df) == 100
    assert set(df["churn"].unique()).issubset({0, 1})

def test_churn_model_fit_and_predict():
    df = generate_churn_data(600, seed=99)
    train = df.iloc[:500]
    test = df.iloc[500:]
    
    model = CustomerChurnModel(lr=0.1, n_epochs=150)
    model.fit(train)
    
    probs = model.predict_proba(test)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    metrics = model.evaluate(test)
    assert metrics["accuracy"] > 0.65
