"""Tests for House Price Prediction Pipeline."""
import pytest
import numpy as np
from housing_data_loader import generate_housing_data
from model import HousePricePredictor

def test_data_loader():
    df = generate_housing_data(100)
    assert len(df) == 100
    assert "sale_price" in df.columns
    assert (df["sale_price"] > 0).all()

def test_house_price_pipeline():
    df = generate_housing_data(500, seed=123)
    train_df = df.iloc[:400]
    test_df = df.iloc[400:]
    
    predictor = HousePricePredictor(alpha=5.0)
    predictor.fit(train_df)
    
    preds = predictor.predict(test_df)
    assert len(preds) == 100
    assert not np.isnan(preds).any()
    
    metrics = predictor.evaluate(test_df)
    assert metrics["r2"] > 0.70
    assert metrics["mae"] < 50000.0
