"""Tests for Drift Monitoring Pipeline."""
import numpy as np
from monitor import calculate_psi, DriftMonitoringPipeline

def test_psi_identical():
    np.random.seed(123)
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.02

def test_drift_pipeline():
    np.random.seed(42)
    base = {"feature_a": np.random.normal(50, 10, 800)}
    shifted = {"feature_a": np.random.normal(90, 25, 400)}
    
    pipeline = DriftMonitoringPipeline(psi_threshold=0.25)
    res = pipeline.evaluate_batch(base, shifted)
    assert res["drift_detected"] is True
    assert res["recommended_action"] == "TRIGGER_RETRAINING"
