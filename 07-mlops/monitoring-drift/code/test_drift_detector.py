import numpy as np
import pytest
from drift_detector import (
    kolmogorov_smirnov_2sample,
    population_stability_index,
    wasserstein_distance_1d,
    DataDriftDetector,
    ConceptDriftDetector,
    PrometheusMetricsExporter
)


def test_ks_and_psi_identical_distributions():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=500).tolist()
    cur = np.random.normal(loc=0.0, scale=1.0, size=500).tolist()

    d_stat, p_val = kolmogorov_smirnov_2sample(ref, cur)
    psi = population_stability_index(ref, cur)
    w_dist = wasserstein_distance_1d(ref, cur)

    assert d_stat < 0.15
    assert p_val > 0.05       # no significant difference
    assert psi < 0.10         # stable population
    assert w_dist < 0.15


def test_ks_and_psi_shifted_distributions():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=500).tolist()
    cur = np.random.normal(loc=2.5, scale=1.0, size=500).tolist()  # significant mean shift

    d_stat, p_val = kolmogorov_smirnov_2sample(ref, cur)
    psi = population_stability_index(ref, cur)
    w_dist = wasserstein_distance_1d(ref, cur)

    assert d_stat > 0.50
    assert p_val < 0.001      # significant drift detected!
    assert psi > 0.25         # action required threshold!
    assert w_dist > 2.0


def test_data_drift_detector():
    np.random.seed(0)
    baseline = {
        "income": np.random.normal(50000, 10000, 400).tolist(),
        "age": np.random.normal(40, 10, 400).tolist()
    }
    detector = DataDriftDetector(baseline, p_value_threshold=0.01, psi_threshold=0.25)

    # Batch with drifted income but normal age
    production_batch = {
        "income": np.random.normal(85000, 15000, 200).tolist(),  # shifted!
        "age": np.random.normal(40, 10, 200).tolist()             # unchanged
    }

    report = detector.evaluate_batch(production_batch)
    assert report["income"].is_drifted is True
    assert report["age"].is_drifted is False


def test_concept_drift_detector():
    # Baseline model accuracy is 0.90
    detector = ConceptDriftDetector(baseline_metric=0.90, threshold_drop=0.15, window_size=50)

    # 40 correct predictions
    for _ in range(40):
        detector.log_prediction(1.0)

    # Performance degrades to 0.50 accuracy
    alert = None
    for _ in range(50):
        res = detector.log_prediction(0.50)
        if res:
            alert = res
            break

    assert alert is not None
    assert alert["alert"] == "CONCEPT_DRIFT_DETECTED"
    assert alert["current"] <= 0.75


def test_prometheus_exporter():
    exporter = PrometheusMetricsExporter("fraud_detector", "v1.2.0")
    for lat in [12.0, 15.0, 14.0, 18.0, 35.0]:
        exporter.record_request(latency_ms=lat, has_drift=False)
    exporter.record_request(latency_ms=50.0, has_drift=True)

    metrics_text = exporter.export_metrics()
    assert "ml_requests_total" in metrics_text
    assert 'version="v1.2.0"' in metrics_text
    assert "ml_drift_alerts_total" in metrics_text
    assert " 1\n" in metrics_text  # 1 drift alert
