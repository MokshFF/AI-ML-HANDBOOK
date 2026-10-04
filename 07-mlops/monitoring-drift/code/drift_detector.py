"""
Production Model Monitoring & Drift Detection Engine:
1. Statistical Data Drift Metrics: Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), Wasserstein Distance.
2. Concept Drift Detector: Tracking performance degradation and conditional distribution shift P(Y|X).
3. Telemetry & Prometheus metric exporter for MLOps observability.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Statistical Drift Metrics
# --------------------------------------------------------------------------- #
def kolmogorov_smirnov_2sample(reference: Sequence[float], current: Sequence[float]) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous feature drift.
    Computes D = sup_x |F_ref(x) - F_cur(x)| and asymptotic p-value.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    n1, n2 = len(ref), len(cur)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combined sorted values
    combined = np.concatenate([ref, cur])
    # Empirical CDFs
    cdf_ref = np.searchsorted(ref, combined, side="right") / n1
    cdf_cur = np.searchsorted(cur, combined, side="right") / n2

    d_stat = float(np.max(np.abs(cdf_ref - cdf_cur)))
    # Asymptotic p-value approximation (Smirnov distribution)
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    # Kolmogorov distribution approximation
    if lambda_val <= 0:
        p_val = 1.0
    else:
        p_val = 2.0 * math.exp(-2.0 * lambda_val * lambda_val)
        p_val = max(0.0, min(1.0, p_val))
    return d_stat, p_val


def population_stability_index(reference: Sequence[float], current: Sequence[float], num_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) using quantile binning on reference:
    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)
    Thresholds:
      PSI < 0.10: No significant shift
      0.10 <= PSI < 0.25: Moderate shift (warning)
      PSI >= 0.25: Significant shift (retrain trigger)
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Quantile bin edges based on reference
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    # Deduplicate edges if distribution has repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Fractions with Laplace smoothing
    ref_fractions = (ref_counts + 1e-4) / (len(ref) + 1e-4 * len(ref_counts))
    cur_fractions = (cur_counts + 1e-4) / (len(cur) + 1e-4 * len(cur_counts))

    psi = np.sum((cur_fractions - ref_fractions) * np.log(cur_fractions / ref_fractions))
    return float(psi)


def wasserstein_distance_1d(reference: Sequence[float], current: Sequence[float]) -> float:
    """Computes 1D Wasserstein-1 (Earth Mover's Distance) between two sample sets."""
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    combined = np.sort(np.unique(np.concatenate([ref, cur])))
    if len(combined) <= 1:
        return 0.0

    cdf_ref = np.searchsorted(ref, combined, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, combined, side="right") / len(cur)
    deltas = np.diff(combined)
    return float(np.sum(np.abs(cdf_ref[:-1] - cdf_cur[:-1]) * deltas))


# --------------------------------------------------------------------------- #
# 2. End-to-End Drift & Concept Degradation Engine
# --------------------------------------------------------------------------- #
@dataclass
class FeatureDriftSummary:
    feature_name: str
    ks_stat: float
    ks_pvalue: float
    psi: float
    wasserstein: float
    is_drifted: bool


class DataDriftDetector:
    """Monitors incoming production features against a golden training baseline."""
    def __init__(self, reference_data: Dict[str, Sequence[float]], p_value_threshold: float = 0.05, psi_threshold: float = 0.25):
        self.reference = reference_data
        self.p_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def evaluate_batch(self, current_batch: Dict[str, Sequence[float]]) -> Dict[str, FeatureDriftSummary]:
        results = {}
        for feat, cur_vals in current_batch.items():
            if feat not in self.reference:
                continue
            ref_vals = self.reference[feat]
            ks_stat, p_val = kolmogorov_smirnov_2sample(ref_vals, cur_vals)
            psi = population_stability_index(ref_vals, cur_vals)
            w_dist = wasserstein_distance_1d(ref_vals, cur_vals)
            is_drifted = (p_val < self.p_threshold) or (psi >= self.psi_threshold)
            results[feat] = FeatureDriftSummary(
                feature_name=feat,
                ks_stat=round(ks_stat, 4),
                ks_pvalue=round(p_val, 4),
                psi=round(psi, 4),
                wasserstein=round(w_dist, 4),
                is_drifted=is_drifted
            )
        return results


class ConceptDriftDetector:
    """Detects concept drift / model degradation by monitoring rolling error metrics."""
    def __init__(self, baseline_metric: float, threshold_drop: float = 0.10, window_size: int = 100):
        self.baseline_metric = baseline_metric
        self.threshold_drop = threshold_drop
        self.window_size = window_size
        self.window: List[float] = []

    def log_prediction(self, is_correct: float) -> Optional[Dict[str, Any]]:
        self.window.append(is_correct)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if len(self.window) >= self.window_size:
            current_metric = sum(self.window) / len(self.window)
            drop = self.baseline_metric - current_metric
            if drop >= self.threshold_drop:
                return {
                    "alert": "CONCEPT_DRIFT_DETECTED",
                    "baseline": self.baseline_metric,
                    "current": round(current_metric, 4),
                    "drop": round(drop, 4)
                }
        return None


# --------------------------------------------------------------------------- #
# 3. Observability & Prometheus Metrics Exporter
# --------------------------------------------------------------------------- #
class PrometheusMetricsExporter:
    """Formats serving and drift telemetry into Prometheus exposition format."""
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.request_count = 0
        self.drift_alerts = 0
        self.latencies: List[float] = []

    def record_request(self, latency_ms: float, has_drift: bool = False):
        self.request_count += 1
        self.latencies.append(latency_ms)
        if has_drift:
            self.drift_alerts += 1

    def export_metrics(self) -> str:
        p95 = np.percentile(self.latencies, 95) if self.latencies else 0.0
        return f"""# HELP ml_requests_total Total number of inference requests
# TYPE ml_requests_total counter
ml_requests_total{{model="{self.model_name}",version="{self.model_version}"}} {self.request_count}

# HELP ml_latency_p95_ms 95th percentile inference latency in ms
# TYPE ml_latency_p95_ms gauge
ml_latency_p95_ms{{model="{self.model_name}",version="{self.model_version}"}} {p95:.2f}

# HELP ml_drift_alerts_total Total drift alerts fired
# TYPE ml_drift_alerts_total counter
ml_drift_alerts_total{{model="{self.model_name}",version="{self.model_version}"}} {self.drift_alerts}
"""
