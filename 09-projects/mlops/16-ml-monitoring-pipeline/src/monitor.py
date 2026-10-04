"""Automated Data Drift and PSI Monitoring Pipeline."""
import numpy as np
from typing import Dict, Any, List

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two samples."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    c_counts, _ = np.histogram(current, bins=bin_edges)
    
    b_probs = (b_counts + 1e-4) / (len(baseline) + 1e-4 * num_bins)
    c_probs = (c_counts + 1e-4) / (len(current) + 1e-4 * num_bins)
    
    psi_value = np.sum((c_probs - b_probs) * np.log(c_probs / b_probs))
    return float(psi_value)

class DriftMonitoringPipeline:
    def __init__(self, psi_threshold: float = 0.25):
        self.psi_threshold = psi_threshold

    def evaluate_batch(self, baseline_features: Dict[str, np.ndarray], current_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        report = {}
        drift_detected = False
        
        for feat_name, base_data in baseline_features.items():
            curr_data = current_features.get(feat_name)
            if curr_data is None:
                continue
                
            psi = calculate_psi(base_data, curr_data)
            status = "CRITICAL_DRIFT" if psi >= self.psi_threshold else ("MODERATE_DRIFT" if psi >= 0.1 else "STABLE")
            if psi >= self.psi_threshold:
                drift_detected = True
                
            report[feat_name] = {
                "psi": round(psi, 4),
                "status": status
            }
            
        return {
            "drift_detected": drift_detected,
            "feature_reports": report,
            "recommended_action": "TRIGGER_RETRAINING" if drift_detected else "NONE"
        }

if __name__ == "__main__":
    np.random.seed(42)
    base = {"amount": np.random.normal(100, 20, 1000)}
    curr_clean = {"amount": np.random.normal(102, 21, 500)}
    curr_drift = {"amount": np.random.normal(180, 45, 500)}
    
    pipeline = DriftMonitoringPipeline()
    res1 = pipeline.evaluate_batch(base, curr_clean)
    print("Clean Batch Monitoring:", res1)
    res2 = pipeline.evaluate_batch(base, curr_drift)
    print("Drift Batch Monitoring:", res2)
