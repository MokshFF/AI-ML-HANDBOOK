"""
CI/CD for Machine Learning (MLOps CI/CD):
1. Data Validation Suite (Great Expectations pattern).
2. Challenger vs Champion Model Qualification Gate with Slice Testing.
3. Continuous Training (CT) triggers and GitHub Actions workflow synthesis.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 1. Data Validation Suite (Great Expectations Pattern)
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    expectation_name: str
    column: str
    success: bool
    details: str


class DataValidator:
    """Runs declarative schema and data quality expectations on incoming datasets."""
    def __init__(self):
        self.results: List[ValidationResult] = []

    def expect_column_values_to_not_be_null(self, data: Sequence[Dict[str, Any]], column: str) -> bool:
        null_count = sum(1 for row in data if row.get(column) is None)
        success = null_count == 0
        self.results.append(ValidationResult(
            expectation_name="expect_column_values_to_not_be_null",
            column=column,
            success=success,
            details=f"Found {null_count} nulls out of {len(data)} rows"
        ))
        return success

    def expect_column_values_to_be_between(
        self,
        data: Sequence[Dict[str, Any]],
        column: str,
        min_val: float,
        max_val: float
    ) -> bool:
        violations = [row[column] for row in data if column in row and row[column] is not None and not (min_val <= row[column] <= max_val)]
        success = len(violations) == 0
        self.results.append(ValidationResult(
            expectation_name="expect_column_values_to_be_between",
            column=column,
            success=success,
            details=f"{len(violations)} violations outside [{min_val}, {max_val}]"
        ))
        return success

    def expect_column_values_to_be_in_set(
        self,
        data: Sequence[Dict[str, Any]],
        column: str,
        allowed_set: set
    ) -> bool:
        violations = [row[column] for row in data if column in row and row[column] not in allowed_set]
        success = len(violations) == 0
        self.results.append(ValidationResult(
            expectation_name="expect_column_values_to_be_in_set",
            column=column,
            success=success,
            details=f"{len(violations)} invalid categories detected"
        ))
        return success

    def is_all_passed(self) -> bool:
        return all(r.success for r in self.results)


# --------------------------------------------------------------------------- #
# 2. Challenger vs Champion Model Gate & Slice Testing
# --------------------------------------------------------------------------- #
@dataclass
class SliceEvaluation:
    slice_name: str
    champion_metric: float
    challenger_metric: float
    passed: bool


@dataclass
class ModelGateDecision:
    promoted: bool
    overall_champ_metric: float
    overall_challenger_metric: float
    slice_evaluations: List[SliceEvaluation]
    reasons: List[str]


def evaluate_model_gate(
    champion_predict_fn: Callable[[Dict[str, Any]], float],
    challenger_predict_fn: Callable[[Dict[str, Any]], float],
    test_dataset: Sequence[Dict[str, Any]],
    label_key: str = "label",
    slice_key: str = "device_type",
    metric_threshold_gain: float = 0.0  # challenger must match or exceed champion
) -> ModelGateDecision:
    """
    Evaluates Candidate (Challenger) vs Production (Champion):
    1. Overall accuracy / score must meet threshold gain.
    2. Critical slices (e.g. iOS vs Android, demographic groups) must not regress significantly (>2%).
    """
    reasons = []

    # Overall calculation
    def calc_acc(pred_fn, rows):
        if not rows:
            return 0.0
        correct = 0
        for r in rows:
            p = 1 if pred_fn(r) >= 0.5 else 0
            correct += (p == r[label_key])
        return correct / len(rows)

    champ_overall = calc_acc(champion_predict_fn, test_dataset)
    challenger_overall = calc_acc(challenger_predict_fn, test_dataset)

    overall_passed = challenger_overall >= (champ_overall + metric_threshold_gain)
    if not overall_passed:
        reasons.append(f"Challenger overall accuracy ({challenger_overall:.4f}) failed to beat champion ({champ_overall:.4f})")

    # Slice evaluation
    slices = sorted(set(r[slice_key] for r in test_dataset if slice_key in r))
    slice_results = []
    for s in slices:
        subset = [r for r in test_dataset if r.get(slice_key) == s]
        c_score = calc_acc(champion_predict_fn, subset)
        ch_score = calc_acc(challenger_predict_fn, subset)
        s_passed = ch_score >= (c_score - 0.02)  # max allowable regression on slice is 2%
        if not s_passed:
            reasons.append(f"Slice '{s}' regressed from {c_score:.4f} to {ch_score:.4f}")
        slice_results.append(SliceEvaluation(slice_name=str(s), champion_metric=round(c_score, 4), challenger_metric=round(ch_score, 4), passed=s_passed))

    promoted = overall_passed and all(sr.passed for sr in slice_results)
    return ModelGateDecision(
        promoted=promoted,
        overall_champ_metric=round(champ_overall, 4),
        overall_challenger_metric=round(challenger_overall, 4),
        slice_evaluations=slice_results,
        reasons=reasons
    )


# --------------------------------------------------------------------------- #
# 3. GitHub Actions Workflow Generator
# --------------------------------------------------------------------------- #
def generate_github_actions_workflow() -> str:
    return """name: ML CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  validate-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8

    - name: Lint code & formatting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Run Unit Tests
      run: |
        pytest tests/ -v

    - name: Validate Training Data Expectations
      run: |
        python -m scripts.validate_data --dataset data/training/current.parquet

    - name: Evaluate Challenger vs Champion Gate
      run: |
        python -m scripts.evaluate_gate --candidate-run-id ${{ github.sha }}

    - name: Build & Push Inference Docker Image
      if: github.ref == 'refs/heads/main'
      run: |
        docker build -t registry.internal/ml-service:${{ github.sha }} .
        docker push registry.internal/ml-service:${{ github.sha }}
"""
