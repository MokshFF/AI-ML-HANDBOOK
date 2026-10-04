import pytest
from ml_pipeline_ci import (
    DataValidator,
    evaluate_model_gate,
    generate_github_actions_workflow
)


def test_data_validator_expectations():
    validator = DataValidator()
    data = [
        {"age": 25, "income": 50000, "status": "active"},
        {"age": 42, "income": 80000, "status": "pending"},
        {"age": 31, "income": 65000, "status": "active"}
    ]

    assert validator.expect_column_values_to_not_be_null(data, "age") is True
    assert validator.expect_column_values_to_be_between(data, "age", 18, 70) is True
    assert validator.expect_column_values_to_be_in_set(data, "status", {"active", "pending", "closed"}) is True
    assert validator.is_all_passed() is True

    # Test failure case (age out of bounds)
    bad_data = [{"age": 12, "income": 50000, "status": "active"}]
    assert validator.expect_column_values_to_be_between(bad_data, "age", 18, 70) is False
    assert validator.is_all_passed() is False


def test_model_gate_evaluation():
    test_rows = [
        {"device_type": "ios", "feat": 1.0, "label": 1},
        {"device_type": "ios", "feat": 0.0, "label": 0},
        {"device_type": "android", "feat": 1.0, "label": 1},
        {"device_type": "android", "feat": 0.0, "label": 0}
    ]

    # Champion model achieves 75% accuracy
    def champ(r):
        return 0.8 if r["feat"] == 1.0 and r["device_type"] == "ios" else 0.4

    # Superior Challenger model achieves 100% accuracy
    def challenger_good(r):
        return 0.9 if r["feat"] == 1.0 else 0.1

    gate_pass = evaluate_model_gate(champ, challenger_good, test_rows, slice_key="device_type")
    assert gate_pass.promoted is True
    assert gate_pass.overall_challenger_metric > gate_pass.overall_champ_metric

    # Inferior Challenger model fails gate
    def challenger_bad(r):
        return 0.2  # predicts 0 everywhere

    gate_fail = evaluate_model_gate(champ, challenger_bad, test_rows, slice_key="device_type")
    assert gate_fail.promoted is False
    assert len(gate_fail.reasons) > 0


def test_github_actions_workflow_generation():
    wf = generate_github_actions_workflow()
    assert "name: ML CI/CD Pipeline" in wf
    assert "pytest tests/ -v" in wf
    assert "docker build" in wf
