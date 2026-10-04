import pytest
from tracker import ExperimentTracker, ModelRegistry


def test_experiment_tracker_run_lifecycle():
    tracker = ExperimentTracker()

    run = tracker.start_run(experiment_name="churn_prediction", run_name="xgb_baseline", tags={"env": "dev"})
    assert run.status == "RUNNING"

    tracker.log_params({"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100})
    tracker.log_metric("loss", 0.65, step=1)
    tracker.log_metric("loss", 0.42, step=2)
    tracker.log_metric("val_auc", 0.88, step=2)
    tracker.log_artifact("model_weights", {"weights": [0.1, 0.2, 0.3]})

    finished_run = tracker.end_run(status="FINISHED")
    assert finished_run.status == "FINISHED"
    assert finished_run.end_time is not None

    latest_metrics = finished_run.get_latest_metrics()
    assert latest_metrics["loss"] == 0.42
    assert latest_metrics["val_auc"] == 0.88
    assert finished_run.params["max_depth"] == 6


def test_search_and_compare_runs():
    tracker = ExperimentTracker()

    r1 = tracker.start_run("sentiment_analysis", "distilbert_lr1e4")
    tracker.log_param("lr", 1e-4)
    tracker.log_metric("accuracy", 0.89)
    tracker.end_run()

    r2 = tracker.start_run("sentiment_analysis", "roberta_lr2e5")
    tracker.log_param("lr", 2e-5)
    tracker.log_metric("accuracy", 0.94)
    tracker.end_run()

    # Search runs sorted by accuracy
    sorted_runs = tracker.search_runs("sentiment_analysis", order_by_metric="accuracy", descending=True)
    assert len(sorted_runs) == 2
    assert sorted_runs[0].run_id == r2.run_id
    assert sorted_runs[1].run_id == r1.run_id

    # Comparison table
    cmp = tracker.compare_runs([r1.run_id, r2.run_id])
    assert len(cmp) == 2
    assert cmp[0]["metrics"]["accuracy"] == 0.89
    assert cmp[1]["metrics"]["accuracy"] == 0.94


def test_model_registry_versioning_and_stage_transitions():
    registry = ModelRegistry()

    # 1. Register version 1
    v1 = registry.register_model(
        model_name="fraud_detector",
        run_id="run_abc123",
        artifact_name="model.onnx",
        description="Initial logistic baseline"
    )
    assert v1.version == 1
    assert v1.stage == "None"

    # Transition v1 to Staging then Production
    registry.transition_stage("fraud_detector", version=1, target_stage="Staging")
    assert v1.stage == "Staging"

    registry.transition_stage("fraud_detector", version=1, target_stage="Production")
    assert v1.stage == "Production"
    assert registry.get_production_model("fraud_detector").version == 1

    # 2. Register version 2 and promote to Production
    v2 = registry.register_model(
        model_name="fraud_detector",
        run_id="run_def456",
        artifact_name="model_v2.onnx",
        description="LightGBM model"
    )
    assert v2.version == 2
    registry.transition_stage("fraud_detector", version=2, target_stage="Production", archive_existing_versions=True)

    # v2 is now Production; v1 should be automatically archived!
    assert v2.stage == "Production"
    assert v1.stage == "Archived"
    assert registry.get_production_model("fraud_detector").version == 2
