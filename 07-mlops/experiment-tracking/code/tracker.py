"""
Lightweight, reproducible Experiment Tracking & Model Registry Engine:
Mirrors core architecture of MLflow and Weights & Biases (W&B):
1. Run lifecycle management (params, metrics across steps, artifacts, tags).
2. Experiment querying, filtering, and comparative analysis.
3. Model Registry with semantic versioning and stage transitions (Staging -> Production -> Archived).
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class MetricRecord:
    step: int
    value: float
    timestamp: float


@dataclass
class Run:
    run_id: str
    experiment_name: str
    run_name: str
    status: str = "RUNNING"  # "RUNNING", "FINISHED", "FAILED"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[MetricRecord]] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if key not in self.metrics:
            self.metrics[key] = []
        cur_step = step if step is not None else len(self.metrics[key])
        self.metrics[key].append(MetricRecord(step=cur_step, value=float(value), timestamp=time.time()))

    def log_artifact(self, name: str, artifact_obj: Any) -> None:
        self.artifacts[name] = artifact_obj

    def get_latest_metrics(self) -> Dict[str, float]:
        return {k: v[-1].value for k, v in self.metrics.items() if v}

    def finish(self, status: str = "FINISHED") -> None:
        self.status = status
        self.end_time = time.time()


class ExperimentTracker:
    """Central tracking server storing runs across experiments."""
    def __init__(self):
        self.runs: Dict[str, Run] = {}
        self.active_run: Optional[Run] = None

    def start_run(self, experiment_name: str = "default", run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Run:
        rid = f"run_{uuid.uuid4().hex[:8]}"
        rname = run_name or f"run_{len(self.runs) + 1}"
        run = Run(run_id=rid, experiment_name=experiment_name, run_name=rname, tags=tags or {})
        self.runs[rid] = run
        self.active_run = run
        return run

    def log_param(self, key: str, value: Any) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run.log_param(key, value)

    def log_params(self, params_dict: Dict[str, Any]) -> None:
        for k, v in params_dict.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run.log_metric(key, value, step)

    def log_artifact(self, name: str, artifact_obj: Any) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run.log_artifact(name, artifact_obj)

    def end_run(self, status: str = "FINISHED") -> Optional[Run]:
        if self.active_run:
            run = self.active_run
            run.finish(status)
            self.active_run = None
            return run
        return None

    def search_runs(
        self,
        experiment_name: Optional[str] = None,
        order_by_metric: Optional[str] = None,
        descending: bool = True
    ) -> List[Run]:
        results = list(self.runs.values())
        if experiment_name:
            results = [r for r in results if r.experiment_name == experiment_name]
        if order_by_metric:
            def metric_val(r: Run) -> float:
                latest = r.get_latest_metrics()
                return latest.get(order_by_metric, float("-inf") if descending else float("inf"))
            results.sort(key=metric_val, reverse=descending)
        return results

    def compare_runs(self, run_ids: Sequence[str]) -> List[Dict[str, Any]]:
        comparison = []
        for rid in run_ids:
            if rid in self.runs:
                r = self.runs[rid]
                entry = {
                    "run_id": r.run_id,
                    "run_name": r.run_name,
                    "status": r.status,
                    "params": r.params,
                    "metrics": r.get_latest_metrics()
                }
                comparison.append(entry)
        return comparison


# --------------------------------------------------------------------------- #
# Model Registry (MLflow Model Registry Pattern)
# --------------------------------------------------------------------------- #
VALID_STAGES = {"None", "Staging", "Production", "Archived"}


@dataclass
class ModelVersion:
    model_name: str
    version: int
    run_id: str
    artifact_name: str
    stage: str = "None"
    description: str = ""
    created_at: float = field(default_factory=time.time)


class ModelRegistry:
    """Central registry tracking model versions, metadata, and lifecycle stages."""
    def __init__(self):
        self.registered_models: Dict[str, List[ModelVersion]] = {}

    def register_model(
        self,
        model_name: str,
        run_id: str,
        artifact_name: str,
        description: str = ""
    ) -> ModelVersion:
        if model_name not in self.registered_models:
            self.registered_models[model_name] = []
        version = len(self.registered_models[model_name]) + 1
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            run_id=run_id,
            artifact_name=artifact_name,
            stage="None",
            description=description
        )
        self.registered_models[model_name].append(mv)
        return mv

    def transition_stage(
        self,
        model_name: str,
        version: int,
        target_stage: str,
        archive_existing_versions: bool = True
    ) -> ModelVersion:
        if target_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{target_stage}'. Must be one of {VALID_STAGES}")
        if model_name not in self.registered_models:
            raise KeyError(f"Model '{model_name}' not found in registry")

        versions = self.registered_models[model_name]
        target_mv = next((v for v in versions if v.version == version), None)
        if not target_mv:
            raise KeyError(f"Version {version} not found for model '{model_name}'")

        # If promoting to Production and archive_existing is True, demote current Production to Archived
        if target_stage == "Production" and archive_existing_versions:
            for v in versions:
                if v.stage == "Production" and v.version != version:
                    v.stage = "Archived"

        target_mv.stage = target_stage
        return target_mv

    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        if model_name not in self.registered_models:
            return None
        for v in reversed(self.registered_models[model_name]):
            if v.stage == "Production":
                return v
        return None

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        return list(self.registered_models.get(model_name, []))
