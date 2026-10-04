# Cheat Sheet: Production MLOps

| Pillar | Industry Tools | Key Requirements | Production Checklist |
| :--- | :--- | :--- | :--- |
| **Experiment Tracking** | MLflow, Weights & Biases | Log hyperparameters, git commit hash, artifacts | Reproducibility guarantee |
| **Data Versioning** | DVC, LakeFS | Content-addressable storage, pointer manifests | Immutable data lineage |
| **Feature Store** | Feast, Tecton | Dual storage: Redis (online) + Snowflake (offline) | Point-in-time join (no time-travel leakage) |
| **Serving** | FastAPI, Triton Inference Server, TorchServe | Healthz/ready probes, dynamic micro-batching | P99 latency SLA $< 20\text{ ms}$ |
| **CI/CD for ML** | GitHub Actions, CML | Automated model gate: Challenger must beat Champion | Fails build if slice accuracy regresses |
| **Drift Monitoring** | Evidently, Prometheus, Grafana | Kolmogorov-Smirnov test, Population Stability Index | PSI $\ge 0.25$ triggers automated retraining alert |
