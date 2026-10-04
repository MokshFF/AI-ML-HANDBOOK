# Production MLOps & Infrastructure Tooling

Production-grade tools for containerization, workflow orchestration, serving, and observability.

## Model Serving & Runtime Acceleration
- **vLLM**: PagedAttention-powered LLM inference engine supporting continuous batching and speculative decoding.
- **Triton Inference Server**: NVIDIA enterprise server supporting dynamic micro-batching and multi-model pipelines.
- **TensorRT / ONNX Runtime**: Graph optimizations, operator fusion, and FP8/INT8 engine compilation.

## Workflow Orchestration & Pipelines
- **Kubeflow / Argo Workflows**: Kubernetes-native workflow engines for automated DAG pipelines.
- **Airflow**: Programmatic scheduling and orchestration of complex ETL and training tasks.
- **Prefect**: Modern, Python-native workflow orchestration with robust retry and error semantics.

## Observability & Monitoring
- **Prometheus + Grafana**: Standard time-series telemetry and dashboarding for latency, throughput, and GPU utilization.
- **Evidently AI**: Automated evaluation and reporting for data drift, target drift, and model degradation.
- **Arize / TruEra**: Production ML observability for embedding drift and LLM hallucination tracking.
