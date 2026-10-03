# MLOps & Production Engineering (`07-mlops`)

## Overview
Lifecycle management, experiment tracking, continuous integration/deployment, serving infrastructure, and monitoring.

## Subtopics & Navigation
| Directory | Topic | Scope |
| :--- | :--- | :--- |
| [`experiment-tracking/`](./experiment-tracking/) | **Experiment Tracking** | Metric logging, artifact storage, hyperparameter comparison with tools like MLflow and Weights & Biases. |
| [`data-versioning/`](./data-versioning/) | **Data Versioning** | Data lineage, dataset hashing, and dataset versioning with tools such as DVC. |
| [`serving/`](./serving/) | **Serving** | Model packaging, REST/gRPC inference APIs, FastAPI, Triton Inference Server, and TorchServe. |
| [`ci-cd-for-ml/`](./ci-cd-for-ml/) | **CI CD For ML** | Automated model testing, linting, regression testing, continuous integration, and automated deployment pipelines. |
| [`monitoring-drift/`](./monitoring-drift/) | **Monitoring Drift** | Data drift, concept drift detection, performance degradation alerts, and continuous evaluation. |
| [`llmops/`](./llmops/) | **Llmops** | Prompt versioning, LLM gateway management, rate limiting, cost tracking, caching, and latency telemetry. |

## Standard Directory Schema
Every topic directory in this module follows our standard five-component structure:
- `README.md` - Module introduction, learning objectives, and concept matrix
- `notebook.ipynb` - Reproducible, runnable interactive notebook
- `code/` - Clean, modular Python scripts and helper utilities
- `interview.md` - Technical screening questions, edge cases, and design discussions
- `references.md` - Research papers, textbooks, and documentation

## Prerequisites
Before beginning this module, review:
- Foundational math and coding prerequisites in [`../00-prerequisites/`](../00-prerequisites/)
- The end-to-end learning pathways defined in [`../ROADMAP.md`](../ROADMAP.md)
