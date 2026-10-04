# MLOps Experiment Tracking Interview Questions & Answers

### Q1: Why is Git insufficient for tracking machine learning experiments?
**Answer:**
1. **Large Binary Weights**: Storing gigabyte-scale neural network weight checkpoints (`.pt`, `.bin`) bloats Git repositories, making cloning and diffing prohibitively slow.
2. **Non-Code Variables**: ML results depend on code, data, hyperparameters, hardware configurations (CUDA version, GPU architecture), and stochastic seeds. Git only tracks source code.
3. **Dynamic Metric Streams**: Git cannot record high-frequency scalar metrics (step-by-step training loss curves, GPU memory utilization) or visualize metric trajectories.
4. **Specialized Tooling**: Tools like MLflow and Weights & Biases (W&B) store lightweight metadata in relational databases and large binary artifacts in object stores (S3/GCS), linking each run to the specific Git commit hash for full auditability.

---

### Q2: What are the three components of MLflow, and how do they interact?
**Answer:**
1. **MLflow Tracking**: An API and UI for logging parameters, code versions, metrics, and artifacts during ML code execution.
2. **MLflow Models / Projects**: A standard packaging format for model dependencies (`conda.yaml`, `requirements.txt`) and unified flavors (PyTorch, Scikit-learn, ONNX) allowing models to be loaded uniformly anywhere.
3. **MLflow Model Registry**: A centralized model store with APIs and UI for managing the full lifecycle of ML models, including versioning, stage transitions (Staging -> Production -> Archived), and approval annotations.

---

### Q3: How do you design an automated promotion pipeline from Staging to Production in a Model Registry?
**Answer:**
1. **Continuous Integration (CI)**: When a training run concludes, the candidate model is registered in stage `None`.
2. **Automated Offline Gates**: A CI job loads the candidate model and runs automated evaluation tests:
   - Performance gate: Candidate AUC/F1 must be $\ge$ Production Champion baseline by $+0.5\%$.
   - Safety/Fairness gate: Demographic parity and slice evaluation tests must pass.
   - Latency/SLA gate: 99th percentile inference latency on target hardware must be under 50 ms.
3. **Staging Promotion**: If all offline gates pass, the registry automatically transitions the model to `Staging`.
4. **Canary / Shadow Testing**: The model receives 5% of production traffic or runs in shadow mode (asynchronous scoring without returning predictions to users).
5. **Production Promotion**: If error rates and business KPIs remain stable over 48 hours, the candidate is promoted to `Production`, and the previous version is transitioned to `Archived`.
