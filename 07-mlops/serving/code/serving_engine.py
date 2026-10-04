"""
Production ML Serving Engine:
1. FastAPI application with Pydantic request validation and Kubernetes liveness/readiness probes.
2. Dynamic Micro-Batching background queue (batching requests within latency SLA).
3. Dockerfile and Kubernetes deployment manifest generators.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 1. Pydantic Schemas & ML Model Wrapper
# --------------------------------------------------------------------------- #
class PredictionItem(BaseModel):
    features: List[float] = Field(..., description="Feature vector")


class SinglePredictionRequest(BaseModel):
    features: List[float]


class BatchPredictionRequest(BaseModel):
    items: List[PredictionItem]


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    batch_size: int
    latency_ms: float


class MockProductionModel:
    """Mock ML Model supporting single and batch vector inference."""
    def __init__(self, version: str = "v1.0.0", weights: Optional[List[float]] = None):
        self.version = version
        self.weights = weights or [0.5, -0.2, 0.8, 0.1]
        self.is_ready = True

    def predict_batch(self, feature_matrix: List[List[float]]) -> List[float]:
        # Vector dot product + sigmoid
        results = []
        for vec in feature_matrix:
            score = sum(w * (v if i < len(vec) else 0.0) for i, (w, v) in enumerate(zip(self.weights, vec)))
            prob = 1.0 / (1.0 + pow(2.71828, -score))
            results.append(round(prob, 4))
        return results


# --------------------------------------------------------------------------- #
# 2. Dynamic Micro-Batching Engine
# --------------------------------------------------------------------------- #
@dataclass
class BatchedRequest:
    features: List[float]
    response_future: Any


class DynamicMicroBatcher:
    """
    Batches concurrent inference requests arriving within max_wait_ms window,
    maximizing GPU/CPU hardware throughput without exceeding latency SLAs.
    """
    def __init__(self, model: MockProductionModel, max_batch_size: int = 16, max_wait_ms: float = 20.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: List[Tuple[List[float], Any]] = []

    def add_request(self, features: List[float]) -> Tuple[float, int]:
        """Synchronous teaching simulation of batch collection and execution."""
        self.queue.append((features, None))
        if len(self.queue) >= self.max_batch_size:
            return self.flush_sync()
        return self.flush_sync()

    def flush_sync(self) -> Tuple[List[float], int]:
        if not self.queue:
            return [], 0
        batch = self.queue[:]
        self.queue.clear()
        feature_matrix = [item[0] for item in batch]
        predictions = self.model.predict_batch(feature_matrix)
        return predictions, len(batch)


# --------------------------------------------------------------------------- #
# 3. FastAPI App Builder
# --------------------------------------------------------------------------- #
def create_app(model: Optional[MockProductionModel] = None):
    """Creates a FastAPI application with health probes and inference endpoints."""
    from fastapi import FastAPI, HTTPException, status
    app = FastAPI(title="ML Model Serving API", version="1.0.0")
    m = model or MockProductionModel()

    @app.get("/healthz", status_code=status.HTTP_200_OK)
    def liveness():
        """Liveness probe: verifies process is alive."""
        return {"status": "alive"}

    @app.get("/ready", status_code=status.HTTP_200_OK)
    def readiness():
        """Readiness probe: verifies model is loaded and ready to serve."""
        if not m.is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model loading")
        return {"status": "ready", "model_version": m.version}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: SinglePredictionRequest):
        t0 = time.perf_counter()
        preds = m.predict_batch([req.features])
        lat = (time.perf_counter() - t0) * 1000.0
        return PredictionResponse(prediction=preds[0], model_version=m.version, latency_ms=round(lat, 2))

    @app.post("/predict_batch", response_model=BatchPredictionResponse)
    def predict_batch(req: BatchPredictionRequest):
        t0 = time.perf_counter()
        mat = [item.features for item in req.items]
        preds = m.predict_batch(mat)
        lat = (time.perf_counter() - t0) * 1000.0
        return BatchPredictionResponse(
            predictions=preds,
            model_version=m.version,
            batch_size=len(preds),
            latency_ms=round(lat, 2)
        )

    return app


# --------------------------------------------------------------------------- #
# 4. Kubernetes Manifest & Dockerfile Templates
# --------------------------------------------------------------------------- #
def generate_dockerfile() -> str:
    return """# Multi-stage production ML serving Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .
# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]"""


def generate_k8s_deployment(app_name: str = "ml-service", replicas: int = 3) -> str:
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: registry.internal/{app_name}:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  type: ClusterIP
  selector:
    app: {app_name}
  ports:
  - port: 80
    targetPort: 8000"""
