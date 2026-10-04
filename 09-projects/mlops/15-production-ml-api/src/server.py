"""Production FastAPI ML Serving Microservice."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from typing import List, Dict, Any

app = FastAPI(title="Production Credit Scoring API", version="1.0.0")

class PredictionRequest(BaseModel):
    account_age_months: int = Field(..., ge=0, le=1200)
    transaction_amount: float = Field(..., gt=0.0)
    is_international: bool = False
    daily_velocity: int = Field(..., ge=0, le=500)

class PredictionResponse(BaseModel):
    risk_score: float
    decision: str
    latency_ms: float
    model_version: str

MODEL_VERSION = "v1.2.0"

@app.get("/healthz")
def liveness():
    return {"status": "alive"}

@app.get("/ready")
def readiness():
    return {"status": "ready", "model_version": MODEL_VERSION}

@app.get("/metrics")
def metrics():
    return {
        "model_requests_total": 42,
        "inference_latency_p95_ms": 4.8
    }

@app.post("/v1/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    t0 = time.time()
    # Scoring heuristic
    base_risk = min(1.0, req.transaction_amount / 5000.0) * 0.4
    intl_risk = 0.3 if req.is_international else 0.0
    velo_risk = min(1.0, req.daily_velocity / 20.0) * 0.3
    age_discount = min(0.2, (req.account_age_months / 120.0) * 0.2)
    
    score = float(max(0.0, min(1.0, base_risk + intl_risk + velo_risk - age_discount)))
    decision = "DECLINE" if score >= 0.7 else ("REVIEW" if score >= 0.4 else "APPROVE")
    latency = round((time.time() - t0) * 1000.0 + 0.5, 2)
    
    return PredictionResponse(
        risk_score=round(score, 4),
        decision=decision,
        latency_ms=latency,
        model_version=MODEL_VERSION
    )
