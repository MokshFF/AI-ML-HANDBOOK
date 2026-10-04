"""Tests for Production ML Serving API."""
from starlette.testclient import TestClient
from server import app

client = TestClient(app)

def test_health_endpoints():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"

def test_predict_success():
    payload = {
        "account_age_months": 24,
        "transaction_amount": 150.0,
        "is_international": False,
        "daily_velocity": 2
    }
    res = client.post("/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["decision"] in ["APPROVE", "REVIEW", "DECLINE"]

def test_predict_validation_error():
    bad_payload = {
        "account_age_months": -5,  # Violates ge=0
        "transaction_amount": 10.0,
        "daily_velocity": 1
    }
    res = client.post("/v1/predict", json=bad_payload)
    assert res.status_code == 422
