import pytest
from starlette.testclient import TestClient
from serving_engine import (
    MockProductionModel,
    create_app,
    DynamicMicroBatcher,
    generate_dockerfile,
    generate_k8s_deployment
)


def test_mock_production_model():
    model = MockProductionModel(version="v2.1", weights=[1.0, 2.0])
    preds = model.predict_batch([[1.0, 1.0], [0.0, 0.0]])
    assert len(preds) == 2
    assert preds[0] > preds[1]  # positive weights -> higher probability


def test_fastapi_endpoints():
    model = MockProductionModel(version="v1.0.0")
    app = create_app(model)
    client = TestClient(app)

    # 1. Health checks
    res_liveness = client.get("/healthz")
    assert res_liveness.status_code == 200
    assert res_liveness.json()["status"] == "alive"

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["model_version"] == "v1.0.0"

    # 2. Single prediction
    payload = {"features": [1.0, 0.5, -0.5, 0.2]}
    res_pred = client.post("/predict", json=payload)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "prediction" in data
    assert data["model_version"] == "v1.0.0"

    # 3. Batch prediction
    batch_payload = {
        "items": [
            {"features": [1.0, 0.0, 0.0, 0.0]},
            {"features": [0.0, 1.0, 0.0, 0.0]}
        ]
    }
    res_batch = client.post("/predict_batch", json=batch_payload)
    assert res_batch.status_code == 200
    bdata = res_batch.json()
    assert len(bdata["predictions"]) == 2
    assert bdata["batch_size"] == 2


def test_dynamic_micro_batcher():
    model = MockProductionModel()
    batcher = DynamicMicroBatcher(model, max_batch_size=4)

    # Add single requests
    preds, count = batcher.add_request([0.5, 0.5, 0.5, 0.5])
    assert count == 1
    assert len(preds) == 1


def test_manifest_generators():
    df = generate_dockerfile()
    assert "FROM python" in df
    assert "USER appuser" in df

    k8s = generate_k8s_deployment(app_name="fraud-service", replicas=5)
    assert "kind: Deployment" in k8s
    assert "replicas: 5" in k8s
    assert "livenessProbe" in k8s
    assert "readinessProbe" in k8s
