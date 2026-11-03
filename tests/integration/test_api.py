from fastapi.testclient import TestClient
import pytest

from aeromind.api.main import app

client = TestClient(app)


def test_api_health_check():
    response = client.get("/api/v1/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["service"] == "AeroMind Ops Enterprise API"


def test_api_list_fleet():
    response = client.get("/api/v1/engines?limit=5")
    assert response.status_code == 200
    engines = response.json()
    assert len(engines) <= 5
    if len(engines) > 0:
        assert "engine_id" in engines[0]
        assert "estimated_rul" in engines[0]
        assert "degradation_stage" in engines[0]


def test_api_predict_endpoint():
    # 30 cycles, 15 features
    dummy_sequence = [[0.1] * 15 for _ in range(30)]
    payload = {
        "engine_id": 42,
        "sequence": dummy_sequence,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 42
    assert "rul_point_estimate" in data
    assert "uncertainty_intervals" in data
    assert "q10" in data["uncertainty_intervals"]
    assert "q90" in data["uncertainty_intervals"]


def test_api_copilot_chat():
    payload = {"message": "Explain the current health and status of engine 1."}
    response = client.post("/api/v1/copilot/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["status"] == "SUCCESS"
    assert len(data["tool_calls"]) > 0

