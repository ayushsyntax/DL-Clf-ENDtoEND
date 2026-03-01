"""Integration tests for FastAPI inference endpoints."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_check_reachable():
    """Health endpoint must return 200 with a status field regardless of model state."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health_check_schema():
    """Health response must contain both status and model_loaded fields."""
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_without_file_returns_422():
    """Predict endpoint must reject requests with no file upload."""
    response = client.post("/predict")
    assert response.status_code == 422
