from fastapi.testclient import TestClient

from src.api.app import app


def test_service_health_check():
    """
    Confirms the public health endpoint is reachable and healthy.
    """
    fastapi_client = TestClient(app)
    api_response = fastapi_client.get("/health")

    assert api_response.status_code == 200
    assert api_response.json() == {"status": "healthy"}


def test_predict_requires_authentication():
    """
    Asserts that the /predict endpoint is protected by API key middleware.
    """
    fastapi_client = TestClient(app)
    unauthorized_response = fastapi_client.post("/predict")

    assert unauthorized_response.status_code == 403
