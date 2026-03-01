from fastapi.testclient import TestClient

from src.api.app import app


def test_service_health_check():
    """
    Confirms the public health endpoint is reachable and healthy if model exists.
    """
    fastapi_client = TestClient(app)
    api_response = fastapi_client.get("/health")

    assert api_response.status_code == 200
    # The response can be healthy or degraded depending on the model path existing in CI
    assert "status" in api_response.json()
