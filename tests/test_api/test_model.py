"""Model metadata endpoint tests."""

from fastapi.testclient import TestClient


def test_model_info_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.get("/api/v1/model")

    assert response.status_code == 503
    assert response.json()["code"] == "model_not_ready"
