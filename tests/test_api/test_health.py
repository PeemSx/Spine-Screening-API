"""Health endpoint tests."""

from fastapi.testclient import TestClient


def test_liveness(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "model_not_ready"
