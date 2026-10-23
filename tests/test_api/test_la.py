"""LA API boundary tests."""

from fastapi.testclient import TestClient


def test_la_readiness_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.get("/api/v1/la/health/ready")

    assert response.status_code == 503
    assert response.json()["code"] == "model_not_ready"


def test_la_model_info_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.get("/api/v1/la/model")

    assert response.status_code == 503
    assert response.json()["code"] == "model_not_ready"


def test_la_prediction_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.post(
        "/api/v1/la/predictions",
        files={"image": ("lateral.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "model_not_ready"
