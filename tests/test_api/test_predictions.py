"""Prediction endpoint boundary tests."""

from fastapi.testclient import TestClient


def test_prediction_reports_unavailable_runtime(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predictions",
        files={"image": ("radiograph.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "model_not_ready"
