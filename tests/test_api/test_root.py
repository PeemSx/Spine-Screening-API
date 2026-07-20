"""Tests for the browser-facing application root."""

from fastapi.testclient import TestClient


def test_root_redirects_to_swagger_docs(client: TestClient) -> None:
    """Opening a Docker Space should lead users to its API documentation."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
