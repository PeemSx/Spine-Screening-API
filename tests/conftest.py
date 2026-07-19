"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.inference.runtime import UnavailableInferenceRuntime
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Return an in-process client with application lifespan enabled."""
    app.state.runtime_factory = lambda settings: UnavailableInferenceRuntime(
        "Inference runtime disabled for API boundary tests."
    )
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        del app.state.runtime_factory
