"""Real-checkpoint smoke test, enabled explicitly because it is comparatively expensive."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inference.runtime import CenterNetRuntime
from app.main import create_app
from app.schemas.prediction import PredictionResponse
from app.services.screening import ScreeningService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _encoded_test_image() -> bytes:
    image = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (512, 1))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    return buffer.tobytes()


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("SPINE_RUN_MODEL_TESTS") != "1",
    reason="Set SPINE_RUN_MODEL_TESTS=1 to execute the real checkpoint.",
)
def test_real_checkpoint_runs_complete_pipeline() -> None:
    runtime = CenterNetRuntime(
        PROJECT_ROOT / "app" / "weight" / "best_center_f1.pt",
        device_name="cpu",
    )
    service = ScreeningService(runtime, max_upload_bytes=20 * 1024 * 1024)
    try:
        service.start()
        response = PredictionResponse.model_validate(
            service.predict(_encoded_test_image(), "image/png")
        )
    finally:
        service.close()

    assert response.model.backbone == "hrnet_w18"
    assert response.image.width == 256
    assert response.image.height == 512
    assert response.counts.raw >= response.counts.selected


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("SPINE_RUN_MODEL_TESTS") != "1",
    reason="Set SPINE_RUN_MODEL_TESTS=1 to execute the real checkpoint.",
)
def test_real_checkpoint_is_available_through_http() -> None:
    with TestClient(create_app()) as client:
        readiness = client.get("/api/v1/health/ready")
        model = client.get("/api/v1/model")
        prediction = client.post(
            "/api/v1/predictions",
            files={"image": ("radiograph.png", _encoded_test_image(), "image/png")},
        )

    assert readiness.status_code == 200
    assert model.status_code == 200
    assert prediction.status_code == 200
    assert prediction.json()["model"]["backbone"] == "hrnet_w18"
