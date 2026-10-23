"""Readiness isolation between the AP/PA and LA pipelines."""

from typing import Any

from fastapi.testclient import TestClient

from app.inference.manifest import (
    LA_KEYPOINT_ORDER,
    LAModelManifest,
    ModelManifest,
    SpineChainConfig,
)
from app.inference.runtime import UnavailableInferenceRuntime
from app.main import create_app

CHAIN = SpineChainConfig(0.18, 0.35, 0.18, 3.0, 3)
AP_MANIFEST = ModelManifest(
    model_id="ready-ap",
    artifact_type="spine_centernet_inference",
    format_version=1,
    backbone="hrnet_w18",
    input_size=1024,
    down_ratio=4,
    peak_threshold=0.1,
    topk=50,
    sha256="a" * 64,
    spine_chain=CHAIN,
)
LA_MANIFEST = LAModelManifest(
    model_id="ready-la",
    artifact_type="spine_centernet_la_six_keypoint_checkpoint",
    format_version=1,
    backbone="hrnet_w18",
    input_size=1024,
    down_ratio=4,
    peak_threshold=0.1,
    topk=50,
    sha256="b" * 64,
    checkpoint_epoch=33,
    spine_chain=CHAIN,
    keypoint_order=LA_KEYPOINT_ORDER,
)


class ReadyRuntime:
    ready = True
    status_detail = "ready"

    def __init__(self, manifest: ModelManifest | LAModelManifest) -> None:
        self.manifest = manifest

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def predict(self, image_bytes: bytes, media_type: str) -> Any:
        del image_bytes, media_type
        raise AssertionError("Invalid uploads must be rejected before inference")


def test_ap_pa_remains_ready_when_la_is_unavailable() -> None:
    application = create_app()
    application.state.runtime_factory = lambda settings: ReadyRuntime(AP_MANIFEST)
    application.state.la_runtime_factory = lambda settings: UnavailableInferenceRuntime(
        "LA unavailable"
    )

    with TestClient(application) as client:
        ap_response = client.get("/api/v1/health/ready")
        la_response = client.get("/api/v1/la/health/ready")

    assert ap_response.status_code == 200
    assert ap_response.json()["model_id"] == "ready-ap"
    assert la_response.status_code == 503


def test_la_remains_ready_and_validates_uploads_when_ap_pa_is_unavailable() -> None:
    application = create_app()
    application.state.runtime_factory = lambda settings: UnavailableInferenceRuntime(
        "AP/PA unavailable"
    )
    application.state.la_runtime_factory = lambda settings: ReadyRuntime(LA_MANIFEST)

    with TestClient(application) as client:
        ap_response = client.get("/api/v1/health/ready")
        la_response = client.get("/api/v1/la/health/ready")
        invalid_upload = client.post(
            "/api/v1/la/predictions",
            files={"image": ("notes.txt", b"not a radiograph", "text/plain")},
        )

    assert ap_response.status_code == 503
    assert la_response.status_code == 200
    assert la_response.json()["model_id"] == "ready-la"
    assert invalid_upload.status_code == 415
    assert invalid_upload.json()["code"] == "unsupported_media_type"
