"""End-to-end orchestration tests using a deterministic fake runtime."""

import numpy as np

from app.inference.manifest import ModelManifest, SpineChainConfig
from app.inference.types import RawInferenceResult, RawPrediction
from app.schemas.prediction import PredictionResponse
from app.services.screening import ScreeningService


class FakeRuntime:
    ready = True
    status_detail = "ready"
    manifest = ModelManifest(
        model_id="test-model",
        artifact_type="spine_centernet_inference",
        format_version=1,
        backbone="hrnet_w18",
        input_size=1024,
        down_ratio=4,
        peak_threshold=0.05,
        topk=50,
        sha256="a" * 64,
        spine_chain=SpineChainConfig(0.18, 0.35, 0.18, 3.0, 3),
    )

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def predict(self, image_bytes: bytes, media_type: str) -> RawInferenceResult:
        del image_bytes
        centers = np.asarray([[50, 10], [50, 30], [50, 50]], dtype=np.float32)
        corners = np.asarray(
            [
                [[40, 5], [60, 5], [40, 15], [60, 15]],
                [[40, 25], [60, 25], [40, 35], [60, 35]],
                [[40, 45], [60, 45], [40, 55], [60, 55]],
            ],
            dtype=np.float32,
        )
        return RawInferenceResult(
            prediction=RawPrediction(
                scores=np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
                centers=centers,
                corners=corners,
            ),
            original_width=100,
            original_height=100,
            media_type=media_type,
        )


def test_service_returns_schema_compatible_measurements() -> None:
    service = ScreeningService(FakeRuntime(), max_upload_bytes=1024)

    payload = service.predict(b"\x89PNG\r\n\x1a\ncontent", "image/png")
    response = PredictionResponse.model_validate(payload)

    assert response.model.model_id == "test-model"
    assert response.counts.raw == 3
    assert response.counts.selected == 3
    assert len(response.selected_vertebrae) == 3
    assert response.cobb.valid
    assert len(response.morphology) == 3
