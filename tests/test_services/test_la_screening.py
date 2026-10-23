"""End-to-end LA orchestration tests using a deterministic fake runtime."""

import numpy as np

from app.inference.manifest import LA_KEYPOINT_ORDER, LAModelManifest, SpineChainConfig
from app.inference.types import RawSixPointInferenceResult, RawSixPointPrediction
from app.schemas.la_prediction import LAPredictionResponse
from app.services.la_screening import LA_COBB_SCOPE, LAScreeningService


class FakeLARuntime:
    ready = True
    status_detail = "ready"
    manifest = LAModelManifest(
        model_id="test-la-model",
        artifact_type="spine_centernet_la_six_keypoint_checkpoint",
        format_version=1,
        backbone="hrnet_w18",
        input_size=1024,
        down_ratio=4,
        peak_threshold=0.1,
        topk=50,
        sha256="b" * 64,
        checkpoint_epoch=33,
        spine_chain=SpineChainConfig(0.18, 0.35, 0.18, 3.0, 3),
        keypoint_order=LA_KEYPOINT_ORDER,
    )

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def predict(self, image_bytes: bytes, media_type: str) -> RawSixPointInferenceResult:
        del image_bytes
        centers = np.asarray([[50, 10], [50, 30], [50, 50]], dtype=np.float32)
        keypoints = np.asarray(
            [
                [[40, 5], [50, 5], [60, 5], [40, 15], [50, 15], [60, 15]],
                [[40, 25], [50, 25], [60, 25], [40, 35], [50, 35], [60, 35]],
                [[40, 45], [50, 45], [60, 45], [40, 55], [50, 55], [60, 55]],
            ],
            dtype=np.float32,
        )
        return RawSixPointInferenceResult(
            prediction=RawSixPointPrediction(
                scores=np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
                centers=centers,
                keypoints=keypoints,
            ),
            original_width=100,
            original_height=100,
            media_type=media_type,
        )


def test_la_service_returns_six_points_and_outer_landmark_measurements() -> None:
    service = LAScreeningService(FakeLARuntime(), max_upload_bytes=1024)

    payload = service.predict(b"\x89PNG\r\n\x1a\ncontent", "image/png")
    response = LAPredictionResponse.model_validate(payload)

    assert response.projection == "LA"
    assert response.keypoint_order == LA_KEYPOINT_ORDER
    assert response.counts.raw == 3
    assert response.counts.selected == 3
    assert response.selected_vertebrae[0].keypoints.top_middle.x == 50.0
    assert response.selected_vertebrae[0].keypoints.bottom_middle.y == 15.0
    assert response.cobb.valid
    assert response.cobb.scope == LA_COBB_SCOPE
    assert len(response.morphology) == 3
    assert response.warnings[-1] == LA_COBB_SCOPE
