"""LA six-keypoint screening use case and measurement orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.errors import InvalidImageError, ModelNotReadyError
from app.inference.manifest import LAModelManifest
from app.inference.preprocessing import ImageDecodingError
from app.inference.runtime import LAInferenceRuntime, RuntimeUnavailableError
from app.inference.types import RawPrediction
from app.postprocessing.cobb import calculate_cobb_angles
from app.postprocessing.morphology import extract_chain_morphology
from app.postprocessing.spine_chain import SpineCandidate, select_spine_chain
from app.services.screening import quality_warnings
from app.utils.image_validation import validate_image_upload

LA_COBB_SCOPE = (
    "LA geometry agreement calculated from the four outer landmarks; "
    "not a clinical AP/PA Cobb measurement."
)
OUTER_KEYPOINT_INDICES = (0, 2, 3, 5)


class LAScreeningService:
    """Coordinate validation, six-point inference, chain selection, and measurements."""

    def __init__(self, runtime: LAInferenceRuntime, max_upload_bytes: int) -> None:
        self._runtime = runtime
        self.max_upload_bytes = max_upload_bytes

    @property
    def ready(self) -> bool:
        return self._runtime.ready

    @property
    def status_detail(self) -> str:
        return self._runtime.status_detail

    @property
    def manifest(self) -> LAModelManifest | None:
        return self._runtime.manifest

    def start(self) -> None:
        self._runtime.start()

    def close(self) -> None:
        self._runtime.close()

    def ensure_ready(self) -> None:
        if not self.ready:
            raise ModelNotReadyError(self.status_detail)

    def predict(
        self,
        image_bytes: bytes,
        declared_media_type: str | None,
    ) -> Mapping[str, Any]:
        """Execute the complete LA six-keypoint pipeline."""
        self.ensure_ready()
        image = validate_image_upload(
            image_bytes,
            declared_media_type,
            self.max_upload_bytes,
        )
        try:
            raw_result = self._runtime.predict(image.data, image.media_type)
        except RuntimeUnavailableError as exc:
            raise ModelNotReadyError(str(exc)) from exc
        except ImageDecodingError as exc:
            raise InvalidImageError(str(exc)) from exc

        manifest = self.manifest
        if manifest is None:
            raise ModelNotReadyError(self.status_detail)
        prediction = raw_result.prediction
        outer_corners = prediction.keypoints[:, OUTER_KEYPOINT_INDICES, :]
        chain_input = RawPrediction(
            scores=prediction.scores,
            centers=prediction.centers,
            corners=outer_corners,
        )
        chain = manifest.spine_chain
        selection = select_spine_chain(
            chain_input,
            duplicate_iou_threshold=chain.duplicate_iou_threshold,
            duplicate_center_scale=chain.duplicate_center_scale,
            score_threshold=chain.score_threshold,
            score_weight=chain.score_weight,
            min_chain_len=chain.min_chain_len,
        )
        cobb = calculate_cobb_angles(selection.selected)
        morphology = extract_chain_morphology(selection.selected)
        warnings = quality_warnings(
            selection.selected,
            raw_count=len(selection.raw),
            width=raw_result.original_width,
            height=raw_result.original_height,
            score_threshold=chain.score_threshold,
            min_chain_len=chain.min_chain_len,
            cobb_valid=cobb.valid,
        )
        warnings.append(LA_COBB_SCOPE)
        return {
            "prediction_id": uuid4(),
            "projection": "LA",
            "model": _model_payload(manifest),
            "image": {
                "width": raw_result.original_width,
                "height": raw_result.original_height,
                "media_type": raw_result.media_type,
            },
            "counts": {
                "raw": len(selection.raw),
                "deduplicated": len(selection.deduplicated),
                "selected": len(selection.selected),
            },
            "raw_predictions": [
                _candidate_payload(candidate, rank, prediction.keypoints)
                for rank, candidate in enumerate(selection.raw, start=1)
            ],
            "selected_vertebrae": [
                _candidate_payload(candidate, rank, prediction.keypoints)
                for rank, candidate in enumerate(selection.selected, start=1)
            ],
            "cobb": {**cobb.to_payload(), "scope": LA_COBB_SCOPE},
            "morphology": [feature.to_payload() for feature in morphology],
            "warnings": warnings,
        }


def _model_payload(manifest: LAModelManifest) -> dict[str, Any]:
    chain = manifest.spine_chain
    return {
        "model_id": manifest.model_id,
        "artifact_type": manifest.artifact_type,
        "format_version": manifest.format_version,
        "backbone": manifest.backbone,
        "input_size": manifest.input_size,
        "down_ratio": manifest.down_ratio,
        "peak_threshold": manifest.peak_threshold,
        "topk": manifest.topk,
        "sha256": manifest.sha256,
        "checkpoint_epoch": manifest.checkpoint_epoch,
        "keypoint_order": manifest.keypoint_order,
        "spine_chain": {
            "duplicate_iou_threshold": chain.duplicate_iou_threshold,
            "duplicate_center_scale": chain.duplicate_center_scale,
            "score_threshold": chain.score_threshold,
            "score_weight": chain.score_weight,
            "min_chain_len": chain.min_chain_len,
        },
    }


def _point_payload(point: np.ndarray) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1])}


def _candidate_payload(
    candidate: SpineCandidate,
    rank: int,
    keypoints: np.ndarray,
) -> dict[str, Any]:
    points = keypoints[candidate.candidate_id]
    return {
        "candidate_id": candidate.candidate_id,
        "rank": rank,
        "detector_score": candidate.score,
        "center": _point_payload(candidate.center),
        "keypoints": {
            "top_left": _point_payload(points[0]),
            "top_middle": _point_payload(points[1]),
            "top_right": _point_payload(points[2]),
            "bottom_left": _point_payload(points[3]),
            "bottom_middle": _point_payload(points[4]),
            "bottom_right": _point_payload(points[5]),
        },
    }
