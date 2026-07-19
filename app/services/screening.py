"""Application use case coordinating validation, inference, and measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.errors import InvalidImageError, ModelNotReadyError
from app.inference.manifest import ModelManifest
from app.inference.preprocessing import ImageDecodingError
from app.inference.runtime import InferenceRuntime, RuntimeUnavailableError
from app.postprocessing.cobb import calculate_cobb_angles
from app.postprocessing.morphology import extract_chain_morphology
from app.postprocessing.spine_chain import SpineCandidate, select_spine_chain
from app.utils.image_validation import validate_image_upload


class ScreeningService:
    """Keep HTTP endpoints independent of model and numerical processing details."""

    def __init__(self, runtime: InferenceRuntime, max_upload_bytes: int) -> None:
        self._runtime = runtime
        self.max_upload_bytes = max_upload_bytes

    @property
    def ready(self) -> bool:
        return self._runtime.ready

    @property
    def status_detail(self) -> str:
        return self._runtime.status_detail

    @property
    def manifest(self) -> ModelManifest | None:
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
        """Execute the complete active pipeline and return an API-ready mapping."""
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
        chain_config = manifest.spine_chain
        selection = select_spine_chain(
            raw_result.prediction,
            duplicate_iou_threshold=chain_config.duplicate_iou_threshold,
            duplicate_center_scale=chain_config.duplicate_center_scale,
            score_threshold=chain_config.score_threshold,
            score_weight=chain_config.score_weight,
            min_chain_len=chain_config.min_chain_len,
        )
        cobb = calculate_cobb_angles(selection.selected)
        morphology = extract_chain_morphology(selection.selected)
        warnings = _quality_warnings(
            selection.selected,
            raw_count=len(selection.raw),
            width=raw_result.original_width,
            height=raw_result.original_height,
            score_threshold=chain_config.score_threshold,
            min_chain_len=chain_config.min_chain_len,
            cobb_valid=cobb.valid,
        )
        return {
            "prediction_id": uuid4(),
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
                _candidate_payload(candidate, rank)
                for rank, candidate in enumerate(selection.raw, start=1)
            ],
            "selected_vertebrae": [
                _candidate_payload(candidate, rank)
                for rank, candidate in enumerate(selection.selected, start=1)
            ],
            "cobb": cobb.to_payload(),
            "morphology": [feature.to_payload() for feature in morphology],
            "warnings": warnings,
        }


def _model_payload(manifest: ModelManifest) -> dict[str, Any]:
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
        "corner_order": manifest.corner_order,
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


def _candidate_payload(candidate: SpineCandidate, rank: int) -> dict[str, Any]:
    top_left, top_right, bottom_left, bottom_right = candidate.corners
    return {
        "candidate_id": candidate.candidate_id,
        "rank": rank,
        "detector_score": candidate.score,
        "center": _point_payload(candidate.center),
        "corners": {
            "top_left": _point_payload(top_left),
            "top_right": _point_payload(top_right),
            "bottom_left": _point_payload(bottom_left),
            "bottom_right": _point_payload(bottom_right),
        },
    }


def _quality_warnings(
    selected: tuple[SpineCandidate, ...],
    *,
    raw_count: int,
    width: int,
    height: int,
    score_threshold: float,
    min_chain_len: int,
    cobb_valid: bool,
) -> list[str]:
    warnings = []
    if raw_count == 0:
        warnings.append("No vertebral candidates were detected.")
    if len(selected) < min_chain_len:
        warnings.append(
            f"The selected spine chain contains fewer than {min_chain_len} vertebrae."
        )
    if any(candidate.score < score_threshold for candidate in selected):
        warnings.append("The selected chain contains low-confidence detector candidates.")
    if any(
        np.any(candidate.corners[:, 0] < 0.0)
        or np.any(candidate.corners[:, 0] >= float(width))
        or np.any(candidate.corners[:, 1] < 0.0)
        or np.any(candidate.corners[:, 1] >= float(height))
        for candidate in selected
    ):
        warnings.append("One or more selected corners lie outside the original image.")
    if not cobb_valid:
        warnings.append("Cobb geometry could not be calculated from the selected chain.")
    return warnings

