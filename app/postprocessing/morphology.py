"""Measurement-only vertebral morphology features for a selected chain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.postprocessing.spine_chain import SpineCandidate


@dataclass(frozen=True, slots=True)
class MorphologyFeature:
    """Original-pixel and dimensionless geometry for one candidate."""

    rank: int
    candidate_id: int
    detector_score: float
    superior_width_px: float
    inferior_width_px: float
    left_height_px: float
    right_height_px: float
    mean_height_px: float
    left_right_height_ratio: float
    height_asymmetry_fraction: float
    width_height_ratio: float
    superior_endplate_angle_deg: float
    inferior_endplate_angle_deg: float
    endplate_nonparallel_deg: float
    neighbor_reference_height_px: float | None = None
    height_ratio_to_neighbors: float | None = None
    relative_height_deviation: float | None = None
    previous_center_spacing_px: float | None = None
    next_center_spacing_px: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(second) - np.asarray(first)))


def _line_angle(first: np.ndarray, second: np.ndarray) -> float:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _angle_difference(first: float, second: float) -> float:
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), 1e-6))


def extract_chain_morphology(
    candidates: tuple[SpineCandidate, ...],
) -> tuple[MorphologyFeature, ...]:
    """Extract features from a top-to-bottom selected spine chain."""
    ordered = sorted(candidates, key=lambda candidate: (candidate.y, candidate.x))
    base_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ordered, start=1):
        points = np.asarray(candidate.corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        top_left, top_right, bottom_left, bottom_right = points
        superior_width = _distance(top_left, top_right)
        inferior_width = _distance(bottom_left, bottom_right)
        left_height = _distance(top_left, bottom_left)
        right_height = _distance(top_right, bottom_right)
        mean_height = 0.5 * (left_height + right_height)
        mean_width = 0.5 * (superior_width + inferior_width)
        superior_angle = _line_angle(top_left, top_right)
        inferior_angle = _line_angle(bottom_left, bottom_right)
        base_rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "detector_score": candidate.score,
                "superior_width_px": superior_width,
                "inferior_width_px": inferior_width,
                "left_height_px": left_height,
                "right_height_px": right_height,
                "mean_height_px": mean_height,
                "left_right_height_ratio": _safe_ratio(left_height, right_height),
                "height_asymmetry_fraction": _safe_ratio(
                    abs(left_height - right_height),
                    mean_height,
                ),
                "width_height_ratio": _safe_ratio(mean_width, mean_height),
                "superior_endplate_angle_deg": superior_angle,
                "inferior_endplate_angle_deg": inferior_angle,
                "endplate_nonparallel_deg": _angle_difference(
                    superior_angle,
                    inferior_angle,
                ),
            }
        )

    for index, row in enumerate(base_rows):
        neighbor_heights = []
        if index > 0:
            neighbor_heights.append(float(base_rows[index - 1]["mean_height_px"]))
        if index + 1 < len(base_rows):
            neighbor_heights.append(float(base_rows[index + 1]["mean_height_px"]))
        reference_height = float(np.mean(neighbor_heights)) if neighbor_heights else None
        current_height = float(row["mean_height_px"])
        row["neighbor_reference_height_px"] = reference_height
        row["height_ratio_to_neighbors"] = (
            _safe_ratio(current_height, reference_height) if reference_height is not None else None
        )
        row["relative_height_deviation"] = (
            _safe_ratio(abs(current_height - reference_height), reference_height)
            if reference_height is not None
            else None
        )
        center = ordered[index].center
        row["previous_center_spacing_px"] = (
            _distance(ordered[index - 1].center, center) if index > 0 else None
        )
        row["next_center_spacing_px"] = (
            _distance(center, ordered[index + 1].center)
            if index + 1 < len(ordered)
            else None
        )
    return tuple(MorphologyFeature(**row) for row in base_rows)

