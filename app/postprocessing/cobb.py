"""Measurement-only Cobb geometry with stable vertebral candidate IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.postprocessing.spine_chain import SpineCandidate


@dataclass(frozen=True, slots=True)
class CobbLine:
    """One endplate-axis line tied to a stable raw candidate ID."""

    start: tuple[float, float]
    end: tuple[float, float]
    sorted_index: int
    candidate_id: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "start": {"x": self.start[0], "y": self.start[1]},
            "end": {"x": self.end[0], "y": self.end[1]},
            "vertebra_candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class CobbResult:
    """Computational angles and the exact lines used to derive them."""

    valid: bool
    vertebra_count: int
    cobb_1_deg: float | None
    cobb_2_deg: float | None
    cobb_3_deg: float | None
    major_lines: tuple[CobbLine, CobbLine] | None
    cobb_2_lines: tuple[CobbLine, CobbLine] | None
    cobb_3_lines: tuple[CobbLine, CobbLine] | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "vertebra_count": self.vertebra_count,
            "cobb_1_deg": self.cobb_1_deg,
            "cobb_2_deg": self.cobb_2_deg,
            "cobb_3_deg": self.cobb_3_deg,
            "major_lines": [line.to_payload() for line in self.major_lines or ()],
            "cobb_2_lines": [line.to_payload() for line in self.cobb_2_lines or ()],
            "cobb_3_lines": [line.to_payload() for line in self.cobb_3_lines or ()],
        }


def invalid_cobb_result(vertebra_count: int = 0) -> CobbResult:
    return CobbResult(
        valid=False,
        vertebra_count=int(vertebra_count),
        cobb_1_deg=None,
        cobb_2_deg=None,
        cobb_3_deg=None,
        major_lines=None,
        cobb_2_lines=None,
        cobb_3_lines=None,
    )


def _prepare_candidates(
    candidates: tuple[SpineCandidate, ...],
) -> tuple[np.ndarray, list[SpineCandidate]]:
    valid = [
        candidate
        for candidate in candidates
        if candidate.corners.shape == (4, 2) and np.isfinite(candidate.corners).all()
    ]
    valid.sort(key=lambda candidate: float(candidate.corners.mean(axis=0)[1]))
    if not valid:
        return np.zeros((0, 4, 2), dtype=np.float32), []
    return np.asarray([candidate.corners for candidate in valid], dtype=np.float32), valid


def _axis_lines(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_midpoints = (corners[:, 0, :] + corners[:, 2, :]) * 0.5
    right_midpoints = (corners[:, 1, :] + corners[:, 3, :]) * 0.5
    return left_midpoints, right_midpoints, right_midpoints - left_midpoints


def _pairwise_line_angles(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    denominator = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.divide(
            vectors @ vectors.T,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float32),
            where=denominator > 1e-6,
        )
    return np.degrees(np.arccos(np.clip(np.abs(cosines), 0.0, 1.0))).astype(np.float32)


def _line(
    left_midpoints: np.ndarray,
    right_midpoints: np.ndarray,
    candidates: list[SpineCandidate],
    index: int,
) -> CobbLine:
    return CobbLine(
        start=(float(left_midpoints[index, 0]), float(left_midpoints[index, 1])),
        end=(float(right_midpoints[index, 0]), float(right_midpoints[index, 1])),
        sorted_index=index,
        candidate_id=candidates[index].candidate_id,
    )


def _lines_for_pair(
    left_midpoints: np.ndarray,
    right_midpoints: np.ndarray,
    candidates: list[SpineCandidate],
    first_index: int,
    second_index: int,
) -> tuple[CobbLine, CobbLine]:
    return (
        _line(left_midpoints, right_midpoints, candidates, first_index),
        _line(left_midpoints, right_midpoints, candidates, second_index),
    )


def _secondary_angles(
    angles: np.ndarray,
    first_index: int,
    second_index: int,
) -> tuple[float | None, tuple[int, int] | None, float | None, tuple[int, int] | None]:
    vertebra_count = int(angles.shape[0])
    if vertebra_count < 3:
        return None, None, None, None
    top_index = min(first_index, second_index)
    bottom_index = max(first_index, second_index)
    top_angle = None
    top_pair = None
    if top_index > 0:
        top_slice = angles[:top_index, top_index]
        top_partner = int(np.argmax(top_slice))
        top_angle = float(top_slice[top_partner])
        top_pair = (top_partner, top_index)
    bottom_angle = None
    bottom_pair = None
    if bottom_index < vertebra_count - 1:
        bottom_slice = angles[bottom_index, bottom_index + 1 :]
        bottom_partner = int(bottom_index + 1 + np.argmax(bottom_slice))
        bottom_angle = float(angles[bottom_index, bottom_partner])
        bottom_pair = (bottom_index, bottom_partner)
    return top_angle, top_pair, bottom_angle, bottom_pair


def calculate_cobb_angles(candidates: tuple[SpineCandidate, ...]) -> CobbResult:
    """Calculate major and optional secondary angles for the selected chain."""
    corners, ordered = _prepare_candidates(candidates)
    vertebra_count = len(ordered)
    if vertebra_count < 2:
        return invalid_cobb_result(vertebra_count)
    left_midpoints, right_midpoints, vectors = _axis_lines(corners)
    valid_lines = np.linalg.norm(vectors, axis=1) > 1e-6
    if int(valid_lines.sum()) < 2:
        return invalid_cobb_result(vertebra_count)
    if not valid_lines.all():
        left_midpoints = left_midpoints[valid_lines]
        right_midpoints = right_midpoints[valid_lines]
        vectors = vectors[valid_lines]
        ordered = [candidate for candidate, keep in zip(ordered, valid_lines) if keep]
        vertebra_count = len(ordered)

    angles = _pairwise_line_angles(vectors)
    candidates_angles = angles.copy()
    np.fill_diagonal(candidates_angles, -1.0)
    first_index, second_index = np.unravel_index(
        int(np.argmax(candidates_angles)),
        candidates_angles.shape,
    )
    major_angle = float(angles[first_index, second_index])
    cobb_2, cobb_2_pair, cobb_3, cobb_3_pair = _secondary_angles(
        angles,
        int(first_index),
        int(second_index),
    )
    major_lines = _lines_for_pair(
        left_midpoints,
        right_midpoints,
        ordered,
        int(first_index),
        int(second_index),
    )
    cobb_2_lines = (
        _lines_for_pair(left_midpoints, right_midpoints, ordered, *cobb_2_pair)
        if cobb_2_pair is not None
        else None
    )
    cobb_3_lines = (
        _lines_for_pair(left_midpoints, right_midpoints, ordered, *cobb_3_pair)
        if cobb_3_pair is not None
        else None
    )
    return CobbResult(
        valid=True,
        vertebra_count=vertebra_count,
        cobb_1_deg=major_angle,
        cobb_2_deg=cobb_2,
        cobb_3_deg=cobb_3,
        major_lines=major_lines,
        cobb_2_lines=cobb_2_lines,
        cobb_3_lines=cobb_3_lines,
    )

