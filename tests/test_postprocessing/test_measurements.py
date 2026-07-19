"""Cobb and morphology measurement tests."""

import math

import numpy as np
import pytest

from app.postprocessing.cobb import calculate_cobb_angles
from app.postprocessing.morphology import extract_chain_morphology
from app.postprocessing.spine_chain import SpineCandidate, box_from_corners


def _candidate(candidate_id: int, y_coord: float, angle_deg: float) -> SpineCandidate:
    angle = math.radians(angle_deg)
    horizontal = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32) * 20.0
    vertical = np.asarray([0.0, 5.0], dtype=np.float32)
    center = np.asarray([50.0, y_coord], dtype=np.float32)
    left = center - horizontal
    right = center + horizontal
    corners = np.asarray(
        [left - vertical, right - vertical, left + vertical, right + vertical],
        dtype=np.float32,
    )
    return SpineCandidate(
        candidate_id=candidate_id,
        score=0.9,
        center=center,
        corners=corners,
        box=box_from_corners(corners, center),
    )


def test_cobb_uses_stable_candidate_ids() -> None:
    candidates = (
        _candidate(10, 20.0, -10.0),
        _candidate(20, 50.0, 0.0),
        _candidate(30, 80.0, 10.0),
    )

    result = calculate_cobb_angles(candidates)

    assert result.valid
    assert result.cobb_1_deg == pytest.approx(20.0, abs=1e-4)
    assert result.major_lines is not None
    assert {line.candidate_id for line in result.major_lines} == {10, 30}


def test_morphology_extracts_rectangle_geometry() -> None:
    candidates = (
        _candidate(10, 20.0, 0.0),
        _candidate(20, 50.0, 0.0),
        _candidate(30, 80.0, 0.0),
    )

    features = extract_chain_morphology(candidates)

    assert len(features) == 3
    assert features[1].superior_width_px == pytest.approx(40.0)
    assert features[1].mean_height_px == pytest.approx(10.0)
    assert features[1].height_ratio_to_neighbors == pytest.approx(1.0)
