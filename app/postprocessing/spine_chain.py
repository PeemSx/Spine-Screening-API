"""Duplicate suppression and dynamic-programming spine-chain selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.inference.types import RawPrediction


@dataclass(frozen=True, slots=True)
class SpineCandidate:
    """One finite detector candidate with a stable raw candidate identifier."""

    candidate_id: int
    score: float
    center: np.ndarray
    corners: np.ndarray
    box: np.ndarray

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    @property
    def height(self) -> float:
        return float(max(self.box[3] - self.box[1], 1.0))


@dataclass(frozen=True, slots=True)
class ChainSelection:
    """Raw, deduplicated, and selected candidates kept visibly separate."""

    raw: tuple[SpineCandidate, ...]
    deduplicated: tuple[SpineCandidate, ...]
    selected: tuple[SpineCandidate, ...]
    debug: dict[str, Any]


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    width = max(0.0, ix2 - ix1)
    height = max(0.0, iy2 - iy1)
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def box_from_corners(corners: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Create a candidate box, falling back to a one-pixel center box."""
    if corners.shape == (4, 2) and np.isfinite(corners).all():
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        if np.all(maximum > minimum):
            return np.asarray(
                [minimum[0], minimum[1], maximum[0], maximum[1]],
                dtype=np.float32,
            )
    x_coord, y_coord = float(center[0]), float(center[1])
    return np.asarray(
        [x_coord - 0.5, y_coord - 0.5, x_coord + 0.5, y_coord + 0.5],
        dtype=np.float32,
    )


def prediction_to_candidates(prediction: RawPrediction) -> list[SpineCandidate]:
    """Assign stable raw IDs and discard malformed or non-finite candidates."""
    candidates = []
    for candidate_id, (score, center, corners) in enumerate(
        zip(prediction.scores, prediction.centers, prediction.corners)
    ):
        center = np.asarray(center, dtype=np.float32)
        corners = np.asarray(corners, dtype=np.float32)
        if center.shape != (2,) or corners.shape != (4, 2):
            continue
        if not np.isfinite(center).all() or not np.isfinite(corners).all():
            continue
        candidates.append(
            SpineCandidate(
                candidate_id=candidate_id,
                score=float(score),
                center=center,
                corners=corners,
                box=box_from_corners(corners, center),
            )
        )
    return candidates


def suppress_duplicate_candidates(
    candidates: list[SpineCandidate],
    *,
    iou_threshold: float = 0.18,
    center_scale: float = 0.35,
) -> list[SpineCandidate]:
    """Greedily retain the highest-confidence representative of each duplicate."""
    kept = []
    for candidate in sorted(candidates, key=lambda item: -item.score):
        duplicate = False
        for previous in kept:
            overlap = box_iou(candidate.box, previous.box)
            distance = math.hypot(candidate.x - previous.x, candidate.y - previous.y)
            radius = float(center_scale) * min(candidate.height, previous.height)
            if overlap >= float(iou_threshold) or distance <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.y, item.x))


def estimate_spacing(candidates: list[SpineCandidate]) -> tuple[float, float, float]:
    """Estimate expected vertical vertebral spacing from candidate geometry."""
    if not candidates:
        return 32.0, 16.0, 80.0
    heights = np.asarray([candidate.height for candidate in candidates], dtype=np.float32)
    median_height = float(np.median(heights)) if len(heights) else 32.0
    ys = np.asarray(sorted(candidate.y for candidate in candidates), dtype=np.float32)
    gaps = np.diff(ys)
    plausible = gaps[(gaps >= 0.35 * median_height) & (gaps <= 2.20 * median_height)]
    target_dy = (
        float(np.median(plausible))
        if len(plausible) >= 2
        else max(0.85 * median_height, 12.0)
    )
    minimum_dy = max(0.35 * target_dy, 8.0)
    maximum_dy = max(2.10 * target_dy, minimum_dy + 8.0)
    return target_dy, minimum_dy, maximum_dy


def transition_score(
    previous: SpineCandidate,
    current: SpineCandidate,
    target_dy: float,
    minimum_dy: float,
    maximum_dy: float,
) -> float | None:
    """Score anatomical continuity between two top-to-bottom candidates."""
    delta_y = current.y - previous.y
    if delta_y <= 0.0 or delta_y < minimum_dy or delta_y > maximum_dy:
        return None
    delta_x = current.x - previous.x
    if abs(delta_x) > 1.35 * max(target_dy, 1.0):
        return None
    spacing_penalty = ((delta_y - target_dy) / max(target_dy, 1.0)) ** 2
    lateral_penalty = (delta_x / max(target_dy, 1.0)) ** 2
    size_penalty = abs(math.log(max(current.height, 1.0) / max(previous.height, 1.0)))
    return -(0.55 * spacing_penalty + 0.35 * lateral_penalty + 0.20 * size_penalty)


def dynamic_spine_chain(
    candidates: list[SpineCandidate],
    *,
    score_threshold: float = 0.18,
    score_weight: float = 3.0,
    min_chain_len: int = 3,
) -> tuple[list[SpineCandidate], dict[str, Any]]:
    """Select the highest-scoring continuous chain using dynamic programming."""
    ordered = sorted(candidates, key=lambda item: (item.y, item.x))
    if not ordered:
        return [], {
            "score": 0.0,
            "target_dy": 0.0,
            "min_dy": 0.0,
            "max_dy": 0.0,
            "node_score_threshold": float(score_threshold),
            "num_candidates": 0,
            "num_selected": 0,
        }

    target_dy, minimum_dy, maximum_dy = estimate_spacing(ordered)
    count = len(ordered)
    best = np.full(count, -np.inf, dtype=np.float64)
    lengths = np.ones(count, dtype=np.int32)
    parents = np.full(count, -1, dtype=np.int32)
    node_values = np.asarray(
        [float(score_weight) * (candidate.score - score_threshold) for candidate in ordered],
        dtype=np.float64,
    )
    best[:] = node_values
    for current_index in range(count):
        for previous_index in range(current_index):
            edge = transition_score(
                ordered[previous_index],
                ordered[current_index],
                target_dy,
                minimum_dy,
                maximum_dy,
            )
            if edge is None:
                continue
            value = best[previous_index] + node_values[current_index] + edge
            if value > best[current_index]:
                best[current_index] = value
                lengths[current_index] = lengths[previous_index] + 1
                parents[current_index] = previous_index

    valid_ends = np.where(lengths >= int(min_chain_len))[0]
    if len(valid_ends) == 0:
        valid_ends = np.arange(count)
    end_index = int(valid_ends[np.argmax(best[valid_ends])])
    chain_indices = []
    current = end_index
    while current >= 0:
        chain_indices.append(current)
        current = int(parents[current])
    chain_indices.reverse()
    chain = [ordered[index] for index in chain_indices]
    return chain, {
        "score": float(best[end_index]),
        "target_dy": float(target_dy),
        "min_dy": float(minimum_dy),
        "max_dy": float(maximum_dy),
        "node_score_threshold": float(score_threshold),
        "num_candidates": count,
        "num_selected": len(chain),
    }


def select_spine_chain(
    prediction: RawPrediction,
    *,
    duplicate_iou_threshold: float = 0.18,
    duplicate_center_scale: float = 0.35,
    score_threshold: float = 0.18,
    score_weight: float = 3.0,
    min_chain_len: int = 3,
) -> ChainSelection:
    """Run the complete active raw-to-chain policy."""
    raw = prediction_to_candidates(prediction)
    deduplicated = suppress_duplicate_candidates(
        raw,
        iou_threshold=duplicate_iou_threshold,
        center_scale=duplicate_center_scale,
    )
    selected, debug = dynamic_spine_chain(
        deduplicated,
        score_threshold=score_threshold,
        score_weight=score_weight,
        min_chain_len=min_chain_len,
    )
    debug["num_raw"] = len(raw)
    debug["num_deduplicated"] = len(deduplicated)
    return ChainSelection(
        raw=tuple(raw),
        deduplicated=tuple(deduplicated),
        selected=tuple(selected),
        debug=debug,
    )

