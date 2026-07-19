"""Framework-independent value objects shared by the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TransformMeta:
    """Geometry needed to map padded-model coordinates back to the upload."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float
    input_size: int

    def __post_init__(self) -> None:
        if min(
            self.original_width,
            self.original_height,
            self.resized_width,
            self.resized_height,
            self.input_size,
        ) <= 0:
            raise ValueError("Image and input dimensions must be positive")
        if self.scale <= 0.0:
            raise ValueError("Transform scale must be positive")


@dataclass(frozen=True, slots=True)
class RawPrediction:
    """CenterNet candidates in original-image pixel coordinates."""

    scores: np.ndarray
    centers: np.ndarray
    corners: np.ndarray

    def __post_init__(self) -> None:
        if self.scores.ndim != 1:
            raise ValueError("scores must have shape (N,)")
        if self.centers.shape != (len(self.scores), 2):
            raise ValueError("centers must have shape (N, 2)")
        if self.corners.shape != (len(self.scores), 4, 2):
            raise ValueError("corners must have shape (N, 4, 2)")
        if not all(np.isfinite(values).all() for values in self.arrays()):
            raise ValueError("Predictions must contain finite values")

    @property
    def count(self) -> int:
        return int(len(self.scores))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.scores, self.centers, self.corners


@dataclass(frozen=True, slots=True)
class RawInferenceResult:
    """Raw predictions paired with non-identifying decoded-image metadata."""

    prediction: RawPrediction
    original_width: int
    original_height: int
    media_type: str

