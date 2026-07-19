"""Typed description of one deployable inference artifact."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpineChainConfig:
    """Versioned parameters controlling duplicate suppression and chain selection."""

    duplicate_iou_threshold: float
    duplicate_center_scale: float
    score_threshold: float
    score_weight: float
    min_chain_len: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between zero and one")
        if self.duplicate_center_scale < 0.0:
            raise ValueError("duplicate_center_scale cannot be negative")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between zero and one")
        if self.score_weight <= 0.0 or self.min_chain_len <= 0:
            raise ValueError("score_weight and min_chain_len must be positive")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Model and preprocessing values that make a prediction reproducible."""

    model_id: str
    artifact_type: str
    format_version: int
    backbone: str
    input_size: int
    down_ratio: int
    peak_threshold: float
    topk: int
    sha256: str
    spine_chain: SpineChainConfig
    corner_order: tuple[str, str, str, str] = ("TL", "TR", "BL", "BR")

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.down_ratio <= 0:
            raise ValueError("input_size and down_ratio must be positive")
        if self.input_size % self.down_ratio != 0:
            raise ValueError("input_size must be divisible by down_ratio")
        if not 0.0 <= self.peak_threshold <= 1.0:
            raise ValueError("peak_threshold must be between zero and one")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("sha256 must contain hexadecimal characters") from exc
        if self.corner_order != ("TL", "TR", "BL", "BR"):
            raise ValueError("corner_order must be TL, TR, BL, BR")
