"""Public contract describing the deployed model release."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from app.inference.manifest import ModelManifest


class SpineChainInfo(BaseModel):
    """Public deployed spine-chain parameters."""

    model_config = ConfigDict(extra="forbid")

    duplicate_iou_threshold: float = Field(ge=0.0, le=1.0)
    duplicate_center_scale: float = Field(ge=0.0)
    score_threshold: float = Field(ge=0.0, le=1.0)
    score_weight: float = Field(gt=0.0)
    min_chain_len: int = Field(gt=0)


class ModelInfoResponse(BaseModel):
    """Safe model metadata exposed without filesystem or training details."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    artifact_type: str
    format_version: int = Field(ge=1)
    backbone: str
    input_size: int = Field(gt=0)
    down_ratio: int = Field(gt=0)
    peak_threshold: float = Field(ge=0.0, le=1.0)
    topk: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    corner_order: tuple[str, str, str, str]
    spine_chain: SpineChainInfo

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> Self:
        """Translate the framework-independent manifest into an API schema."""
        return cls(
            model_id=manifest.model_id,
            artifact_type=manifest.artifact_type,
            format_version=manifest.format_version,
            backbone=manifest.backbone,
            input_size=manifest.input_size,
            down_ratio=manifest.down_ratio,
            peak_threshold=manifest.peak_threshold,
            topk=manifest.topk,
            sha256=manifest.sha256,
            corner_order=manifest.corner_order,
            spine_chain=SpineChainInfo(
                duplicate_iou_threshold=manifest.spine_chain.duplicate_iou_threshold,
                duplicate_center_scale=manifest.spine_chain.duplicate_center_scale,
                score_threshold=manifest.spine_chain.score_threshold,
                score_weight=manifest.spine_chain.score_weight,
                min_chain_len=manifest.spine_chain.min_chain_len,
            ),
        )
