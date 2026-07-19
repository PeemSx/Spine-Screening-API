"""Strict single-radiograph prediction response contracts."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.model import ModelInfoResponse

RESEARCH_DISCLAIMER = (
    "Research screening-support measurements only. These outputs are not a diagnosis "
    "and require clinical interpretation."
)


class PredictionSchema(BaseModel):
    """Shared strict and finite-number behavior for prediction schemas."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(PredictionSchema):
    """A floating-point point in original-image pixel coordinates."""

    x: float
    y: float


class OrderedCorners(PredictionSchema):
    """Canonical vertebral landmark order: TL, TR, BL, BR."""

    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point


class VertebraPrediction(PredictionSchema):
    """One detector candidate or selected spine-chain vertebra."""

    candidate_id: int = Field(ge=0)
    rank: int = Field(ge=1)
    detector_score: float = Field(ge=0.0, le=1.0)
    center: Point
    corners: OrderedCorners


class CobbLine(PredictionSchema):
    """Endplate-axis line used by a Cobb measurement."""

    start: Point
    end: Point
    vertebra_candidate_id: int = Field(ge=0)


class CobbResult(PredictionSchema):
    """Computational Cobb geometry; validity is not a clinical conclusion."""

    valid: bool
    vertebra_count: int = Field(ge=0)
    cobb_1_deg: float | None = None
    cobb_2_deg: float | None = None
    cobb_3_deg: float | None = None
    major_lines: list[CobbLine] = Field(default_factory=list, max_length=2)
    cobb_2_lines: list[CobbLine] = Field(default_factory=list, max_length=2)
    cobb_3_lines: list[CobbLine] = Field(default_factory=list, max_length=2)


class MorphologyFeature(PredictionSchema):
    """Measurement-only geometry for one selected vertebra."""

    rank: int = Field(ge=1)
    candidate_id: int = Field(ge=0)
    detector_score: float = Field(ge=0.0, le=1.0)
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


class ImageInfo(PredictionSchema):
    """Non-identifying properties of the decoded input image."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    media_type: str


class PredictionCounts(PredictionSchema):
    """Keep detector and spine-chain counts visibly distinct."""

    raw: int = Field(ge=0)
    deduplicated: int = Field(ge=0)
    selected: int = Field(ge=0)


class PredictionResponse(PredictionSchema):
    """Complete stateless screening-support result for one radiograph."""

    prediction_id: UUID
    model: ModelInfoResponse
    image: ImageInfo
    coordinate_space: Literal["original_image_pixels"] = "original_image_pixels"
    coordinate_origin: Literal["top_left"] = "top_left"
    corner_order: tuple[
        Literal["TL"],
        Literal["TR"],
        Literal["BL"],
        Literal["BR"],
    ] = ("TL", "TR", "BL", "BR")
    counts: PredictionCounts
    raw_predictions: list[VertebraPrediction] | None = None
    selected_vertebrae: list[VertebraPrediction] = Field(default_factory=list)
    cobb: CobbResult
    morphology: list[MorphologyFeature] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clinical_review_required: Literal[True] = True
    disclaimer: str = RESEARCH_DISCLAIMER
