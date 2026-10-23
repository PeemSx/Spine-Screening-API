"""Strict response contract for LA six-keypoint screening."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.model import LAModelInfoResponse
from app.schemas.prediction import (
    RESEARCH_DISCLAIMER,
    CobbResult,
    ImageInfo,
    MorphologyFeature,
    Point,
    PredictionCounts,
    PredictionSchema,
)

LA_COBB_SCOPE = (
    "LA geometry agreement calculated from the four outer landmarks; "
    "not a clinical AP/PA Cobb measurement."
)


class SixKeypoints(PredictionSchema):
    """Canonical LA vertebral landmark order."""

    top_left: Point
    top_middle: Point
    top_right: Point
    bottom_left: Point
    bottom_middle: Point
    bottom_right: Point


class LAVertebraPrediction(PredictionSchema):
    """One raw or selected LA vertebral candidate."""

    candidate_id: int = Field(ge=0)
    rank: int = Field(ge=1)
    detector_score: float = Field(ge=0.0, le=1.0)
    center: Point
    keypoints: SixKeypoints


class LACobbResult(CobbResult):
    """Outer-landmark geometry with an explicit non-clinical LA scope."""

    scope: Literal[
        "LA geometry agreement calculated from the four outer landmarks; "
        "not a clinical AP/PA Cobb measurement."
    ] = LA_COBB_SCOPE


class LAPredictionResponse(PredictionSchema):
    """Complete stateless result for one lateral radiograph."""

    prediction_id: UUID
    projection: Literal["LA"] = "LA"
    model: LAModelInfoResponse
    image: ImageInfo
    coordinate_space: Literal["original_image_pixels"] = "original_image_pixels"
    coordinate_origin: Literal["top_left"] = "top_left"
    keypoint_order: tuple[
        Literal["top_left"],
        Literal["top_middle"],
        Literal["top_right"],
        Literal["bottom_left"],
        Literal["bottom_middle"],
        Literal["bottom_right"],
    ] = (
        "top_left",
        "top_middle",
        "top_right",
        "bottom_left",
        "bottom_middle",
        "bottom_right",
    )
    measurement_landmarks: tuple[
        Literal["top_left"],
        Literal["top_right"],
        Literal["bottom_left"],
        Literal["bottom_right"],
    ] = ("top_left", "top_right", "bottom_left", "bottom_right")
    counts: PredictionCounts
    raw_predictions: list[LAVertebraPrediction] = Field(default_factory=list)
    selected_vertebrae: list[LAVertebraPrediction] = Field(default_factory=list)
    cobb: LACobbResult
    morphology: list[MorphologyFeature] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clinical_review_required: Literal[True] = True
    disclaimer: str = RESEARCH_DISCLAIMER
