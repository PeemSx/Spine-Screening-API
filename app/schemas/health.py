"""Health endpoint schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Process health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Response returned only after the model runtime is ready."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    model_id: str
