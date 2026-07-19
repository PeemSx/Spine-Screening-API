"""Client-visible problem response contract."""

from pydantic import BaseModel, ConfigDict, Field


class ProblemDetail(BaseModel):
    """RFC 9457-style error body with a stable machine-readable code."""

    model_config = ConfigDict(extra="forbid")

    type: str
    title: str
    status: int = Field(ge=400, le=599)
    detail: str
    code: str
    instance: str
    request_id: str | None = None
