"""Process liveness and model readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_screening_service
from app.core.errors import ModelNotReadyError
from app.schemas.error import ProblemDetail
from app.schemas.health import HealthResponse, ReadinessResponse
from app.services.screening import ScreeningService

router = APIRouter()


@router.get("/live", response_model=HealthResponse, summary="Check process liveness")
async def liveness() -> HealthResponse:
    """Confirm that the API process can serve requests."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ProblemDetail}},
    summary="Check model readiness",
)
async def readiness(
    service: Annotated[ScreeningService, Depends(get_screening_service)],
) -> ReadinessResponse:
    """Confirm that the shared model runtime can accept inference work."""
    manifest = service.manifest
    if not service.ready or manifest is None:
        raise ModelNotReadyError(service.status_detail)
    return ReadinessResponse(status="ready", model_id=manifest.model_id)
