"""Deployed model metadata endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_screening_service
from app.core.errors import ModelNotReadyError
from app.schemas.error import ProblemDetail
from app.schemas.model import ModelInfoResponse
from app.services.screening import ScreeningService

router = APIRouter()


@router.get(
    "",
    response_model=ModelInfoResponse,
    responses={503: {"model": ProblemDetail}},
    summary="Describe the deployed model",
)
async def model_info(
    service: Annotated[ScreeningService, Depends(get_screening_service)],
) -> ModelInfoResponse:
    """Expose reproducibility metadata without internal paths or training state."""
    manifest = service.manifest
    if not service.ready or manifest is None:
        raise ModelNotReadyError(service.status_detail)
    return ModelInfoResponse.from_manifest(manifest)
