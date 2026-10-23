"""LA six-keypoint model, readiness, and prediction endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_la_screening_service
from app.core.errors import ModelNotReadyError
from app.schemas.error import ProblemDetail
from app.schemas.health import ReadinessResponse
from app.schemas.la_prediction import LAPredictionResponse
from app.schemas.model import LAModelInfoResponse
from app.services.la_screening import LAScreeningService

router = APIRouter()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ProblemDetail}},
    summary="Check LA model readiness",
)
async def readiness(
    service: Annotated[LAScreeningService, Depends(get_la_screening_service)],
) -> ReadinessResponse:
    """Confirm that the LA runtime can accept inference work."""
    manifest = service.manifest
    if not service.ready or manifest is None:
        raise ModelNotReadyError(service.status_detail)
    return ReadinessResponse(status="ready", model_id=manifest.model_id)


@router.get(
    "/model",
    response_model=LAModelInfoResponse,
    responses={503: {"model": ProblemDetail}},
    summary="Describe the deployed LA six-keypoint model",
)
async def model_info(
    service: Annotated[LAScreeningService, Depends(get_la_screening_service)],
) -> LAModelInfoResponse:
    """Expose reproducibility metadata without internal checkpoint paths."""
    manifest = service.manifest
    if not service.ready or manifest is None:
        raise ModelNotReadyError(service.status_detail)
    return LAModelInfoResponse.from_manifest(manifest)


@router.post(
    "/predictions",
    response_model=LAPredictionResponse,
    responses={
        413: {"model": ProblemDetail},
        415: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
    summary="Screen one lateral radiograph with six landmarks",
)
async def create_prediction(
    image: Annotated[UploadFile, File(description="Lateral raster radiograph")],
    service: Annotated[LAScreeningService, Depends(get_la_screening_service)],
) -> LAPredictionResponse:
    """Validate one upload and run blocking LA inference outside the event loop."""
    service.ensure_ready()
    declared_media_type = image.content_type
    try:
        image_bytes = await image.read(service.max_upload_bytes + 1)
    finally:
        await image.close()

    result = await run_in_threadpool(service.predict, image_bytes, declared_media_type)
    return LAPredictionResponse.model_validate(result)
