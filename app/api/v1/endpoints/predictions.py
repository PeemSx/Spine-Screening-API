"""Synchronous single-radiograph prediction endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_screening_service
from app.schemas.error import ProblemDetail
from app.schemas.prediction import PredictionResponse
from app.services.screening import ScreeningService

router = APIRouter()


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        413: {"model": ProblemDetail},
        415: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
    summary="Screen one radiograph",
)
async def create_prediction(
    image: Annotated[UploadFile, File(description="AP or PA raster radiograph")],
    service: Annotated[ScreeningService, Depends(get_screening_service)],
) -> PredictionResponse:
    """Validate one upload and run blocking inference outside the event loop."""
    service.ensure_ready()
    declared_media_type = image.content_type
    try:
        image_bytes = await image.read(service.max_upload_bytes + 1)
    finally:
        await image.close()

    result = await run_in_threadpool(service.predict, image_bytes, declared_media_type)
    return PredictionResponse.model_validate(result)
