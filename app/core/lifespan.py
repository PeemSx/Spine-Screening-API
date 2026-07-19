"""Startup and shutdown ownership for process-wide application resources."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.inference.runtime import CenterNetRuntime, InferenceRuntime
from app.services.screening import ScreeningService

logger = logging.getLogger("app.lifecycle")
RuntimeFactory = Callable[[Settings], InferenceRuntime]


def build_runtime(settings: Settings) -> InferenceRuntime:
    """Build the production runtime from validated environment settings."""
    return CenterNetRuntime(
        settings.model_path,
        device_name=settings.model_device,
        max_image_pixels=settings.max_image_pixels,
        inference_concurrency=settings.inference_concurrency,
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create shared services once and close them during application shutdown."""
    settings = get_settings()
    runtime_factory: RuntimeFactory = getattr(
        application.state,
        "runtime_factory",
        build_runtime,
    )
    runtime = runtime_factory(settings)
    service = ScreeningService(runtime, max_upload_bytes=settings.max_upload_bytes)
    application.state.screening_service = service
    logger.info("Starting screening service")
    try:
        service.start()
    except Exception:
        logger.exception("Model runtime startup failed; liveness remains available")
    try:
        yield
    finally:
        logger.info("Stopping screening service")
        service.close()
