"""Startup and shutdown ownership for process-wide application resources."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.inference.runtime import (
    CenterNetRuntime,
    InferenceRuntime,
    LACenterNetRuntime,
    LAInferenceRuntime,
)
from app.services.la_screening import LAScreeningService
from app.services.screening import ScreeningService

logger = logging.getLogger("app.lifecycle")
RuntimeFactory = Callable[[Settings], InferenceRuntime]
LARuntimeFactory = Callable[[Settings], LAInferenceRuntime]


def build_runtime(
    settings: Settings,
    *,
    inference_limiter: threading.BoundedSemaphore | None = None,
) -> InferenceRuntime:
    """Build the production runtime from validated environment settings."""
    return CenterNetRuntime(
        settings.model_path,
        device_name=settings.model_device,
        max_image_pixels=settings.max_image_pixels,
        inference_concurrency=settings.inference_concurrency,
        inference_limiter=inference_limiter,
    )


def build_la_runtime(
    settings: Settings,
    *,
    inference_limiter: threading.BoundedSemaphore | None = None,
) -> LAInferenceRuntime:
    """Build the LA six-keypoint runtime from validated settings."""
    return LACenterNetRuntime(
        settings.la_model_path,
        device_name=settings.model_device,
        max_image_pixels=settings.max_image_pixels,
        inference_concurrency=settings.inference_concurrency,
        inference_limiter=inference_limiter,
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create shared services once and close them during application shutdown."""
    settings = get_settings()
    inference_limiter = threading.BoundedSemaphore(settings.inference_concurrency)
    runtime_factory: RuntimeFactory | None = getattr(
        application.state, "runtime_factory", None
    )
    la_runtime_factory: LARuntimeFactory | None = getattr(
        application.state, "la_runtime_factory", None
    )
    runtime = (
        runtime_factory(settings)
        if runtime_factory is not None
        else build_runtime(settings, inference_limiter=inference_limiter)
    )
    la_runtime = (
        la_runtime_factory(settings)
        if la_runtime_factory is not None
        else build_la_runtime(settings, inference_limiter=inference_limiter)
    )
    service = ScreeningService(runtime, max_upload_bytes=settings.max_upload_bytes)
    la_service = LAScreeningService(la_runtime, max_upload_bytes=settings.max_upload_bytes)
    application.state.screening_service = service
    application.state.la_screening_service = la_service
    logger.info("Starting AP/PA screening service")
    try:
        service.start()
    except Exception:
        logger.exception("AP/PA model startup failed; other pipelines remain available")
    logger.info("Starting LA screening service")
    try:
        la_service.start()
    except Exception:
        logger.exception("LA model startup failed; other pipelines remain available")
    try:
        yield
    finally:
        logger.info("Stopping screening services")
        try:
            la_service.close()
        finally:
            service.close()
