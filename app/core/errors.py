"""Stable application errors and their HTTP representation."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An expected API failure safe to expose to a client."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        title: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.title = title
        self.detail = detail
        self.headers = headers


class ModelNotReadyError(ApiError):
    """Raised when a request requires a model that is not ready."""

    def __init__(self, detail: str = "The inference model is not ready.") -> None:
        super().__init__(
            status_code=503,
            code="model_not_ready",
            title="Model not ready",
            detail=detail,
            headers={"Retry-After": "30"},
        )


class ImageTooLargeError(ApiError):
    """Raised when an upload exceeds the configured byte limit."""

    def __init__(self, max_upload_bytes: int) -> None:
        super().__init__(
            status_code=413,
            code="image_too_large",
            title="Image too large",
            detail=f"The image exceeds the {max_upload_bytes}-byte upload limit.",
        )


class UnsupportedImageTypeError(ApiError):
    """Raised when bytes are not one of the supported raster containers."""

    def __init__(self) -> None:
        super().__init__(
            status_code=415,
            code="unsupported_media_type",
            title="Unsupported media type",
            detail="Upload a JPEG, PNG, BMP, or single-image TIFF radiograph.",
        )


class InvalidImageError(ApiError):
    """Raised when an upload is empty or cannot be treated as an image."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=422,
            code="invalid_image",
            title="Invalid image",
            detail=detail,
        )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Return an RFC 9457-style problem document for an expected failure."""
    request_id = getattr(request.state, "request_id", None)
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": exc.title,
        "status": exc.status_code,
        "detail": exc.detail,
        "code": exc.code,
        "instance": str(request.url.path),
    }
    if request_id is not None:
        body["request_id"] = str(request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=exc.headers,
        media_type="application/problem+json",
    )


def register_exception_handlers(application: FastAPI) -> None:
    """Register handlers in one place so the application factory stays small."""
    application.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
