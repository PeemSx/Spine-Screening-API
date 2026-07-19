"""Bounded validation of uploaded raster-image containers."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import ImageTooLargeError, InvalidImageError, UnsupportedImageTypeError

GENERIC_MEDIA_TYPES = {"", "application/octet-stream"}
MEDIA_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/tif": "image/tiff"}


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    """Image bytes paired with the media type detected from their signature."""

    data: bytes
    media_type: str


def detect_image_media_type(data: bytes) -> str | None:
    """Detect the supported container using its signature, not its filename."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return None


def validate_image_upload(
    data: bytes,
    declared_media_type: str | None,
    max_upload_bytes: int,
) -> ValidatedImage:
    """Reject empty, oversized, unsupported, or mislabeled image containers."""
    if not data:
        raise InvalidImageError("The uploaded image is empty.")
    if len(data) > max_upload_bytes:
        raise ImageTooLargeError(max_upload_bytes)

    detected_media_type = detect_image_media_type(data)
    if detected_media_type is None:
        raise UnsupportedImageTypeError()

    declared = (declared_media_type or "").lower().split(";", maxsplit=1)[0].strip()
    declared = MEDIA_TYPE_ALIASES.get(declared, declared)
    if declared not in GENERIC_MEDIA_TYPES and declared != detected_media_type:
        raise UnsupportedImageTypeError()
    return ValidatedImage(data=data, media_type=detected_media_type)
