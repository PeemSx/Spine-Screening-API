"""Image-container validation tests."""

import pytest

from app.core.errors import ImageTooLargeError, UnsupportedImageTypeError
from app.utils.image_validation import validate_image_upload


def test_detects_png_from_bytes() -> None:
    image = validate_image_upload(b"\x89PNG\r\n\x1a\ncontent", "image/png", 100)

    assert image.media_type == "image/png"


def test_rejects_mismatched_declared_type() -> None:
    with pytest.raises(UnsupportedImageTypeError):
        validate_image_upload(b"\x89PNG\r\n\x1a\ncontent", "image/jpeg", 100)


def test_rejects_upload_over_limit() -> None:
    with pytest.raises(ImageTooLargeError):
        validate_image_upload(b"\xff\xd8\xffcontent", "image/jpeg", 4)
