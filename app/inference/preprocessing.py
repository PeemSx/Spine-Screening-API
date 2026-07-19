"""Deterministic image decoding, resize/pad, and coordinate transforms."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from app.inference.types import TransformMeta


class ImageDecodingError(ValueError):
    """Raised when image bytes cannot satisfy the model input contract."""


def decode_image_bytes(image_bytes: bytes, max_image_pixels: int) -> np.ndarray:
    """Decode one raster image as uint8 RGB and enforce a decoded-pixel limit."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ImageDecodingError("The uploaded raster could not be decoded as one RGB image.")

    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageDecodingError("The decoded image has invalid dimensions.")
    if height * width > max_image_pixels:
        raise ImageDecodingError(
            f"The decoded image exceeds the {max_image_pixels}-pixel safety limit."
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_pad_image(
    image_rgb: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, TransformMeta]:
    """Fit the longest side to a square input and apply centered black padding."""
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    original_height, original_width = image_rgb.shape[:2]
    scale = float(input_size) / float(max(original_height, original_width))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    meta = TransformMeta(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
        input_size=input_size,
    )
    return padded, meta


def image_to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Apply the training-time normalization and create one NCHW batch."""
    normalized = image_rgb.astype(np.float32) / 255.0 - 0.5
    nchw = np.transpose(normalized, (2, 0, 1))[None]
    return torch.from_numpy(np.ascontiguousarray(nchw)).to(device)


def preprocess_image_bytes(
    image_bytes: bytes,
    *,
    input_size: int,
    max_image_pixels: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, TransformMeta]:
    """Decode and transform upload bytes into the model tensor and geometry metadata."""
    image_rgb = decode_image_bytes(image_bytes, max_image_pixels=max_image_pixels)
    padded_rgb, meta = resize_pad_image(image_rgb, input_size=input_size)
    return image_rgb, image_to_tensor(padded_rgb, device), meta


def map_points_to_original(points: np.ndarray, meta: TransformMeta) -> np.ndarray:
    """Reverse padding and scale without rounding or clamping floating-point points."""
    mapped = points.copy().astype(np.float32)
    mapped[..., 0] = (mapped[..., 0] - float(meta.pad_left)) / meta.scale
    mapped[..., 1] = (mapped[..., 1] - float(meta.pad_top)) / meta.scale
    return mapped


def valid_center_mask(centers: np.ndarray, meta: TransformMeta) -> np.ndarray:
    """Identify centers inside the original image; corners remain deliberately unclamped."""
    if len(centers) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (centers[:, 0] >= 0.0)
        & (centers[:, 0] < float(meta.original_width))
        & (centers[:, 1] >= 0.0)
        & (centers[:, 1] < float(meta.original_height))
    )

