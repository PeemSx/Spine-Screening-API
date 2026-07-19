"""Deterministic preprocessing and geometry tests."""

import cv2
import numpy as np
import torch

from app.inference.preprocessing import (
    decode_image_bytes,
    image_to_tensor,
    map_points_to_original,
    resize_pad_image,
)


def test_resize_pad_and_reverse_mapping() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    padded, meta = resize_pad_image(image, input_size=1024)

    assert padded.shape == (1024, 1024, 3)
    assert meta.resized_width == 1024
    assert meta.resized_height == 512
    assert meta.pad_left == 0
    assert meta.pad_top == 256
    original_points = np.asarray([[0.0, 0.0], [199.0, 99.0]], dtype=np.float32)
    model_points = original_points * meta.scale
    model_points += np.asarray([meta.pad_left, meta.pad_top], dtype=np.float32)
    np.testing.assert_allclose(map_points_to_original(model_points, meta), original_points)


def test_training_normalization_is_preserved() -> None:
    image = np.asarray([[[0, 127, 255]]], dtype=np.uint8)

    tensor = image_to_tensor(image, torch.device("cpu"))

    assert tensor.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(
        tensor.numpy().reshape(3),
        np.asarray([-0.5, 127.0 / 255.0 - 0.5, 0.5], dtype=np.float32),
        atol=1e-7,
    )


def test_decodes_png_bytes_as_rgb() -> None:
    bgr = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", bgr)
    assert encoded

    rgb = decode_image_bytes(buffer.tobytes(), max_image_pixels=10)

    assert rgb.tolist() == [[[30, 20, 10]]]
