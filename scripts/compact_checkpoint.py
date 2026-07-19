"""Convert a trusted training checkpoint into a compact inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

RUNTIME_ARG_NAMES = (
    "backbone",
    "input_size",
    "down_ratio",
    "peak_thresh",
    "eval_topk",
)


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    """Load the existing trusted checkpoint, including legacy Path values."""
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected the source checkpoint to contain a dictionary")
    return checkpoint


def build_inference_artifact(
    checkpoint: Mapping[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    """Keep model weights and the deterministic runtime contract only."""
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Source checkpoint has no non-empty model_state_dict")

    source_args = checkpoint.get("args", {})
    if not isinstance(source_args, Mapping):
        raise TypeError("Source checkpoint args must be a mapping")
    missing_args = [name for name in RUNTIME_ARG_NAMES if name not in source_args]
    if missing_args:
        raise ValueError(f"Source checkpoint is missing runtime args: {missing_args}")

    runtime_args = {name: source_args[name] for name in RUNTIME_ARG_NAMES}
    experiment_name = source_args.get("experiment_name")
    return {
        "artifact_type": "spine_centernet_inference",
        "format_version": 1,
        "model_state_dict": state_dict,
        "args": runtime_args,
        "preprocessing": {
            "color_space": "RGB",
            "resize": "longest_side_preserving_aspect_ratio",
            "padding": "centered_zero_padding",
            "normalization": "pixel_value / 255.0 - 0.5",
        },
        "postprocessing": {
            "spine_chain_enabled": True,
            "duplicate_iou_threshold": 0.18,
            "duplicate_center_scale": 0.35,
            "score_threshold": 0.18,
            "score_weight": 3.0,
            "min_chain_len": 3,
        },
        "output_contract": {
            "corner_order": ["TL", "TR", "BL", "BR"],
            "point_order": ["x", "y"],
            "coordinate_space": "original_image_pixels",
        },
        "provenance": {
            "source_checkpoint_sha256": source_sha256,
            "source_epoch": int(checkpoint["epoch"]) if "epoch" in checkpoint else None,
            "experiment_name": str(experiment_name) if experiment_name is not None else None,
        },
    }


def validate_artifact(
    source_state_dict: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
) -> None:
    """Verify safe loading retained every tensor without modification."""
    compact_state_dict = artifact.get("model_state_dict")
    if not isinstance(compact_state_dict, Mapping):
        raise TypeError("Compact artifact has no model_state_dict mapping")
    if list(source_state_dict) != list(compact_state_dict):
        raise ValueError("State-dict keys or their ordering changed during export")
    for name, source_tensor in source_state_dict.items():
        compact_tensor = compact_state_dict[name]
        if not isinstance(compact_tensor, torch.Tensor):
            raise TypeError(f"Compact state value is not a tensor: {name}")
        if not torch.equal(source_tensor, compact_tensor):
            raise ValueError(f"Tensor changed during export: {name}")


def compact_checkpoint(path: Path) -> tuple[int, int, str, str]:
    """Validate and atomically replace ``path`` with its compact artifact."""
    path = path.resolve(strict=True)
    original_size = path.stat().st_size
    original_sha256 = file_sha256(path)
    checkpoint = load_trusted_checkpoint(path)
    artifact = build_inference_artifact(checkpoint, original_sha256)
    source_state_dict = checkpoint["model_state_dict"]
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        torch.save(artifact, temp_path)
        safely_loaded = torch.load(temp_path, map_location="cpu", weights_only=True)
        if not isinstance(safely_loaded, dict):
            raise TypeError("Compact artifact did not safely load as a dictionary")
        validate_artifact(source_state_dict, safely_loaded)

        compact_size = temp_path.stat().st_size
        compact_sha256 = file_sha256(temp_path)
        os.replace(temp_path, path)
        temp_path = None
        return original_size, compact_size, original_sha256, compact_sha256
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted checkpoint to compact in place")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_size, compact_size, original_hash, compact_hash = compact_checkpoint(args.checkpoint)
    reduction = (1.0 - compact_size / original_size) * 100.0
    print(f"original_size_bytes={original_size}")
    print(f"compact_size_bytes={compact_size}")
    print(f"size_reduction_percent={reduction:.2f}")
    print(f"original_sha256={original_hash}")
    print(f"compact_sha256={compact_hash}")


if __name__ == "__main__":
    main()
