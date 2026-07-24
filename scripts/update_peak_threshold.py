"""Atomically update the decoder peak threshold in an inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_peak_threshold(path: Path, threshold: float) -> tuple[float, str, str]:
    """Update ``args.peak_thresh`` while preserving every model tensor."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between zero and one")

    path = path.resolve(strict=True)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError("Expected the artifact to contain a dictionary")
    if artifact.get("artifact_type") != "spine_centernet_inference":
        raise ValueError("Expected a spine_centernet_inference artifact")

    args = artifact.get("args")
    if not isinstance(args, dict) or "peak_thresh" not in args:
        raise ValueError("Artifact is missing args.peak_thresh")
    state_dict = artifact.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Artifact has no non-empty model_state_dict")

    previous = float(args["peak_thresh"])
    original_hash = file_sha256(path)
    args["peak_thresh"] = float(threshold)

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
        reloaded: Any = torch.load(temp_path, map_location="cpu", weights_only=True)
        if not isinstance(reloaded, Mapping):
            raise TypeError("Updated artifact did not safely load as a mapping")
        reloaded_args = reloaded.get("args")
        if not isinstance(reloaded_args, Mapping):
            raise TypeError("Updated artifact has no args mapping")
        if float(reloaded_args.get("peak_thresh", -1.0)) != threshold:
            raise ValueError("Updated artifact did not retain the requested threshold")

        reloaded_state = reloaded.get("model_state_dict")
        if not isinstance(reloaded_state, Mapping):
            raise TypeError("Updated artifact has no model_state_dict mapping")
        if list(state_dict) != list(reloaded_state):
            raise ValueError("Model state-dict keys or ordering changed")
        for name, tensor in state_dict.items():
            updated_tensor = reloaded_state[name]
            if not isinstance(tensor, torch.Tensor) or not isinstance(updated_tensor, torch.Tensor):
                raise TypeError(f"Model state value is not a tensor: {name}")
            if not torch.equal(tensor, updated_tensor):
                raise ValueError(f"Model tensor changed while updating threshold: {name}")

        updated_hash = file_sha256(temp_path)
        os.replace(temp_path, path)
        temp_path = None
        return previous, original_hash, updated_hash
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous, original_hash, updated_hash = update_peak_threshold(
        args.artifact,
        args.threshold,
    )
    print(f"previous_peak_threshold={previous}")
    print(f"updated_peak_threshold={args.threshold}")
    print(f"original_sha256={original_hash}")
    print(f"updated_sha256={updated_hash}")


if __name__ == "__main__":
    main()
