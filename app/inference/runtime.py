"""Safe lifecycle and raw prediction runtime for the CenterNet artifact."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn

from app.inference.decoder import decode_six_points_to_original, decode_to_original
from app.inference.manifest import (
    LA_KEYPOINT_ORDER,
    LAModelManifest,
    ModelManifest,
    SpineChainConfig,
)
from app.inference.model import SUPPORTED_BACKBONES, build_centernet_model
from app.inference.preprocessing import preprocess_image_bytes
from app.inference.types import RawInferenceResult, RawSixPointInferenceResult


class RuntimeUnavailableError(RuntimeError):
    """Raised when the runtime cannot currently execute inference."""


class InferenceRuntime(Protocol):
    """Interface the service layer needs from any model implementation."""

    @property
    def ready(self) -> bool: ...

    @property
    def status_detail(self) -> str: ...

    @property
    def manifest(self) -> ModelManifest | None: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def predict(self, image_bytes: bytes, media_type: str) -> RawInferenceResult: ...


class LAInferenceRuntime(Protocol):
    """Interface required by the LA six-keypoint screening service."""

    @property
    def ready(self) -> bool: ...

    @property
    def status_detail(self) -> str: ...

    @property
    def manifest(self) -> LAModelManifest | None: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def predict(self, image_bytes: bytes, media_type: str) -> RawSixPointInferenceResult: ...


def file_sha256(path: Path) -> str:
    """Calculate an artifact digest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently ignoring an explicit CUDA request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeUnavailableError("CUDA was requested but is not available.")
    if requested not in {"cpu", "cuda"}:
        raise RuntimeUnavailableError(f"Unsupported inference device: {requested}")
    return torch.device(requested)


def _mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Artifact field '{key}' must be a mapping")
    return value


class CenterNetRuntime:
    """Load one trusted artifact and execute bounded single-image inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
        max_image_pixels: int = 50_000_000,
        inference_concurrency: int = 1,
        inference_limiter: threading.BoundedSemaphore | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device_name = device_name
        self._max_image_pixels = max_image_pixels
        self._semaphore = inference_limiter or threading.BoundedSemaphore(inference_concurrency)
        self._model: nn.Module | None = None
        self._manifest: ModelManifest | None = None
        self._device: torch.device | None = None
        self._ready = False
        self._status_detail = "The model runtime has not started."

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status_detail(self) -> str:
        return self._status_detail

    @property
    def manifest(self) -> ModelManifest | None:
        return self._manifest

    @property
    def device(self) -> torch.device | None:
        return self._device

    def start(self) -> None:
        """Safely load, strictly validate, move, and warm the model once."""
        if self._ready:
            return
        path = self._checkpoint_path.resolve(strict=True)
        self._status_detail = "The model runtime is loading."
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(artifact, Mapping):
                raise ValueError("Inference artifact must contain a mapping")
            manifest = self._parse_manifest(artifact, artifact_sha256=file_sha256(path))
            state_dict = _mapping(artifact, "model_state_dict")
            device = resolve_device(self._device_name)
            model = build_centernet_model(manifest.backbone, pretrained=False)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            self._warm_up(model, device, manifest)
            self._model = model
            self._device = device
            self._manifest = manifest
            self._ready = True
            self._status_detail = "The model runtime is ready."
        except Exception:
            self._model = None
            self._device = None
            self._manifest = None
            self._ready = False
            self._status_detail = "The model runtime failed to start."
            raise

    def close(self) -> None:
        """Release process-owned model resources."""
        device = self._device
        self._model = None
        self._device = None
        self._manifest = None
        self._ready = False
        self._status_detail = "The model runtime is stopped."
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image_bytes: bytes, media_type: str) -> RawInferenceResult:
        """Decode, preprocess, infer, and return raw original-pixel candidates."""
        model = self._model
        device = self._device
        manifest = self._manifest
        if not self._ready or model is None or device is None or manifest is None:
            raise RuntimeUnavailableError(self._status_detail)

        with self._semaphore:
            image_rgb, tensor, meta = preprocess_image_bytes(
                image_bytes,
                input_size=manifest.input_size,
                max_image_pixels=self._max_image_pixels,
                device=device,
            )
            with torch.inference_mode():
                outputs = model(tensor)
            self._validate_outputs(outputs, manifest)
            prediction = decode_to_original(
                outputs,
                meta,
                down_ratio=manifest.down_ratio,
                peak_threshold=manifest.peak_threshold,
                topk=manifest.topk,
            )
        height, width = image_rgb.shape[:2]
        return RawInferenceResult(
            prediction=prediction,
            original_width=width,
            original_height=height,
            media_type=media_type,
        )

    @staticmethod
    def _parse_manifest(
        artifact: Mapping[str, Any],
        *,
        artifact_sha256: str,
    ) -> ModelManifest:
        artifact_type = artifact.get("artifact_type")
        format_version = artifact.get("format_version")
        if artifact_type != "spine_centernet_inference":
            raise ValueError(f"Unsupported artifact_type: {artifact_type}")
        if format_version != 1:
            raise ValueError(f"Unsupported format_version: {format_version}")

        args = _mapping(artifact, "args")
        backbone = str(args.get("backbone"))
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone in artifact: {backbone}")
        input_size = int(args["input_size"])
        down_ratio = int(args["down_ratio"])
        peak_threshold = float(args["peak_thresh"])
        topk = int(args["eval_topk"])

        preprocessing = _mapping(artifact, "preprocessing")
        if preprocessing.get("normalization") != "pixel_value / 255.0 - 0.5":
            raise ValueError("Artifact normalization does not match the deployed preprocessor")
        output_contract = _mapping(artifact, "output_contract")
        corner_order = tuple(output_contract.get("corner_order", ()))
        postprocessing = _mapping(artifact, "postprocessing")
        if postprocessing.get("spine_chain_enabled") is not True:
            raise ValueError("The deployed artifact must enable spine-chain selection")
        spine_chain = SpineChainConfig(
            duplicate_iou_threshold=float(postprocessing["duplicate_iou_threshold"]),
            duplicate_center_scale=float(postprocessing["duplicate_center_scale"]),
            score_threshold=float(postprocessing["score_threshold"]),
            score_weight=float(postprocessing["score_weight"]),
            min_chain_len=int(postprocessing["min_chain_len"]),
        )
        provenance = _mapping(artifact, "provenance")
        experiment = str(provenance.get("experiment_name") or "centernet")
        epoch = provenance.get("source_epoch")
        model_id = f"{experiment}-epoch-{int(epoch)}" if epoch is not None else experiment
        return ModelManifest(
            model_id=model_id,
            artifact_type=str(artifact_type),
            format_version=int(format_version),
            backbone=backbone,
            input_size=input_size,
            down_ratio=down_ratio,
            peak_threshold=peak_threshold,
            topk=topk,
            sha256=artifact_sha256,
            spine_chain=spine_chain,
            corner_order=corner_order,  # type: ignore[arg-type]
        )

    @classmethod
    def _warm_up(
        cls,
        model: nn.Module,
        device: torch.device,
        manifest: ModelManifest,
    ) -> None:
        dummy = torch.zeros(
            (1, 3, manifest.input_size, manifest.input_size),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            outputs = model(dummy)
        cls._validate_outputs(outputs, manifest)

    @staticmethod
    def _validate_outputs(
        outputs: dict[str, torch.Tensor],
        manifest: ModelManifest,
    ) -> None:
        _validate_output_tensors(outputs, manifest, keypoint_count=4)


def _validate_output_tensors(
    outputs: dict[str, torch.Tensor],
    manifest: ModelManifest | LAModelManifest,
    *,
    keypoint_count: int,
) -> None:
    """Validate CenterNet tensor keys, shapes, finiteness, and heatmap range."""
    output_size = manifest.input_size // manifest.down_ratio
    expected_shapes = {
        "hm": (1, 1, output_size, output_size),
        "reg": (1, 2, output_size, output_size),
        "wh": (1, keypoint_count * 2, output_size, output_size),
    }
    if set(outputs) != set(expected_shapes):
        raise ValueError(f"Unexpected model output keys: {sorted(outputs)}")
    for name, expected_shape in expected_shapes.items():
        value = outputs[name]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Output '{name}' has shape {tuple(value.shape)}, expected {expected_shape}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Output '{name}' contains non-finite values")
    heatmap = outputs["hm"]
    if float(heatmap.min()) < 0.0 or float(heatmap.max()) > 1.0:
        raise ValueError("Heatmap output must be within zero and one")


class LACenterNetRuntime:
    """Strict runtime for the supplied LA six-keypoint training checkpoint."""

    _EXPECTED_CHECKPOINT_KEYS = {
        "epoch",
        "model",
        "keypoint_order",
        "backbone",
        "input_size",
        "down_ratio",
        "val_metrics",
        "seed",
        "batch_size",
    }

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
        max_image_pixels: int = 50_000_000,
        inference_concurrency: int = 1,
        inference_limiter: threading.BoundedSemaphore | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device_name = device_name
        self._max_image_pixels = max_image_pixels
        self._semaphore = inference_limiter or threading.BoundedSemaphore(inference_concurrency)
        self._model: nn.Module | None = None
        self._manifest: LAModelManifest | None = None
        self._device: torch.device | None = None
        self._ready = False
        self._status_detail = "The LA model runtime has not started."

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status_detail(self) -> str:
        return self._status_detail

    @property
    def manifest(self) -> LAModelManifest | None:
        return self._manifest

    @property
    def device(self) -> torch.device | None:
        return self._device

    def start(self) -> None:
        """Safely load and strictly validate the LA training checkpoint."""
        if self._ready:
            return
        path = self._checkpoint_path.resolve(strict=True)
        self._status_detail = "The LA model runtime is loading."
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(artifact, Mapping):
                raise ValueError("LA checkpoint must contain a mapping")
            manifest = self._parse_manifest(
                artifact,
                artifact_sha256=file_sha256(path),
                checkpoint_stem=path.stem,
            )
            state_dict = _mapping(artifact, "model")
            device = resolve_device(self._device_name)
            model = build_centernet_model(
                manifest.backbone,
                pretrained=False,
                keypoint_count=6,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            self._warm_up(model, device, manifest)
            self._model = model
            self._device = device
            self._manifest = manifest
            self._ready = True
            self._status_detail = "The LA model runtime is ready."
        except Exception:
            self._model = None
            self._device = None
            self._manifest = None
            self._ready = False
            self._status_detail = "The LA model runtime failed to start."
            raise

    def close(self) -> None:
        """Release process-owned LA model resources."""
        device = self._device
        self._model = None
        self._device = None
        self._manifest = None
        self._ready = False
        self._status_detail = "The LA model runtime is stopped."
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image_bytes: bytes, media_type: str) -> RawSixPointInferenceResult:
        """Return six-point candidates in original-image pixel coordinates."""
        model = self._model
        device = self._device
        manifest = self._manifest
        if not self._ready or model is None or device is None or manifest is None:
            raise RuntimeUnavailableError(self._status_detail)

        with self._semaphore:
            image_rgb, tensor, meta = preprocess_image_bytes(
                image_bytes,
                input_size=manifest.input_size,
                max_image_pixels=self._max_image_pixels,
                device=device,
            )
            with torch.inference_mode():
                outputs = model(tensor)
            _validate_output_tensors(outputs, manifest, keypoint_count=6)
            prediction = decode_six_points_to_original(
                outputs,
                meta,
                down_ratio=manifest.down_ratio,
                peak_threshold=manifest.peak_threshold,
                topk=manifest.topk,
            )
        height, width = image_rgb.shape[:2]
        return RawSixPointInferenceResult(
            prediction=prediction,
            original_width=width,
            original_height=height,
            media_type=media_type,
        )

    @classmethod
    def _parse_manifest(
        cls,
        artifact: Mapping[str, Any],
        *,
        artifact_sha256: str,
        checkpoint_stem: str,
    ) -> LAModelManifest:
        actual_keys = set(artifact)
        if actual_keys != cls._EXPECTED_CHECKPOINT_KEYS:
            missing = sorted(cls._EXPECTED_CHECKPOINT_KEYS - actual_keys)
            unexpected = sorted(actual_keys - cls._EXPECTED_CHECKPOINT_KEYS)
            raise ValueError(
                f"Unexpected LA checkpoint fields; missing={missing}, unexpected={unexpected}"
            )
        epoch = int(artifact["epoch"])
        backbone = str(artifact["backbone"])
        input_size = int(artifact["input_size"])
        down_ratio = int(artifact["down_ratio"])
        keypoint_order = tuple(artifact["keypoint_order"])
        if epoch != 33:
            raise ValueError(f"Expected LA checkpoint epoch 33, got {epoch}")
        if backbone != "hrnet_w18":
            raise ValueError(f"Expected LA backbone hrnet_w18, got {backbone}")
        if input_size != 1024 or down_ratio != 4:
            raise ValueError("Expected LA input_size=1024 and down_ratio=4")
        if keypoint_order != LA_KEYPOINT_ORDER:
            raise ValueError(f"Unexpected LA keypoint order: {keypoint_order}")
        _mapping(artifact, "model")
        _mapping(artifact, "val_metrics")
        return LAModelManifest(
            model_id=f"{checkpoint_stem}-epoch-{epoch}",
            artifact_type="spine_centernet_la_six_keypoint_checkpoint",
            format_version=1,
            backbone=backbone,
            input_size=input_size,
            down_ratio=down_ratio,
            peak_threshold=0.1,
            topk=50,
            sha256=artifact_sha256,
            checkpoint_epoch=epoch,
            spine_chain=SpineChainConfig(
                duplicate_iou_threshold=0.18,
                duplicate_center_scale=0.35,
                score_threshold=0.18,
                score_weight=3.0,
                min_chain_len=3,
            ),
            keypoint_order=keypoint_order,  # type: ignore[arg-type]
        )

    @staticmethod
    def _warm_up(
        model: nn.Module,
        device: torch.device,
        manifest: LAModelManifest,
    ) -> None:
        dummy = torch.zeros(
            (1, 3, manifest.input_size, manifest.input_size),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            outputs = model(dummy)
        _validate_output_tensors(outputs, manifest, keypoint_count=6)


class UnavailableInferenceRuntime:
    """Test/degraded-mode runtime that never advertises readiness."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def ready(self) -> bool:
        return False

    @property
    def status_detail(self) -> str:
        return self._reason

    @property
    def manifest(self) -> None:
        return None

    def start(self) -> None:
        """No-op startup preserving the real runtime lifecycle contract."""

    def close(self) -> None:
        """No-op shutdown preserving the real runtime lifecycle contract."""

    def predict(self, image_bytes: bytes, media_type: str) -> RawInferenceResult:
        del image_bytes, media_type
        raise RuntimeUnavailableError(self._reason)
