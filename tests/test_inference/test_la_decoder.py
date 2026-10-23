"""LA six-keypoint decoder and runtime-contract tests."""

import numpy as np
import pytest
import torch

from app.inference.decoder import decode_six_points_to_original
from app.inference.manifest import LA_KEYPOINT_ORDER
from app.inference.model import CenterNetHRNet
from app.inference.runtime import LACenterNetRuntime
from app.inference.types import TransformMeta


def test_six_point_model_head_has_twelve_channels() -> None:
    model = CenterNetHRNet(pretrained=False, keypoint_count=6)

    assert model.wh[-1].out_channels == 12


def test_decodes_six_keypoints_in_canonical_order() -> None:
    heatmap = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    heatmap[0, 0, 1, 2] = 0.9
    regression = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    offsets = torch.zeros((1, 12, 4, 4), dtype=torch.float32)
    offsets[0, :, 1, 2] = torch.tensor(
        [
            1.0,
            1.0,
            0.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
        ]
    )
    meta = TransformMeta(
        original_width=16,
        original_height=16,
        resized_width=16,
        resized_height=16,
        pad_left=0,
        pad_top=0,
        scale=1.0,
        input_size=16,
    )

    prediction = decode_six_points_to_original(
        {"hm": heatmap, "reg": regression, "wh": offsets},
        meta,
        down_ratio=4,
        peak_threshold=0.5,
        topk=10,
    )

    assert prediction.count == 1
    np.testing.assert_allclose(prediction.centers[0], [8.0, 4.0])
    np.testing.assert_allclose(
        prediction.keypoints[0],
        [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0], [4.0, 8.0], [8.0, 8.0], [12.0, 8.0]],
    )


def _checkpoint(**overrides: object) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "epoch": 33,
        "model": {"weight": torch.zeros(1)},
        "keypoint_order": list(LA_KEYPOINT_ORDER),
        "backbone": "hrnet_w18",
        "input_size": 1024,
        "down_ratio": 4,
        "val_metrics": {},
        "seed": 20260627,
        "batch_size": 48,
    }
    checkpoint.update(overrides)
    return checkpoint


def test_la_manifest_uses_explicit_inference_defaults() -> None:
    manifest = LACenterNetRuntime._parse_manifest(
        _checkpoint(),
        artifact_sha256="a" * 64,
        checkpoint_stem="LA_best_usable_recall",
    )

    assert manifest.model_id == "LA_best_usable_recall-epoch-33"
    assert manifest.keypoint_order == LA_KEYPOINT_ORDER
    assert manifest.peak_threshold == 0.1
    assert manifest.topk == 50
    assert manifest.spine_chain.score_threshold == 0.18


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", 32),
        ("backbone", "hrnet_w32"),
        ("input_size", 512),
        ("down_ratio", 2),
        ("keypoint_order", list(reversed(LA_KEYPOINT_ORDER))),
    ],
)
def test_la_manifest_rejects_incompatible_checkpoint(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        LACenterNetRuntime._parse_manifest(
            _checkpoint(**{field: value}),
            artifact_sha256="a" * 64,
            checkpoint_stem="LA_best_usable_recall",
        )
