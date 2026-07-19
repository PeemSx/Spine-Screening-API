"""CenterNet decoding tests."""

import numpy as np
import torch

from app.inference.decoder import decode_to_original
from app.inference.types import TransformMeta


def test_decodes_center_and_tl_tr_bl_br_corners() -> None:
    heatmap = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    heatmap[0, 0, 1, 2] = 0.9
    regression = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    regression[0, :, 1, 2] = torch.tensor([0.25, 0.5])
    corner_offsets = torch.zeros((1, 8, 4, 4), dtype=torch.float32)
    corner_offsets[0, :, 1, 2] = torch.tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
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

    prediction = decode_to_original(
        {"hm": heatmap, "reg": regression, "wh": corner_offsets},
        meta,
        down_ratio=4,
        peak_threshold=0.5,
        topk=10,
    )

    assert prediction.count == 1
    np.testing.assert_allclose(prediction.centers[0], [9.0, 6.0])
    np.testing.assert_allclose(
        prediction.corners[0],
        [[5.0, 2.0], [13.0, 2.0], [5.0, 10.0], [13.0, 10.0]],
    )
