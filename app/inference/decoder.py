"""Decode CenterNet output tensors and restore original-image coordinates."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from app.inference.preprocessing import map_points_to_original, valid_center_mask
from app.inference.types import RawPrediction, TransformMeta


def nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Retain local heatmap maxima using the reference 3x3 operation."""
    pad = (kernel - 1) // 2
    maximum = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)
    return heatmap * (maximum == heatmap).float()


def _gather_feature_map(feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    feature = feature.permute(0, 2, 3, 1).contiguous()
    feature = feature.view(feature.size(0), -1, feature.size(3))
    dimension = feature.size(2)
    expanded = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dimension)
    return feature.gather(1, expanded)


@torch.inference_mode()
def decode_centernet_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    down_ratio: int,
    peak_threshold: float,
    topk: int,
) -> list[RawPrediction]:
    """Decode one-class center peaks and four ordered corner offsets."""
    heatmap = nms_heatmap(outputs["hm"])
    batch_size, _, height, width = heatmap.shape
    actual_topk = min(int(topk), height * width)
    scores, indices = torch.topk(heatmap.view(batch_size, -1), actual_topk)
    ys = torch.div(indices, width, rounding_mode="floor").float()
    xs = (indices % width).float()
    regression = _gather_feature_map(outputs["reg"], indices)
    corner_offsets = _gather_feature_map(outputs["wh"], indices).view(
        batch_size,
        actual_topk,
        4,
        2,
    )
    centers = torch.stack(
        [xs + regression[:, :, 0], ys + regression[:, :, 1]],
        dim=2,
    )
    corners = centers[:, :, None, :] - corner_offsets
    centers *= float(down_ratio)
    corners *= float(down_ratio)

    decoded = []
    for batch_index in range(batch_size):
        keep = scores[batch_index] >= float(peak_threshold)
        decoded.append(
            RawPrediction(
                scores=scores[batch_index, keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                centers=centers[batch_index, keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                corners=corners[batch_index, keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            )
        )
    return decoded


def decode_to_original(
    outputs: dict[str, torch.Tensor],
    meta: TransformMeta,
    *,
    down_ratio: int,
    peak_threshold: float,
    topk: int,
) -> RawPrediction:
    """Decode a single model result and reverse resize/padding geometry."""
    decoded = decode_centernet_outputs(
        outputs,
        down_ratio=down_ratio,
        peak_threshold=peak_threshold,
        topk=topk,
    )
    if len(decoded) != 1:
        raise ValueError(f"Expected one decoded image, got {len(decoded)}")
    prediction = decoded[0]
    centers = map_points_to_original(prediction.centers, meta)
    corners = map_points_to_original(prediction.corners, meta)
    mask = valid_center_mask(centers, meta)
    return RawPrediction(
        scores=prediction.scores[mask],
        centers=centers[mask],
        corners=corners[mask],
    )
