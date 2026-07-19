"""Duplicate and spine-chain selection tests."""

import numpy as np

from app.inference.types import RawPrediction
from app.postprocessing.spine_chain import select_spine_chain


def _corners(center_x: float, center_y: float) -> list[list[float]]:
    return [
        [center_x - 10.0, center_y - 5.0],
        [center_x + 10.0, center_y - 5.0],
        [center_x - 10.0, center_y + 5.0],
        [center_x + 10.0, center_y + 5.0],
    ]


def test_suppresses_duplicate_and_preserves_raw_ids() -> None:
    centers = np.asarray([[50, 10], [51, 10], [50, 30], [50, 50]], dtype=np.float32)
    prediction = RawPrediction(
        scores=np.asarray([0.9, 0.8, 0.9, 0.9], dtype=np.float32),
        centers=centers,
        corners=np.asarray([_corners(*center) for center in centers], dtype=np.float32),
    )

    selection = select_spine_chain(prediction)

    assert len(selection.raw) == 4
    assert [candidate.candidate_id for candidate in selection.deduplicated] == [0, 2, 3]
    assert [candidate.candidate_id for candidate in selection.selected] == [0, 2, 3]
