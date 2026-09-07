# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_g1_act_observation import _read_first_row


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "dataset"
    parquet = root / "data/chunk-000/file-000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet)
    return root


def _row(episode: int, frame: int, timestamp: float | None = None) -> dict:
    return {
        "episode_index": episode,
        "frame_index": frame,
        "timestamp": frame / 30.0 if timestamp is None else timestamp,
        "observation.state": np.zeros(28, dtype=np.float32),
    }


def test_default_selects_first_row(tmp_path: Path) -> None:
    state, episode, frame, timestamp = _read_first_row(_dataset(tmp_path, [_row(7, 0), _row(7, 1)]))
    assert state.shape == (28,)
    assert episode == 7
    assert frame == 0
    assert timestamp == 0.0


def test_selects_requested_frame_with_valid_timestamp(tmp_path: Path) -> None:
    state, episode, frame, timestamp = _read_first_row(
        _dataset(tmp_path, [_row(7, 0), _row(7, 1), _row(7, 2)]), 2
    )
    assert state.shape == (28,)
    assert (episode, frame, timestamp) == (7, 2, 2 / 30.0)


def test_rejects_frame_from_later_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one row in first episode"):
        _read_first_row(_dataset(tmp_path, [_row(7, 0), _row(8, 1)]), 1)


def test_rejects_unknown_offset_and_bad_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one row"):
        _read_first_row(_dataset(tmp_path, [_row(7, 0)]), 3)
    with pytest.raises(ValueError, match="expected approximately"):
        _read_first_row(_dataset(tmp_path, [_row(7, 0), _row(7, 1, timestamp=0.25)]), 1)


def test_rejects_negative_frame_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer >= 0"):
        _read_first_row(_dataset(tmp_path, [_row(7, 0)]), -1)


def test_requires_source_timeline_to_begin_at_episode_frame_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="first parquet row"):
        _read_first_row(_dataset(tmp_path, [_row(7, 1), _row(7, 2)]), 1)
