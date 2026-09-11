"""The inspector displays recorded values, including invalid/missing feedback."""

import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.view_pico_hands import CaptureViewer
from gear_sonic.utils.teleop.pico_hand_log import INPUT_SCHEMA


@pytest.fixture
def viewer(tmp_path):
    pytest.importorskip("pinocchio")
    poses = json.loads((Path(__file__).parent / "fixtures/dex3_thumb_poses.json").read_text())
    a = {key: np.zeros((2, *shape), dtype=dtype) for key, (dtype, shape) in INPUT_SCHEMA.items()}
    for side, name in enumerate(("left", "right")):
        a["poses"][:, side, 1:, :3] = poses["straight"][name]
    a["tick_ns"][:] = [1_000_000, 21_000_000]
    a["source_timestamp_ns"][:] = [[10, 10], [11, 10]]
    a["active"][:] = 1
    a["flags"][:] = 15
    a["flags"][0, 0, 0] = 2**63 + 15
    a.update(
        profile=np.array(["dex3"]),
        measured=np.full((2, 2, 7), 0.12, dtype=np.float64),
        captured_target=np.full((2, 2, 7), 0.34, dtype=np.float32),
        captured_command=np.full((2, 2, 7), 0.23, dtype=np.float32),
        captured_state=np.array([[3, 2], [1, 0]], dtype=np.int32),
        captured_reason=np.array([[0, 0], [8, 10]], dtype=np.int32),
        hand_sent=np.array([[True, True], [False, False]]),
    )
    a["measured"][1, 1] = np.nan
    np.savez(tmp_path / "capture_000000.npz", **a)
    return CaptureViewer(tmp_path), a


def test_preserves_actual_pipeline_values_and_missing_feedback(viewer):
    subject, arrays = viewer
    result = json.loads(subject.capture("capture_000000.npz"))
    first, last = result["frames"]
    for key, stored in (("target", "captured_target"), ("command", "captured_command"), ("measured", "measured")):
        np.testing.assert_allclose(first[key], arrays[stored][0], rtol=0, atol=0)
    assert first["flags"][0][0] == str(2**63 + 15)
    assert last["measured"][1] == [None] * 7
    assert last["robot"]["measured"][1] is None
    assert last["state"] == ["HOLDING", "WAITING"]
    assert last["reason"] == ["SOURCE_STALE", "FEEDBACK_UNAVAILABLE"]
    assert last["sent"] == [False, False]
    assert first["source_age_ms"] == [None, None]
    assert last["source_age_ms"] == [0, None]
    assert np.asarray(first["pico"]).shape == (2, 26, 3)
    np.testing.assert_allclose(np.asarray(first["pico"])[:, 1], 0, atol=1e-12)
    for side in range(2):
        assert [len(chain) for chain in first["robot"]["target"][side]] == [4, 3, 3]
        assert first["robot"]["target"][side] != first["robot"]["command"][side]


def test_rejects_path_escape_and_missing_actual_mapping(viewer):
    subject, arrays = viewer
    for name in ("../capture_000000.npz", "/tmp/capture_000000.npz", "capture_000009.npz"):
        with pytest.raises(ValueError, match="filename"):
            subject.capture(name)
    arrays.pop("captured_target")
    np.savez(subject.directory / "capture_000001.npz", **arrays)
    with pytest.raises(ValueError, match="captured_target"):
        subject.capture("capture_000001.npz")
