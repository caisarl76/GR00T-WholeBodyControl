import hashlib
import json

import numpy as np
import pytest

from gear_sonic.utils.inference.loop_motion import READY_POSE_PATH, load_ready_pose

SHAPES = dict(
    token=(64,), hands=(14,), state=(28,), raise_tokens=(83, 64), raise_hands=(83, 14),
    lower_tokens=(83, 64), lower_hands=(83, 14),
)


def _good():
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


def test_load_ready_pose_shapes_and_sha():
    pose = load_ready_pose()
    for name, shape in SHAPES.items():
        arr = getattr(pose, name)
        assert arr.shape == shape and arr.dtype == np.float32
    meta = json.loads(READY_POSE_PATH.with_suffix(".json").read_text())
    assert hashlib.sha256(READY_POSE_PATH.read_bytes()).hexdigest() == meta["sha256"]


def test_load_ready_pose_rejects_bad_shapes(tmp_path):
    load_ready_pose(_save(tmp_path, _good()))
    bad = _good()
    bad["raise_tokens"] = np.zeros((82, 64), np.float32)
    with pytest.raises(ValueError, match="raise_tokens"):
        load_ready_pose(_save(tmp_path, bad))
    nan = _good()
    nan["token"][3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        load_ready_pose(_save(tmp_path, nan))


def test_load_ready_pose_rejects_missing_key_and_large_token(tmp_path):
    missing = _good()
    del missing["state"]
    with pytest.raises(ValueError, match="state"):
        load_ready_pose(_save(tmp_path, missing))
    big = _good()
    big["token"][0] = 1.3
    with pytest.raises(ValueError, match="1.25"):
        load_ready_pose(_save(tmp_path, big))


def _save(tmp_path, arrays):
    path = tmp_path / "pose.npz"
    np.savez(path, **arrays)
    return path
