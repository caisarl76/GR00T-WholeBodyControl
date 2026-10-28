import json
import time

import numpy as np
import pytest

from gear_sonic.utils.inference import loop_records
from gear_sonic.utils.inference.loop_records import TrialRecorder, TrialWriter, next_trial

ARGS = {"fps": 50, "chunk": 40}


def _record(session, trial, phases=("blend", "episode", "episode")):
    rec = TrialRecorder(session, trial, ARGS, "model-x")
    for i, ph in enumerate(phases):
        rec.tick(100.0 + i * 0.02, i * 0.02, ph, i, np.full(64, i), np.zeros(14), 100.0, np.zeros(28))
    rec.frame(100.01, "head", b"\xff\xd8abc")
    rec.frame(100.03, "head", b"\xff\xd8de")
    return rec


def _load_frames(path):  # mirrors lerobot eval_session.load_frames
    z = np.load(path)
    raw, offsets = z["jpeg"].tobytes(), z["offsets"]
    return z["t_recv"], z["names"], [raw[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def test_trial_files_match_lerobot_layout(tmp_path):
    _record(tmp_path, 0).close("handover")
    assert TrialWriter.flush(5)
    z = np.load(tmp_path / "trial_000.npz")
    assert set(z.files) == {
        "wall", "t_ep", "phase", "frame", "token", "hands", "chunk_t0", "state", "joint_ref",
        "episode_end", "args", "model", "wall_start", "wall_stop",
    }  # fmt: skip
    assert z["token"].shape == (3, 64) and z["token"].dtype == np.float32
    assert z["hands"].shape == (3, 14) and z["state"].shape == (3, 28)
    assert z["joint_ref"].shape == (3, 31) and z["joint_ref"].dtype == np.float32
    assert np.isnan(z["joint_ref"]).all()
    assert z["wall"].dtype == np.float64 and z["frame"].dtype == np.int64
    assert z["phase"].tolist() == ["blend", "episode", "episode"]
    assert json.loads(str(z["args"])) == ARGS
    assert str(z["model"]) == "model-x" and str(z["episode_end"]) == "handover"
    assert z["wall_start"] == pytest.approx(100.02) and z["wall_stop"] == pytest.approx(100.04)
    t_recv, names, blobs = _load_frames(tmp_path / "trial_000_frames.npz")
    assert t_recv.tolist() == [100.01, 100.03] and names.tolist() == ["head", "head"]
    assert blobs == [b"\xff\xd8abc", b"\xff\xd8de"]
    assert not (tmp_path / "results.jsonl").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_no_episode_rows_gives_nan_wall(tmp_path):
    _record(tmp_path, 0, phases=("blend",)).close("cap")
    assert TrialWriter.flush(5)
    z = np.load(tmp_path / "trial_000.npz")
    assert np.isnan(z["wall_start"]) and np.isnan(z["wall_stop"])


def test_next_trial_counts_trial_files(tmp_path):
    assert next_trial(tmp_path) == 0
    for i in range(2):
        (tmp_path / f"trial_{i:03d}_frames.npz").write_bytes(b"")
        (tmp_path / f"trial_{i:03d}.npz").write_bytes(b"")
    assert next_trial(tmp_path) == 2


def test_next_trial_is_one_past_the_highest_not_a_count(tmp_path, monkeypatch):  # review: sparse trial numbers
    for i in (0, 2):
        (tmp_path / f"trial_{i:03d}_frames.npz").write_bytes(b"")
    (tmp_path / "trial_x_frames.npz").write_bytes(b"")  # not a trial number: ignored
    assert next_trial(tmp_path) == 3  # a count (2) would hand out trial 2 again
    real = loop_records._save_npz
    monkeypatch.setattr(loop_records, "_save_npz", lambda *a, **k: (time.sleep(0.5), real(*a, **k)))
    _record(tmp_path, 7).close("handover")  # queued above the highest file
    assert next_trial(tmp_path) == 8
    assert TrialWriter.flush(10)
    assert next_trial(tmp_path) == 8


def test_close_returns_before_slow_write(tmp_path, monkeypatch):
    real = loop_records._save_npz

    def slow(*a, **k):
        time.sleep(2)
        real(*a, **k)

    monkeypatch.setattr(loop_records, "_save_npz", slow)
    rec = _record(tmp_path, 0)
    t0 = time.monotonic()
    rec.close("handover")
    assert time.monotonic() - t0 < 0.05
    assert TrialWriter.flush(10)
    assert (tmp_path / "trial_000.npz").exists() and (tmp_path / "trial_000_frames.npz").exists()


def test_next_trial_counts_queued(tmp_path, monkeypatch):
    real = loop_records._save_npz
    monkeypatch.setattr(loop_records, "_save_npz", lambda *a, **k: (time.sleep(0.5), real(*a, **k)))
    _record(tmp_path, 0).close("handover")
    assert next_trial(tmp_path) == 1  # queued, not yet on disk
    assert next_trial(tmp_path / "other") == 0
    assert TrialWriter.flush(10)
    assert next_trial(tmp_path) == 1


def test_writer_error_does_not_kill_thread(tmp_path, monkeypatch):
    real = loop_records._save_npz
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        real(*a, **k)

    monkeypatch.setattr(loop_records, "_save_npz", flaky)
    TrialWriter.errors.clear()
    _record(tmp_path, 0).close("handover")
    assert TrialWriter.flush(5)
    assert len(TrialWriter.errors) == 1 and "disk full" in str(TrialWriter.errors[0])
    assert next_trial(tmp_path) == 1  # the failed number stays counted (R12)
    _record(tmp_path, 1).close("handover")
    assert TrialWriter.flush(5)
    assert (tmp_path / "trial_001.npz").exists()
    TrialWriter.errors.clear()


def test_close_twice_raises(tmp_path):
    rec = _record(tmp_path, 0)
    rec.close("handover")
    with pytest.raises(RuntimeError):
        rec.close("handover")
    with pytest.raises(RuntimeError):
        rec.tick(0.0, 0.0, "episode", 0, np.zeros(64), np.zeros(14), 0.0, np.zeros(28))
    with pytest.raises(RuntimeError):
        rec.frame(0.0, "head", b"x")
    assert TrialWriter.flush(5)


def test_relative_and_absolute_session_dir_match(tmp_path, monkeypatch):
    real = loop_records._save_npz
    monkeypatch.setattr(loop_records, "_save_npz", lambda *a, **k: (time.sleep(0.3), real(*a, **k)))
    monkeypatch.chdir(tmp_path)
    _record("sess", 0).close("handover")
    assert next_trial(tmp_path / "sess") == 1 and next_trial("sess") == 1
    assert TrialWriter.flush(10)
    assert next_trial("sess") == 1
