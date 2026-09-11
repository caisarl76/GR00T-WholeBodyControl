"""Synthetic fixtures test offline tooling; they are not real qualification data."""

import json

import numpy as np
import pytest

from gear_sonic.scripts.replay_pico_hands import load_captures, replay, write_results
from gear_sonic.utils.teleop.pico_hand_log import HandCaptureLog, snapshot_at
from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, HandTracker, TrackingState


class FakeRetargeter:
    def __init__(self, target=0.0, size=7):
        self.lower = np.full(size, -1.0 if size == 7 else 0.0)
        self.upper = np.ones(size)
        self.target = target

    def retarget(self, points):
        return np.full(len(self.lower), self.target)


def synthetic_capture(count=40, size=7):
    pose = np.zeros((26, 7), np.float64)
    for digit, (start, stop) in enumerate(((2, 6), (6, 11), (11, 16), (16, 21), (21, 26))):
        for index in range(start, stop):
            pose[index, :3] = (0.02 * (index - start + 1), 0.02 * (digit - 2), 0)
    return {
        "poses": np.broadcast_to(pose, (count, 2, 26, 7)).copy(),
        "flags": np.full((count, 2, 26), 10, np.uint64),
        "radius": np.full((count, 2, 26), 0.005, np.float64),
        "scale": np.ones((count, 2), np.float64),
        "active": np.ones((count, 2), np.int32),
        "source_timestamp_ns": np.repeat(np.arange(1, count + 1, dtype=np.int64)[:, None], 2, axis=1),
        "binding_generation": np.repeat(np.arange(1, count + 1, dtype=np.uint64)[:, None], 2, axis=1),
        "body_wrist": np.zeros((count, 2, 3), np.float64),
        "measured": np.zeros((count, 2, size), np.float64),
        "tick_ns": np.arange(count, dtype=np.int64) * 20_000_000,
    }


def test_frozen_left_holds_at_100ms_while_right_advances_then_five_frame_recovery():
    arrays = synthetic_capture()
    arrays["source_timestamp_ns"][10:22, 0] = 10
    result, report = replay(arrays, retargeters=[FakeRetargeter(0.3), FakeRetargeter(0.3)])
    assert result["reason"][14, 0] == HandReason.SOURCE_STALE
    assert result["age_ns"][14, 0] == 100_000_000
    assert result["state"][14, 0] == TrackingState.HOLDING
    np.testing.assert_array_equal(result["emitted"][14:22, 0], np.repeat(result["emitted"][13:14, 0], 8, axis=0))
    assert np.all(result["reason"][14:22, 1] == HandReason.OK)
    assert np.all(result["state"][22:26, 0] == TrackingState.HOLDING)
    assert result["state"][26, 0] == TrackingState.RECOVERING
    assert report["counts"]["late_or_nonholding_freezes"] == 0
    assert report["counts"]["early_recovery_admissions"] == 0
    assert report["replay_safety_checks_pass"]
    assert report["synthetic"] and not report["d0_qualified"]
    assert report["qualification_status"] == "incomplete"


def test_bad_retarget_candidates_are_counted_without_unsafe_emissions():
    arrays = synthetic_capture(10)
    output, report = replay(arrays, retargeters=[FakeRetargeter(2), FakeRetargeter(float("nan"))])
    assert report["counts"]["out_of_bounds_targets_rejected"] > 0
    assert report["counts"]["nonfinite_targets_rejected"] > 0
    assert report["counts"]["out_of_bounds_emitted"] == 0
    assert report["counts"]["nonfinite_emitted"] == 0
    assert output["raw_target"][4, 0, 0] == 2
    assert output["reason"][4, 0] == HandReason.RETARGET_FAILED
    assert report["replay_safety_checks_pass"]


def test_capture_shards_copy_inputs_preserve_outputs_and_replay(tmp_path):
    arrays = synthetic_capture(12)
    arrays["timestamp_source"] = np.tile(np.array([1, 0], np.int8), (12, 1))
    tracker = [HandTracker(FakeRetargeter()) for _ in range(2)]
    logger = HandCaptureLog(tmp_path / "capture", "dex3", shard_size=5, source_kind="synthetic")
    for index, tick in enumerate(arrays["tick_ns"]):
        snapshots = [snapshot_at(arrays, index, side) for side in range(2)]
        outputs = [
            tracker[side].step(
                snapshots[side], arrays["body_wrist"][index, side], arrays["measured"][index, side], int(tick)
            )
            for side in range(2)
        ]
        logger.record(
            int(tick),
            snapshots,
            arrays["body_wrist"][index],
            arrays["measured"][index],
            outputs,
            pose_label="open",
            body_sent=True,
            hand_sent=[True, True],
            sample_generation=index,
        )
        # This original source storage is mutable; capture must already own its copy.
        arrays["poses"][index] = 999
        if index in (4, 9):
            for pending in logger._pending:
                pending.result()
    logger.close()
    paths = sorted((tmp_path / "capture").glob("capture_*.npz"))
    assert len(paths) == 3
    captured, kind = load_captures(paths, "dex3")
    assert captured["poses"].shape == (12, 2, 26, 7)
    assert captured["poses"].max() < 1
    np.testing.assert_array_equal(captured["timestamp_source"], arrays["timestamp_source"])
    assert kind == "synthetic"
    np.testing.assert_array_equal(captured["sample_generation"], np.arange(12))
    outputs, report = replay(captured, retargeters=[FakeRetargeter(), FakeRetargeter()], source_kind=kind)
    assert report["captured_command_comparison"]["max_abs_difference"] == 0
    assert "host content-change timing" in json.dumps(report)
    write_results(tmp_path / "replay", outputs, report)
    saved = json.loads((tmp_path / "replay/report.json").read_text())
    assert saved["source_kind"] == "synthetic" and not saved["d0_qualified"]
    assert (tmp_path / "capture/transitions.jsonl").read_text()
    with pytest.raises(FileExistsError):
        write_results(tmp_path / "replay", outputs, report)


def test_missing_feedback_and_snapshot_are_recorded_as_missing_not_zero_commands(tmp_path):
    logger = HandCaptureLog(tmp_path, "inspire_ftp", source_kind="synthetic")
    logger.record(0, [None, None], [None, None], [None, None])
    logger.close()
    arrays, kind = load_captures([tmp_path / "capture_000000.npz"], "inspire_ftp")
    assert not arrays["snapshot_present"].any()
    assert np.isnan(arrays["measured"]).all()
    result, report = replay(
        arrays, "inspire_ftp", retargeters=[FakeRetargeter(size=6), FakeRetargeter(size=6)], source_kind=kind
    )
    assert not result["command_present"].any()
    assert np.isnan(result["emitted"]).all()
    assert report["counts"]["nonfinite_emitted"] == 0
    assert not report["d0_qualified"]


@pytest.mark.parametrize(
    "key,change",
    [
        ("flags", lambda value: value.astype(np.int64)),
        ("poses", lambda value: value.astype(np.float32)),
        ("tick_ns", lambda value: value[::-1]),
        ("measured", lambda value: value[:, :, :6]),
    ],
)
def test_replay_rejects_structural_storage_errors(key, change):
    arrays = synthetic_capture(8)
    arrays[key] = change(arrays[key])
    with pytest.raises(ValueError):
        replay(arrays, retargeters=[FakeRetargeter(), FakeRetargeter()])


def test_load_rejects_overlapping_or_wrong_profile_shards(tmp_path):
    arrays = synthetic_capture(8)
    np.savez(tmp_path / "one.npz", **arrays)
    with pytest.raises(ValueError, match="increasing"):
        load_captures([tmp_path / "one.npz", tmp_path / "one.npz"], "dex3")
    np.savez(tmp_path / "wrong.npz", **arrays, profile=np.array(["inspire_ftp"]))
    with pytest.raises(ValueError, match="profile"):
        load_captures([tmp_path / "wrong.npz"], "dex3")


def test_report_requires_labels_and_live_cadence_evidence():
    arrays = synthetic_capture(10)
    _, report = replay(arrays, retargeters=[FakeRetargeter(), FakeRetargeter()], source_kind="runtime_capture")
    assert report["synthetic"]  # Injected implementation never masquerades as real retargeting.
    assert report["body_cadence"] is None
    assert not report["opposite_hand_evidence"]
    assert report["pose_label_frame_deficits"]["open"] == 1000
    assert report["duration_s"] < 600
    assert len(report["missing_qualification_evidence"]) >= 6


def test_available_publication_evidence_is_checked():
    arrays = synthetic_capture(40)
    arrays["source_timestamp_ns"][10:30, 0] = 10
    arrays["body_sent"] = np.ones(40, np.bool_)
    arrays["hand_sent"] = np.ones((40, 2), np.bool_)
    _, report = replay(arrays, retargeters=[FakeRetargeter(), FakeRetargeter()])
    assert report["body_cadence"]["pass"]
    assert report["opposite_hand_evidence"][0]["other_hand_hz"] == 50
    assert not report["d0_qualified"]
    arrays["body_sent"][1::2] = False
    _, report = replay(arrays, retargeters=[FakeRetargeter(), FakeRetargeter()])
    assert not report["body_cadence"]["pass"]
