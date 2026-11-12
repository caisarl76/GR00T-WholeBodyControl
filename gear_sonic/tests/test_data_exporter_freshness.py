import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.features_sonic_vla import get_features_sonic_vla
from gear_sonic.scripts import run_data_exporter as exporter
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


@pytest.fixture
def collector(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(exporter.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(exporter.time, "time", lambda: 1000.0)
    obj = exporter.GrootDataCollector.__new__(exporter.GrootDataCollector)
    obj.legacy_recording_controls = True
    obj._body_packets = {}
    obj.current_stream_mode = 1
    obj.latest_sonic_msg = None
    obj.latest_planner_msg = None
    obj.sonic_timing_monitor = SimpleNamespace(log_time_delta=lambda _: None, failure_count=0)
    obj._print_and_say = lambda *args, **kwargs: None
    return obj, clock


def _planner_packet(stationary=False):
    return pack_pose_message({
        "mode": np.array([0 if stationary else 2], dtype=np.int32),
        "movement": np.array([0 if stationary else 0.4, 0, 0], dtype=np.float32),
        "facing": np.array([1, 0, 0], dtype=np.float32),
        "speed": np.array([0.5], dtype=np.float32),
        "left_hand_joints": np.arange(7, dtype=np.float32),
    }, topic="planner")


def _pose_packet():
    return pack_pose_message({
        "smpl_joints": np.ones((1, 24, 3), dtype=np.float32),
        "smpl_pose": np.ones((1, 21, 3), dtype=np.float32),
        "frame_index": np.array([5], dtype=np.int64),
    })


@pytest.mark.parametrize("mode", [2, 3, 5])
def test_planner_commands_and_age_recorded_in_all_planner_modes(collector, mode):
    obj, clock = collector
    obj.current_stream_mode = mode
    obj._handle_planner_message(_planner_packet())
    clock.now += 0.05
    frame = {}
    obj._add_sonic_pose_features(frame)
    assert frame["teleop.planner_mode"].item() == 2
    np.testing.assert_allclose(frame["teleop.planner_movement"], [0.4, 0, 0])
    np.testing.assert_array_equal(frame["teleop.left_hand_joints"], np.arange(7))
    assert frame["teleop.planner_receive_age_ms"].item() == pytest.approx(50)
    assert frame["teleop.pose_receive_age_ms"].item() == -1


def test_pose_age_uses_monotonic_time_and_missing_packets_age_out(collector, monkeypatch):
    obj, clock = collector
    obj._handle_pose_message(_pose_packet())
    clock.now += 0.05
    monkeypatch.setattr(exporter.time, "time", lambda: -5000.0)
    frame = {}
    obj._add_sonic_pose_features(frame)
    assert frame["teleop.pose_receive_age_ms"].item() == pytest.approx(50)
    assert np.all(frame["teleop.smpl_pose"] == 1)
    clock.now += 0.2
    obj._add_sonic_pose_features(frame)
    assert frame["teleop.pose_receive_age_ms"].item() == pytest.approx(250)
    assert np.all(frame["teleop.smpl_pose"] == 0)


def test_identical_planner_hold_refreshes_age_but_silence_does_not(collector):
    obj, clock = collector
    obj.current_stream_mode = 2
    obj._handle_planner_message(_planner_packet(stationary=True))
    clock.now += 1
    frame = {}
    obj._add_sonic_pose_features(frame)
    assert frame["teleop.planner_receive_age_ms"].item() == pytest.approx(1000)
    obj._handle_planner_message(_planner_packet(stationary=True))
    obj._add_sonic_pose_features(frame)
    assert frame["teleop.planner_receive_age_ms"].item() == 0
    assert frame["teleop.planner_mode"].item() == 0
    assert np.all(frame["teleop.planner_movement"] == 0)


def test_missing_sources_have_finite_sentinel_matching_schema(collector):
    obj, _ = collector
    frame = {}
    obj._add_sonic_pose_features(frame)
    features = get_features_sonic_vla(SimpleNamespace(joint_names=["joint"], num_joints=1))
    for key in ("teleop.pose_receive_age_ms", "teleop.planner_receive_age_ms"):
        assert frame[key].tolist() == [-1]
        assert frame[key].shape == features[key]["shape"] == (1,)
        assert str(frame[key].dtype) == features[key]["dtype"] == "float64"


def test_recorded_planner_hold_survives_cleaning_but_receive_gap_does_not(collector, tmp_path):
    import pandas as pd

    from gear_sonic.scripts.process_dataset import process_single_dataset

    obj, clock = collector
    obj.current_stream_mode = 2
    frames = []
    for index in range(3):
        clock.now += 1
        if index != 1:
            obj._handle_planner_message(_planner_packet(stationary=True))
        frame = {"frame_index": index, "timestamp": index / 50}
        obj._add_sonic_pose_features(frame)
        frames.append(frame)
    (tmp_path / "meta").mkdir()
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame(frames).to_parquet(tmp_path / "data/chunk-000/episode_000000.parquet")
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 50, "features": {}}))
    (tmp_path / "meta/episodes.jsonl").write_text(json.dumps({
        "episode_index": 0, "length": 3, "tasks": ["hold position"],
    }) + "\n")
    stats, episodes, _ = process_single_dataset(tmp_path, remove_stale_smpl=True)
    assert stats["stale_input_frames"] == 1
    assert episodes[0]["valid_indices"].tolist() == [0, 2]
    np.testing.assert_array_equal(
        episodes[0]["df"]["teleop.left_hand_joints"].iloc[0],
        episodes[0]["df"]["teleop.left_hand_joints"].iloc[1],
    )


def test_inspire_export_keeps_quality_metadata_out_of_action_modalities(monkeypatch):
    recorded = {}
    model = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(41)], num_joints=41)
    monkeypatch.setattr(exporter, "get_g1_robot_model", lambda **kwargs: model)
    monkeypatch.setattr(exporter, "get_modality_config_sonic_vla", lambda _: {
        "state": {}, "action": {"left_hand_joints": {}, "right_hand_joints": {}},
    })
    monkeypatch.setattr(exporter, "poll_robot_config_zmq", lambda *args: {})
    monkeypatch.setattr(exporter.Gr00tDataExporter, "create", lambda **kwargs: recorded.update(kwargs))
    monkeypatch.setattr(exporter, "GrootDataCollector", lambda **kwargs: SimpleNamespace(run=lambda: None))
    exporter.main(exporter.SonicDataExporterConfig(hand_profile="inspire_ftp", text_to_speech=False))
    features = recorded["features"]
    assert features["observation.state"]["shape"] == features["action.wbc"]["shape"] == (41,)
    assert "teleop.left_hand_joints" not in features
    assert features["teleop.left_inspire_hand_command"]["shape"] == (6,)
    assert "teleop.left_hand_source_timestamp_ns" in features
    assert "teleop.pose_receive_age_ms" in features
    actions = recorded["modality_config"]["action"]
    assert "left_inspire_hand_applied" in actions
    assert "left_inspire_hand_command" in actions
    assert "left_hand_source_timestamp_ns" not in actions
    assert "left_hand_held" not in actions
    assert "hand_sample_generation" not in actions
