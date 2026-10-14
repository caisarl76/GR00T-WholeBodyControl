"""Exercise managed exporter recording and schemas without cameras or robot DDS."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.features_sonic_vla import get_features_sonic_vla, get_g1_robot_model
from gear_sonic.data.pico_hand_features import hand_episode_features, join_hand_frame
from gear_sonic.scripts import run_data_exporter as runtime
from gear_sonic.utils.teleop.pico_inspire_protocol import STATUS_SCHEMA
from gear_sonic.utils.teleop.pico_recording import (
    ManagerRecording,
    RecordingCommand as C,
    RecordingState as S,
    make_pv,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message


class Socket:
    def __init__(self):
        self.messages = []

    def send(self, data, *args):
        self.messages.append(data)

    def recv(self, *args):
        if self.messages:
            return self.messages.pop(0)
        raise runtime.zmq.Again()

    def close(self, **kwargs):
        pass

    def setsockopt(self, *args):
        pass

    def setsockopt_string(self, *args):
        pass

    def connect(self, *args):
        pass

    def bind(self, *args):
        pass


class Context:
    def socket(self, *args):
        return Socket()

    def term(self):
        pass


class Exporter:
    def __init__(self, root):
        self.meta = SimpleNamespace(root=root)
        self.episode_buffer = {"episode_index": 0, "size": 0, "values": []}
        self.features = {}
        self.frames = []
        self.saved = []

    def add_frame(self, frame):
        self.frames.append(frame)
        self.episode_buffer["size"] += 1

    def save_episode(self):
        self.saved.append("save")
        self.episode_buffer = {"episode_index": 1, "size": 0, "values": []}

    def save_episode_as_discarded(self):
        self.saved.append("discard")
        self.episode_buffer = {"episode_index": 1, "size": 0, "values": []}


@pytest.fixture
def collector(tmp_path, monkeypatch):
    clock = [1_000_000_000]
    monkeypatch.setattr(runtime.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runtime.zmq, "Context", Context)
    subscriber = SimpleNamespace(get_msg=lambda **kwargs: None, close=lambda: None)
    monkeypatch.setattr(runtime, "ZMQStateSubscriber", lambda **kwargs: subscriber)
    c = runtime.GrootDataCollector(
        "localhost", 5555, Exporter(tmp_path), get_g1_robot_model(), use_dummy_camera=True
    )
    c.clock = clock
    c._proprio_received_ns = clock[0]
    yield c
    c._save_executor.shutdown(wait=True)


def manager_message(m, mode=1):
    return pack_pose_message(m.fields(0, mode), topic="manager_state", version=4)


def adopt(c):
    m = ManagerRecording(b"m" * 16, c.hand_profile)
    c._handle_manager_state(manager_message(m))
    m.receive_status(c.recorder.status_fields(0), c.clock[0])
    return m


def start(c):
    m = adopt(c)
    assert m.enqueue(C.START, c.clock[0], tracking=True)
    c._handle_manager_state(manager_message(m))
    m.receive_status(c.recorder.status_fields(0), c.clock[0])
    assert c.recorder.capture_active
    return m


def packets(profile="dex3", generation=1):
    pv = make_pv(b"m" * 16, 0, 1)
    body = {"pv": pv, "sample_generation": np.array([generation], np.int64), "received_ns": 1_000_000_000}
    hand = {
        **body,
        "hand_profile": np.array([0 if profile == "dex3" else 1], np.int32),
        "hand_input": np.array([0], np.int32),
        "body_sent": np.array([True]),
    }
    for side in ("left", "right"):
        q = np.zeros(7 if profile == "dex3" else 6, np.float32)
        body[f"{side}_hand_joints"] = q
        hand.update(
            {
                f"{side}_state": np.array([1], np.int32),
                f"{side}_source_epoch": np.array([0], np.int64),
                f"{side}_source_timestamp_ns": np.array([100], np.int64),
                f"{side}_valid": np.array([False]),
                f"{side}_command": q,
            }
        )
    if profile == "inspire_ftp":
        hand["inspire_message_seq"] = np.array([15], np.int64)
    return body, hand


def test_managed_controls_disable_keyboard_and_ignore_legacy_pulse(collector):
    c = collector
    assert c._keyboard_listener is None
    legacy = {"stream_mode": np.array([1], np.int32), "toggle_data_collection": np.array([True])}
    c._handle_manager_state(pack_pose_message(legacy, topic="manager_state"))
    assert c.recorder.state == S.IDLE
    m = start(c)
    c._handle_manager_state(manager_message(m))
    assert c.recorder.state == S.RECORDING


def test_stop_stops_capture_before_save_and_status_remains_live(collector):
    c = collector
    m = start(c)
    c.data_exporter.episode_buffer["size"] = 1
    release = threading.Event()
    original = c.data_exporter.save_episode

    def wait_save():
        assert release.wait(2)
        original()

    c.data_exporter.save_episode = wait_save
    assert m.enqueue(C.STOP_AND_SAVE, c.clock[0], tracking=True)
    c._handle_manager_state(manager_message(m))
    assert c.recorder.state == S.SAVING and not c.recorder.capture_active
    c._service_recording()
    status = unpack_pose_message(c._recording_status_socket.messages[-1], topic="recording_status")
    assert status["recording_state"].item() == S.SAVING
    release.set()
    c._save_future.result(timeout=2)
    c._service_recording()
    assert c.recorder.state == S.IDLE and c.data_exporter.saved == ["save"]


def test_failed_save_preserves_original_and_cleanup_never_retries(collector):
    c = collector
    m = start(c)
    c.data_exporter.episode_buffer.update(size=1, values=[np.array([42])])
    attempts = []

    def fail():
        attempts.append(1)
        c.data_exporter.episode_buffer.pop("size")
        (c.data_exporter.meta.root / "partial.parquet").write_text("partial")
        raise OSError("disk failed")

    c.data_exporter.save_episode = fail
    m.enqueue(C.STOP_AND_SAVE, c.clock[0], tracking=True)
    c._handle_manager_state(manager_message(m))
    with pytest.raises(OSError):
        c._save_future.result(timeout=2)
    c._service_recording()
    assert c.recorder.state == S.ERROR
    assert c.data_exporter.episode_buffer["size"] == 1
    assert c.data_exporter.episode_buffer["values"][0].item() == 42
    c.save_and_cleanup()
    assert len(attempts) == 1
    assert (c.data_exporter.meta.root / "partial.parquet").exists()


def test_robot_timeout_discards_while_manager_is_live(collector):
    c = collector
    m = start(c)
    c.data_exporter.episode_buffer["size"] = 1
    c.clock[0] += 500_000_000
    c._handle_manager_state(manager_message(m))
    c._service_recording()
    assert not c.recorder.capture_active
    c._save_future.result(timeout=2)
    c._service_recording()
    assert c.data_exporter.saved == ["discard"]


def test_invalid_robot_packet_cannot_refresh_receive_age(collector):
    c = collector
    c._state_subscriber.get_msg = lambda **kwargs: {"body_q": np.zeros(28), "last_action": np.zeros(29)}
    c.clock[0] += 600_000_000
    c._poll_state_zmq()
    assert c._proprio_received_ns == 1_000_000_000


def test_hand_join_allows_real_hold_but_rejects_missing_generation():
    body, hand = packets()
    result = join_hand_frame(body, hand, "dex3", 1_000_000_000, b"m" * 16, 1)
    assert result["teleop.left_hand_held"].item()
    for malformed in (
        None,
        {**hand, "sample_generation": np.array([2], np.int64)},
        {**hand, "body_sent": np.array([False])},
    ):
        with pytest.raises(ValueError):
            join_hand_frame(body, malformed, "dex3", 1_000_000_000, b"m" * 16, 1)
    with pytest.raises(ValueError):
        join_hand_frame(body, hand, "dex3", 1_100_000_000, b"m" * 16, 1)


def test_real_dex3_frame_drops_absent_hands_and_orders_state(collector):
    c = collector
    start(c)
    c.latest_image_msg = {"images": {}}
    c.latest_proprio_msg = {"body_q": np.zeros(29), "last_action": np.zeros(29)}
    for side in ("left", "right"):
        c.latest_proprio_msg.update(
            {
                f"{side}_hand_q": np.arange(7) * 0.01,
                f"last_{side}_hand_action": np.arange(7) * 0.02,
                f"{side}_hand_feedback_valid": True,
                f"{side}_hand_feedback_age_ns": 0,
            }
        )
    assert not c._add_data_frame()
    assert not c.data_exporter.frames
    body, hand = packets()
    c._body_packets["pose"], c._hand_diagnostics = body, hand
    assert c._add_data_frame()
    frame = c.data_exporter.frames[0]
    assert frame["observation.state"].shape == (43,)
    idx = c.robot_model.dof_index("left_hand_index_0_joint")
    assert frame["observation.state"][idx] == 0.05
    assert frame["teleop.left_hand_joints"].shape == (7,)
    assert not c._add_data_frame()  # same generation is never appended twice

    # Physical sample age includes time spent in transport and the local cache.
    c.clock[0] += 50_000_000
    c._body_packets["pose"], c._hand_diagnostics = packets(generation=2)
    c.latest_proprio_msg["left_hand_feedback_age_ns"] = 50_000_000
    assert not c._add_data_frame()  # exactly 100 ms old despite each age being <100 ms
    assert len(c.data_exporter.frames) == 1
    c.latest_proprio_msg["left_hand_feedback_age_ns"] = 49_999_999
    assert c._add_data_frame()


def test_managed_recording_requires_v4_wire_message(collector):
    c = collector
    m = ManagerRecording(b"m" * 16, c.hand_profile)
    c._handle_manager_state(pack_pose_message(m.fields(0, 1), topic="manager_state", version=3))
    assert c.recorder.manager_session_id is None
    c._handle_manager_state(manager_message(m))
    assert c.recorder.manager_session_id == b"m" * 16


def test_inspire_model_schema_and_fk_are_real_41_dof():
    model = get_g1_robot_model(hand_profile="inspire_ftp")
    assert model.num_joints == 41
    assert model.joint_names[29:35] == [
        f"left_{name}_joint" for name in ("little_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1")
    ]
    q = model.get_configuration_from_actuated_joints(np.zeros(29), np.zeros(6), np.zeros(6))
    model.cache_forward_kinematics(q)
    assert np.isfinite(model.frame_placement("left_wrist_yaw_link").translation).all()
    assert get_features_sonic_vla(model)["observation.state"]["shape"] == (41,)
    assert "teleop.left_hand_joints" not in hand_episode_features("inspire_ftp")


def status_packet(seq=0, session=b"p" * 16):
    fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in STATUS_SCHEMA}
    fields["pc2_session_id"][:] = np.frombuffer(session, np.uint8)
    fields["status_seq"][0] = seq
    fields["accepted_pv"][:] = make_pv(b"m" * 16, 0, 1)
    fields["bridge_state"][0] = 2
    fields["feedback_healthy"][0] = True
    fields["last_applied_message_seq"][0] = 12
    for side in ("left", "right"):
        fields[f"{side}_angle_act"][:] = 900
        fields[f"{side}_applied"][:] = 0.8
    return pack_pose_message(fields, topic="inspire_hand_status", version=1)


def test_pc2_restart_rearms_and_inspire_join_uses_physical_values(collector):
    c = collector
    c._inspire_status_socket = Socket()
    c._inspire_status_socket.messages = [status_packet(i) for i in range(10)]
    c._poll_inspire_status()
    assert c._inspire_status["ready"]
    body, hand = packets("inspire_ftp")
    result = join_hand_frame(body, hand, "inspire_ftp", c.clock[0], b"m" * 16, 1, c._inspire_status)
    np.testing.assert_allclose(result["observation.left_inspire_hand_state"], 0.9)
    np.testing.assert_allclose(result["action.left_inspire_hand_applied"], 0.8)
    c._inspire_status_socket.messages = [status_packet(0, b"q" * 16)]
    c._poll_inspire_status()
    assert not c._inspire_status["ready"]
    with pytest.raises(ValueError):
        join_hand_frame(body, hand, "inspire_ftp", c.clock[0], b"m" * 16, 1, c._inspire_status)


def test_real_exporter_save_preserves_buffer_on_disk_failure_and_flushes_success(tmp_path, monkeypatch):
    from gear_sonic.data.exporter import Gr00tDataExporter

    exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "real",
        fps=50,
        features={"observation.state": {"dtype": "float64", "shape": (41,), "names": [str(i) for i in range(41)]}},
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
        task="test",
        script_config={"hand_profile": "inspire_ftp"},
    )
    exporter.add_frame({"observation.state": np.zeros(41, np.float64)})
    original = exporter._save_episode_table

    def fail_table(*args):
        raise OSError("table write failed")

    monkeypatch.setattr(exporter, "_save_episode_table", fail_table)
    with pytest.raises(OSError):
        exporter.save_episode()
    assert exporter.episode_buffer["size"] == 1
    assert isinstance(exporter.episode_buffer["task"], list)
    assert exporter.episode_buffer["observation.state"][0].shape == (41,)
    monkeypatch.setattr(exporter, "_save_episode_table", original)
    writer_indices = []
    original_factory = exporter.create_video_writer

    def create_next_writer(episode_index=None):
        writer_indices.append(episode_index)
        return original_factory(episode_index)

    monkeypatch.setattr(exporter, "create_video_writer", create_next_writer)
    exporter.save_episode()
    assert writer_indices == [1]  # Never reopen/truncate the episode just saved.
    assert exporter.episode_buffer["size"] == 0
    assert exporter.meta.total_episodes == 1
    assert len(list((tmp_path / "real").rglob("*.parquet"))) == 1
    with pytest.raises(ValueError, match="profile"):
        Gr00tDataExporter.create(
            save_root=tmp_path / "real",
            fps=50,
            features={},
            modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
            task="test",
            script_config={"hand_profile": "dex3"},
        )


def test_inspire_full_frame_is_41dof_with_physical_state_and_applied_action(collector):
    from gear_sonic.data.pico_hand_features import INSPIRE_RANGES
    from gear_sonic.utils.teleop.pico_recording import RecorderProtocol

    c = collector
    c.hand_profile = "inspire_ftp"
    c.recorder = RecorderProtocol("inspire_ftp")
    c.robot_model = get_g1_robot_model(hand_profile="inspire_ftp")
    start(c)
    c.latest_image_msg = {"images": {}}
    c.latest_proprio_msg = {"body_q": np.zeros(29), "last_action": np.ones(29) * 0.01}
    c._inspire_status_socket = Socket()
    c._inspire_status_socket.messages = [status_packet(i) for i in range(10)]
    c._poll_inspire_status()
    c._body_packets["pose"], c._hand_diagnostics = packets("inspire_ftp")
    assert c._add_data_frame()
    frame = c.data_exporter.frames[0]
    assert frame["observation.state"].shape == frame["action.wbc"].shape == (41,)
    np.testing.assert_allclose(frame["observation.state"][29:35], 0.1 * INSPIRE_RANGES, rtol=1e-6)
    np.testing.assert_allclose(frame["action.wbc"][29:35], 0.2 * INSPIRE_RANGES, rtol=1e-6)
    assert "teleop.left_hand_joints" not in frame
    assert frame["teleop.hand_input"].item() == 0


def test_missing_robot_hand_field_is_dropped_without_crashing(collector):
    c = collector
    start(c)
    c.latest_image_msg = {"images": {}}
    c.latest_proprio_msg = {"body_q": np.zeros(29), "last_action": np.zeros(29)}
    c._body_packets["pose"], c._hand_diagnostics = packets()
    assert not c._add_data_frame()
    assert c.dropped_hand_frames == 1
    assert c.recorder.capture_active


def test_matching_old_mode_epoch_is_still_ineligible():
    body, hand = packets()
    with pytest.raises(ValueError, match="epoch"):
        join_hand_frame(body, hand, "dex3", 1_000_000_000, b"m" * 16, 1, manager_epoch=2)


def test_monitor_and_opening_inspire_status_never_record_actions(collector):
    c = collector
    c._inspire_status_socket = Socket()
    c._inspire_status_socket.messages = [status_packet(i) for i in range(10)]
    c._poll_inspire_status()
    body, hand = packets("inspire_ftp")
    for state in (0, 1, 3, 4, 5):
        status = {**c._inspire_status, "bridge_state": np.array([state], np.int32)}
        with pytest.raises(ValueError, match="actively"):
            join_hand_frame(body, hand, "inspire_ftp", c.clock[0], b"m" * 16, 1, status)


def test_inspire_application_ahead_of_manager_intent_is_rejected(collector):
    c = collector
    c._inspire_status_socket = Socket()
    c._inspire_status_socket.messages = [status_packet(i) for i in range(10)]
    c._poll_inspire_status()
    body, hand = packets("inspire_ftp")
    hand["inspire_message_seq"][:] = 11
    with pytest.raises(ValueError, match="ahead"):
        join_hand_frame(body, hand, "inspire_ftp", c.clock[0], b"m" * 16, 1, c._inspire_status)


@pytest.mark.parametrize("profile", ["dex3", "inspire_ftp"])
def test_hand_episode_schema_records_timestamp_source(profile):
    for side in ("left", "right"):
        assert hand_episode_features(profile)[f"teleop.{side}_hand_timestamp_source"] == {
            "dtype": "int32",
            "shape": (1,),
            "names": ["timestamp_source"],
        }


def test_hand_join_preserves_independent_clock_sources_and_legacy_unknown():
    body, hand = packets()
    # Existing diagnostic packets remain readable without inventing device time.
    legacy = join_hand_frame(body, hand, "dex3", 1_000_000_000, b"m" * 16, 1)
    assert legacy["teleop.left_hand_timestamp_source"].item() == -1
    assert legacy["teleop.right_hand_timestamp_source"].item() == -1
    hand["left_timestamp_source"] = np.array([0], np.int32)
    hand["right_timestamp_source"] = np.array([1], np.int32)
    joined = join_hand_frame(body, hand, "dex3", 1_000_000_000, b"m" * 16, 1)
    assert joined["teleop.left_hand_timestamp_source"].item() == 0
    assert joined["teleop.right_hand_timestamp_source"].item() == 1
    hand["right_timestamp_source"][0] = 0
    assert joined["teleop.right_hand_timestamp_source"].item() == 1


@pytest.mark.parametrize(
    "source",
    [
        np.array([2], np.int32),
        np.array([0], np.int64),
        np.array([True]),
        np.array([0, 1], np.int32),
        np.array([-2], np.int32),
    ],
)
def test_hand_join_rejects_invalid_clock_source(source):
    body, hand = packets()
    hand["left_timestamp_source"] = source
    with pytest.raises(ValueError, match="timestamp.source"):
        join_hand_frame(body, hand, "dex3", 1_000_000_000, b"m" * 16, 1)


def test_real_exporter_saves_clock_provenance_scalars(tmp_path):
    import pyarrow.parquet as pq

    from gear_sonic.data.exporter import Gr00tDataExporter

    features = {
        key: value for key, value in hand_episode_features("dex3").items() if key.endswith("timestamp_source")
    }
    exporter = Gr00tDataExporter.create(
        save_root=tmp_path / "clock",
        fps=50,
        features=features,
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
        task="clock provenance",
        script_config={"hand_profile": "dex3"},
    )
    for left, right in [(0, 1), (1, 0), (-1, -1)]:
        exporter.add_frame(
            {
                "teleop.left_hand_timestamp_source": np.array([left], np.int32),
                "teleop.right_hand_timestamp_source": np.array([right], np.int32),
            }
        )
    exporter.save_episode()
    table = pq.read_table(next((tmp_path / "clock").rglob("*.parquet")))
    assert table["teleop.left_hand_timestamp_source"].to_pylist() == [0, 1, -1]
    assert table["teleop.right_hand_timestamp_source"].to_pylist() == [1, 0, -1]


@pytest.mark.parametrize("interruption,expected", [(RuntimeError, "save"), (KeyboardInterrupt, "discard")])
def test_legacy_run_exit_saves_nonempty_episode_once(collector, interruption, expected):
    c = collector
    c.legacy_recording_controls = True
    c.data_exporter.episode_buffer["size"] = 2

    def fail_poll():
        raise interruption("collection stopped")

    c._poll_state_zmq = fail_poll
    if interruption is RuntimeError:
        with pytest.raises(RuntimeError, match="collection stopped"):
            c.run()
    else:
        c.run()
    assert c.data_exporter.saved == [expected]
    assert c.data_exporter.episode_buffer["size"] == 0


@pytest.mark.parametrize("trigger", ["stop", "discard", "interrupt", "cleanup"])
def test_legacy_failed_save_is_not_retried_and_resources_close(collector, trigger):
    c = collector
    c.legacy_recording_controls = True
    c.data_exporter.episode_buffer["size"] = 2
    attempts = []
    closed = []
    c._state_subscriber.close = lambda: closed.append(True)

    def fail_save():
        attempts.append(True)
        raise OSError("disk failed")

    c.data_exporter.save_episode = c.data_exporter.save_episode_as_discarded = fail_save

    def stop_loop():
        if trigger == "stop":
            c._episode_state.state = c._episode_state.NEED_TO_SAVE
            c._finalize_frame(runtime.time.monotonic())
        elif trigger == "discard":
            c._episode_state.state = c._episode_state.RECORDING
            c._keyboard_listener = SimpleNamespace(read_msg=lambda: "x", close=lambda: None)
            c._check_recording_commands()
        elif trigger == "interrupt":
            raise KeyboardInterrupt()
        else:
            raise RuntimeError("camera failed")

    c._poll_state_zmq = stop_loop
    with pytest.raises(RuntimeError if trigger == "cleanup" else OSError):
        c.run()
    assert attempts == [True]
    assert closed == [True]
    assert c.data_exporter.episode_buffer["size"] == 2
    c.save_and_cleanup()
    assert attempts == [True]


def test_legacy_successful_save_does_not_block_next_episode_cleanup(collector):
    c = collector
    c.legacy_recording_controls = True
    c.data_exporter.episode_buffer["size"] = 2

    def stop_loop():
        c._episode_state.state = c._episode_state.NEED_TO_SAVE
        c._finalize_frame(runtime.time.monotonic())
        c.data_exporter.episode_buffer["size"] = 1
        raise RuntimeError("next episode camera failed")

    c._poll_state_zmq = stop_loop
    with pytest.raises(RuntimeError, match="next episode"):
        c.run()
    assert c.data_exporter.saved == ["save", "save"]


def test_legacy_interrupt_during_save_does_not_attempt_discard_or_retry(collector):
    c = collector
    c.legacy_recording_controls = True
    c.data_exporter.episode_buffer["size"] = 2
    attempts = []

    def interrupted_save():
        attempts.append(True)
        raise KeyboardInterrupt()

    def stop_loop():
        c._episode_state.state = c._episode_state.NEED_TO_SAVE
        c._finalize_frame(runtime.time.monotonic())

    c.data_exporter.save_episode = interrupted_save
    c._poll_state_zmq = stop_loop
    c.run()
    assert attempts == [True]
    assert c.data_exporter.saved == []
    assert c.data_exporter.episode_buffer["size"] == 2
