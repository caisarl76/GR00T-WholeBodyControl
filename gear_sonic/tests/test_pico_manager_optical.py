"""Exercise the real manager loop with in-process XR, recorder and transport peers."""

from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.utils.teleop import pico_hand_runtime as hr
from gear_sonic.utils.teleop.pico_hand_log import HandCaptureLog, validate_capture
from gear_sonic.utils.teleop.pico_inspire_protocol import STATUS_SCHEMA, validate_inspire_hand
from gear_sonic.utils.teleop.pico_recording import RecorderProtocol, RecordingCommand, RecordingState
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message


@pytest.fixture
def manager(monkeypatch):
    # Import the actual module without importing Torch, robot IK or GUI packages.
    stubs = {
        "torch": (),
        "gear_sonic.trl.utils.rotation_conversion": ("decompose_rotation_aa",),
        "gear_sonic.trl.utils.torch_transform": (
            "angle_axis_to_quaternion",
            "compute_human_joints",
            "quat_apply",
            "quat_inv",
            "quaternion_to_angle_axis",
            "quaternion_to_rotation_matrix",
        ),
        "gear_sonic.isaac_utils.rotations": ("remove_smpl_base_rot", "smpl_root_ytoz_up"),
        "gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver": ("G1GripperInverseKinematicsSolver",),
        "gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer": ("VR3PtPoseVisualizer", "get_g1_key_frame_poses"),
        "xrobotoolkit_sdk": (),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected robot/model operation")

    for name, exports in stubs.items():
        stub = ModuleType(name)
        for export in exports:
            setattr(stub, export, forbidden)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).parents[1] / "scripts/pico_manager_thread_server.py"
    spec = importlib.util.spec_from_file_location("_pico_manager_wiring_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optical_snapshot(stamp):
    pose = np.zeros((26, 7), np.float64)
    for digit, (start, stop) in enumerate(((2, 6), (6, 11), (11, 16), (16, 21), (21, 26))):
        for index in range(start, stop):
            pose[index, :3] = (0.02 * (index - start + 1), 0.02 * (digit - 2), 0)
    return {
        "pose": pose,
        "location_flags": np.full(26, 10, np.uint64),
        "radius": np.full(26, 0.005),
        "scale": 1.0,
        "is_active": 1,
        "source_timestamp_ns": stamp,
        "binding_generation": stamp,
    }


class Harness:
    def __init__(
        self,
        monkeypatch,
        manager,
        *,
        profile="dex3",
        keys=None,
        end=30,
        stop_tick=None,
        feedback_from=0,
        lost_left_from=None,
        pending_start=False,
        finish_save=True,
        skip_body_ticks=(),
        suppress_ack_ticks=(),
        malformed_shutdown_ack=False,
    ):
        self.manager, self.profile, self.end = manager, profile, end
        self.keys, self.stop_tick = keys or {}, stop_tick
        self.feedback_from, self.lost_left_from = feedback_from, lost_left_from
        self.pending_start, self.finish_save = pending_start, finish_save
        self.skip_body_ticks = set(skip_body_ticks)
        self.suppress_ack_ticks = set(suppress_ack_ticks)
        self.malformed_shutdown_ack = malformed_shutdown_ack
        self.hand_snapshot_reads, self.hand_outputs = [], {}
        self.now, self.tick = 1_000_000_000, -1
        self.messages, self.sleeps, self.steps, self.actions = [], [], [], []
        self.ack_queue = deque()
        self.protocol = RecorderProtocol(profile)
        self.start_sends = 0
        self.service_calls = []
        self.sockets = []
        self.hands = None
        owner = self

        class Clock:
            @staticmethod
            def monotonic_ns():
                return owner.now

            @staticmethod
            def monotonic():
                return owner.now / 1e9

            time = monotonic

            @staticmethod
            def sleep(seconds):
                owner.sleeps.append((owner.tick, seconds))
                owner.now += round(seconds * 1e9)

        class Socket:
            def __init__(self, kind):
                self.kind, self.topic, self.closed = kind, "", False
                owner.sockets.append(self)

            def bind(self, endpoint):
                self.endpoint = endpoint

            def connect(self, endpoint):
                self.endpoint = endpoint

            def setsockopt_string(self, option, value):
                if option == zmq.SUBSCRIBE:
                    self.topic = value

            def setsockopt(self, *args):
                pass

            def send(self, raw):
                owner.sent(raw)

            def recv(self, flags):
                if self.topic == "recording_status":
                    if owner.ack_queue:
                        return owner.ack_queue.popleft()
                    raise zmq.Again()
                if self.topic in ("g1_debug", "inspire_hand_status") and owner.tick >= owner.feedback_from:
                    return owner.feedback_wire(self.topic)
                raise zmq.Again()

            def close(self):
                self.closed = True

        class Context:
            def socket(self, kind):
                return Socket(kind)

            def term(self):
                pass

        class SDK:
            def init(self):
                owner.service_calls.append("sdk.init")

            def is_body_data_available(self):
                return True

            def get_left_controller_snapshot(self):
                return owner.controller("left")

            def get_right_controller_snapshot(self):
                return owner.controller("right")

            def get_left_hand_snapshot(self):
                value = (
                    None
                    if owner.lost_left_from is not None and owner.tick >= owner.lost_left_from
                    else optical_snapshot(owner.tick + 1)
                )
                owner.hand_snapshot_reads.append((owner.tick, "left", value))
                return value

            def get_right_hand_snapshot(self):
                value = optical_snapshot(owner.tick + 1)
                owner.hand_snapshot_reads.append((owner.tick, "right", value))
                return value

        class Keyboard:
            def open(self):
                pass

            def poll(self):
                owner.tick += 1
                if owner.tick >= owner.end:
                    if owner.malformed_shutdown_ack:
                        owner.ack_queue.appendleft(b"recording_status{broken")
                    raise KeyboardInterrupt
                return list(owner.keys.get(owner.tick, ""))

            def close(self):
                pass

        class Reader:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def get_timestamp_ns(self):
                return owner.tick + 1

            def get_latest(self):
                return {"body_poses_np": np.zeros((24, 7)), "timestamp_monotonic": owner.now / 1e9}

        class ThreePoint:
            def __init__(self, **kwargs):
                pass

            def calibrate_now(self, body):
                pass

            def close(self):
                pass

        class Streamer:
            def __init__(self, socket, **kwargs):
                self.socket = socket
                self.hand_input = kwargs.get("hand_input", "controller")
                self.hand_profile = kwargs.get("hand_profile", "dex3")
                self.mode = manager.LocomotionMode.IDLE
                self.feedback_reader = SimpleNamespace(poller=SimpleNamespace(close=lambda: None))

            def reset_yaw(self):
                pass

            def on_mode_exit(self):
                pass

            def run_once(self, mode=None):
                assert self.managed
                if owner.tick in owner.skip_body_ticks:
                    return
                topic = "pose" if mode is None else "planner"
                fields = {
                    "body_q": np.zeros(29, np.float32),
                    "left_hand_joints": np.full(7, 0.9, np.float32),
                    "right_hand_joints": np.full(7, 0.9, np.float32),
                }
                if self.hand_input == "controller":
                    commands = manager.compute_hand_joints_from_inputs(
                        None, None, 0.0, 0.0, 0.0, 0.0, hand_profile=self.hand_profile
                    )
                    for side, command in zip(("left", "right"), commands):
                        fields[f"{side}_hand_joints"] = command.reshape(-1)
                self.socket.send(pack_pose_message(fields, topic, 3 if mode is None else 1))

        class Retargeter:
            def __init__(self, profile, side):
                self.lower = np.full(7, -1.0) if profile == "dex3" else np.zeros(6)
                self.upper = np.ones(self.lower.size)

            def retarget(self, points):
                return np.full(self.lower.size, 0.5, np.float32)

        class Hands(hr.PicoHandRuntime):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owner.hands = self

            def step(self, sample, now_ns, *, enabled):
                owner.steps.append((owner.tick, now_ns, enabled))
                result = super().step(sample, now_ns, enabled=enabled)
                owner.hand_outputs[owner.tick] = tuple(self.outputs)
                return result

        monkeypatch.setattr(hr, "HandRetargeter", Retargeter)
        monkeypatch.setattr(manager, "PicoHandRuntime", Hands)
        monkeypatch.setattr(manager, "time", Clock)
        monkeypatch.setattr(
            manager,
            "zmq",
            SimpleNamespace(
                Context=Context,
                PUB=zmq.PUB,
                SUB=zmq.SUB,
                SUBSCRIBE=zmq.SUBSCRIBE,
                LINGER=zmq.LINGER,
                NOBLOCK=zmq.NOBLOCK,
                Again=zmq.Again,
            ),
        )
        monkeypatch.setattr(manager, "xrt", SDK())
        monkeypatch.setattr(
            manager, "subprocess", SimpleNamespace(Popen=lambda cmd: self.service_calls.append(tuple(cmd)))
        )
        monkeypatch.setattr(manager, "ManagerKeyboard", Keyboard)
        monkeypatch.setattr(manager, "PicoReader", Reader)
        monkeypatch.setattr(manager, "ThreePointPose", ThreePoint)
        monkeypatch.setattr(manager, "PoseStreamer", Streamer)
        monkeypatch.setattr(manager, "PlannerStreamer", Streamer)

    def controller(self, side):
        pressed = (
            3 <= self.tick <= 5 or self.stop_tick is not None and self.stop_tick <= self.tick <= self.stop_tick + 2
        )
        return {
            "binding_generation": self.tick + 1,
            "receipt_age_ns": 0,
            "primary_button": pressed,
            "secondary_button": pressed,
            "axis_click": False,
            "menu_button": False,
            "grip": 0.0,
            "trigger": 0.0,
            "axis": (0.0, 0.0),
        }

    def feedback_wire(self, topic):
        if topic == "g1_debug":
            fields = {"index": self.tick + 1}
            for side in ("left", "right"):
                fields[f"{side}_hand_feedback_valid"] = True
                fields[f"{side}_hand_feedback_age_ns"] = 0
                fields[f"{side}_hand_q"] = [0.0] * 7
                fields[f"{side}_hand_q_measured"] = [2.0] * 7  # Visualization is not DDS feedback.
            return topic.encode() + msgpack.packb(fields)
        fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in STATUS_SCHEMA}
        fields["pc2_session_id"][:] = 1
        fields["status_seq"][0] = self.tick + 1
        fields["last_applied_message_seq"][0] = -1
        fields["bridge_state"][0] = 1
        for side in ("left", "right"):
            fields[f"{side}_angle_act"][:] = 1000
            fields[f"{side}_applied"][:] = 1
        fields["feedback_healthy"][0] = True
        return pack_pose_message(fields, topic, 1)

    def sent(self, raw):
        topic = raw[: raw.index(b"{")].decode()
        fields = unpack_pose_message(raw, topic)
        self.messages.append((self.tick, topic, fields))
        if topic != "manager_state":
            return
        command = fields["recording_command"][0]
        event = self.protocol.receive(fields, self.now)
        if event:
            self.actions.append(event)
        if event in ("save", "abort") and self.finish_save:
            self.protocol.finish_save(True)
        if command == RecordingCommand.START:
            self.start_sends += 1
            if self.pending_start and self.start_sends == 1:
                return
        if self.tick not in self.suppress_ack_ticks:
            self.ack_queue.append(pack_pose_message(self.protocol.status_fields(0), "recording_status", 1))

    def run(self, **kwargs):
        self.manager.run_pico_manager(
            hand_input=kwargs.pop("hand_input", "optical"), hand_profile=self.profile, **kwargs
        )
        assert all(socket.closed for socket in self.sockets)
        expected = (
            []
            if kwargs.get("input_source") == "isaac-teleop"
            else [("bash", "/opt/apps/roboticsservice/runService.sh"), "sdk.init"]
        )
        assert self.service_calls == expected
        return self

    def states(self):
        return {tick: fields for tick, topic, fields in self.messages if topic == "manager_state"}


@pytest.mark.parametrize("profile", ["dex3", "inspire_ftp"])
def test_manager_orders_state_body_and_optical_generation_once_per_tick(monkeypatch, manager, profile):
    h = Harness(monkeypatch, manager, profile=profile, keys={9: "t", 12: "o"}, end=18).run()
    states = h.states()
    assert states[9]["stream_mode"][0] == 1
    for tick in range(9 if profile == "inspire_ftp" else 5, 18):
        messages = [(topic, fields) for t, topic, fields in h.messages if t == tick]
        assert messages[0][0] == "manager_state" and messages[0][1]["version"] == 4
        assert messages[0][1]["hand_profile"][0] == (profile == "inspire_ftp")
        body = next(fields for topic, fields in messages if topic in ("pose", "planner"))
        diagnostics = next(fields for topic, fields in messages if topic == "hand_tracking")
        assert body["sample_generation"][0] == diagnostics["sample_generation"][0] == tick
        assert diagnostics["body_sent"][0]
        np.testing.assert_array_equal(body["pv"], messages[0][1]["pv"])
        if profile == "inspire_ftp":
            assert "left_hand_joints" not in body and "right_hand_joints" not in body
            inspire = next(fields for topic, fields in messages if topic == "inspire_hand")
            validate_inspire_hand(inspire)
            assert inspire["sample_generation"][0] == tick
            assert [topic for topic, _ in messages].index("inspire_hand") > 1
        else:
            np.testing.assert_array_equal(body["left_hand_joints"], diagnostics["left_command"])
    assert len(h.steps) == 18 and len({tick for tick, _, _ in h.steps}) == 18
    assert all(seconds == pytest.approx(0.02) for tick, seconds in h.sleeps if tick >= 0)
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")


def test_feedback_gates_entry_and_one_missing_hand_does_not_stop_body(monkeypatch, manager):
    h = Harness(monkeypatch, manager, keys={9: "t", 11: "t"}, feedback_from=10, lost_left_from=19, end=28).run()
    assert h.states()[9]["stream_mode"][0] == 2
    assert h.states()[11]["stream_mode"][0] == 1
    diagnostics = {tick: fields for tick, topic, fields in h.messages if topic == "hand_tracking"}
    for tick in range(19, 28):
        assert h.states()[tick]["stream_mode"][0] == 1
        assert diagnostics[tick]["left_state"][0] == 1
        assert diagnostics[tick]["body_sent"][0]
        np.testing.assert_array_equal(diagnostics[tick]["left_command"], diagnostics[18]["left_command"])
    assert np.any(diagnostics[27]["right_command"] != diagnostics[18]["right_command"])


def test_c_s_controls_and_tracking_exit_wait_for_recording_and_saving(monkeypatch, manager):
    h = Harness(
        monkeypatch, manager, keys={9: "t", 10: "c", 12: "t", 14: "s", 16: "t"}, finish_save=False, end=19
    ).run()
    assert h.actions == ["start", "save"]
    assert h.states()[12]["stream_mode"][0] == h.states()[16]["stream_mode"][0] == 1
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")


def test_abxy_global_stop_delivers_abort_before_sonic_stop_without_wait(monkeypatch, manager):
    h = Harness(monkeypatch, manager, keys={9: "t", 10: "c"}, stop_tick=12, end=25).run()
    assert h.actions == ["start", "abort"]
    stop = [(tick, fields) for tick, topic, fields in h.messages if topic == "command" and fields["stop"][0]]
    assert len(stop) == 1 and stop[0][0] == 14
    assert h.tick == 14


def test_ctrl_c_pending_start_eventually_sends_stop_and_save_without_sonic_stop(monkeypatch, manager):
    h = Harness(monkeypatch, manager, keys={9: "t", 10: "c"}, pending_start=True, end=11).run()
    assert h.actions == ["start", "save"]
    assert h.protocol.state is RecordingState.IDLE
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")


def test_managed_pose_missing_body_has_no_inner_sleep(monkeypatch, manager):
    streamer = manager.PoseStreamer.__new__(manager.PoseStreamer)
    streamer.reader = SimpleNamespace(get_latest=lambda: None)
    streamer.managed = True
    sleeps = []
    monkeypatch.setattr(manager, "time", SimpleNamespace(sleep=sleeps.append))
    streamer.run_once()
    assert sleeps == []


def test_body_sent_is_reset_when_this_tick_has_no_body_frame(monkeypatch, manager):
    h = Harness(monkeypatch, manager, keys={9: "t"}, skip_body_ticks={13}, end=16).run()
    diagnostics = {tick: fields for tick, topic, fields in h.messages if topic == "hand_tracking"}
    assert diagnostics[12]["body_sent"][0]
    assert not diagnostics[13]["body_sent"][0]
    assert diagnostics[14]["body_sent"][0]
    assert diagnostics[13]["sample_generation"][0] == 13
    assert not any(tick == 13 and topic in ("pose", "planner") for tick, topic, _ in h.messages)
    assert sum(tick == 13 for tick, _, _ in h.steps) == 1


def test_actual_planner_abxy_startup_sends_only_neutral_planner_and_one_start(monkeypatch, manager):
    planner_class = manager.PlannerStreamer
    h = Harness(monkeypatch, manager, end=12)
    # Keep real planner computation and serialization; replace only its feedback I/O.
    monkeypatch.setattr(manager, "PlannerStreamer", planner_class)
    monkeypatch.setattr(
        manager, "FeedbackReader", lambda **kwargs: SimpleNamespace(poller=SimpleNamespace(close=lambda: None))
    )
    monkeypatch.setattr(manager.xrt, "get_time_stamp_ns", lambda: h.tick + 1, raising=False)
    h.run()

    commands = [(tick, fields) for tick, topic, fields in h.messages if topic == "command"]
    assert len(commands) == 1
    start_tick, command = commands[0]
    assert start_tick == 5  # The harness holds ABXY at ticks 3–5 after neutral release.
    assert command["start"].tolist() == [1]
    assert command["stop"].tolist() == [0]
    assert command["planner"].tolist() == [1]
    assert not any(
        topic in ("planner", "pose", "command", "inspire_hand")
        for tick, topic, _ in h.messages
        if tick < start_tick
    )
    assert not any(topic == "pose" for _, topic, _ in h.messages)
    startup_topics = [topic for tick, topic, _ in h.messages if tick == start_tick]
    assert startup_topics.index("planner") < startup_topics.index("command")
    planners = [(tick, fields) for tick, topic, fields in h.messages if topic == "planner"]
    assert [tick for tick, _ in planners] == list(range(start_tick, h.end))
    for _, fields in planners:
        assert fields["mode"].tolist() == [0]
        assert fields["movement"].tolist() == [0.0, 0.0, 0.0]
        assert fields["facing"].tolist() == [1.0, 0.0, 0.0]
        assert fields["speed"].tolist() == fields["height"].tolist() == [-1.0]
        assert not any(name.startswith(("upper_body_", "vr_")) or name == "body_q" for name in fields)
        for side in ("left", "right"):
            np.testing.assert_array_equal(fields[f"{side}_hand_joints"], np.zeros(7))
    assert all(not enabled for _, _, enabled in h.steps)


@pytest.mark.parametrize("kind", ["pose", "planner"])
def test_actual_streamers_use_shared_atomic_controls(monkeypatch, manager, kind):
    class ReachedBodyProcessing(BaseException):
        pass

    def reached(*args, **kwargs):
        raise ReachedBodyProcessing

    def legacy(*args):
        pytest.fail("managed streamer re-read legacy controller SDK")

    monkeypatch.setattr(manager, "get_abxy_buttons", legacy)
    monkeypatch.setattr(manager, "get_controller_inputs", legacy)
    monkeypatch.setattr(manager, "get_controller_axes", legacy)
    frame = {"abxy": (False,) * 4, "inputs": (False, 0.0, 0.0, 0.0, 0.0), "axes": (0.0,) * 4}
    if kind == "pose":
        obj = manager.PoseStreamer.__new__(manager.PoseStreamer)
        obj.reader = SimpleNamespace(get_latest=lambda: {"body_poses_np": np.zeros((24, 7))})
        obj.parent_indices, obj.device = [], None
        obj.data_collection_chords = manager.DataCollectionChordTracker()
        obj.hand_profile = "dex3"
        obj.left_hand_ik_solver = obj.right_hand_ik_solver = None
        monkeypatch.setattr(manager, "compute_from_body_poses", lambda *args: {})
        monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", reached)
        run = obj.run_once
    else:
        obj = manager.PlannerStreamer.__new__(manager.PlannerStreamer)
        obj.controller_3pt, obj.last_xrt_timestamp = False, None
        obj.reader = SimpleNamespace(get_timestamp_ns=lambda: 1)
        obj.prev_ab = obj.prev_xy = False
        obj.dt = 0.02
        obj.yaw_accumulator = SimpleNamespace(update=reached)
        monkeypatch.setattr(manager, "xrt", SimpleNamespace(get_time_stamp_ns=lambda: 1))

        def run():
            obj.run_once(manager.StreamMode.PLANNER)

    obj.managed, obj.control_frame = True, frame
    with pytest.raises(ReachedBodyProcessing):
        run()


@pytest.mark.parametrize("profile", ["dex3", "inspire_ftp"])
def test_real_capture_uses_same_snapshots_and_flushes_wire_evidence_on_shutdown(
    monkeypatch, manager, tmp_path, profile
):
    logs = []

    class ObservedLog(HandCaptureLog):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorded_snapshot_ids = []
            self.close_calls = 0
            logs.append(self)

        def record(self, tick_ns, snapshots, *args, **kwargs):
            self.recorded_snapshot_ids.append(tuple(id(value) for value in snapshots))
            return super().record(tick_ns, snapshots, *args, **kwargs)

        def close(self):
            self.close_calls += 1
            # This capture is shorter than one shard: only close() can persist it.
            assert not list(self.output_dir.glob("capture_*.npz"))
            return super().close()

    monkeypatch.setattr(manager, "HandCaptureLog", ObservedLog)
    directory = tmp_path / profile
    h = Harness(
        monkeypatch,
        manager,
        profile=profile,
        keys={9: "t"},
        skip_body_ticks={13},
        lost_left_from=15,
        end=18,
    ).run(hand_log_dir=str(directory))
    assert len(logs) == 1 and logs[0].close_calls == 1
    assert len(h.hand_snapshot_reads) == 2 * h.end
    assert len(logs[0].recorded_snapshot_ids) == h.end
    files = list(directory.glob("capture_*.npz"))
    assert len(files) == 1
    assert not list(directory.glob("*.tmp"))
    with np.load(files[0], allow_pickle=False) as capture:
        assert validate_capture(capture, profile) == h.end
        assert capture["source_kind"].item() == "runtime_capture"
        assert capture["profile"].item() == profile
        assert capture["sample_generation"].dtype == np.int64
        np.testing.assert_array_equal(capture["sample_generation"], np.arange(h.end))
        ticks = dict((tick, now) for tick, now, _ in h.steps)
        states = h.states()
        for tick in range(h.end):
            assert capture["tick_ns"][tick] == ticks[tick]
            np.testing.assert_array_equal(capture["pv"][tick], states[tick]["pv"])
            reads = [(side, snapshot) for read_tick, side, snapshot in h.hand_snapshot_reads if read_tick == tick]
            assert [side for side, _ in reads] == ["left", "right"]
            assert logs[0].recorded_snapshot_ids[tick] == tuple(id(snapshot) for _, snapshot in reads)
            for index, (side, snapshot) in enumerate(reads):
                assert capture["snapshot_present"][tick, index] == (snapshot is not None)
                if snapshot is not None:
                    for key, field in (
                        ("poses", "pose"),
                        ("flags", "location_flags"),
                        ("radius", "radius"),
                        ("scale", "scale"),
                        ("active", "is_active"),
                        ("source_timestamp_ns", "source_timestamp_ns"),
                        ("binding_generation", "binding_generation"),
                    ):
                        np.testing.assert_array_equal(capture[key][tick, index], snapshot[field])
                output = h.hand_outputs[tick][index]
                assert capture["captured_state"][tick, index] == output.state
                assert capture["captured_reason"][tick, index] == output.reason
                if output.command is None:
                    assert np.isnan(capture["captured_command"][tick, index]).all()
                else:
                    np.testing.assert_array_equal(capture["captured_command"][tick, index], output.command)
                if output.target is not None:
                    np.testing.assert_array_equal(capture["captured_target"][tick, index], output.target)
            body = next(
                (fields for t, topic, fields in h.messages if t == tick and topic in ("pose", "planner")), None
            )
            assert capture["body_sent"][tick] == (body is not None)
            if body is not None:
                assert capture["sample_generation"][tick] == body["sample_generation"][0]
            diagnostics = next(
                (fields for t, topic, fields in h.messages if t == tick and topic == "hand_tracking"), None
            )
            if diagnostics is not None:
                assert capture["sample_generation"][tick] == diagnostics["sample_generation"][0]
                for index, side in enumerate(("left", "right")):
                    np.testing.assert_array_equal(
                        capture["captured_command"][tick, index], diagnostics[f"{side}_command"]
                    )
        assert capture["body_sent"][12] and not capture["body_sent"][13] and capture["body_sent"][14]
        assert capture["hand_sent"][13].all() == (profile == "inspire_ftp")
        assert not capture["snapshot_present"][15:, 0].any()
        assert capture["snapshot_present"][:, 1].all()
    events = [json.loads(line) for line in (directory / "transitions.jsonl").read_text().splitlines()]
    assert any(event["side"] == "left" and event["reason"] == "MISSING" for event in events)


def test_capture_record_error_disables_logging_but_body_continues_and_close_flushes(
    monkeypatch, manager, tmp_path, capsys
):
    logs = []

    class FailingLog(HandCaptureLog):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.record_calls = self.close_calls = 0
            logs.append(self)

        def record(self, *args, **kwargs):
            self.record_calls += 1
            if self.record_calls == 13:
                raise OSError("injected capture disk error")
            return super().record(*args, **kwargs)

        def close(self):
            self.close_calls += 1
            return super().close()

    monkeypatch.setattr(manager, "HandCaptureLog", FailingLog)
    h = Harness(monkeypatch, manager, keys={9: "t"}, end=18).run(hand_log_dir=str(tmp_path))
    assert logs[0].record_calls == 13 and logs[0].close_calls == 1
    assert len(h.steps) == 18
    for tick in range(12, 18):
        assert any(t == tick and topic == "pose" for t, topic, _ in h.messages)
        assert any(t == tick and topic == "hand_tracking" for t, topic, _ in h.messages)
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")
    assert capsys.readouterr().out.count("Hand capture disabled after error: injected capture disk error") == 1
    with np.load(tmp_path / "capture_000000.npz", allow_pickle=False) as capture:
        assert validate_capture(capture, "dex3") == 12
        np.testing.assert_array_equal(capture["sample_generation"], np.arange(12))


def test_ctrl_c_ignores_malformed_ack_then_saves_and_cleans_up(monkeypatch, manager):
    h = Harness(
        monkeypatch,
        manager,
        keys={9: "t", 10: "c"},
        pending_start=True,
        malformed_shutdown_ack=True,
        end=11,
    ).run()
    assert h.actions == ["start", "save"]
    assert h.protocol.state is RecordingState.IDLE
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")


def test_ctrl_c_retries_save_when_stale_recorder_ack_resumes(monkeypatch, manager):
    h = Harness(
        monkeypatch,
        manager,
        keys={9: "t", 10: "c"},
        suppress_ack_ticks=range(11, 40),
        end=40,
    ).run()
    assert h.actions == ["start", "save"]
    assert h.protocol.state is RecordingState.IDLE
    shutdown = [fields for tick, topic, fields in h.messages if tick == 40 and topic == "manager_state"]
    assert len(shutdown) >= 2
    assert shutdown[0]["recording_command"][0] == RecordingCommand.NONE
    assert any(fields["recording_command"][0] == RecordingCommand.STOP_AND_SAVE for fields in shutdown)
    assert not any(fields["stop"][0] for _, topic, fields in h.messages if topic == "command")


@pytest.mark.parametrize("profile", ["dex3", "inspire_ftp"])
def test_managed_controller_profile_keeps_main_command_contract(monkeypatch, manager, profile):
    h = Harness(monkeypatch, manager, profile=profile, end=12)
    h.run(hand_input="controller")
    assert h.hands is None
    size = 7 if profile == "dex3" else 6
    planner_messages = [fields for _, topic, fields in h.messages if topic == "planner"]
    assert planner_messages
    for fields in planner_messages:
        for side in ("left", "right"):
            assert fields[f"{side}_hand_joints"].shape == (size,)
    diagnostics = [fields for _, topic, fields in h.messages if topic == "hand_tracking"]
    assert diagnostics
    assert all(fields["hand_profile"][0] == (profile == "inspire_ftp") for fields in diagnostics)


def test_managed_isaac_controls_use_reader_without_xrt(monkeypatch, manager):
    h = Harness(monkeypatch, manager, end=12)
    reader = manager.PicoReader()
    monkeypatch.setattr(manager, "xrt", None)
    monkeypatch.setattr(manager, "_init_input_source", lambda source, size: reader)
    reads = []

    def controls(kind, value):
        def get(actual_reader):
            assert actual_reader is reader
            reads.append((h.tick, kind))
            return value() if callable(value) else value

        return get

    monkeypatch.setattr(manager, "get_abxy_buttons", controls("abxy", lambda: (3 <= h.tick <= 5,) * 4))
    monkeypatch.setattr(manager, "get_controller_inputs", controls("inputs", (False, 0.0, 0.0, 0.0, 0.0)))
    monkeypatch.setattr(manager, "get_controller_axes", controls("axes", (0.0,) * 4))
    monkeypatch.setattr(manager, "get_axis_clicks", controls("clicks", (False, False)))
    h.run(hand_input="controller", input_source="isaac-teleop")
    assert len(reads) == 4 * h.end
    assert any(topic == "planner" for _, topic, _ in h.messages)
    assert h.hands is None


def test_optical_rejects_isaac_input_before_initialization(monkeypatch, manager):
    monkeypatch.setattr(manager, "xrt", None)

    def unexpected(*args):
        pytest.fail("invalid source opened resources")

    monkeypatch.setattr(manager, "_init_input_source", unexpected)
    with pytest.raises(ValueError, match="Optical hands require"):
        manager.run_pico_manager(hand_input="optical", input_source="isaac-teleop")
