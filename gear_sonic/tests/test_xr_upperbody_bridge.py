import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    BridgeConfig,
    DEX3_LEFT_STOP_HAND_TUCKED,
    DEX3_RIGHT_STOP_HAND_TUCKED,
    G1_CALIB_FULL_UPPER_BODY,
    G1_STANDING_UPPER_BODY,
    G1_UPPER_BODY_JOINT_INDICES,
    StopReleaseSchedule,
    UpperBodyFilter,
    anchor_frame_planner_heading,
    build_heading_hold_planner_message,
    build_due_stop_release_messages,
    build_manager_state_message,
    build_frame_planner_message,
    build_inspection_report,
    build_start_control_messages,
    build_stop_standing_message,
    feedback_payload_heading_yaw,
    collect_range_warnings,
    frame_with_current_upper_body_position,
    load_frames,
    normalize_live_source_payload,
    run_local_zmq_smoke,
    split_xr_dex3_dual_hand_joints,
    start_stop_release_schedule,
    synthesize_missing_velocities,
    unpack_bridge_message,
    xr_g1_29_arm_to_sonic_upper_body,
    _send_zmq_json_loop,
)


def _valid_position(offset: float = 0.0) -> list[float]:
    return [offset + 0.01 * i for i in range(17)]


def _decode_header(message: bytes, topic: str = "planner") -> dict:
    start = len(topic)
    raw = message[start : start + HEADER_SIZE].rstrip(b"\x00")
    return json.loads(raw.decode("utf-8"))


def test_calib_full_upper_body_matches_teleop_all_zero_reference() -> None:
    assert G1_CALIB_FULL_UPPER_BODY.shape == (17,)
    assert G1_CALIB_FULL_UPPER_BODY.dtype == np.float32
    assert np.allclose(G1_CALIB_FULL_UPPER_BODY, np.zeros(17, dtype=np.float32))


def test_load_jsonl_frames_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "upperbody.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": 0.0,
                        "upper_body_position": _valid_position(),
                        "left_hand_joints": [0.1] * 7,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": 0.02,
                        "upper_body_position": _valid_position(0.1),
                        "upper_body_velocity": [0.2] * 17,
                        "right_hand_joints": [0.3] * 7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert frames[0].upper_body_position.shape == (17,)
    assert np.allclose(frames[0].upper_body_velocity, np.zeros(17))
    assert np.allclose(frames[0].left_hand_joints, np.full(7, 0.1))
    assert frames[0].right_hand_joints is None
    assert np.allclose(frames[1].upper_body_velocity, np.full(17, 0.2))


def test_load_npz_frames(tmp_path: Path) -> None:
    path = tmp_path / "upperbody.npz"
    np.savez(
        path,
        timestamp=np.array([0.0, 0.02], dtype=np.float64),
        upper_body_position=np.stack([_valid_position(), _valid_position(0.1)]).astype(np.float32),
        left_hand_joints=np.ones((2, 7), dtype=np.float32),
    )

    frames = load_frames(path)

    assert len(frames) == 2
    assert frames[1].timestamp == 0.02
    assert np.allclose(frames[1].left_hand_joints, np.ones(7))
    assert frames[1].right_hand_joints is None


def test_load_jsonl_dual_arm_position_maps_to_sonic_upper_body(tmp_path: Path) -> None:
    path = tmp_path / "arms_only.jsonl"
    arm = [float(i) for i in range(14)]
    path.write_text(
        json.dumps(
            {
                "timestamp": 0.0,
                "dual_arm_position": arm,
                "left_hand_joints": [0.1] * 7,
                "right_hand_joints": [0.2] * 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert np.allclose(
        frames[0].upper_body_position,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            7.0,
            1.0,
            8.0,
            2.0,
            9.0,
            3.0,
            10.0,
            4.0,
            11.0,
            5.0,
            12.0,
            6.0,
            13.0,
        ],
    )


def test_load_jsonl_dual_hand_joints_splits_left_and_right(tmp_path: Path) -> None:
    path = tmp_path / "hands.jsonl"
    path.write_text(
        json.dumps(
            {
                "dual_arm_position": [0.0] * 14,
                "dual_hand_joints": [float(i) for i in range(14)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert np.allclose(frames[0].left_hand_joints, np.arange(7, dtype=np.float32))
    assert np.allclose(frames[0].right_hand_joints, np.arange(7, 14, dtype=np.float32))


def test_live_payload_control_flags_are_source_lifecycle_metadata() -> None:
    frame = normalize_live_source_payload(
        {
            "dual_arm_position": [0.0] * 14,
            "dual_hand_joints": [0.0] * 14,
            "mode": 4,
            "movement": [0.0, 0.0, 0.0],
            "facing": [1.0, 0.0, 0.0],
            "speed": 0.0,
            "height": 0.5,
            "toggle_data_collection": True,
            "toggle_data_abort": True,
            "stop": True,
            "ramp_phase": "in",
            "ramp_alpha": 0.25,
        }
    )

    assert frame.mode == 4
    assert frame.height == 0.5
    assert frame.toggle_data_collection is True
    assert frame.toggle_data_abort is True
    assert frame.stop is True
    assert frame.ramp_phase == "in"
    assert frame.ramp_alpha == 0.25

    planner_message = build_frame_planner_message(frame, BridgeConfig())
    planner_header = _decode_header(planner_message)
    planner_fields = {field["name"] for field in planner_header["fields"]}
    manager_message = build_manager_state_message(
        stream_mode=5,
        toggle_data_collection=frame.toggle_data_collection,
        toggle_data_abort=frame.toggle_data_abort,
    )
    manager_header = _decode_header(manager_message, topic="manager_state")
    manager_fields = {field["name"] for field in manager_header["fields"]}

    assert "stop" not in planner_fields
    assert "stop" not in manager_fields


def test_feedback_payload_heading_yaw_uses_base_quat_wxyz() -> None:
    yaw = math.pi / 2.0
    payload = {"base_quat": [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]}

    assert feedback_payload_heading_yaw(payload) == pytest.approx(yaw)


def test_anchor_frame_planner_heading_rotates_unitree_local_vectors() -> None:
    frame = normalize_live_source_payload(
        {
            "dual_arm_position": [0.0] * 14,
            "movement": [1.0, 0.0, 0.0],
            "facing": [1.0, 0.0, 0.0],
        }
    )

    anchored = anchor_frame_planner_heading(frame, math.pi / 2.0)

    assert anchored.movement == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert anchored.facing == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_heading_hold_planner_message_preserves_current_heading() -> None:
    upper_body = np.full(17, 0.2, dtype=np.float32)
    message = build_heading_hold_planner_message(
        math.pi / 2.0,
        upper_body_position=upper_body,
    )
    decoded = unpack_bridge_message(message, topic="planner")

    assert int(decoded["mode"][0]) == 0
    assert np.allclose(decoded["movement"], np.zeros(3))
    assert decoded["facing"] == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-6)
    assert float(decoded["speed"][0]) == 0.0
    assert float(decoded["height"][0]) == -1.0
    assert np.allclose(decoded["upper_body_position"], upper_body)
    assert np.allclose(decoded["upper_body_velocity"], np.zeros(17))


def test_start_control_messages_send_seeded_hold_before_start_command() -> None:
    upper_body = np.full(17, 0.2, dtype=np.float32)
    messages = build_start_control_messages(
        heading_yaw=math.pi / 2.0,
        upper_body_position=upper_body,
    )

    assert len(messages) == 2
    planner = unpack_bridge_message(messages[0], topic="planner")
    command = unpack_bridge_message(messages[1], topic="command")
    assert np.allclose(planner["upper_body_position"], upper_body)
    assert int(command["start"][0]) == 1
    assert int(command["stop"][0]) == 0
    assert int(command["planner"][0]) == 1


def test_rejects_bad_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"upper_body_position": [0.0] * 16}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="upper_body_position"):
        load_frames(path)


def test_xr_g1_29_arm_to_sonic_upper_body_with_waist() -> None:
    arm = np.arange(14, dtype=np.float32)
    upper = xr_g1_29_arm_to_sonic_upper_body(arm, waist=[0.1, 0.2, 0.3])

    assert np.allclose(upper[:3], [0.1, 0.2, 0.3])
    assert np.allclose(upper[3:], [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13])


def test_split_xr_dex3_dual_hand_joints() -> None:
    left, right = split_xr_dex3_dual_hand_joints(np.arange(14, dtype=np.float32))

    assert np.allclose(left, [0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(right, [7, 8, 9, 10, 11, 12, 13])


def test_calibrated_controller_pinch_json_splits_exact_dex3_sides() -> None:
    left = [-0.379616, 0.516712, 0.121406, 0.0, 0.0, -1.273903, -0.419393]
    right = [-0.379617, -0.516714, -0.121407, 0.0, 0.0, 1.273907, 0.419395]
    payload = json.loads(
        json.dumps(
            {
                "dual_arm_position": [0.0] * 14,
                "dual_hand_joints": left + right,
            }
        )
    )

    frame = normalize_live_source_payload(payload)

    np.testing.assert_allclose(frame.left_hand_joints, left, rtol=0, atol=1e-6)
    np.testing.assert_allclose(frame.right_hand_joints, right, rtol=0, atol=1e-6)


def test_tucked_stop_hand_preset_uses_thumb_opposition_joint() -> None:
    assert DEX3_LEFT_STOP_HAND_TUCKED[1] > 0.0
    assert DEX3_RIGHT_STOP_HAND_TUCKED[1] < 0.0
    assert DEX3_LEFT_STOP_HAND_TUCKED[2] > 0.0
    assert DEX3_RIGHT_STOP_HAND_TUCKED[2] < 0.0


def test_synthesize_missing_velocities_from_timestamps() -> None:
    frames = load_frames(
        [
            {"timestamp": 0.0, "dual_arm_position": [0.0] * 14},
            {"timestamp": 0.5, "dual_arm_position": [0.5] * 14},
        ]
    )

    synthesized = synthesize_missing_velocities(frames, default_hz=50.0)

    assert np.allclose(synthesized[0].upper_body_velocity, np.zeros(17))
    assert np.allclose(synthesized[1].upper_body_velocity[:3], np.zeros(3))
    assert np.allclose(synthesized[1].upper_body_velocity[3:], np.full(14, 1.0))


def test_range_warning_detects_degrees_like_values() -> None:
    frames = load_frames(
        [
            {
                "dual_arm_position": [0.0] * 13 + [180.0],
                "dual_hand_joints": [0.0] * 14,
            }
        ]
    )

    warnings = collect_range_warnings(frames, arm_warn_rad=6.5, hand_warn_rad=3.2)

    assert any("upper_body_position" in warning for warning in warnings)


def test_inspection_report_includes_mapping_and_warnings() -> None:
    frames = load_frames(
        [
            {
                "timestamp": 0.0,
                "dual_arm_position": [0.1] * 14,
                "dual_hand_joints": [0.2] * 14,
            }
        ]
    )

    report = build_inspection_report(frames, source="sample.jsonl")

    assert "sample.jsonl" in report
    assert "frames: 1" in report
    assert "first upper_body_position[17]" in report
    assert "left_hand_joints range" in report


def test_local_zmq_smoke_receives_planner_and_manager_state() -> None:
    frames = load_frames([{"dual_arm_position": [0.0] * 14, "dual_hand_joints": [0.0] * 14}])

    result = run_local_zmq_smoke(frames, BridgeConfig(), port=0, timeout_s=2.0)

    assert int(result["planner"]["mode"][0]) == 0
    assert result["planner"]["upper_body_position"].shape == (17,)
    assert int(result["manager_state"]["stream_mode"][0]) == 5


def test_start_control_defers_live_upper_body_until_late_feedback_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import queue
    import sys

    msgpack = pytest.importorskip("msgpack")

    measured_upper_body = np.full(17, 0.2, dtype=np.float32)
    measured_body_q = np.zeros(29, dtype=np.float32)
    measured_body_q[G1_UPPER_BODY_JOINT_INDICES] = measured_upper_body
    source_payload = json.dumps(
        {
            "upper_body_position": [1.0] * 17,
            "upper_body_velocity": [0.0] * 17,
        }
    ).encode("utf-8")

    class FakeAgain(Exception):
        pass

    class FakeContext:
        def __init__(self) -> None:
            self.source_messages: queue.Queue[bytes] = queue.Queue()
            self.feedback_messages: queue.Queue[bytes] = queue.Queue()
            self.sent_messages: list[bytes] = []
            self._condition = threading.Condition()

        def socket(self, kind: int) -> "FakeSocket":
            return FakeSocket(self, kind)

        def term(self) -> None:
            return None

        def queue_for_endpoint(self, endpoint: str) -> queue.Queue[bytes]:
            if endpoint.endswith(":5560"):
                return self.source_messages
            if endpoint.endswith(":5557"):
                return self.feedback_messages
            raise AssertionError(f"unexpected fake ZMQ endpoint: {endpoint}")

        def record_send(self, message: bytes) -> None:
            with self._condition:
                self.sent_messages.append(message)
                self._condition.notify_all()

        def wait_for_sent(self, predicate, timeout_s: float) -> bytes | None:
            deadline = time.monotonic() + timeout_s
            with self._condition:
                while time.monotonic() < deadline:
                    for message in self.sent_messages:
                        if predicate(message):
                            return message
                    self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            return None

        def sent_since(self, start_index: int) -> list[bytes]:
            with self._condition:
                return list(self.sent_messages[start_index:])

        def sent_count(self) -> int:
            with self._condition:
                return len(self.sent_messages)

    class FakeSocket:
        def __init__(self, context: FakeContext, kind: int) -> None:
            self._context = context
            self._kind = kind
            self._queue: queue.Queue[bytes] | None = None
            self._timeout_s = 1.0

        def setsockopt_string(self, _option: int, _value: str) -> None:
            return None

        def setsockopt(self, option: int, value: int) -> None:
            if option == fake_zmq.RCVTIMEO:
                self._timeout_s = max(float(value) / 1000.0, 0.001)

        def connect(self, endpoint: str) -> None:
            self._queue = self._context.queue_for_endpoint(endpoint)

        def bind(self, _endpoint: str) -> None:
            return None

        def send(self, message: bytes) -> None:
            self._context.record_send(message)

        def recv(self) -> bytes:
            assert self._queue is not None
            try:
                return self._queue.get(timeout=self._timeout_s)
            except queue.Empty as exc:
                raise FakeAgain() from exc

        def close(self, linger: int = 0) -> None:
            return None

    fake_context = FakeContext()

    class FakeContextFactory:
        @staticmethod
        def instance() -> FakeContext:
            return fake_context

    class fake_zmq:
        PUB = 1
        SUB = 2
        RCVTIMEO = 3
        SUBSCRIBE = 4
        Again = FakeAgain
        Context = FakeContextFactory

    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)

    args = argparse.Namespace(
        bind_host="127.0.0.1",
        port=5556,
        source_host="127.0.0.1",
        source_port=5560,
        source_topic="xr_teleop",
        source_timeout_s=0.05,
        feedback_host="127.0.0.1",
        feedback_port=5557,
        feedback_topic="g1_debug",
        feedback_prime_timeout_s=0.05,
        no_feedback_prime=False,
        allow_unseeded_start_control=False,
        start_control=True,
        start_command_repeat_s=1.0,
        start_command_interval_s=0.05,
        pub_warmup_s=0.05,
        hz=50.0,
        max_abs_joint=3.14,
        max_joint_step=0.03,
        stream_mode=5,
        stop_release_s=0.0,
        stop_final_hold_s=0.0,
        stop_hand_preset="tucked-thumb",
        stop_upper_body_preset="straight",
        send_stop_on_exit=False,
        anchor_planner_heading=True,
        once=True,
        debug_live=False,
        debug_interval=0.5,
    )
    result: dict[str, int | None] = {"rc": None}

    def _run_bridge() -> None:
        result["rc"] = _send_zmq_json_loop(args)

    thread = threading.Thread(target=_run_bridge, daemon=True)
    thread.start()

    try:
        start_raw = fake_context.wait_for_sent(
            lambda message: (
                message.startswith(b"command")
                and int(unpack_bridge_message(message, topic="command")["start"][0]) == 1
            ),
            timeout_s=2.0,
        )
        assert start_raw is not None

        count_before_unseeded_source = fake_context.sent_count()
        fake_context.source_messages.put(b"xr_teleop" + source_payload)
        time.sleep(0.2)
        assert not any(
            message.startswith(b"planner")
            for message in fake_context.sent_since(count_before_unseeded_source)
        )

        fake_context.feedback_messages.put(
            b"g1_debug"
            + msgpack.packb(
                {
                    "body_q_measured": measured_body_q.tolist(),
                    "base_quat": [1.0, 0.0, 0.0, 0.0],
                },
                use_bin_type=True,
            )
        )
        fake_context.source_messages.put(b"xr_teleop" + source_payload)

        planner_raw = fake_context.wait_for_sent(
            lambda message: (
                message.startswith(b"planner")
                and "upper_body_position" in unpack_bridge_message(message, topic="planner")
            ),
            timeout_s=2.0,
        )

        assert planner_raw is not None
        planner = unpack_bridge_message(planner_raw, topic="planner")
        assert np.allclose(planner["upper_body_position"], measured_upper_body + 0.03)
    finally:
        thread.join(timeout=2.0)

    assert result["rc"] == 0


def test_normalize_live_source_payload_from_genonai_style_json() -> None:
    frame = normalize_live_source_payload(
        {
            "timestamp": 1.25,
            "dual_arm_position": [0.1] * 14,
            "dual_hand_joints": [0.2] * 14,
        }
    )

    assert frame.timestamp == 1.25
    assert frame.upper_body_position.shape == (17,)
    assert np.allclose(frame.upper_body_position[:3], np.zeros(3))
    assert np.allclose(frame.left_hand_joints, np.full(7, 0.2))
    assert np.allclose(frame.right_hand_joints, np.full(7, 0.2))


def test_filter_clips_and_limits_position_steps() -> None:
    config = BridgeConfig(max_abs_joint=1.0, max_joint_step=0.25)
    filt = UpperBodyFilter(config)
    first = filt.apply(np.full(17, 0.9, dtype=np.float32))
    second = filt.apply(np.full(17, 2.0, dtype=np.float32))

    assert np.allclose(first, np.full(17, 0.9))
    assert np.allclose(second, np.full(17, 1.0))


def test_filter_can_seed_from_measured_position() -> None:
    config = BridgeConfig(max_abs_joint=1.0, max_joint_step=0.25)
    filt = UpperBodyFilter(config)
    filt.seed(np.zeros(17, dtype=np.float32))
    first = filt.apply(np.full(17, 0.9, dtype=np.float32))

    assert np.allclose(first, np.full(17, 0.25))


def test_start_ramp_blends_from_feedback_seed_to_target() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=2.0)
    filt = UpperBodyFilter(config)
    filt.seed(np.zeros(17, dtype=np.float32))
    frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "ramp_phase": "in",
                "ramp_alpha": 0.5,
            }
        ]
    )[0]

    message = build_frame_planner_message(frame, config, filt)
    upper = unpack_bridge_message(message, topic="planner")["upper_body_position"]

    assert np.allclose(upper, np.full(17, 0.5))


def test_start_ramp_uses_fixed_origin_and_hard_step_limit() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=0.03)
    filt = UpperBodyFilter(config)
    seed = np.zeros(17, dtype=np.float32)
    target = np.ones(17, dtype=np.float32)
    filt.seed(seed)

    outputs = []
    for frame_index in range(51):
        raw_alpha = frame_index / 50.0
        alpha = raw_alpha * raw_alpha * (3.0 - 2.0 * raw_alpha)
        outputs.append(
            filt.apply_ramped(target, ramp_phase="in", ramp_alpha=alpha)
        )

    output_array = np.stack(outputs)
    assert np.allclose(output_array[0], seed)
    assert (
        np.max(np.abs(np.diff(output_array, axis=0)))
        <= config.max_joint_step + 1e-6
    )
    assert np.allclose(
        output_array[25],
        np.full(17, 0.5, dtype=np.float32),
        atol=1e-6,
    )


def test_start_ramp_late_first_packet_still_honors_step_limit() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=0.03)
    filt = UpperBodyFilter(config)
    filt.seed(np.zeros(17, dtype=np.float32))

    output = filt.apply_ramped(
        np.ones(17, dtype=np.float32),
        ramp_phase="in",
        ramp_alpha=0.9,
    )

    assert np.allclose(output, np.full(17, config.max_joint_step))


def test_resume_ramp_captures_new_fixed_origin() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=2.0)
    filt = UpperBodyFilter(config)
    filt.seed(np.zeros(17, dtype=np.float32))
    target = np.ones(17, dtype=np.float32)

    filt.apply_ramped(target, ramp_phase="in", ramp_alpha=0.5)
    tracked = filt.apply(np.full(17, 0.8, dtype=np.float32))
    resume_start = filt.apply_ramped(target, ramp_phase="resume", ramp_alpha=0.0)
    resume_half = filt.apply_ramped(
        np.zeros(17, dtype=np.float32),
        ramp_phase="resume",
        ramp_alpha=0.5,
    )
    repeated_resume_half = filt.apply_ramped(
        np.zeros(17, dtype=np.float32),
        ramp_phase="resume",
        ramp_alpha=0.5,
    )

    assert np.allclose(resume_start, tracked)
    assert np.allclose(resume_half, np.full(17, 0.4, dtype=np.float32))
    assert np.allclose(repeated_resume_half, resume_half)


def test_exit_ramp_uses_sender_ramped_target_without_rate_limiter() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=0.2)
    filt = UpperBodyFilter(config)
    filt.seed(np.ones(17, dtype=np.float32))
    frame = load_frames(
        [
            {
                "upper_body_position": [0.25] * 17,
                "ramp_phase": "out",
                "ramp_alpha": 0.75,
            }
        ]
    )[0]

    message = build_frame_planner_message(frame, config, filt)
    upper = unpack_bridge_message(message, topic="planner")["upper_body_position"]

    assert np.allclose(upper, np.full(17, 0.25))


def test_build_planner_message_fields() -> None:
    frames = load_frames(
        [
            {
                "upper_body_position": _valid_position(),
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
            }
        ]
    )
    message = build_frame_planner_message(frames[0], BridgeConfig())
    header = _decode_header(message)
    fields = {field["name"]: field for field in header["fields"]}

    assert message.startswith(b"planner")
    assert fields["mode"]["shape"] == [1]
    assert fields["movement"]["shape"] == [3]
    assert fields["facing"]["shape"] == [3]
    assert fields["upper_body_position"]["shape"] == [17]
    assert fields["upper_body_velocity"]["shape"] == [17]
    assert fields["left_hand_joints"]["shape"] == [7]
    assert fields["right_hand_joints"]["shape"] == [7]


def test_build_planner_message_tucks_hands_during_exit_ramp_out() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": _valid_position(),
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "ramp_phase": "out",
                "ramp_alpha": 1.0,
            }
        ]
    )[0]

    message = build_frame_planner_message(
        frame,
        BridgeConfig(),
        stop_hand_preset="tucked-thumb",
    )
    decoded = unpack_bridge_message(message, topic="planner")

    assert np.allclose(decoded["left_hand_joints"], DEX3_LEFT_STOP_HAND_TUCKED)
    assert np.allclose(decoded["right_hand_joints"], DEX3_RIGHT_STOP_HAND_TUCKED)


def test_stop_standing_message_ramps_upper_body_to_final_standing_by_default() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "stop": True,
            }
        ]
    )[0]

    message = build_stop_standing_message(frame, alpha=1.0)
    fields = {field["name"]: field for field in _decode_header(message)["fields"]}
    decoded = unpack_bridge_message(message, topic="planner")

    assert message.startswith(b"planner")
    assert fields["upper_body_position"]["shape"] == [17]
    assert fields["upper_body_velocity"]["shape"] == [17]
    assert fields["left_hand_joints"]["shape"] == [7]
    assert fields["right_hand_joints"]["shape"] == [7]
    assert int(decoded["mode"][0]) == 0
    assert np.allclose(decoded["movement"], np.zeros(3))
    assert float(decoded["speed"][0]) == 0.0
    assert np.allclose(decoded["upper_body_position"], G1_STANDING_UPPER_BODY)
    assert np.allclose(decoded["upper_body_velocity"], np.zeros(17))
    assert np.allclose(decoded["left_hand_joints"], DEX3_LEFT_STOP_HAND_TUCKED)
    assert np.allclose(decoded["right_hand_joints"], DEX3_RIGHT_STOP_HAND_TUCKED)


def test_stop_standing_message_can_use_deploy_standing_preset() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "stop": True,
            }
        ]
    )[0]

    message = build_stop_standing_message(
        frame,
        alpha=1.0,
        upper_body_preset="deploy-standing",
    )
    decoded = unpack_bridge_message(message, topic="planner")

    assert np.allclose(decoded["upper_body_position"], G1_STANDING_UPPER_BODY)


def test_stop_standing_message_can_use_calib_full_preset() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "stop": True,
            }
        ]
    )[0]

    message = build_stop_standing_message(
        frame,
        alpha=1.0,
        upper_body_preset="calib-full",
    )
    decoded = unpack_bridge_message(message, topic="planner")

    assert np.allclose(decoded["upper_body_position"], G1_CALIB_FULL_UPPER_BODY)


def test_stop_standing_message_can_omit_hand_targets() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": [0.0] * 17,
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "stop": True,
            }
        ]
    )[0]

    message = build_stop_standing_message(frame, alpha=1.0, hand_preset="none")
    fields = {field["name"]: field for field in _decode_header(message)["fields"]}

    assert "left_hand_joints" not in fields
    assert "right_hand_joints" not in fields


def test_stop_release_schedule_sends_exact_final_then_holds() -> None:
    frame = load_frames(
        [
            {
                "upper_body_position": [0.0] * 17,
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
                "stop": True,
            }
        ]
    )[0]
    schedule = StopReleaseSchedule.start(
        frame,
        now=10.0,
        release_s=1.0,
        final_hold_s=0.5,
    )

    assert schedule.next_alpha(now=10.5, hz=50.0) == pytest.approx(0.5)
    assert schedule.next_alpha(now=11.01, hz=50.0) == pytest.approx(1.0)
    assert schedule.next_alpha(now=11.2, hz=50.0) == pytest.approx(1.0)
    assert schedule.active is True
    assert schedule.next_alpha(now=11.6, hz=50.0) is None
    assert schedule.active is False


def test_exit_ramp_starts_from_last_tracking_frame_not_raw_exit_frame() -> None:
    last_tracking_frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "upper_body_velocity": [0.0] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
            }
        ]
    )[0]
    raw_exit_frame = load_frames(
        [
            {
                "upper_body_position": [0.0] * 17,
                "upper_body_velocity": [0.0] * 17,
                "left_hand_joints": [0.0] * 7,
                "right_hand_joints": [0.0] * 7,
                "ramp_phase": "out",
                "ramp_alpha": 1.0,
            }
        ]
    )[0]

    schedule = start_stop_release_schedule(
        raw_exit_frame,
        last_tracking_frame=last_tracking_frame,
        now=20.0,
        release_s=1.0,
        final_hold_s=0.0,
    )
    messages = build_due_stop_release_messages(
        schedule,
        now=20.0,
        hz=50.0,
        stream_mode=5,
        hand_preset="tucked-thumb",
        upper_body_preset="straight",
    )

    assert messages is not None
    planner_message, _manager_state_message = messages
    decoded = unpack_bridge_message(planner_message, topic="planner")
    assert np.allclose(decoded["upper_body_position"], np.ones(17))


def test_exit_ramp_starts_from_last_published_filtered_frame() -> None:
    config = BridgeConfig(max_abs_joint=2.0, max_joint_step=0.2)
    filt = UpperBodyFilter(config)
    filt.seed(np.zeros(17, dtype=np.float32))
    raw_tracking_frame = load_frames(
        [
            {
                "upper_body_position": [1.0] * 17,
                "upper_body_velocity": [0.0] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
            }
        ]
    )[0]
    raw_exit_frame = load_frames(
        [
            {
                "upper_body_position": [0.0] * 17,
                "upper_body_velocity": [0.0] * 17,
                "left_hand_joints": [0.0] * 7,
                "right_hand_joints": [0.0] * 7,
                "ramp_phase": "out",
                "ramp_alpha": 1.0,
            }
        ]
    )[0]

    build_frame_planner_message(raw_tracking_frame, config, filt)
    last_published_frame = frame_with_current_upper_body_position(raw_tracking_frame, filt)
    schedule = start_stop_release_schedule(
        raw_exit_frame,
        last_tracking_frame=last_published_frame,
        now=20.0,
        release_s=1.0,
        final_hold_s=0.0,
    )
    messages = build_due_stop_release_messages(
        schedule,
        now=20.0,
        hz=50.0,
        stream_mode=5,
        hand_preset="tucked-thumb",
        upper_body_preset="straight",
    )

    assert messages is not None
    planner_message, _manager_state_message = messages
    decoded = unpack_bridge_message(planner_message, topic="planner")
    assert np.allclose(decoded["upper_body_position"], np.full(17, 0.2))


def test_build_manager_state_message_for_data_exporter() -> None:
    message = build_manager_state_message(stream_mode=5)
    header = _decode_header(message, topic="manager_state")
    fields = {field["name"]: field for field in header["fields"]}

    assert message.startswith(b"manager_state")
    assert fields["stream_mode"]["dtype"] == "i32"
    assert fields["toggle_data_collection"]["dtype"] == "bool"
    assert fields["toggle_data_abort"]["dtype"] == "bool"


def test_cli_dry_run_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gear_sonic.scripts.xr_upperbody_bridge import main

    path = tmp_path / "upperbody.jsonl"
    path.write_text(
        json.dumps({"upper_body_position": _valid_position()}) + "\n",
        encoding="utf-8",
    )

    rc = main(["--input", str(path), "--dry-run", "--once", "--hz", "50"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Loaded 1 frame" in captured.out
    assert "Dry run published 1 frame" in captured.out
