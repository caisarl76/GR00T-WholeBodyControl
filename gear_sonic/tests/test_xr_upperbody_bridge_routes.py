import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.teleop.bridge_planner_publisher import BridgePlannerPublisher
import gear_sonic.utils.teleop.xr_upperbody_bridge as bridge
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    _send_loop,
    _send_zmq_json_loop,
    load_frames,
    unpack_bridge_message,
)


class FakeAgain(Exception):
    pass


class FakeTransportError(RuntimeError):
    pass


class EndOfScript(RuntimeError):
    pass


def bridge_args(**overrides) -> argparse.Namespace:
    values = dict(
        bind_host="127.0.0.1",
        port=5556,
        source_host="127.0.0.1",
        source_port=5560,
        source_topic="xr_teleop",
        source_timeout_s=0.001,
        feedback_host="127.0.0.1",
        feedback_port=5557,
        feedback_topic="g1_debug",
        feedback_prime_timeout_s=0.001,
        no_feedback_prime=True,
        allow_unseeded_start_control=True,
        start_control=False,
        start_command_repeat_s=0.0,
        start_command_interval_s=0.2,
        pub_warmup_s=0.0,
        hz=50.0,
        max_abs_joint=3.14,
        max_joint_step=0.03,
        stream_mode=5,
        stop_release_s=1.0,
        stop_final_hold_s=1.0,
        stop_hand_preset="tucked-thumb",
        stop_upper_body_preset="straight",
        send_stop_on_exit=False,
        anchor_planner_heading=False,
        once=True,
        debug_live=False,
        debug_interval=0.5,
        dry_run=False,
        loop=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class ScriptedZmq:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        recv_script=(),
        *,
        fail_first_planner_send: bool = False,
        clock_step_s: float = 0.1,
    ) -> None:
        self.recv_script = list(recv_script)
        self.recv_step = 0
        self.clock_step_s = clock_step_s
        self.sent: list[tuple[int, bytes]] = []
        self.fail_first_planner_send = fail_first_planner_send
        self._planner_failed = False
        self.sub_socket = _ScriptedSocket(self, "sub")
        self.pub_socket = _ScriptedSocket(self, "pub")
        self.context = _ScriptedContext(self)
        fake_module = SimpleNamespace(
            PUB=1,
            SUB=2,
            RCVTIMEO=3,
            SUBSCRIBE=4,
            Again=FakeAgain,
            Context=SimpleNamespace(instance=lambda: self.context),
        )
        monkeypatch.setitem(sys.modules, "zmq", fake_module)
        monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            bridge.time,
            "monotonic",
            lambda: self.recv_step * self.clock_step_s,
        )

    @property
    def planner_sends(self) -> list[tuple[int, bytes]]:
        return [(step, raw) for step, raw in self.sent if raw.startswith(b"planner")]


class _ScriptedContext:
    def __init__(self, owner: ScriptedZmq) -> None:
        self.owner = owner

    def socket(self, kind: int):
        if kind == 2:
            return self.owner.sub_socket
        if kind == 1:
            return self.owner.pub_socket
        raise AssertionError(f"unsupported socket kind: {kind}")

    def term(self) -> None:
        return None


class _ScriptedSocket:
    def __init__(self, owner: ScriptedZmq, role: str) -> None:
        self.owner = owner
        self.role = role

    def bind(self, _endpoint: str) -> None:
        return None

    def connect(self, _endpoint: str) -> None:
        return None

    def setsockopt(self, _option: int, _value) -> None:
        return None

    def setsockopt_string(self, _option: int, _value: str) -> None:
        return None

    def close(self, linger: int = 0) -> None:
        return None

    def recv(self) -> bytes:
        if not self.owner.recv_script:
            raise EndOfScript("receive script exhausted")
        self.owner.recv_step += 1
        item = self.owner.recv_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def send(self, message: bytes) -> None:
        assert self.role == "pub"
        if (
            self.owner.fail_first_planner_send
            and message.startswith(b"planner")
            and not self.owner._planner_failed
        ):
            self.owner._planner_failed = True
            raise FakeTransportError("planner transport failed")
        self.owner.sent.append((self.owner.recv_step, message))


def _source_message(payload) -> bytes:
    return b"xr_teleop" + json.dumps(payload).encode("utf-8")


def _planner_payloads(scripted: ScriptedZmq) -> list[dict[str, np.ndarray]]:
    return [unpack_bridge_message(raw, topic="planner") for _step, raw in scripted.planner_sends]


def test_send_loop_replay_routes_consecutive_frames_through_one_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedZmq(monkeypatch)
    frames = load_frames(
        [
            {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14},
            {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [-1.0] * 14},
        ]
    )

    assert _send_loop(bridge_args(once=False, loop=False), frames) == 0

    first, second = _planner_payloads(scripted)
    np.testing.assert_allclose(first["left_hand_joints"], [0.25] * 7)
    np.testing.assert_allclose(second["left_hand_joints"], [0.0] * 7)


def test_zmq_live_loop_routes_valid_frame_through_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedZmq(
        monkeypatch,
        [_source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14})],
    )

    assert _send_zmq_json_loop(bridge_args(once=True)) == 0

    (planner,) = _planner_payloads(scripted)
    np.testing.assert_allclose(planner["left_hand_joints"], [0.25] * 7)
    np.testing.assert_allclose(planner["right_hand_joints"], [0.25] * 7)


def test_zmq_stop_release_routes_immediate_timeout_and_final_hold_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[str] = []
    real_publisher = bridge.BridgePlannerPublisher

    class PhaseRecordingPublisher(real_publisher):
        def prepare(self, command, *, ramp_phase):
            phases.append(ramp_phase)
            return super().prepare(command, ramp_phase=ramp_phase)

    monkeypatch.setattr(bridge, "BridgePlannerPublisher", PhaseRecordingPublisher)
    tracking = _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14})
    exiting = _source_message(
        {
            "dual_arm_position": [0.0] * 14,
            "dual_hand_joints": [0.0] * 14,
            "ramp_phase": "out",
        }
    )
    scripted = ScriptedZmq(
        monkeypatch,
        [tracking, exiting, FakeAgain(), FakeAgain(), FakeAgain(), FakeAgain(), FakeAgain(), EndOfScript()],
        clock_step_s=0.3,
    )

    with pytest.raises(EndOfScript):
        _send_zmq_json_loop(bridge_args(once=False))

    planner_sends = scripted.planner_sends
    assert any(step == 2 for step, _raw in planner_sends)
    assert any(step > 2 for step, _raw in planner_sends)
    assert "stop-release" in phases
    assert "final-hold" in phases
    decoded = _planner_payloads(scripted)
    for previous, current in zip(decoded, decoded[1:]):
        for side in ("left_hand_joints", "right_hand_joints"):
            assert np.max(np.abs(current[side] - previous[side])) <= 0.25 + 1e-7


def test_zmq_invalid_frames_continue_and_warning_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overflow_mode_payload = (
        b'xr_teleop{"dual_arm_position":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"mode":1e309}'
    )
    scripted = ScriptedZmq(
        monkeypatch,
        [
            b"xr_teleop[]",
            _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [0.0] * 13}),
            overflow_mode_payload,
            _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}),
        ],
        clock_step_s=0.1,
    )

    assert _send_zmq_json_loop(bridge_args(once=True)) == 0

    output = capsys.readouterr().out
    assert output.count("invalid live frame skipped") == 1
    assert len(scripted.planner_sends) == 1


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"speed": 1e309},
        {"height": -1e309},
        {"ramp_phase": "start", "ramp_alpha": float("nan")},
        {"timestamp": float("inf")},
    ],
)
def test_zmq_nonfinite_scalar_frame_is_skipped_before_next_valid_frame(
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: dict,
) -> None:
    invalid = {
        "dual_arm_position": [0.0] * 14,
        "dual_hand_joints": [1.0] * 14,
        **invalid_field,
    }
    scripted = ScriptedZmq(
        monkeypatch,
        [
            _source_message(invalid),
            _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}),
        ],
    )

    assert _send_zmq_json_loop(bridge_args(once=True)) == 0

    assert scripted.recv_step == 2
    (planner,) = _planner_payloads(scripted)
    assert float(planner["speed"][0]) == -1.0
    np.testing.assert_allclose(planner["upper_body_position"], [0.0] * 17)


def test_zmq_rejected_planner_preparation_rolls_back_upper_body_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedZmq(
        monkeypatch,
        [
            _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [0.0] * 14}),
            _source_message(
                {
                    "dual_arm_position": [1.0] * 14,
                    "dual_hand_joints": [1.0] * 14,
                    "speed": 1e308,
                }
            ),
            _source_message({"dual_arm_position": [1.0] * 14, "dual_hand_joints": [1.0] * 14}),
            EndOfScript(),
        ],
    )

    with pytest.raises(EndOfScript):
        _send_zmq_json_loop(bridge_args(once=False))

    first, second = _planner_payloads(scripted)
    np.testing.assert_allclose(first["upper_body_position"], [0.0] * 17)
    delta = second["upper_body_position"] - first["upper_body_position"]
    assert np.max(np.abs(delta)) == pytest.approx(0.03, abs=1e-7)


def test_zmq_encoding_failure_does_not_commit_before_next_valid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedZmq(
        monkeypatch,
        [
            _source_message(
                {
                    "dual_arm_position": [0.0] * 14,
                    "dual_hand_joints": [1.0] * 14,
                    "speed": 1e308,
                }
            ),
            _source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}),
        ],
    )

    assert _send_zmq_json_loop(bridge_args(once=True)) == 0

    (planner,) = _planner_payloads(scripted)
    np.testing.assert_allclose(planner["left_hand_joints"], [0.25] * 7)
    np.testing.assert_allclose(planner["right_hand_joints"], [0.25] * 7)


def test_zmq_transport_failure_propagates_and_does_not_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[BridgePlannerPublisher] = []
    real_publisher = bridge.BridgePlannerPublisher

    class RecordingPublisher(real_publisher):
        def __init__(self, socket) -> None:
            super().__init__(socket)
            constructed.append(self)

    monkeypatch.setattr(bridge, "BridgePlannerPublisher", RecordingPublisher)
    ScriptedZmq(
        monkeypatch,
        [_source_message({"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14})],
        fail_first_planner_send=True,
    )

    with pytest.raises(FakeTransportError, match="planner transport failed"):
        _send_zmq_json_loop(bridge_args(once=True))

    assert len(constructed) == 1
    np.testing.assert_allclose(constructed[0].last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(constructed[0].last_emitted("right"), [0.0] * 7)
