import struct

import numpy as np
import pytest

import gear_sonic.utils.teleop.bridge_planner_publisher as publisher_module
from gear_sonic.utils.teleop.bridge_planner_publisher import (
    MAX_HAND_JOINT_STEP_RAD,
    BridgePlannerPublisher,
    HandCommandLimiter,
    PlannerCommand,
    PlannerPreparationError,
)
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message


class RecordingSocket:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.sent = []

    def send(self, message: bytes):
        if self.fail:
            raise RuntimeError("send failed")
        self.sent.append(message)


class RejectingSocket:
    def send(self, message: bytes):
        return False


def command(left=None, right=None) -> PlannerCommand:
    return PlannerCommand(
        mode=0,
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=0.0,
        height=-1.0,
        upper_body_position=(0.0,) * 17,
        upper_body_velocity=(0.0,) * 17,
        left_hand_position=left,
        right_hand_position=right,
    )


def test_preview_limits_each_included_hand_from_command_zero() -> None:
    assert MAX_HAND_JOINT_STEP_RAD == 0.25
    limiter = HandCommandLimiter()
    preview = limiter.preview([1.0] * 7, [-1.0] * 7)
    np.testing.assert_allclose(preview.left, [0.25] * 7)
    np.testing.assert_allclose(preview.right, [-0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.0] * 7)


def test_commit_updates_both_included_hands_together() -> None:
    limiter = HandCommandLimiter()
    preview = limiter.preview([1.0] * 7, [-1.0] * 7)
    limiter.commit(preview)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [-0.25] * 7)


def test_omitted_hand_is_not_committed() -> None:
    limiter = HandCommandLimiter()
    limiter.commit(limiter.preview([1.0] * 7, None))
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [0.0] * 7)


@pytest.mark.parametrize("bad", [[1.0] * 6, [1.0] * 6 + [float("nan")], "bad"])
def test_invalid_included_hand_rejects_preview_without_state_change(bad) -> None:
    limiter = HandCommandLimiter()
    with pytest.raises(ValueError):
        limiter.preview([1.0] * 7, bad)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [0.0] * 7)


def test_pause_reemits_last_command_without_advancing() -> None:
    limiter = HandCommandLimiter()
    limiter.commit(limiter.preview([1.0] * 7, None))
    paused = limiter.preview([1.0] * 7, None, pause=True)
    np.testing.assert_allclose(paused.left, [0.25] * 7)
    limiter.commit(paused)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    resumed = limiter.preview([1.0] * 7, None)
    np.testing.assert_allclose(resumed.left, [0.5] * 7)


def test_successful_planner_send_commits_limited_state() -> None:
    socket = RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    publisher.publish(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    assert len(socket.sent) == 1
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [-0.25] * 7)


def test_successful_planner_send_encodes_limited_hand_values() -> None:
    socket = RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    publisher.publish(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    decoded = unpack_bridge_message(socket.sent[0], topic="planner")
    np.testing.assert_allclose(decoded["left_hand_joints"], [0.25] * 7)
    np.testing.assert_allclose(decoded["right_hand_joints"], [-0.25] * 7)


def test_failed_planner_send_commits_neither_hand() -> None:
    publisher = BridgePlannerPublisher(RecordingSocket(fail=True))
    with pytest.raises(RuntimeError, match="send failed"):
        publisher.publish(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)


def test_encoding_failure_is_preparation_error_without_commit(monkeypatch) -> None:
    def fail_encode(*args, **kwargs):
        raise struct.error("float out of range")

    monkeypatch.setattr(publisher_module, "build_planner_message", fail_encode)
    socket = RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    with pytest.raises(PlannerPreparationError, match="invalid planner command"):
        publisher.prepare(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    assert socket.sent == []
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)


def test_send_transport_failure_propagates_without_commit() -> None:
    publisher = BridgePlannerPublisher(RecordingSocket(fail=True))
    prepared = publisher.prepare(
        command([1.0] * 7, [-1.0] * 7),
        ramp_phase="track",
    )
    with pytest.raises(RuntimeError, match="send failed"):
        publisher.send(prepared)
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)


def test_prepared_commit_snapshot_is_immutable() -> None:
    publisher = BridgePlannerPublisher(RecordingSocket())
    prepared = publisher.prepare(
        command([1.0] * 7, [-1.0] * 7),
        ramp_phase="track",
    )
    with pytest.raises(TypeError):
        prepared.hand_preview.left[0] = 99.0
    publisher.send(prepared)
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [-0.25] * 7)


def test_socket_literal_false_rejects_without_commit() -> None:
    publisher = BridgePlannerPublisher(RejectingSocket())
    prepared = publisher.prepare(
        command([1.0] * 7, [-1.0] * 7),
        ramp_phase="track",
    )
    with pytest.raises(RuntimeError, match="planner socket rejected message"):
        publisher.send(prepared)
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)
