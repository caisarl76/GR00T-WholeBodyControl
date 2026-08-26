import json

import numpy as np
import pytest
import zmq

from gear_sonic.utils.mujoco_sim.inspire_ftp_hand import (
    InspireCommandState,
    InspireFtpZmqSubscriber,
)
from gear_sonic.utils.teleop.inspire_ftp import OPEN
from gear_sonic.utils.teleop.zmq.zmq_message_decoder import unpack_pose_message
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_planner_message,
    pack_pose_message,
)

LEFT = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
RIGHT = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)


def _pose_message(left=LEFT, right=RIGHT):
    return pack_pose_message(
        {
            "left_hand_joints": np.asarray(left, dtype=np.float32),
            "right_hand_joints": np.asarray(right, dtype=np.float32),
        }
    )


def _planner_message(left=LEFT, right=RIGHT):
    return build_planner_message(
        mode=0,
        movement=[0.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        left_hand_position=left,
        right_hand_position=right,
    )


def _wire(topic, fields, payload):
    header = json.dumps(
        {"v": 1, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode()
    return topic.encode() + header.ljust(HEADER_SIZE, b"\x00") + payload


@pytest.mark.parametrize(("topic", "message"), (("pose", _pose_message()), ("planner", _planner_message())))
def test_decoder_reads_six_motor_hand_fields_from_both_topics(topic, message):
    decoded = unpack_pose_message(message, topic=topic)

    assert decoded["left_hand_joints"].shape == (6,)
    assert decoded["right_hand_joints"].shape == (6,)
    np.testing.assert_allclose(decoded["left_hand_joints"], LEFT)
    np.testing.assert_allclose(decoded["right_hand_joints"], RIGHT)


@pytest.mark.parametrize(
    "message",
    (
        b"pose" + b"{}",
        _wire("pose", [{"name": "x", "dtype": "unknown", "shape": [1]}], b"\x00"),
        _wire("pose", [{"name": "x", "dtype": "f32", "shape": [-1]}], b""),
        _wire(
            "pose",
            [
                {"name": "x", "dtype": "f32", "shape": [1]},
                {"name": "x", "dtype": "f32", "shape": [1]},
            ],
            np.zeros(2, dtype=np.float32).tobytes(),
        ),
        _wire(
            "pose",
            [{"name": "x", "dtype": "f32", "shape": [2]}],
            np.zeros(1, dtype=np.float32).tobytes(),
        ),
        _wire(
            "pose",
            [{"name": "x", "dtype": "f32", "shape": [1]}],
            np.zeros(2, dtype=np.float32).tobytes(),
        ),
    ),
)
def test_decoder_rejects_malformed_header_or_payload(message):
    with pytest.raises(ValueError):
        unpack_pose_message(message, topic="pose")


def test_command_state_starts_open_and_follows_fresh_command():
    state = InspireCommandState(stale_after_s=0.25)

    startup_left, startup_right = state.advance(now=10.0, dt=0.01)
    np.testing.assert_array_equal(startup_left, OPEN)
    np.testing.assert_array_equal(startup_right, OPEN)

    state.accept(LEFT, RIGHT, now=10.0)
    fresh_left, fresh_right = state.advance(now=10.249, dt=0.01)
    np.testing.assert_allclose(fresh_left, LEFT)
    np.testing.assert_allclose(fresh_right, RIGHT)


def test_command_state_rejects_hand_pair_atomically():
    state = InspireCommandState()
    state.accept(LEFT, RIGHT, now=1.0)

    with pytest.raises(ValueError):
        state.accept(np.zeros(6), np.zeros(7), now=2.0)

    left, right = state.advance(now=1.1, dt=0.01)
    np.testing.assert_allclose(left, LEFT)
    np.testing.assert_allclose(right, RIGHT)
    assert state.last_receive_monotonic == 1.0


def test_stale_command_monotonically_ramps_toward_open_at_configured_speed():
    speed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    state = InspireCommandState(stale_after_s=0.25, max_open_speed=speed)
    state.accept(np.zeros(6), np.zeros(6), now=0.0)
    state.advance(now=0.1, dt=0.1)

    first_left, first_right = state.advance(now=0.251, dt=0.5)
    second_left, second_right = state.advance(now=0.751, dt=0.5)

    np.testing.assert_allclose(first_left, speed * 0.5)
    np.testing.assert_allclose(first_right, speed * 0.5)
    np.testing.assert_allclose(second_left, speed)
    np.testing.assert_allclose(second_right, speed)
    assert np.all(second_left >= first_left)
    assert np.all(second_left <= OPEN)


class _FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self, flags):
        assert flags == zmq.NOBLOCK
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


def test_subscriber_drains_to_latest_valid_complete_hand_pair():
    newest_left = np.full(6, 0.75, dtype=np.float32)
    newest_right = np.full(6, 0.25, dtype=np.float32)
    socket = _FakeSocket(
        [
            _pose_message(np.zeros(6), np.zeros(6)),
            _pose_message(np.zeros(7), np.zeros(6)),
            _planner_message(newest_left, newest_right),
        ]
    )
    subscriber = InspireFtpZmqSubscriber.from_socket(socket)

    assert subscriber.poll(now=5.0)
    left, right = subscriber.state.advance(now=5.1, dt=0.01)
    np.testing.assert_allclose(left, newest_left)
    np.testing.assert_allclose(right, newest_right)
    assert socket.messages == []


def test_subscriber_ignores_message_missing_one_hand():
    socket = _FakeSocket([pack_pose_message({"left_hand_joints": LEFT}), _pose_message(LEFT, RIGHT)])
    subscriber = InspireFtpZmqSubscriber.from_socket(socket)

    assert subscriber.poll(now=7.0)
    left, right = subscriber.state.advance(now=7.1, dt=0.01)
    np.testing.assert_allclose(left, LEFT)
    np.testing.assert_allclose(right, RIGHT)
