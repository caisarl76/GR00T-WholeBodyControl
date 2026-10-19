import struct
from types import SimpleNamespace

import numpy as np
import pytest

import gear_sonic.scripts.pico_manager_thread_server as pico
from gear_sonic.scripts.pico_manager_thread_server import (
    LocomotionMode,
    PlannerStreamer,
    StreamMode,
    YawAccumulator,
    compute_planner_movement_from_left_stick,
)


@pytest.mark.parametrize("rx, direction", [(1.0, -1), (-1.0, 1)])
def test_right_stick_turns_in_requested_direction(rx, direction):
    yaw = YawAccumulator(yaw_gain=1.5, deadzone=0.15)

    facing = yaw.update(rx=rx, dt=0.1)

    assert direction * yaw.yaw_angle_change() > 0.0
    assert direction * facing[1] > 0.0


def test_left_stick_forward_follows_current_facing():
    facing = [0.0, -1.0, 0.0]

    movement = compute_planner_movement_from_left_stick(lx=0.0, ly=1.0, mag=1.0, facing=facing)

    assert movement == pytest.approx([0.0, -1.0, 0.0])


def test_left_stick_right_strafes_right_relative_to_current_facing():
    facing = [0.0, -1.0, 0.0]

    movement = compute_planner_movement_from_left_stick(lx=1.0, ly=0.0, mag=1.0, facing=facing)

    assert movement == pytest.approx([-1.0, 0.0, 0.0])


def test_left_stick_left_strafes_left_relative_to_current_facing():
    movement = compute_planner_movement_from_left_stick(lx=-1.0, ly=0.0, mag=1.0, facing=[0.0, -1.0, 0.0])

    assert movement == pytest.approx([1.0, 0.0, 0.0])


def test_left_stick_deadzone_zeroes_movement():
    movement = compute_planner_movement_from_left_stick(
        lx=0.5,
        ly=0.5,
        mag=0.0,
        facing=[1.0, 0.0, 0.0],
    )

    assert np.asarray(movement) == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "axes, expected_mode, expected_speed",
    [
        ((0.0, 1.0, 0.0, 0.0), LocomotionMode.SLOW_WALK, 0.4),
        ((0.0, 0.5, 0.0, 0.0), LocomotionMode.SLOW_WALK, 0.2235294118),
        ((0.05, 0.05, 0.0, 0.0), LocomotionMode.IDLE, -1.0),
    ],
)
def test_planner_publishes_slow_walk_speed(monkeypatch, axes, expected_mode, expected_speed):
    sent = []
    monkeypatch.setattr(pico, "get_controller_axes", lambda: axes)
    monkeypatch.setattr(pico, "get_abxy_buttons", lambda: (False, False, False, False))
    monkeypatch.setattr(pico, "xrt", SimpleNamespace(get_time_stamp_ns=lambda: 1))
    streamer = PlannerStreamer.__new__(PlannerStreamer)
    streamer.controller_3pt = False
    streamer.reader = None
    streamer.socket = SimpleNamespace(send=sent.append)
    streamer.dt = 0.0
    streamer.last_send = 0.0
    streamer.last_xrt_timestamp = None
    streamer.prev_ab = streamer.prev_xy = False
    streamer.mode = LocomotionMode.SLOW_WALK
    streamer.yaw_accumulator = YawAccumulator()
    streamer.freeze_vr3pt_target_once = False
    streamer._last_vr3pt_pose = None

    streamer.run_once(StreamMode.PLANNER)

    assert len(sent) == 1 and sent[0].startswith(b"planner")
    # Base planner wire payload: mode, movement[3], facing[3], speed, height.
    mode, *values = struct.unpack("<i8f", sent[0][-36:])
    assert mode == expected_mode
    assert values[6] == pytest.approx(expected_speed)
