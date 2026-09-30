import numpy as np
import pytest

from gear_sonic.scripts.pico_manager_thread_server import (
    YawAccumulator,
    compute_planner_movement_from_left_stick,
)


def test_right_stick_right_turns_facing_clockwise():
    yaw = YawAccumulator(yaw_gain=1.5, deadzone=0.15)

    # XRoboToolkit/PICO reports stick-right as negative X on the current setup.
    facing = yaw.update(rx=-1.0, dt=0.1)

    assert yaw.yaw_angle_change() < 0.0
    assert facing[1] < 0.0


def test_left_stick_forward_follows_current_facing():
    facing = [0.0, -1.0, 0.0]

    movement = compute_planner_movement_from_left_stick(lx=0.0, ly=1.0, mag=1.0, facing=facing)

    assert movement == pytest.approx([0.0, -1.0, 0.0])


def test_left_stick_right_strafes_right_relative_to_current_facing():
    facing = [0.0, -1.0, 0.0]

    # XRoboToolkit/PICO reports stick-right as negative X on the current setup.
    movement = compute_planner_movement_from_left_stick(lx=-1.0, ly=0.0, mag=1.0, facing=facing)

    assert movement == pytest.approx([-1.0, 0.0, 0.0])


def test_left_stick_deadzone_zeroes_movement():
    movement = compute_planner_movement_from_left_stick(
        lx=0.5,
        ly=0.5,
        mag=0.0,
        facing=[1.0, 0.0, 0.0],
    )

    assert np.asarray(movement) == pytest.approx([0.0, 0.0, 0.0])
