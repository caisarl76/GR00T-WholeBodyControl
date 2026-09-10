from dataclasses import replace
import math

import numpy as np
import pytest

from gear_sonic.utils.inference.planner_heading_frame import planner_command_in_reference_frame
from gear_sonic.utils.teleop.bridge_planner_publisher import PlannerCommand


def command(yaw):
    return PlannerCommand(0, (0, 0, 0), (math.cos(yaw), math.sin(yaw), 0), 0, -1, [0] * 17, [0] * 17)


def state(offset, active=1):
    return {
        "reference_heading_quat": [math.cos(offset / 2), 0, 0, math.sin(offset / 2)],
        "planner_reference_active": [active],
    }


@pytest.mark.parametrize("yaw", [0.8, -1.2, math.pi])
def test_reset_keeps_world_heading_when_reference_reanchors(yaw):
    desired = command(yaw)
    for offset in [0.0, yaw, yaw + 0.03]:
        out = planner_command_in_reference_frame(desired, state(offset))
        actual = math.atan2(out.facing[1], out.facing[0]) + offset
        assert math.cos(actual) == pytest.approx(math.cos(yaw))
        assert math.sin(actual) == pytest.approx(math.sin(yaw))
        np.testing.assert_allclose(out.movement, 0)
        assert out.upper_body_position == desired.upper_body_position


def test_movement_rotates_into_same_reference_as_facing():
    desired = replace(command(math.pi / 2), movement=(0, 1, 0), mode=1, speed=0.2)
    out = planner_command_in_reference_frame(desired, state(math.pi / 2))
    np.testing.assert_allclose(out.facing, (1, 0, 0), atol=1e-7)
    np.testing.assert_allclose(out.movement, (1, 0, 0), atol=1e-7)
    assert out.mode == 1 and out.speed == 0.2


def test_initializing_planner_uses_neutral_heading_and_no_movement():
    desired = replace(command(1.5), movement=(1, 0, 0), mode=1, speed=0.2)
    out = planner_command_in_reference_frame(desired, state(-0.8, active=0))
    assert out.mode == 0 and out.speed == 0
    assert out.facing == (1, 0, 0) and out.movement == (0, 0, 0)


def test_missing_frame_does_not_silently_treat_world_yaw_as_reference_yaw():
    with pytest.raises(ValueError, match="rebuild"):
        planner_command_in_reference_frame(command(1), {})
