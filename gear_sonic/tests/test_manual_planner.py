import math

import pytest

from gear_sonic.utils.inference.manual_planner import ManualPlanner
from gear_sonic.utils.teleop.bridge_planner_publisher import PlannerCommand


def base():
    return PlannerCommand(0, (0, 0, 0), (1, 0, 0), 0, -1, (1,), (2,), (3,), (4,))


@pytest.mark.parametrize(
    "key,direction", [("w", (1, 0, 0)), ("s", (-1, 0, 0)), ("a", (0, 1, 0)), ("d", (0, -1, 0))]
)
def test_translation_stays_on_until_same_key_stops_it(key, direction):
    planner = ManualPlanner(0)
    planner.handle_key(key, 0)
    for _ in range(1000):
        command = planner.command(base(), 0)
        assert command.movement == direction and command.mode == 1 and command.speed == 0.2
    planner.handle_key(key, 0)
    assert planner.command(base(), 0).mode == 0
    assert planner.command(base(), 0).movement == (0, 0, 0)


def test_switch_direction_rotates_from_current_heading_and_stop_holds():
    planner = ManualPlanner(0)
    planner.handle_key("w", 0)
    planner.handle_key("s", math.pi / 2)
    command = planner.command(base(), math.pi / 2)
    assert command.movement == pytest.approx((0, -1, 0))
    assert command.upper_body_position == (1,) and command.left_hand_position == (3,)
    planner.handle_key("z", 0.5)
    stopped = planner.command(base(), 0.8)
    assert stopped.mode == 0 and stopped.speed == 0
    assert stopped.facing == pytest.approx((math.cos(0.5), math.sin(0.5), 0))


@pytest.mark.parametrize("key,sign", [("q", 1), ("e", -1)])
def test_turn_continues_from_measured_heading_until_toggled_off(key, sign):
    planner = ManualPlanner(0)
    planner.handle_key(key, 0)
    for heading in (0, 0.1, 0.2):
        command = planner.command(base(), heading)
        target = heading + sign * math.radians(5)
        assert command.mode == 0 and command.movement == (0, 0, 0)
        assert command.facing == pytest.approx((math.cos(target), math.sin(target), 0))
    planner.handle_key(key, 0.2)
    assert planner.command(base(), 0.3).facing == pytest.approx((math.cos(0.2), math.sin(0.2), 0))


@pytest.mark.parametrize("key", ["w", "q"])
def test_feedback_loss_cancels_and_does_not_resume(key):
    planner = ManualPlanner(0)
    planner.handle_key(key, 0)
    planner.command(base(), 0.3)
    stopped = planner.command(base(), None)
    assert stopped.mode == 0 and stopped.movement == (0, 0, 0)
    assert stopped.facing == pytest.approx((math.cos(0.3), math.sin(0.3), 0))
    assert planner.command(base(), 0.4).mode == 0
    assert planner.active_key is None


def test_reserved_keys_do_not_start_movement_and_invalid_heading_rejected():
    planner = ManualPlanner(0)
    for key in ["c", "x", "f", "b", "ww"]:
        planner.handle_key(key, 0)
        assert planner.command(base(), 0).mode == 0
    with pytest.raises(ValueError):
        ManualPlanner(float("nan"))
