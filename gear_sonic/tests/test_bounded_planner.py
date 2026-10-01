import importlib

import numpy as np
import pytest
from test_harness_control import make, request, start, state


def feedback(yaw=0.0, index=1, planner=1):
    return {**state(index, planner), "base_quat": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]}


def planner(yaw=0.0):
    cls = importlib.import_module("gear_sonic.utils.inference.bounded_planner").BoundedPlanner
    return cls(feedback(yaw), 0.0)


def test_walk_stops_at_deadline_and_preserves_hands():
    p = planner()
    p.begin_walk("backward", 1.0, 0.2)
    command = p.advance(feedback(), 0.5)
    assert command.movement == (-1.0, 0.0, 0.0) and command.speed == 0.2
    assert command.left_hand_position == [0.3] * 7
    assert command.right_hand_position == [0.4] * 7
    stopped = p.advance(feedback(), 1.0)
    assert p.finished and stopped.movement == (0.0, 0.0, 0.0) and stopped.speed == 0.0


@pytest.mark.parametrize(
    "kind,args",
    [
        ("walk", ("forward", 5.01, 0.2)),
        ("walk", ("forward", 1.0, 0.201)),
        ("turn", (np.pi / 4 + 0.001, 0.1)),
        ("turn", (0.1, np.pi / 18 + 0.001)),
    ],
)
def test_hard_bounds(kind, args):
    p = planner()
    with pytest.raises(ValueError):
        getattr(p, "begin_" + kind)(*args)


def test_turn_wraps_pi_uses_measured_yaw_and_dwell():
    p = planner(np.deg2rad(179))
    p.begin_turn(np.deg2rad(2), np.deg2rad(10))
    p.advance(feedback(np.deg2rad(-179), 2), 0.2)
    assert not p.finished
    p.advance(feedback(np.deg2rad(-179), 3), 0.71)
    assert p.finished


def test_turn_rate_and_lead_are_bounded():
    p = planner()
    p.begin_turn(np.pi / 4, np.pi / 18)
    previous = 0.0
    for i in range(1, 101):
        command = p.advance(feedback(0.0, i + 1), i * 0.02)
        yaw = np.arctan2(command.facing[1], command.facing[0])
        assert 0 <= yaw - previous <= np.pi / 18 * 0.02 + 1e-8
        assert yaw <= np.pi / 36 + 1e-8
        previous = yaw
    with pytest.raises(ValueError, match="deadline"):
        p.advance(feedback(), 10.01)


def test_missing_feedback_cannot_advance_or_complete():
    p = planner()
    p.begin_walk("forward", 1.0, 0.2)
    with pytest.raises(ValueError):
        p.advance(None, 0.2)
    assert not p.finished


def enable_locomotion(c):
    c.locomotion_enabled = True
    request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})
    request(c, "reset_standing", {"execution_id": "", "open_hands": False}, 0.01)
    c.phase, c.hold_confirmed = "COMPLETED", True


def test_stale_feedback_and_lease_loss_clear_motion():
    c, h = make()
    enable_locomotion(c)
    assert (
        request(c, "walk_for", {"direction": "forward", "duration_s": 5.0, "speed_mps": 0.2}, 0.02)["error"]
        is None
    )
    c.tick(0.1, state(2), 0.1)
    assert h.command.speed == 0.2 and not c.hold_confirmed
    c.tick(0.7, state(2), 0.7)
    assert c.phase == "FAULT" and not c.hold_confirmed
    assert c.planner is None
    assert h.command.speed == 0.0 and h.command.movement == (0.0, 0.0, 0.0)
    c, h = make()
    enable_locomotion(c)
    request(c, "walk_for", {"direction": "forward", "duration_s": 5.0, "speed_mps": 0.2}, 0.02)
    c.tick(2.01, state(2), 2.01)
    assert c.phase == "INTERRUPTED" and c.owner is None
    assert h.holds[-1][1] is False and c.planner is None


def test_locomotion_cannot_overlap_manipulation():
    c, _ = make()
    c.locomotion_enabled = True
    start(c)
    response = request(c, "walk_for", {"direction": "forward", "duration_s": 1.0, "speed_mps": 0.2}, 0.01)
    assert response["error"]["code"] == "NOT_READY"
    assert c.phase == "MANIPULATING"


def test_lost_planner_activation_interrupts_walking():
    c, _ = make()
    enable_locomotion(c)
    request(c, "walk_for", {"direction": "forward", "duration_s": 1.0, "speed_mps": 0.2}, 0.02)
    c.tick(0.1, state(2, 0), 0.1)
    assert c.phase == "INTERRUPTED" and c.reason == "planner_inactive"
