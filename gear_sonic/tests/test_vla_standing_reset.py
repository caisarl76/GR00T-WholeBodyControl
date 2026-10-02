import numpy as np
import pytest

from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_STANDING_UPPER_BODY,
    G1_UPPER_BODY_JOINT_INDICES,
)


def feedback(value=0.8):
    return {
        "body_q_measured": np.full(29, value),
        "left_hand_q_measured": np.full(7, value),
        "right_hand_q_measured": np.full(7, -value),
        "base_quat": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
    }


def test_explicit_absolute_motor_measurement_wins_over_legacy_joint_fields():
    state = feedback(-9.0)
    state["body_q"] = np.arange(29) * -2.0
    state["body_q_measured_motor"] = np.arange(29) * 0.01
    reset = StandingReset(state, np.zeros(7), np.zeros(7))
    np.testing.assert_allclose(
        reset.command.upper_body_position,
        state["body_q_measured_motor"][G1_UPPER_BODY_JOINT_INDICES],
    )


def test_standing_reset_starts_at_measured_pose_and_reaches_straight_preset():
    state = feedback()
    reset = StandingReset(state, np.zeros(7), np.zeros(7))
    np.testing.assert_allclose(reset.command.upper_body_position, 0.8)
    np.testing.assert_allclose(reset.command.facing, [0, 1, 0], atol=1e-7)
    for _ in range(300):
        old = np.asarray(reset.command.upper_body_position)
        command = reset.advance(state, 0.02)
        assert np.max(np.abs(np.asarray(command.upper_body_position) - old)) <= 0.010001
        state["body_q_measured"][G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        state["left_hand_q_measured"] = command.left_hand_position
        state["right_hand_q_measured"] = command.right_hand_position
    np.testing.assert_allclose(command.upper_body_position, G1_STANDING_UPPER_BODY)
    np.testing.assert_allclose(command.left_hand_position, 0)
    np.testing.assert_allclose(command.right_hand_position, 0)
    assert command.mode == 0 and command.speed == 0
    assert command.movement == (0, 0, 0)


def test_planner_initialization_delay_cannot_run_ramp_far_ahead_of_robot():
    state = feedback()
    reset = StandingReset(state, np.zeros(7), np.zeros(7))
    for _ in range(500):
        command = reset.advance(state, 0.02)
    assert np.max(np.abs(np.asarray(command.upper_body_position) - 0.8)) <= 0.150001
    before = command
    assert reset.advance(None, 10) == before


@pytest.mark.parametrize("field", ["body_q_measured", "left_hand_q_measured", "base_quat"])
def test_reset_rejects_missing_or_invalid_feedback(field):
    state = feedback()
    state[field][0] = np.nan
    with pytest.raises(ValueError):
        StandingReset(state, np.zeros(7), np.zeros(7))


def test_repeat_reset_captures_new_pose_and_preserves_closed_hand_target():
    first = StandingReset(feedback(), np.zeros(7), np.zeros(7))
    second = StandingReset(feedback(-0.5), np.full(7, 0.4), np.zeros(7))
    np.testing.assert_allclose(first.command.upper_body_position, 0.8)
    np.testing.assert_allclose(second.command.upper_body_position, -0.5)
    state = feedback(-0.5)
    for _ in range(300):
        command = second.advance(state, 0.02)
        state["body_q_measured"][G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        state["left_hand_q_measured"] = command.left_hand_position
        state["right_hand_q_measured"] = command.right_hand_position
    np.testing.assert_allclose(command.left_hand_position, 0.4)
