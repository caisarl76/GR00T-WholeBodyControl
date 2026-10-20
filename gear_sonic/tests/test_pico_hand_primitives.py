"""Regression checks for the calibrated PICO Grip/Squeeze pinch mapping."""

import numpy as np
import pytest

from gear_sonic.scripts.pico_manager_thread_server import compute_hand_joints_from_inputs
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)

# Measured right Dex3 poses from the original calibration (commit 12e6468).
RIGHT_OPEN = np.array(
    [0.4025313854, -0.1496349275, -0.0161621254, 0.6290951371, -0.0335988589, -0.0211195406, -0.0305269454]
)
RIGHT_CLOSED = np.array(
    [0.4024966359, -0.7084636688, -0.0161451288, 1.4996159077, -0.0339727886, -0.0211526398, -0.0305269454]
) + np.array([0, -0.03, 0, 0.03, 0, 0, 0])  # extra closure for the ~5 mm hardware gap
MIRROR = np.array([1, -1, -1, -1, -1, -1, -1])


@pytest.fixture
def solvers():
    return tuple(G1GripperInverseKinematicsSolver(side) for side in ("left", "right"))


@pytest.mark.parametrize(
    "grip, expected",
    [
        (0.0, np.zeros(7)),
        (0.1, RIGHT_OPEN / 2),
        (0.2, RIGHT_OPEN),
        (0.6, (RIGHT_OPEN + RIGHT_CLOSED) / 2),
        (1.0, RIGHT_CLOSED),
        (-1.0, np.zeros(7)),
        (2.0, RIGHT_CLOSED),
        (np.nan, np.zeros(7)),
        (np.inf, np.zeros(7)),
        (-np.inf, np.zeros(7)),
    ],
)
def test_grip_drives_calibrated_pinch_on_each_hand(solvers, grip, expected):
    left, right = compute_hand_joints_from_inputs(*solvers, 0.0, grip, 0.0, grip)
    np.testing.assert_allclose(right, expected, atol=1e-9)
    np.testing.assert_allclose(left, expected * MIRROR, atol=1e-9)


LEFT_GRASP_CLOSED = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5])


@pytest.mark.parametrize(
    "trigger, expected_left",
    [
        (0.0, RIGHT_CLOSED * MIRROR),
        (0.05, RIGHT_CLOSED * MIRROR),
        (0.25, 0.25 * LEFT_GRASP_CLOSED),
        (0.5, 0.5 * LEFT_GRASP_CLOSED),
        (1.0, LEFT_GRASP_CLOSED),
        (2.0, LEFT_GRASP_CLOSED),
        (np.nan, RIGHT_CLOSED * MIRROR),
    ],
)
def test_analog_trigger_grasp_priority_is_independent_per_hand(solvers, trigger, expected_left):
    left, right = compute_hand_joints_from_inputs(*solvers, trigger, 1.0, 0.0, 0.2)
    np.testing.assert_allclose(left, expected_left, atol=1e-9)
    np.testing.assert_allclose(right, RIGHT_OPEN, atol=1e-9)
    left, right = compute_hand_joints_from_inputs(*solvers, 0.0, 0.2, trigger, 1.0)
    np.testing.assert_allclose(left, RIGHT_OPEN * MIRROR, atol=1e-9)
    expected_right = RIGHT_CLOSED if np.allclose(expected_left, RIGHT_CLOSED * MIRROR) else -expected_left
    np.testing.assert_allclose(right, expected_right, atol=1e-9)


@pytest.mark.parametrize("missing_side", [0, 1])
def test_missing_solver_preserves_zero_fallback(solvers, missing_side):
    available = list(solvers)
    available[missing_side] = None
    left, right = compute_hand_joints_from_inputs(*available, 1.0, 1.0, 1.0, 1.0)
    np.testing.assert_array_equal(left, np.zeros((1, 7)))
    np.testing.assert_array_equal(right, np.zeros((1, 7)))
