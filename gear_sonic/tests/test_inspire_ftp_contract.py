import numpy as np
import pytest

from gear_sonic.scripts.pico_manager_thread_server import (
    PlannerStreamer,
    PoseStreamer,
    compute_hand_joints_from_inputs,
)
from gear_sonic.utils.teleop.inspire_ftp import (
    CLOSED_RADIANS,
    MOTOR_NAMES,
    map_pico_controls,
    normalized_to_ftp_counts,
    normalized_to_radians,
    radians_to_normalized,
    validate_normalized,
)


def test_motor_order_matches_ftp_driver_contract():
    assert MOTOR_NAMES == (
        "pinky",
        "ring",
        "middle",
        "index",
        "thumb_bend",
        "thumb_rotation",
    )


def test_normalized_endpoints_follow_ftp_convention():
    np.testing.assert_allclose(normalized_to_radians(np.ones(6)), np.zeros(6))
    np.testing.assert_allclose(normalized_to_radians(np.zeros(6)), CLOSED_RADIANS)
    np.testing.assert_allclose(
        CLOSED_RADIANS,
        [1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641],
    )


def test_normalized_radian_round_trip_at_midpoints():
    normalized = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(radians_to_normalized(normalized_to_radians(normalized)), normalized, atol=1e-12)


def test_normalized_values_scale_to_ftp_counts():
    np.testing.assert_array_equal(
        normalized_to_ftp_counts([0.0, 0.25, 0.5, 0.75, 1.0, 0.5004]),
        [0, 250, 500, 750, 1000, 500],
    )


def test_pico_trigger_closes_five_bend_motors_and_grip_rotates_thumb():
    np.testing.assert_allclose(map_pico_controls(1.0, 0.0, deadzone=0.0), [0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(map_pico_controls(0.0, 1.0, deadzone=0.0), [1, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(map_pico_controls(1.0, 1.0, deadzone=0.0), np.zeros(6))


def test_pico_deadzone_is_continuous_and_reaches_endpoints():
    np.testing.assert_allclose(map_pico_controls(0.05, 0.05), np.ones(6))
    np.testing.assert_allclose(map_pico_controls(0.95, 0.95), np.zeros(6))
    np.testing.assert_allclose(map_pico_controls(0.5, 0.5), np.full(6, 0.5))


@pytest.mark.parametrize(
    "values",
    (
        [0.0] * 5,
        [0.0] * 7,
        [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
        [0.0, 0.0, 0.0, 0.0, 0.0, np.inf],
        [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.01],
    ),
)
def test_validate_normalized_rejects_invalid_commands(values):
    with pytest.raises(ValueError):
        validate_normalized(values)


@pytest.mark.parametrize(
    ("trigger", "grip"),
    ((np.nan, 0.0), (0.0, np.inf), (-0.1, 0.0), (0.0, 1.1)),
)
def test_pico_mapping_rejects_invalid_controls(trigger, grip):
    with pytest.raises(ValueError):
        map_pico_controls(trigger, grip)


@pytest.mark.parametrize("deadzone", (-0.01, 0.5, 1.0))
def test_pico_mapping_rejects_invalid_deadzone(deadzone):
    with pytest.raises(ValueError):
        map_pico_controls(0.0, 0.0, deadzone=deadzone)


def test_pico_manager_inspire_profile_returns_six_motors_per_hand():
    left, right = compute_hand_joints_from_inputs(
        None,
        None,
        left_trigger=1.0,
        left_grip=0.0,
        right_trigger=0.0,
        right_grip=1.0,
        hand_profile="inspire_ftp",
    )

    assert left.shape == (1, 6)
    assert right.shape == (1, 6)
    np.testing.assert_allclose(left[0], [0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(right[0], [1, 1, 1, 1, 1, 0])


def test_pico_manager_inspire_profile_bypasses_dex3_solvers():
    def fail_if_called(_):
        raise AssertionError("Dex3 solver must not own Inspire commands")

    left, right = compute_hand_joints_from_inputs(
        fail_if_called,
        fail_if_called,
        0.0,
        0.0,
        0.0,
        0.0,
        hand_profile="inspire_ftp",
    )

    np.testing.assert_allclose(left, np.ones((1, 6)))
    np.testing.assert_allclose(right, np.ones((1, 6)))


def test_pico_manager_default_profile_preserves_dex3_shape():
    left, right = compute_hand_joints_from_inputs(None, None, 0.0, 0.0, 0.0, 0.0)

    assert left.shape == (1, 7)
    assert right.shape == (1, 7)


def test_pico_manager_rejects_unknown_hand_profile():
    with pytest.raises(ValueError, match="hand profile"):
        compute_hand_joints_from_inputs(
            None,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            hand_profile="unknown",
        )


@pytest.mark.parametrize("streamer_type", (PoseStreamer, PlannerStreamer))
def test_pose_and_planner_streamers_use_selected_hand_profile(streamer_type):
    streamer = object.__new__(streamer_type)
    streamer.left_hand_ik_solver = None
    streamer.right_hand_ik_solver = None
    streamer.hand_profile = "inspire_ftp"

    left, right = streamer.compute_hand_joints(1.0, 0.0, 0.0, 1.0)

    np.testing.assert_allclose(left, [[0, 0, 0, 0, 0, 1]])
    np.testing.assert_allclose(right, [[1, 1, 1, 1, 1, 0]])
