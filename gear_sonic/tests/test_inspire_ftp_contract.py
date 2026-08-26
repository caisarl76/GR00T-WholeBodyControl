import numpy as np
import pytest

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
