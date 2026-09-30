import pytest

from decoupled_wbc.control.teleop.streamers.pico_streamer import pico_joysticks_to_nav_inputs


def test_pico_right_stick_right_maps_to_right_turn():
    _, _, yaw = pico_joysticks_to_nav_inputs(
        left_joystick=[0.0, 0.0],
        right_joystick=[-1.0, 0.0],
    )

    assert yaw < 0.0


def test_pico_left_stick_right_maps_to_right_strafe():
    _, strafe, _ = pico_joysticks_to_nav_inputs(
        left_joystick=[-1.0, 0.0],
        right_joystick=[0.0, 0.0],
    )

    assert strafe > 0.0


def test_pico_left_stick_forward_maps_to_forward():
    forward, strafe, yaw = pico_joysticks_to_nav_inputs(
        left_joystick=[0.0, 1.0],
        right_joystick=[0.0, 0.0],
    )

    assert forward == pytest.approx(1.0)
    assert strafe == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
