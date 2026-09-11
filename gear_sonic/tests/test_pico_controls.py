"""Elapsed-time controller arbitration and nonblocking terminal keys."""

import os
from types import SimpleNamespace

import pytest

from gear_sonic.utils.teleop.pico_controls import ControllerChords, ManagerKeyboard, read_controllers


def press(resolver, keys, milliseconds, **kwargs):
    return resolver.poll(dict.fromkeys(keys, True), milliseconds * 1_000_000, **kwargs)


def armed():
    resolver = ControllerChords()
    assert press(resolver, "", 0) is None
    assert press(resolver, "", 39) is None
    assert resolver.wait_release
    assert press(resolver, "", 40) is None
    assert not resolver.wait_release
    return resolver


def test_elapsed_debounce_and_priority_suppresses_subset():
    resolver = armed()
    assert press(resolver, "ax", 50) is None
    assert press(resolver, "ax", 89) is None
    assert resolver.deadline is None
    assert press(resolver, "ax", 90) is None
    assert resolver.deadline == 290_000_000
    assert press(resolver, "abxy", 180) is None
    assert press(resolver, "abxy", 219) is None
    assert press(resolver, "abxy", 220) == "sonic"
    assert press(resolver, "abxy", 500) is None
    assert press(resolver, "ax", 700) is None


@pytest.mark.parametrize(
    "keys,expected",
    [
        (("a", "x"), "tracking"),
        (("grip", "a", "x"), "record"),
        (("grip", "a", "b", "x"), "abort"),
        (("a", "b"), "locomotion_next"),
    ],
)
def test_exact_deadline_priorities_and_requires_full_neutral(keys, expected):
    resolver = armed()
    press(resolver, keys, 50)
    press(resolver, keys, 90)
    assert press(resolver, keys, 289) is None
    assert press(resolver, keys, 290) == expected
    assert press(resolver, "abxy", 310) is None
    assert press(resolver, "", 400) is None
    assert press(resolver, "", 439) is None
    assert resolver.wait_release
    assert press(resolver, "", 440) is None
    assert not resolver.wait_release


def test_bounce_and_packet_loss_cancel_without_action_and_rearm_neutral():
    resolver = armed()
    press(resolver, "ax", 50)
    press(resolver, "", 70)
    press(resolver, "ax", 80)
    assert press(resolver, "ax", 119) is None
    assert resolver.deadline is None
    press(resolver, "ax", 120)
    assert press(resolver, "ax", 300, fresh=False) is None
    assert press(resolver, "abxy", 400) is None
    press(resolver, "", 410)
    press(resolver, "", 450)
    press(resolver, "abxy", 460)
    assert press(resolver, "abxy", 500) == "sonic"


def sample(**changes):
    return {
        "binding_generation": 1,
        "receipt_age_ns": 0,
        "primary_button": False,
        "secondary_button": False,
        "axis_click": False,
        "menu_button": False,
        "axis": (0.0, 0.0),
        "grip": 0.0,
        "trigger": 0.0,
        **changes,
    }


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        sample(receipt_age_ns=100_000_000),
        sample(receipt_age_ns=True),
        sample(grip=float("nan")),
        sample(axis=(1.0,)),
        sample(primary_button=1),
        sample(binding_generation=True),
        sample(grip="bad"),
    ],
)
def test_malformed_side_is_neutral_and_other_side_remains_available(bad):
    sdk = SimpleNamespace(
        get_left_controller_snapshot=lambda: bad, get_right_controller_snapshot=lambda: sample(primary_button=True)
    )
    result = read_controllers(sdk)
    assert not result["fresh"]
    assert result["abxy"] == (True, False, False, False)
    assert result["inputs"] == (False, 0.0, 0.0, 0.0, 0.0)


def test_getter_error_isolated_and_each_atomic_getter_called_once():
    counts = [0, 0]

    def left():
        counts[0] += 1
        raise RuntimeError("disconnected")

    def right():
        counts[1] += 1
        return sample(primary_button=True, receipt_age_ns=99_999_999)

    result = read_controllers(
        SimpleNamespace(get_left_controller_snapshot=left, get_right_controller_snapshot=right)
    )
    assert counts == [1, 1]
    assert result["buttons"]["a"]
    assert not result["fresh"]


def test_keyboard_ignores_o_and_preserves_order_without_blocking():
    read_fd, write_fd = os.pipe()
    try:
        keyboard = ManagerKeyboard()
        keyboard.fd = read_fd
        assert keyboard.poll() == []
        os.write(write_fd, b"OoTCCSs!\xff")
        assert keyboard.poll() == ["t", "c", "c", "s", "s"]
        assert keyboard.poll() == []
    finally:
        os.close(read_fd)
        os.close(write_fd)
