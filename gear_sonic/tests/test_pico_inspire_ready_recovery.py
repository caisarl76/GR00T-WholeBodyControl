"""Transient idle errors must not permanently block the first optical admission."""

import pytest

from gear_sonic.scripts import run_pico_inspire_bridge as b
from gear_sonic.tests.test_pico_inspire_bridge import (
    BASE,
    OPEN,
    PERIOD,
    PV,
    command,
    feedback,
    ownership,
    ready,
)
from gear_sonic.utils.teleop.inspire_ftp_real_bridge import (
    BridgeState,
    InspireFtpSafetyController,
    ValidatedHandState,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ready_error_pauses_writes_then_recovers_without_restart(side):
    c = ready()
    now = BASE + 10 * PERIOD
    feedback(c, now)
    bad = ValidatedHandState(side, OPEN, (0, 84, 0, 0, 0, 0), now)
    assert not c.accept_hand_state(bad)
    d = c.tick(now)
    assert d.state is BridgeState.READY
    assert not d.publish and d.fault_reason is None
    assert c.open_confirm_ticks == 0
    for tick in range(11, 20):
        accepted, d = command(c, tick, tick)
        assert not accepted and d.state is BridgeState.READY
    accepted, d = command(c, 20, 20)
    assert not accepted and d.state is BridgeState.READY
    assert command(c, 21, 21)[1].state is BridgeState.ACTIVE
    record = b.build_status_record(c, ownership().evidence(), BASE + 21 * PERIOD, None, None)
    assert record["ready_error_feedback"]["errors"] == list(bad.err)
    assert record["ready_error_feedback"]["side"] == side
    assert record["fault_feedback"] is None


def test_transient_ready_error_invalidates_queued_pair_even_if_feedback_recovers_before_tick():
    c = ready()
    now = BASE + 10 * PERIOD
    c.accept_manager(PV, now - 3)
    assert c.accept_pair((0.8,) * 6, (0.8,) * 6, PV, "inspire_hand", now - 2, 0, b"queued")
    c.accept_hand_state(ValidatedHandState("left", OPEN, (84,) * 6, now - 1))
    feedback(c, now)
    assert c.tick(now).state is BridgeState.READY
    assert c.latest_pair is None and c.open_confirm_ticks == 1
    assert c.fault_reason is None


def test_persistent_ready_error_never_publishes_or_enters_active():
    c = ready()
    for tick in range(10, 40):
        now = BASE + tick * PERIOD
        feedback(c, now)
        c.accept_hand_state(ValidatedHandState("right", OPEN, (0, 84, 0, 0, 0, 0), now))
        c.accept_manager(PV, now - 2)
        assert not c.accept_pair((0.8,) * 6, (0.8,) * 6, PV, "inspire_hand", now - 1, tick, b"blocked")
        d = c.tick(now)
        assert d.state is BridgeState.READY and not d.publish
    assert c.fault_reason is None


def test_ready_warning_does_not_hide_stale_feedback_or_other_faults():
    for reason in ("timeout", "writer"):
        c = ready()
        now = BASE + 10 * PERIOD
        feedback(c, now)
        c.accept_hand_state(ValidatedHandState("left", OPEN, (84,) * 6, now))
        c.tick(now)
        if reason == "writer":
            c.latch_fault("left hand command write failed")
            d = c.tick(now + PERIOD)
            assert "write failed" in d.fault_reason
        else:
            for tick in range(1, 6):
                d = c.tick(now + tick * PERIOD)
            assert "timeout" in d.fault_reason
        assert d.state is BridgeState.FAULT_LATCHED


def test_error_after_active_still_latches_and_retains_offending_feedback():
    c = ready()
    command(c, 10, 0)
    now = BASE + 11 * PERIOD
    bad = ValidatedHandState("left", OPEN, (0, 84, 0, 0, 0, 0), now)
    c.accept_hand_state(bad)
    feedback(c, now + 1)
    assert c.tick(now + 1).state is BridgeState.FAULT_LATCHED
    assert c.fault_feedback == bad


def test_errors_still_latch_during_commissioning():
    c = InspireFtpSafetyController(initialize_hands=True)
    c.arm_publish_from_feedback(
        ValidatedHandState("left", OPEN, (0,) * 6, BASE),
        ValidatedHandState("right", OPEN, (0,) * 6, BASE),
        BASE,
    )
    c.accept_hand_state(ValidatedHandState("right", OPEN, (84,) * 6, BASE))
    d = c.tick(BASE)
    assert d.state is BridgeState.FAULT_LATCHED and not d.publish


def test_ready_after_active_does_not_relax_errors():
    c = ready()
    command(c, 10, 0)
    for tick in range(11, 30):
        now = BASE + tick * PERIOD
        feedback(c, now)
        c.tick(now)
    assert c.state is BridgeState.READY
    now += PERIOD
    c.accept_hand_state(ValidatedHandState("left", OPEN, (84,) * 6, now))
    feedback(c, now + 1)
    assert c.tick(now + 1).state is BridgeState.FAULT_LATCHED
