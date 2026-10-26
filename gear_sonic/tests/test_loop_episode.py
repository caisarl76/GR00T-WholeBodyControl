import numpy as np
import pytest

from gear_sonic.utils.inference.loop_episode import (
    ARM_HI,
    ARM_LO,
    ARM_SPEED_TICKS,
    TOKEN_ECHO_TOL,
    EpisodeGuards,
    HandoverDetector,
    max_arm_speed,
    token_echo_err,
)

OPEN, CLOSED = 0.0, 0.38  # mean of the 7 measured right-hand joints, open vs closed around the bottle


def run_detector(samples, **kw):
    """samples: [(t, right_hand_mean)] -> first t at which the detector fires, or None."""
    d = HandoverDetector(**kw)
    for t, h in samples:
        if d.update(h, now=t):
            return t
    return None


def ticks(t0, t1, value):
    return [(round(t, 2), value) for t in np.arange(t0, t1, 0.02)]


def test_handover_fires_after_grasp_then_release():
    s = ticks(0, 2, OPEN) + ticks(2, 4, CLOSED) + ticks(4, 6, OPEN)
    assert run_detector(s, open_s=1.0) == pytest.approx(5.0, abs=0.03)


def test_startup_fist_that_opens_is_not_a_handover():
    s = ticks(0, 1.5, CLOSED) + ticks(1.5, 5, OPEN)
    assert run_detector(s, open_s=1.0) is None


def test_brief_close_is_not_a_grasp():
    s = ticks(0, 1, OPEN) + ticks(1, 1.3, CLOSED) + ticks(1.3, 4, OPEN)
    assert run_detector(s, open_s=1.0) is None


def test_short_reopen_during_a_regrasp_does_not_end_the_trial():
    s = ticks(0, 1, OPEN) + ticks(1, 2, CLOSED) + ticks(2, 2.5, OPEN) + ticks(2.5, 4, CLOSED) + ticks(4, 6, OPEN)
    assert run_detector(s, open_s=1.0) == pytest.approx(5.0, abs=0.03)


def test_missing_state_is_ignored():
    s = ticks(0, 1, OPEN) + ticks(1, 2, CLOSED) + [(2.0, float("nan"))] + ticks(2.02, 4, OPEN)
    assert run_detector(s, open_s=1.0) == pytest.approx(3.02, abs=0.03)


def test_handover_defaults_to_monotonic_clock(monkeypatch):
    from gear_sonic.utils.inference import loop_episode as le

    clock = [0.0]
    monkeypatch.setattr(le.time, "monotonic", lambda: clock[0])
    d = HandoverDetector()
    fired = None
    for _ in range(400):
        clock[0] += 0.02
        if d.update(CLOSED if 1.0 <= clock[0] < 2.0 else OPEN):
            fired = clock[0]
            break
    assert fired == pytest.approx(3.0, abs=0.03)


# ---- guards ----
def _msg(arm=None, token=None):
    body = np.zeros(29)
    body[15:29] = (ARM_LO + ARM_HI) / 2 if arm is None else arm
    return {
        "body_q": body,
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "token_state": np.zeros(64) if token is None else token,
    }


TOKENS = [np.zeros(64, np.float32)]


def _check(g, now, msg=None, state_age=0.0, camera_age=0.0, last_chunk_at=None, tokens=TOKENS):
    return g.check(now, _msg() if msg is None else msg, state_age, camera_age,
                   now if last_chunk_at is None else last_chunk_at, tokens)  # fmt: skip


def _guards():
    g = EpisodeGuards()
    g.start(100.0)
    return g


def test_guard_categories():
    t = 100.0
    assert _check(_guards(), t + 1) is None
    assert _check(_guards(), t + 30.01) == ("end", "completed")
    assert _check(_guards(), t + 5, last_chunk_at=t + 3.99) == ("end", "no valid chunk for 1 s")
    assert _check(_guards(), t + 5, state_age=0.21) == ("freeze", "robot state 0.21 s old")
    assert _check(_guards(), t + 5, camera_age=0.51) == ("freeze", "camera frame 0.51 s old")
    # speed: 3 ticks later joint 2 moved 0.4 rad in 0.06 s = 6.67 rad/s > 6
    g = _guards()
    base = (ARM_LO + ARM_HI) / 2
    for k in range(ARM_SPEED_TICKS):
        assert _check(g, t + 1 + 0.02 * k, _msg(base)) is None
    fast = base.copy()
    fast[2] += 0.4
    assert _check(g, t + 1.06, _msg(fast)) == (
        "end",
        "watchdog: arm joint 2 measured at 6.7 rad/s > 6",
    )
    # range: joint 0 is 0.16 rad below ARM_LO
    low = base.copy()
    low[0] = ARM_LO[0] - 0.16
    g = _guards()
    res = None
    for k in range(ARM_SPEED_TICKS + 1):
        res = _check(g, t + 1 + 0.02 * k, _msg(low))
    assert res == ("end", "watchdog: arm joint 0 at -2.76 rad, 0.16 rad outside the training range")


def test_token_echo_is_a_timer():
    msg_bad = _msg(token=np.ones(64))
    reason = "deploy token_state has not matched a streamed token for 0.5 s (another command source?)"
    g = _guards()
    assert _check(g, 101.0, msg_bad) is None  # first miss starts the timer
    assert _check(g, 101.5, msg_bad) is None  # exactly the timeout: not yet
    assert _check(g, 101.51, msg_bad) == ("freeze", reason)
    # a match clears the timer
    g = _guards()
    assert _check(g, 101.0, msg_bad) is None
    assert _check(g, 101.4, _msg()) is None
    assert _check(g, 101.6, msg_bad) is None  # new timer from 101.6
    assert _check(g, 102.0, msg_bad) is None
    assert _check(g, 102.11, msg_bad) == ("freeze", reason)


def test_arm_speed_is_measured_over_arm_speed_ticks():
    base = (ARM_LO + ARM_HI) / 2
    g = _guards()
    t = 101.0
    # 1 rad/s steady drift: far below the limit
    for k in range(10):
        a = base.copy()
        a[1] += 0.02 * k
        assert _check(g, t + 0.02 * k, _msg(a)) is None
    # a 0.5 rad jump on the newest tick only: speed over 3 ticks (0.06 s) = 8.3 rad/s > 6
    a = base.copy()
    a[1] += 0.18 + 0.5
    res = _check(g, t + 0.2, _msg(a))
    assert res is not None and res[0] == "end" and "arm joint 1 measured at" in res[1]
    # fewer than ARM_SPEED_TICKS+1 entries: no verdict yet
    g = _guards()
    jump = base.copy()
    jump[1] += 1.0
    assert _check(g, 101.0, _msg(base)) is None
    assert _check(g, 101.02, _msg(jump)) is None


def test_start_resets_state():
    base = (ARM_LO + ARM_HI) / 2
    g = _guards()
    bad = _msg(token=np.ones(64))
    _check(g, 101.0, bad)
    for k in range(ARM_SPEED_TICKS):
        _check(g, 101.0 + 0.02 * k, _msg(base))
    g.start(200.0)
    assert _check(g, 201.0, bad) is None  # echo timer restarted
    jump = base.copy()
    jump[1] += 1.0
    assert _check(g, 201.02, _msg(jump)) is None  # arm history cleared
    assert _check(g, 229.0) is None  # cap counts from the new start
    assert _check(g, 230.0) == ("end", "completed")  # the cap boundary: lerobot never runs the tick at 30.0
    assert _check(g, 230.01) == ("end", "completed")


def test_check_before_start_raises():
    with pytest.raises(RuntimeError, match="start"):
        EpisodeGuards().check(123456.0, None, 0.0, 0.0, 123456.0, [])


def test_arm_range_uses_lerobot_limits():
    assert ARM_LO[:3] == pytest.approx([-2.596, -0.33, -1.53])
    assert ARM_HI[:3] == pytest.approx([1.473, 1.637, 1.676])
    assert ARM_LO.shape == ARM_HI.shape == (14,)
    assert ARM_LO[-1] == -1.611 and ARM_HI[8] == 0.284 and ARM_HI[-1] == 1.616


def test_helpers():
    assert token_echo_err(None, TOKENS) == float("inf")
    assert token_echo_err({"token_state": np.zeros(64)}, []) == float("inf")
    assert token_echo_err({"token_state": np.full(64, 0.5)}, TOKENS) == pytest.approx(0.5)
    assert TOKEN_ECHO_TOL == 1e-4
    assert max_arm_speed(1.0, np.zeros(28), 1.0, np.ones(28)) is None
    s1 = np.zeros(28)
    s1[5] = 1.0
    assert max_arm_speed(0.0, np.zeros(28), 0.5, s1) == (5, 2.0)
