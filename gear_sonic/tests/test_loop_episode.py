import numpy as np
import pytest

from gear_sonic.utils.inference.loop_episode import (
    ARM_HI,
    ARM_LO,
    ARM_SPEED_TICKS,
    TAKE_HOLD_S,
    TAKE_REFERENCE_MIN_SAMPLES,
    TOKEN_ECHO_TOL,
    EpisodeGuards,
    HandoverDetector,
    TakeDetector,
    max_arm_speed,
    take_reference,
    take_torque,
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


def test_blend_in_history_feeds_the_first_episode_check():
    """lerobot logs the blend in at t_ep 0, so its arm history holds the blend in's last ticks at episode start."""
    base = (ARM_LO + ARM_HI) / 2
    arm = base.copy()
    for keep in (True, False):
        g = EpisodeGuards()
        for k in range(50):  # the blend in ends moving arm joint 3 at 10 rad/s
            arm = base.copy()
            arm[3] += 0.2 * max(0, k - 45)
            g.record(100.0 + 0.02 * k, _msg(arm))
        g.start(101.0, keep_history=keep)
        res = _check(g, 101.0, _msg(arm))
        if keep:
            assert res is not None and res[0] == "end" and res[1].startswith("watchdog: arm joint 3 measured at")
        else:
            assert res is None
    g = EpisodeGuards()
    g.record(100.0, _msg(token=np.ones(64)))
    with pytest.raises(RuntimeError):
        _check(g, 100.0)  # record() alone does not start the checks
    g.start(101.0, keep_history=True)
    assert _check(g, 130.9) is None and _check(g, 131.0) == ("end", "completed")  # the cap counts from start()


def _torque(sh, el, wy=0.0):
    tau = np.zeros(29)
    tau[22], tau[25], tau[28] = sh, el, wy  # right shoulder pitch, elbow, wrist yaw (hardware order)
    return {"motor_torque": tau}


LOADED, UNLOADED, SPIKE = (-8.1, -5.2), (-5.45, -3.4), (-10.4, -6.0)  # session ready-A-r27 (N*m)


def _r27(take_at=6.5, spike=True, settle=True, duration=12.0, episode=None):
    """(t relative to the hold start, message) at 50 Hz: the episode's last 1.0 s (t < 0; loaded, +-0.1 noise,
    or ``episode(t)`` -> (sh, el)), then the hold: loaded on the palm, a 0.5 s grab spike in the loading direction
    ending at ``take_at``, a 0.2 s unloading ramp, then the unloaded arm."""
    out = []
    for k in range(-50, int(duration * 50)):
        t = k * 0.02
        noise = 0.1 * np.sin(7.3 * t)
        sh, el = LOADED if episode is None or t > 0 else episode(t)
        if spike and take_at - 0.5 <= t < take_at:
            sh, el = SPIKE
        elif settle and take_at <= t:
            w = min(1.0, (t - take_at) / 0.2)
            sh, el = SPIKE[0] + w * (UNLOADED[0] - SPIKE[0]), SPIKE[1] + w * (UNLOADED[1] - SPIKE[1])
        out.append((t, _torque(sh + noise, el + noise)))
    return out


def _detector(samples, start=0.0):
    """As LoopController: the reference from the episode's samples up to the hold start, then a TakeDetector."""
    ring = [(t, arm) for t, msg in samples if t <= start and (arm := take_torque(msg)) is not None]
    reference = take_reference(ring, start)[0]
    return TakeDetector(start, reference)


def _first_take(samples):
    d = _detector(samples)
    return next((t for t, msg in samples if t > 0 and d.update(msg, t)), None)


def test_take_detector_fires_on_the_r27_unload():
    t = _first_take(_r27(take_at=6.5))
    # rises past +1.5 / +1.0 during the 0.2 s ramp after the spike, then holds for TAKE_HOLD_S
    assert t is not None and 6.5 + TAKE_HOLD_S <= t <= 6.7 + TAKE_HOLD_S + 0.02


def test_take_detector_fires_on_an_early_take():
    # r28: the bottle taken +0.6 s into the hold; the reference comes from the episode's end, so no in-hold wait
    t = _first_take(_r27(take_at=0.6))
    assert t is not None and 1.7 <= t <= 1.9


def test_take_detector_ignores_the_grab_spike_and_a_steady_hold():
    assert _first_take(_r27(take_at=6.5, settle=False)) is None  # spike, then back on the palm
    assert _first_take(_r27(spike=False, settle=False, duration=60.0)) is None  # steady loaded hold
    # only the shoulder unloads (the elbow stays loaded): not a take
    one = [(t, _torque(msg["motor_torque"][22], LOADED[1])) for t, msg in _r27()]
    assert _first_take(one) is None


def test_without_a_steady_episode_end_the_reference_is_a_steady_hold_window():  # R30
    # the arm still moving at the episode's end: p90 - p10 over the last 0.5 s > TAKE_REFERENCE_SPREAD_NM
    moving = _r27(episode=lambda t: (LOADED[0] + 3.0 * t, LOADED[1]))  # 3 N*m/s: ~1.2 N*m spread in 0.5 s
    d = _detector(moving)
    assert d.episode_reference is None
    t = _first_take(moving)
    assert t is not None and 6.5 + TAKE_HOLD_S <= t <= 6.7 + TAKE_HOLD_S + 0.02
    # a single outlier tick does not make the episode reference unsteady (p90 - p10, not max - min)
    blip = _r27(episode=lambda t: (LOADED[0] - 3.0, LOADED[1]) if abs(t + 0.2) < 0.01 else LOADED)
    assert _detector(blip).episode_reference is not None and _first_take(blip) is not None
    # too few samples before the hold, or torque only in the hold: the steady loaded hold becomes the reference
    few = [(t, msg) for t, msg in _r27() if t > 0 or t >= -0.02 * (TAKE_REFERENCE_MIN_SAMPLES - 2)]
    assert _detector(few).episode_reference is None and _first_take(few) is not None
    late = [(t, msg if t > 0 else {}) for t, msg in _r27(take_at=3.0)]
    t = _first_take(late)
    assert t is not None and 3.0 + TAKE_HOLD_S <= t <= 3.2 + TAKE_HOLD_S + 0.02
    # no torque at all, or a wrong length: never fires
    for bad in ({"body_q": np.zeros(29)}, {"motor_torque": np.zeros(28)}):
        samples = [(t, bad) for t, _ in _r27()]
        assert _first_take(samples) is None


def test_r33_trial_3_settling_hold_then_take():  # R30, the harness trace of ready-A-r33 trial 3 (N*m)
    """Episode end unsteady (spread 1.06); the hold settles from -8.56/-5.81 to -7.06/-4.75 within 1 s, stays,
    and the take at +5.1 s unloads to -6.06/-3.94 (+1.0/+0.8). Detected about 1 s later, never before the take."""
    def arm(t):
        if t < 0:
            return (-8.6 + 2.0 * (t + 1.0) * (-1) ** int(t * 25), -5.8)  # moving: p90 - p10 well over 1.0
        if t < 1.0:
            w = t / 1.0
            return (-8.56 + w * 1.5, -5.81 + w * 1.06)
        if t < 5.1:
            return (-7.06, -4.75)
        w = min(1.0, (t - 5.1) / 0.5)
        return (-7.06 + w * 1.0, -4.75 + w * 0.81)
    samples = [(k * 0.02, _torque(*(np.array(arm(k * 0.02)) + 0.03 * np.sin(5.1 * k)))) for k in range(-50, 500)]
    d = _detector(samples)
    assert d.episode_reference is None
    t = next((t for t, msg in samples if t > 0 and d.update(msg, t)), None)
    assert d.steady_at is not None and 1.5 <= d.steady_at - d.start <= 2.6  # mostly after settling
    assert t is not None and 5.1 + TAKE_HOLD_S <= t <= 5.6 + TAKE_HOLD_S


def test_an_unsteady_hold_never_sets_a_reference():  # R30
    # +-0.6 N*m through the episode end and the whole hold: no reference, never fires, even on an unload
    samples = [(k * 0.02, _torque(-7.0 + (0.6 if k % 2 else -0.6) + (2.0 if k > 300 else 0.0), -4.8))
               for k in range(-50, 500)]
    d = _detector(samples)
    assert next((t for t, msg in samples if t > 0 and d.update(msg, t)), None) is None and d.reference is None


def test_take_reference_is_the_median_of_the_last_half_second():
    ring = [(-0.02 * k, np.array([-8.0 - 0.01 * k, -5.0])) for k in range(50)]  # 1 s, newest at t = 0
    reference, median, spread, n = take_reference(ring, 0.0)
    assert n == 26 and np.allclose(reference, median) and np.allclose(median, [-8.125, -5.0])
    assert spread[0] == pytest.approx(0.2) and spread[1] == 0.0


def test_take_detector_needs_a_continuous_second_and_skips_messages_without_torque():
    d = TakeDetector(0.0, np.array([-8.0, -5.0, 0.0]))
    assert not d.update(_torque(-6.0, -3.5), 3.0)
    assert not d.update({}, 3.5)  # no torque: ignored, the timer keeps running
    assert not d.update(_torque(-6.0, -3.5), 3.98)
    assert not d.update(_torque(-7.4, -3.5), 4.0)  # the shoulder falls back below +0.8: restart
    assert not d.update(_torque(-6.0, -3.5), 4.02)
    assert not d.update(_torque(-6.0, -3.5), 5.0)
    assert d.update(_torque(-6.0, -3.5), 5.02) and d.update({}, 5.04)  # stays taken
    assert np.allclose(d.rise, [2.0, 1.5, 0.0])


def test_take_detector_r28_reference_and_r30_trial2():
    # r30 trial 2: episode-end reference -7.62/-4.81, the hold settled at about -7.44/-4.69, after the take
    # -6.38/-3.81 (+1.24/+1.00 on the episode reference): fires under R28's 0.8/0.45 on the less-loaded reference
    d = TakeDetector(0.0, np.array([-7.62, -4.81, 0.0]))
    for k in range(15):  # the hold's first 0.3 s, loaded
        assert not d.update(_torque(-7.44, -4.69), k * 0.02)
    assert d.reference is None  # set on the first message after the hold's first 0.3 s
    t, fired = 0.3, None
    for k in range(150):
        t = 0.3 + k * 0.02
        load = (-7.40, -4.66) if t < 1.5 else (-6.38, -3.81)  # loaded drift +0.04, then the take at +1.5 s
        if d.update(_torque(*load), t):
            fired = t
            break
    assert fired is not None and 2.5 <= fired <= 2.55
    assert np.allclose(d.reference, [-7.44, -4.69, 0.0])  # the less loaded of the episode end and the hold's start
    # the loaded drift seen on r27 (+0.44/+0.25 over the reference) never fires
    s2 = TakeDetector(0.0, np.array([-8.19, -5.44, 0.0]))
    assert not any(s2.update(_torque(-8.19 + 0.44, -5.44 + 0.25), k * 0.02) for k in range(500))


def _hold(episode, phases, end=12.0):
    """(t, message) at 50 Hz: the episode's last 1 s at ``episode`` (sh, el, wrist yaw), then the hold's
    ``phases`` [(from t, (sh, el, wy)), ...] with +-0.03 N*m noise."""
    out = []
    for k in range(-50, int(end * 50)):
        t = k * 0.02
        load = episode if t <= 0 else [v for t0, v in phases if t >= t0][-1]
        out.append((t, _torque(*(np.array(load) + 0.03 * np.sin(5.1 * k)))))
    return out


def test_r34_takes_seen_only_by_the_wrist_yaw():  # R31, harness traces of ready-A-r34 holds 3 and 4 (N*m)
    # hold 4: shoulder/elbow -6.50/-4.06 -> -6.25/-3.85 (+0.25/+0.21), wrist yaw +1.35 -> +0.89 (-0.46) at +4.0 s
    h4 = _hold((-6.44, -4.00, 1.28), [(0.0, (-6.50, -4.06, 1.35)), (4.0, (-6.25, -3.85, 0.89))])
    t = _first_take(h4)
    assert t is not None and 4.0 + TAKE_HOLD_S <= t <= 4.1 + TAKE_HOLD_S
    # hold 3: the arm loaded further after the episode's end (-6.56/-4.19 -> -7.06/-4.62), so shoulder/elbow rose
    # only +0.44/+0.31 over the less-loaded reference; wrist yaw +1.46 -> +0.81 at +5.1 s
    h3 = _hold((-6.56, -4.19, 1.36), [(0.0, (-7.06, -4.62, 1.46)), (5.1, (-6.12, -3.88, 0.81))])
    t = _first_take(h3)
    assert t is not None and 5.1 + TAKE_HOLD_S <= t <= 5.2 + TAKE_HOLD_S


def test_wrist_yaw_on_the_palm_does_not_fire():  # R31
    # r33 trial 7: a person's hand on the bottle, still on the palm: wrist yaw -0.31, shoulder/elbow +0.38/+0.47
    t7 = _hold((-7.44, -4.78, 1.19), [(0.0, (-7.44, -4.78, 1.19)), (0.5, (-7.06, -4.31, 0.88))], end=20.0)
    assert _first_take(t7) is None
    # a loaded hold whose wrist yaw rises (r33 trial 9: +1.22 -> +1.30) never fires
    t9 = _hold((-7.44, -5.19, 1.22), [(0.0, (-7.88, -5.44, 1.22)), (1.0, (-8.12, -5.62, 1.30))], end=20.0)
    assert _first_take(t9) is None


def test_a_settling_step_in_the_first_half_second_is_not_a_reference():  # R30, TAKE_STEADY_S
    # unsteady episode end, then 0.6 s flat at the settling value -8.56 and a step to the hold's level -7.12:
    # a 0.5 s window would take -8.56 and fire on the settling (+1.44); the 1 s window waits for the real level
    moving = lambda t: (-8.6 + (0.7 if int(t * 50) % 2 else -0.7), -5.8, 1.2)  # noqa: E731
    h = _hold((0.0, 0.0, 0.0), [(0.0, (-8.56, -5.81, 1.20)), (0.6, (-7.12, -4.81, 1.20))], end=10.0)
    h = [(t, _torque(*moving(t)) if t <= 0 else msg) for t, msg in h]
    d = _detector(h)
    assert d.episode_reference is None
    assert next((t for t, msg in h if t > 0 and d.update(msg, t)), None) is None
    assert d.reference is not None and abs(d.reference[0] + 7.12) < 0.1
