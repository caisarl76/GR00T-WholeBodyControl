"""Episode guards and the handover detector for the autonomous handover loop, ported from
lerobot a6e267c6 (eval_session.py, sonic_policy_streamer.py, stream_backends.py) so the native loop ends and
freezes trials alike."""

import math
import time

import numpy as np

from gear_sonic.utils.inference.lerobot_sonic_policy import lerobot_state

# Arm joint range (motor order 15..28, rad) of observation.state over all Unitree + Humanoid Everyday joint28
# episodes; the watchdog ends an episode that leaves it by more than arm_range_margin.
ARM_LO = np.array(
    [
        -2.596,
        -0.33,
        -1.53,
        -1.046,
        -1.861,
        -1.612,
        -1.61,
        -2.609,
        -1.629,
        -1.627,
        -1.043,
        -1.949,
        -1.618,
        -1.611,
    ]
)
ARM_HI = np.array(
    [1.473, 1.637, 1.676, 1.526, 1.97, 1.618, 1.625, 1.586, 0.284, 1.579, 1.419, 1.599, 1.634, 1.616]
)
TOKEN_ECHO_TOL = 1e-4  # the deploy copies float32 tokens into double token_state: an echo is exact
ARM_SPEED_TICKS = 3  # measured arm speed over 3 ticks (60 ms): robust to one late or repeated state message


class HandoverDetector:
    """End the trial once one pick-and-handover attempt is complete, from the measured right hand
    (mean of its 7 Dex3 joints: ~0 open, ~0.38 closed around the bottle). It fires when the hand, after being seen
    open, stayed closed for at least ``min_hold_s`` (a grasp, not a flick) and then stayed open for ``open_s`` (the
    release; a shorter reopen is a regrasp). A hand that starts closed (the startup fist) and only opens is not a
    handover. One attempt per trial: in s17 (2026-10-23) second attempts within a trial were slower and less
    accurate, because the training episodes end about 2 s after the release.

    Durations use real elapsed time (``time.monotonic()``; ``now`` only for tests), never the streamer's scheduled
    episode time: ticks() catches up late ticks without sleeping, so after a stall the scheduled time races ahead
    and a brief closure would look like a held grasp (PR #47 review)."""

    def __init__(self, open_s: float = 1.0, min_hold_s: float = 0.5, closed_above: float = 0.19):
        self.open_s, self.min_hold_s, self.closed_above = open_s, min_hold_s, closed_above
        self.seen_open, self.held = False, False
        self.closed_since: float | None = None
        self.open_since: float | None = None

    def update(self, right_hand_mean: float, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        if not math.isfinite(right_hand_mean):
            return False
        if right_hand_mean > self.closed_above:
            self.open_since = None
            if self.seen_open:
                if self.closed_since is None:
                    self.closed_since = t
                if t - self.closed_since >= self.min_hold_s:
                    self.held = True
            return False
        self.seen_open, self.closed_since = True, None
        if not self.held:
            return False
        if self.open_since is None:
            self.open_since = t
        return t - self.open_since >= self.open_s


# Bottle-taken detection in await_take from the deploy's right-arm tau_est (g1_debug "motor_torque", hardware
# order). Session ready-A-r27 (3 holds): loaded on the palm sh_pitch -7.9..-8.3, elbow -5.0..-5.5 N*m (+-0.1);
# after the take -5.4..-5.5 / -3.3..-3.5 (a rise toward zero of +2.4..+2.9 / +1.6..+2.2), after a ~0.5 s grab
# spike the other way. r28: people take the bottle as early as +0.6 s into the hold, so the reference is the
# loaded arm at the episode's end, not a window inside the hold (R27). r27: reference-window p90-p10 <= 0.62,
# loaded rise in the first 4 s of the hold <= 0.44 / 0.25.
# R28 (r30): the unload depends on the offer pose: +2.44 / +1.94 down to +1.00 / +0.81 (a pose where the arm
# carries less), while the hold could sit +0.6 above the episode-end reference as the arm settled into it. So the
# reference is the less-loaded of the episode end and the hold's first TAKE_HOLD0_S, and the rise threshold is
# lower: worst loaded drift on that reference 0.44 / 0.25 (r27), weakest real take 1.00 / 0.81 (r30 trial 2).
TAKE_MOTORS = (22, 25)  # right shoulder pitch, right elbow
TAKE_RING_S = 1.0  # the episode's last torque samples kept for the reference
TAKE_REFERENCE_S = 0.5  # reference = median over the last 0.5 s before the hold
TAKE_REFERENCE_MIN_SAMPLES = 10
TAKE_REFERENCE_SPREAD_NM = 1.0  # p90 - p10 per motor over the reference window; wider: inert (operator decides)
TAKE_RISE_NM = (0.8, 0.45)  # rise toward zero (unload) over the reference, per motor
TAKE_HOLD0_S = 0.3  # the hold's first samples also form the reference (the less loaded of the two, per motor)
TAKE_HOLD_S = 1.0  # both rises held continuously this long
# R30 (r33 trial 3): the arm still settling at the episode's end (spread 1.06) left the detector inert and a clean
# +1.06 / +0.87 take at +5.1 s to the operator. Without an episode-end reference, the reference is the median of
# the hold's first TAKE_REFERENCE_S window this steady (still holds: p90 - p10 ~0.06 N*m, r27 at 50 Hz).
TAKE_STEADY_SPREAD_NM = 0.3


def take_torque(msg):
    """(shoulder pitch, elbow) tau_est from a g1_debug message, or None without a 29-value finite motor_torque."""
    tau = np.asarray((msg or {}).get("motor_torque", []), np.float64).reshape(-1)
    if tau.shape != (29,) or not np.isfinite(tau[list(TAKE_MOTORS)]).all():
        return None
    return tau[list(TAKE_MOTORS)]


def take_reference(samples, start):
    """(reference or None, median, p90 - p10 spread, n) from (t, torque) samples in the TAKE_REFERENCE_S before
    ``start``. The reference is None (the detector inert) with fewer than TAKE_REFERENCE_MIN_SAMPLES samples or a
    spread over TAKE_REFERENCE_SPREAD_NM on either motor."""
    arms = np.array([arm for t, arm in samples if start - TAKE_REFERENCE_S <= t <= start])
    if len(arms) < TAKE_REFERENCE_MIN_SAMPLES:
        return None, None, None, len(arms)
    median = np.median(arms, axis=0)
    spread = np.percentile(arms, 90, axis=0) - np.percentile(arms, 10, axis=0)
    return (None if (spread > TAKE_REFERENCE_SPREAD_NM).any() else median), median, spread, len(arms)


class TakeDetector:
    """True once the right arm unloads: shoulder pitch and elbow torque both rise TAKE_RISE_NM over the reference
    for TAKE_HOLD_S. The reference is the less loaded (per motor) of ``reference`` (the episode's end,
    take_reference) and the median of the hold's first TAKE_HOLD0_S, so detection starts after TAKE_HOLD0_S. The
    grab spike loads the arm (more negative), so it cannot trigger. Messages without torque are ignored. With
    ``reference`` None (R30), the reference is the median of the first TAKE_REFERENCE_S hold window whose p90 - p10
    is within TAKE_STEADY_SPREAD_NM on both motors; until then it cannot trigger."""

    def __init__(self, start: float, reference):
        self.start, self.rising_since, self.taken, self.rise = start, None, False, None
        self.episode_reference = None if reference is None else np.asarray(reference, np.float64)
        self.reference, self.hold0 = None, []  # the reference is set once the hold's first TAKE_HOLD0_S is in
        self.steady_at = None  # R30: when the reference came from a steady hold window

    def update(self, msg, now: float) -> bool:
        if self.taken:
            return True
        arm = take_torque(msg)
        if arm is None:
            return False
        if self.reference is None and self.episode_reference is None:  # R30
            self.hold0 = [(t, a) for t, a in self.hold0 if t >= now - TAKE_REFERENCE_S] + [(now, arm)]
            arms = np.array([a for _, a in self.hold0])
            spread = np.percentile(arms, 90, axis=0) - np.percentile(arms, 10, axis=0)
            if (now - self.start >= TAKE_REFERENCE_S and len(arms) >= TAKE_REFERENCE_MIN_SAMPLES
                    and (spread <= TAKE_STEADY_SPREAD_NM).all()):
                self.reference, self.steady_at = np.median(arms, axis=0), now
            return False
        if self.reference is None:
            if now - self.start < TAKE_HOLD0_S:
                self.hold0.append(arm)
                return False
            hold0 = np.median(self.hold0, axis=0) if self.hold0 else self.episode_reference
            self.reference = np.maximum(self.episode_reference, hold0)
        self.rise = arm - self.reference
        if (self.rise >= np.asarray(TAKE_RISE_NM)).all():
            self.rising_since = now if self.rising_since is None else self.rising_since
            self.taken = now - self.rising_since >= TAKE_HOLD_S
        else:
            self.rising_since = None
        return self.taken


def token_echo_err(token_state_msg: dict | None, recent_tokens) -> float:
    """Smallest max-abs difference between the deploy's g1_debug token_state and a recently sent token. In POSE
    mode the deploy copies each received token into token_state; a deploy fed by another command source (or
    still in planner mode) reports other tokens. inf without a 64D token_state or sent tokens."""
    echo = np.asarray((token_state_msg or {}).get("token_state", []), np.float32)
    if echo.shape != (64,) or not len(recent_tokens):
        return float("inf")
    return float(np.abs(np.asarray(recent_tokens) - echo).max(axis=1).min())


def max_arm_speed(wall0: float, state0: np.ndarray, wall1: float, state1: np.ndarray) -> tuple[int, float] | None:
    """(arm joint, rad/s) of the fastest of the 14 arm joints between two measured states; None if no time passed.
    Timed by the wall clock of the ticks, not episode time: the 1 s blend in is logged at t_ep 0 (G1 run48: 3 ticks
    of blend-in motion over 20 ms of t_ep read 9.5 rad/s for a true 2.4 rad/s)."""
    if wall1 <= wall0:
        return None
    speed = np.abs(np.asarray(state1[:14]) - np.asarray(state0[:14])) / (wall1 - wall0)
    j = int(np.argmax(speed))
    return j, float(speed[j])


class EpisodeGuards:
    """Per-tick trial watchdogs in lerobot's order. ``("end", reason)`` stops the trial, ``("freeze", reason)``
    holds the robot. Stateful like lerobot's: call ``start(now)`` when a trial begins, then ``check`` once
    per tick."""

    def __init__(
        self,
        cap_s: float = 30.0,
        max_arm_speed: float = 6.0,
        arm_range_margin: float = 0.15,
        max_chunk_age_s: float = 1.0,
        max_state_age_s: float = 0.2,
        max_camera_age_s: float = 0.5,
        token_echo_timeout_s: float = 0.5,
    ):
        self.cap_s, self.max_arm_speed, self.arm_range_margin = cap_s, max_arm_speed, arm_range_margin
        self.max_chunk_age_s, self.max_state_age_s = max_chunk_age_s, max_state_age_s
        self.max_camera_age_s, self.token_echo_timeout_s = max_camera_age_s, token_echo_timeout_s
        self.t0, self.echo_lost, self.history = None, None, []

    def record(self, now, state_msg) -> None:
        """Append a measured arm state to the speed history only (lerobot logs the blend in at t_ep 0, so its
        watchdog's first episode ticks measure the blend in's end)."""
        if state_msg:
            arm = lerobot_state(state_msg, np.arange(7))[:14]  # the arm slice ignores right_order
            self.history.append((arm, now))
            del self.history[: -(ARM_SPEED_TICKS + 1)]

    def start(self, now: float, keep_history: bool = False) -> None:
        self.t0, self.echo_lost = now, None
        if not keep_history:
            self.history = []

    def check(self, now, state_msg, state_age_s, camera_age_s, last_chunk_at, recent_tokens):
        if self.t0 is None:
            raise RuntimeError("EpisodeGuards.check before start(now)")
        self.record(now, state_msg)
        if now - self.t0 >= self.cap_s:
            return "end", "completed"
        if now - last_chunk_at > self.max_chunk_age_s:
            return "end", f"no valid chunk for {self.max_chunk_age_s:g} s"
        if state_age_s > self.max_state_age_s:
            return "freeze", f"robot state {state_age_s:.2f} s old"
        if camera_age_s > self.max_camera_age_s:
            return "freeze", f"camera frame {camera_age_s:.2f} s old"
        if self.token_echo_timeout_s > 0:
            if token_echo_err(state_msg, recent_tokens) <= TOKEN_ECHO_TOL:
                self.echo_lost = None
            elif self.echo_lost is None:
                self.echo_lost = now
            elif now - self.echo_lost > self.token_echo_timeout_s:
                return "freeze", (
                    f"deploy token_state has not matched a streamed token for {self.token_echo_timeout_s:g} s "
                    "(another command source?)"
                )
        if len(self.history) <= ARM_SPEED_TICKS:
            return None
        (s0, w0), (s1, w1) = self.history[0], self.history[-1]
        fastest = max_arm_speed(w0, s0, w1, s1) if self.max_arm_speed > 0 else None
        if fastest and fastest[1] > self.max_arm_speed:
            return "end", (
                f"watchdog: arm joint {fastest[0]} measured at {fastest[1]:.1f} rad/s > {self.max_arm_speed:g}"
            )
        outside = np.maximum(ARM_LO - s1, s1 - ARM_HI)
        j = int(np.argmax(outside))
        if outside[j] > self.arm_range_margin:
            return "end", (
                f"watchdog: arm joint {j} at {s1[j]:.2f} rad, {outside[j]:.2f} rad outside the training range"
            )
        return None
