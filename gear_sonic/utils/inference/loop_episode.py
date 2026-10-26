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

    def start(self, now: float) -> None:
        self.t0, self.echo_lost, self.history = now, None, []

    def check(self, now, state_msg, state_age_s, camera_age_s, last_chunk_at, recent_tokens):
        if self.t0 is None:
            raise RuntimeError("EpisodeGuards.check before start(now)")
        arm = None
        if state_msg:
            arm = lerobot_state(state_msg, np.arange(7))[:14]  # the arm slice ignores right_order
            self.history.append((arm, now))
            del self.history[: -(ARM_SPEED_TICKS + 1)]
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
