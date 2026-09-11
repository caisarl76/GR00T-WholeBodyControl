"""Independent optical hand admission, source clocks, and bounded recovery.

No body or motor writer lives here. Call once per manager tick, even when the
body sample is unchanged. A missing command means measured feedback is needed.
"""

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class TrackingState(IntEnum):
    WAITING = 0
    HOLDING = 1
    RECOVERING = 2
    TRACKING = 3


class HandReason(IntEnum):
    OK = 0
    MISSING = 1
    MALFORMED = 2
    INACTIVE = 3
    UNTRACKED = 4
    GEOMETRY = 5
    BODY_WRIST_UNAVAILABLE = 6
    WRIST_DISTANCE = 7
    SOURCE_STALE = 8
    SOURCE_RESTART = 9
    FEEDBACK_UNAVAILABLE = 10
    RETARGET_FAILED = 11
    DISABLED = 12


@dataclass(frozen=True)
class HandOutput:
    command: np.ndarray | None
    target: np.ndarray | None
    state: TrackingState
    reason: HandReason
    source_epoch: int
    source_timestamp_ns: int
    binding_generation: int
    age_ns: int
    valid: bool


def _frozen(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


def validate_hand(snapshot, body_wrist):
    """Return canonical wrist + 24 digit positions, or a stable rejection code."""
    if snapshot is None:
        return None, HandReason.MISSING
    try:
        pose = snapshot["pose"]
        flags = snapshot["location_flags"]
        radius = snapshot["radius"]
        if (
            not isinstance(pose, np.ndarray)
            or pose.shape != (26, 7)
            or pose.dtype != np.float64
            or not np.isfinite(pose).all()
            or not isinstance(flags, np.ndarray)
            or flags.shape != (26,)
            or flags.dtype != np.uint64
            or not isinstance(radius, np.ndarray)
            or radius.shape != (26,)
            or radius.dtype != np.float64
            or not np.isfinite(radius).all()
            or not np.isfinite(snapshot["scale"])
            or snapshot["scale"] <= 0
        ):
            return None, HandReason.MALFORMED
        if snapshot["is_active"] != 1:
            return None, HandReason.INACTIVE
        # Original XRoboToolkit reports zero radii even for tracked active hands.
        # Retargeting uses positions; unavailable radius metadata is not invalid geometry.
        if np.any(radius < 0):
            return None, HandReason.MALFORMED
        if not np.all((flags[1:] & 0xA) == 0xA):
            return None, HandReason.UNTRACKED
        points = pose[1:26, :3]
        if np.max(np.linalg.norm(points - points[0], axis=1)) > 0.5:
            return None, HandReason.GEOMETRY
        if np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2)) > 0.5:
            return None, HandReason.GEOMETRY
        for start, stop in ((1, 5), (5, 10), (10, 15), (15, 20), (20, 25)):
            segments = np.linalg.norm(np.diff(points[start:stop], axis=0), axis=1)
            if np.any((segments < 0.002) | (segments > 0.15)):
                return None, HandReason.GEOMETRY
        wrist = np.asarray(body_wrist, dtype=np.float64)
        if wrist.shape != (3,) or not np.isfinite(wrist).all():
            return None, HandReason.BODY_WRIST_UNAVAILABLE
        if np.linalg.norm(points[0] - wrist) > 0.30:
            return None, HandReason.WRIST_DISTANCE
        return points.copy(), HandReason.OK
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, HandReason.MALFORMED


class HandTracker:
    """One side's state; injected retargeter also supplies its output limits."""

    def __init__(self, retargeter, profile="dex3", max_rate=None):
        if profile not in ("dex3", "inspire_ftp"):
            raise ValueError("Unknown hand profile")
        self.retargeter = retargeter
        self.lower = np.asarray(retargeter.lower, dtype=np.float64)
        self.upper = np.asarray(retargeter.upper, dtype=np.float64)
        size = 7 if profile == "dex3" else 6
        if (
            self.lower.shape != (size,)
            or self.upper.shape != (size,)
            or not np.isfinite([self.lower, self.upper]).all()
            or np.any(self.lower >= self.upper)
        ):
            raise ValueError("Invalid retargeter limits")
        nominal_rate = 2.0 if size == 7 else 1.0
        self.max_rate = float(max_rate if max_rate is not None else nominal_rate)
        if not math.isfinite(self.max_rate) or not 0 < self.max_rate <= nominal_rate:
            raise ValueError(f"Hand rate must be positive, finite, and at most {nominal_rate} for {profile}")
        self.tolerance = 0.02 if size == 7 else 0.01
        self.state = TrackingState.WAITING
        self.command = None
        self.target = None
        self.source_timestamp_ns = 0
        self.timestamp_source = None
        self.source_epoch = 0
        self.binding_generation = 0
        self.last_advance_ns = None
        self.previous_tick_ns = None
        self.valid_streak = 0
        self.converged_streak = 0

    def _bounded(self, value, *, measured=False):
        if value is None:
            return None
        try:
            q = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
        # DDS float32 measurements can slightly cross a MuJoCo soft joint limit.
        epsilon = 1e-4 if self.lower.size == 7 else 0.0
        lower_epsilon = np.full(self.lower.shape, epsilon, dtype=np.float64)
        if (
            measured
            and self.lower.size == 7
            and getattr(self.retargeter, "side", None) == "right"
            and self.lower[5] == 0
        ):
            # Dex3 order: thumb 0..2, middle 3..4, index 5..6. Real captures
            # reach -0.000918 rad at right index_0's zero stop. Admit only
            # this measured lower excursion; targets retain the original check.
            lower_epsilon[5] = 1e-3
        if (
            q.shape != self.lower.shape
            or not np.isfinite(q).all()
            or np.any(q < self.lower - lower_epsilon)
            or np.any(q > self.upper + epsilon)
        ):
            return None
        return np.clip(q, self.lower, self.upper)

    def step(self, snapshot, body_wrist, measured, now_ns, *, enabled=True):
        now_ns = int(now_ns)
        advanced = False
        restart = False
        reason = HandReason.OK
        try:
            stamp = snapshot["source_timestamp_ns"]
            generation = snapshot["binding_generation"]
            clock_source = snapshot.get("timestamp_source", 0)
            if (
                isinstance(clock_source, (bool, np.bool_))
                or not isinstance(clock_source, (int, np.integer))
                or clock_source not in (0, 1)
                or isinstance(stamp, (bool, np.bool_))
                or not isinstance(stamp, (int, np.integer))
                or not 0 < stamp <= np.iinfo(np.int64).max
                or isinstance(generation, (bool, np.bool_))
                or not isinstance(generation, (int, np.integer))
                or not 0 < generation <= np.iinfo(np.uint64).max
            ):
                raise ValueError("Invalid source clock")
            self.binding_generation = int(generation)
            clock_changed = self.timestamp_source is not None and clock_source != self.timestamp_source
            if stamp < self.source_timestamp_ns or clock_changed:
                self.source_epoch += 1
                restart = True
            self.timestamp_source = int(clock_source)
            advanced = stamp != self.source_timestamp_ns or clock_changed
            if advanced:
                self.source_timestamp_ns = int(stamp)
                self.last_advance_ns = now_ns
        except (TypeError, KeyError, ValueError, OverflowError):
            reason = HandReason.MISSING if snapshot is None else HandReason.MALFORMED

        age_ns = now_ns - self.last_advance_ns if self.last_advance_ns is not None else -1
        points, sample_reason = validate_hand(snapshot, body_wrist)
        if reason == HandReason.OK:
            reason = sample_reason
        if restart:
            reason = HandReason.SOURCE_RESTART
        elif age_ns >= 100_000_000:
            reason = HandReason.SOURCE_STALE

        measured_q = self._bounded(measured, measured=True)
        if self.command is None and measured_q is not None:
            self.command = measured_q
        elif self.state == TrackingState.WAITING and measured_q is not None:
            self.command = measured_q
        if measured_q is None:
            reason = HandReason.FEEDBACK_UNAVAILABLE
        if not enabled:
            reason = HandReason.DISABLED

        if reason != HandReason.OK:
            # Before first admission, keep refreshing the measured startup baseline.
            # HOLDING preserves a command only after optical control has begun.
            if self.command is not None and self.state != TrackingState.WAITING:
                self.state = TrackingState.HOLDING
            self.valid_streak = self.converged_streak = 0
        else:
            if advanced:
                self.valid_streak += 1
            if self.state in (TrackingState.WAITING, TrackingState.HOLDING):
                if self.valid_streak >= 5:
                    self.state = TrackingState.RECOVERING
                    self.previous_tick_ns = now_ns
                    self.converged_streak = 0
            if self.state in (TrackingState.RECOVERING, TrackingState.TRACKING):
                try:
                    target = self._bounded(self.retargeter.retarget(points))
                    if target is None:
                        raise ValueError("Retargeter returned invalid joints")
                    self.target = target
                    dt = max(0.0, min((now_ns - self.previous_tick_ns) * 1e-9, 0.040))
                    delta = -math.expm1(-dt / 0.060) * (target - self.command)
                    self.command = self.command + np.clip(delta, -self.max_rate * dt, self.max_rate * dt)
                    if np.max(np.abs(self.command - target)) <= self.tolerance:
                        self.converged_streak += 1
                    else:
                        self.converged_streak = 0
                    if self.converged_streak >= 5:
                        self.state = TrackingState.TRACKING
                except (ValueError, TypeError, RuntimeError, FloatingPointError, OverflowError, IndexError):
                    reason = HandReason.RETARGET_FAILED
                    self.state = TrackingState.HOLDING
                    self.valid_streak = self.converged_streak = 0
        self.previous_tick_ns = now_ns
        return HandOutput(
            _frozen(self.command),
            _frozen(self.target),
            self.state,
            reason,
            self.source_epoch,
            self.source_timestamp_ns,
            self.binding_generation,
            age_ns,
            reason == HandReason.OK,
        )
