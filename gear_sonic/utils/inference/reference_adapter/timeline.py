"""Offline timeline validation, interpolation, and action freshness primitives."""

from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np


def validate_timestamps(times: Any) -> np.ndarray:
    if isinstance(times, (str, bytes)):
        raise ValueError("timestamps must be numeric")
    raw = np.asarray(times)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("timestamps must be a nonempty one-dimensional numeric array")
    if raw.dtype.kind in {"b", "c", "O", "U", "S"}:
        raise ValueError("timestamps must be real numeric values")
    try:
        result = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be numeric") from exc
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def resample_trajectory(
    source_times: Any, values: Any, target_times: Any, *, tail_policy: str = "reject"
) -> np.ndarray:
    source = validate_timestamps(source_times)
    target = validate_timestamps(target_times)
    if tail_policy not in {"reject", "hold"}:
        raise ValueError("tail_policy must be 'reject' or 'hold'")
    raw_values = np.asarray(values)
    if raw_values.dtype.kind in {"b", "c", "O", "U", "S"}:
        raise ValueError("values must be finite with shape (N, D)")
    try:
        array = raw_values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be finite with shape (N, D)") from exc
    if array.ndim != 2 or array.shape[0] != source.size or array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite with shape (N, D)")
    if target.size and target[0] < source[0]:
        raise ValueError("target precedes source")
    if target.size and target[-1] > source[-1] and tail_policy == "reject":
        raise ValueError("target exceeds source")
    clipped = np.minimum(target, source[-1])
    return np.column_stack([np.interp(clipped, source, array[:, col]) for col in range(array.shape[1])])


def trajectory_velocities(times: Any, positions: Any, *, held_from: float | None = None) -> np.ndarray:
    """Differentiate final positions, optionally zeroing a declared constant tail.

    Pass the original source end time when resampling with ``tail_policy='hold'``.
    Finite differences alone cannot identify which samples were extrapolated.
    """
    t = validate_timestamps(times)
    raw_positions = np.asarray(positions)
    if raw_positions.dtype.kind in {"b", "c", "O", "U", "S"}:
        raise ValueError("positions must be finite with shape (N, D)")
    try:
        p = raw_positions.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be finite with shape (N, D)") from exc
    if p.ndim != 2 or p.shape[0] != t.size or p.shape[1] == 0 or not np.all(np.isfinite(p)):
        raise ValueError("positions must be finite with shape (N, D)")
    velocities = np.zeros_like(p) if t.size == 1 else np.gradient(p, t, axis=0, edge_order=1)
    if held_from is not None:
        held = t >= _finite_float(held_from, "held_from")
        if np.any(held):
            if not np.all(p[held] == p[held][0]):
                raise ValueError("declared held tail must have constant positions")
            velocities[held] = 0
    return velocities


def select_execution_index(times: Any, now: Any, *, valid_until: Any) -> int | None:
    t = validate_timestamps(times)
    current = _finite_float(now, "now")
    expiry = _finite_float(valid_until, "valid_until")
    if expiry <= t[-1]:
        raise ValueError("valid_until must be greater than the final timestamp")
    if current < t[0] or current >= expiry:
        return None
    index = int(np.searchsorted(t, current, side="right") - 1)
    return index if index >= 0 else None


@dataclass
class InputLease:
    max_action_age_seconds: float
    receipt_timeout_seconds: float

    def __post_init__(self) -> None:
        self.max_action_age_seconds = _finite_float(self.max_action_age_seconds, "max_action_age_seconds")
        self.receipt_timeout_seconds = _finite_float(self.receipt_timeout_seconds, "receipt_timeout_seconds")
        if self.max_action_age_seconds <= 0 or self.receipt_timeout_seconds <= 0:
            raise ValueError("lease limits must be positive")
        self.reset()

    def reset(self) -> None:
        self._sequence = None
        self._receipt = None
        self._observation = None
        self._deadline = None

    def accept(
        self, sequence_id: Any, *, received_at: Any, observation_time: Any, execution_deadline: Any
    ) -> bool:
        if (
            isinstance(sequence_id, (bool, np.bool_))
            or not isinstance(sequence_id, numbers.Integral)
            or sequence_id < 0
        ):
            raise ValueError("sequence_id must be a nonnegative integer")
        receipt = _finite_float(received_at, "received_at")
        observation = _finite_float(observation_time, "observation_time")
        deadline = _finite_float(execution_deadline, "execution_deadline")
        if self._receipt is not None and receipt < self._receipt:
            return False
        if self._sequence is not None and sequence_id <= self._sequence:
            return False
        if observation > receipt or receipt - observation > self.max_action_age_seconds or deadline <= receipt:
            return False
        self._sequence = sequence_id
        self._receipt, self._observation, self._deadline = receipt, observation, deadline
        return True

    def is_fresh(self, now: Any) -> bool:
        current = _finite_float(now, "now")
        return (
            self._receipt is not None
            and current >= self._receipt
            and current - self._receipt < self.receipt_timeout_seconds
            and current - self._observation <= self.max_action_age_seconds
            and current < self._deadline
        )
