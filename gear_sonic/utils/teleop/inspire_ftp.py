"""Hardware-compatible command contract for Unitree G1 Inspire FTP hands.

The FTP driver exposes six normalized motors per hand.  This module is kept
free of DDS, XRoboToolkit, and MuJoCo dependencies so every backend shares the
same order and open/closed convention.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

MOTOR_NAMES = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)

# Active-joint upper limits from Unitree's pinned G1 Inspire FTP URDF.  These
# are physical model radians, not the intermediary ranges used by the XR hand
# retargeter before it normalizes its output.
CLOSED_RADIANS: NDArray[np.float64] = np.array([1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641], dtype=np.float64)
OPEN: NDArray[np.float64] = np.ones(6, dtype=np.float64)


def _as_six_finite(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{label} must have shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array.copy()


def validate_normalized(values: ArrayLike) -> NDArray[np.float64]:
    """Return a validated normalized command without silently clipping it."""

    array = _as_six_finite(values, label="normalized Inspire command")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("normalized Inspire command must lie within [0, 1]")
    return array


def normalized_to_radians(values: ArrayLike) -> NDArray[np.float64]:
    """Map FTP normalized values (one=open, zero=closed) to model radians."""

    normalized = validate_normalized(values)
    return (1.0 - normalized) * CLOSED_RADIANS


def radians_to_normalized(values: ArrayLike) -> NDArray[np.float64]:
    """Map the six active model joint angles back to FTP normalized values."""

    radians = _as_six_finite(values, label="Inspire active-joint radians")
    if np.any(radians < 0.0) or np.any(radians > CLOSED_RADIANS):
        raise ValueError("Inspire active-joint radians exceed the FTP model limits")
    return 1.0 - radians / CLOSED_RADIANS


def normalized_to_ftp_counts(values: ArrayLike) -> NDArray[np.int32]:
    """Scale normalized commands to the FTP driver's inclusive 0..1000 range."""

    normalized = validate_normalized(values)
    return np.rint(normalized * 1000.0).astype(np.int32)


def _calibrate_close_signal(value: float, *, deadzone: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
        raise ValueError("PICO close signals must be finite and lie within [0, 1]")
    if deadzone < 0.0 or deadzone >= 0.5:
        raise ValueError("PICO endpoint deadzone must lie within [0, 0.5)")
    if scalar <= deadzone:
        return 0.0
    if scalar >= 1.0 - deadzone:
        return 1.0
    return (scalar - deadzone) / (1.0 - 2.0 * deadzone)


def map_pico_controls(
    trigger_close: float,
    grip_close: float,
    *,
    deadzone: float = 0.05,
) -> NDArray[np.float64]:
    """Map PICO trigger/grip close signals to the six FTP motor commands.

    Trigger closes the four fingers and bends the thumb. Grip independently
    controls thumb rotation. The output preserves the FTP convention where
    one is open and zero is closed.
    """

    trigger = _calibrate_close_signal(trigger_close, deadzone=deadzone)
    grip = _calibrate_close_signal(grip_close, deadzone=deadzone)
    return np.array([1.0 - trigger] * 5 + [1.0 - grip], dtype=np.float64)


def validate_hand_pair(
    left: Sequence[float] | NDArray[np.floating],
    right: Sequence[float] | NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate left and right commands atomically for receiver backends."""

    return validate_normalized(left), validate_normalized(right)
