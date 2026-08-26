"""Safe six-motor Inspire FTP command reception for MuJoCo simulation."""

from __future__ import annotations

from collections.abc import Sequence
import time

import numpy as np
from numpy.typing import NDArray
import zmq

from gear_sonic.utils.teleop.inspire_ftp import (
    CLOSED_RADIANS,
    OPEN,
    validate_hand_pair,
)
from gear_sonic.utils.teleop.zmq.zmq_message_decoder import unpack_pose_message

DEFAULT_MAX_OPEN_SPEED = 1.0 / CLOSED_RADIANS


class InspireCommandState:
    """Latest valid hand pair with a gradual open-on-stale watchdog."""

    def __init__(
        self,
        *,
        stale_after_s: float = 0.25,
        max_open_speed: Sequence[float] | NDArray[np.floating] = DEFAULT_MAX_OPEN_SPEED,
    ) -> None:
        if not np.isfinite(stale_after_s) or stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be finite and positive")
        speed = np.asarray(max_open_speed, dtype=np.float64)
        if speed.shape != (6,) or not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
            raise ValueError("max_open_speed must contain six finite positive values")

        self.stale_after_s = float(stale_after_s)
        self.max_open_speed = speed.copy()
        self.last_valid: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self.last_receive_monotonic: float | None = None
        self.output = (OPEN.copy(), OPEN.copy())

    def accept(
        self,
        left: Sequence[float] | NDArray[np.floating],
        right: Sequence[float] | NDArray[np.floating],
        *,
        now: float,
    ) -> None:
        """Atomically replace the latest pair after both hands validate."""

        timestamp = float(now)
        if not np.isfinite(timestamp):
            raise ValueError("receive timestamp must be finite")
        validated = validate_hand_pair(left, right)
        self.last_valid = validated
        self.last_receive_monotonic = timestamp

    def advance(self, *, now: float, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the watchdog and return isolated left/right command copies."""

        timestamp = float(now)
        step = float(dt)
        if not np.isfinite(timestamp):
            raise ValueError("current timestamp must be finite")
        if not np.isfinite(step) or step < 0.0:
            raise ValueError("dt must be finite and non-negative")

        if self.last_valid is None or self.last_receive_monotonic is None:
            self.output = (OPEN.copy(), OPEN.copy())
        elif timestamp - self.last_receive_monotonic <= self.stale_after_s:
            self.output = tuple(command.copy() for command in self.last_valid)
        else:
            delta = self.max_open_speed * step
            self.output = tuple(np.minimum(command + delta, OPEN) for command in self.output)
        return tuple(command.copy() for command in self.output)


class InspireFtpZmqSubscriber:
    """Non-publishing subscriber that drains pose/planner commands to latest."""

    TOPICS = ("pose", "planner")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        *,
        state: InspireCommandState | None = None,
    ) -> None:
        self.state = state or InspireCommandState()
        self._context: zmq.Context | None = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        for topic in self.TOPICS:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.connect(f"tcp://{host}:{port}")

    @classmethod
    def from_socket(
        cls,
        socket,
        *,
        state: InspireCommandState | None = None,
    ) -> InspireFtpZmqSubscriber:
        """Construct around an already configured socket, primarily for tests."""

        instance = cls.__new__(cls)
        instance.state = state or InspireCommandState()
        instance._context = None
        instance._socket = socket
        return instance

    @staticmethod
    def _decode_hand_pair(
        message: bytes,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if message.startswith(b"pose"):
            topic = "pose"
        elif message.startswith(b"planner"):
            topic = "planner"
        else:
            raise ValueError("message is not on a subscribed hand-command topic")
        decoded = unpack_pose_message(message, topic=topic)
        if "left_hand_joints" not in decoded or "right_hand_joints" not in decoded:
            raise ValueError("message must contain both Inspire hand fields")
        return validate_hand_pair(decoded["left_hand_joints"], decoded["right_hand_joints"])

    def poll(self, *, now: float | None = None) -> bool:
        """Drain queued messages and atomically accept only the latest valid pair."""

        latest = None
        while True:
            try:
                message = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                latest = self._decode_hand_pair(message)
            except (TypeError, ValueError):
                continue

        if latest is None:
            return False
        timestamp = time.monotonic() if now is None else now
        self.state.accept(*latest, now=timestamp)
        return True

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self) -> InspireFtpZmqSubscriber:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
