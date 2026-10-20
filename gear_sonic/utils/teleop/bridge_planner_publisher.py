from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import struct

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message

HAND_DOF = 7
MAX_HAND_JOINT_STEP_RAD = 0.25


class PlannerPreparationError(ValueError):
    """The planner command could not be validated or encoded."""


@dataclass(frozen=True)
class PlannerCommand:
    mode: int
    movement: Sequence[float]
    facing: Sequence[float]
    speed: float
    height: float
    upper_body_position: Sequence[float]
    upper_body_velocity: Sequence[float]
    left_hand_position: Sequence[float] | None = None
    right_hand_position: Sequence[float] | None = None

    def encode(self) -> bytes:
        try:
            return build_planner_message(
                self.mode,
                self.movement,
                self.facing,
                speed=self.speed,
                height=self.height,
                upper_body_position=self.upper_body_position,
                upper_body_velocity=self.upper_body_velocity,
                left_hand_position=self.left_hand_position,
                right_hand_position=self.right_hand_position,
            )
        except (TypeError, ValueError, OverflowError, struct.error) as exc:
            raise PlannerPreparationError(f"invalid planner command: {exc}") from exc


@dataclass(frozen=True)
class HandCommandPreview:
    left: tuple[float, ...] | None
    right: tuple[float, ...] | None


@dataclass(frozen=True)
class PreparedPlannerPublication:
    message: bytes
    hand_preview: HandCommandPreview


class HandCommandLimiter:
    def __init__(self, max_step: float = MAX_HAND_JOINT_STEP_RAD):
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be finite and positive")
        self._max_step = float(max_step)
        self._last = {
            "left": np.zeros(HAND_DOF, dtype=np.float64),
            "right": np.zeros(HAND_DOF, dtype=np.float64),
        }

    @staticmethod
    def _validated(value, side: str) -> np.ndarray | None:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlannerPreparationError(f"invalid {side} hand command") from exc
        if array.shape != (HAND_DOF,) or not np.all(np.isfinite(array)):
            raise PlannerPreparationError(f"invalid {side} hand command")
        return array

    def preview(self, left, right, *, pause: bool = False) -> HandCommandPreview:
        desired_left = self._validated(left, "left")
        desired_right = self._validated(right, "right")

        def one(side: str, desired: np.ndarray | None) -> tuple[float, ...] | None:
            if desired is None:
                return None
            current = self._last[side]
            if pause:
                emitted = current
            else:
                emitted = current + np.clip(
                    desired - current,
                    -self._max_step,
                    self._max_step,
                )
            return tuple(float(value) for value in emitted)

        return HandCommandPreview(
            left=one("left", desired_left),
            right=one("right", desired_right),
        )

    def commit(self, preview: HandCommandPreview) -> None:
        next_state = {side: value.copy() for side, value in self._last.items()}
        if preview.left is not None:
            next_state["left"] = np.asarray(preview.left, dtype=np.float64).copy()
        if preview.right is not None:
            next_state["right"] = np.asarray(preview.right, dtype=np.float64).copy()
        self._last = next_state

    def last_emitted(self, side: str) -> np.ndarray:
        if side not in self._last:
            raise ValueError(f"unsupported hand side: {side}")
        return self._last[side].copy()


class BridgePlannerPublisher:
    def __init__(self, socket, limiter: HandCommandLimiter | None = None):
        self._socket = socket
        self._limiter = limiter or HandCommandLimiter()

    def prepare(
        self,
        command: PlannerCommand,
        *,
        ramp_phase: str,
    ) -> PreparedPlannerPublication:
        preview = self._limiter.preview(
            command.left_hand_position,
            command.right_hand_position,
            pause=ramp_phase == "pause",
        )
        limited = replace(
            command,
            left_hand_position=preview.left,
            right_hand_position=preview.right,
        )
        return PreparedPlannerPublication(
            message=limited.encode(),
            hand_preview=preview,
        )

    def send(self, prepared: PreparedPlannerPublication) -> bytes:
        result = self._socket.send(prepared.message)
        if result is False:
            raise RuntimeError("planner socket rejected message")
        self._limiter.commit(prepared.hand_preview)
        return prepared.message

    def publish(self, command: PlannerCommand, *, ramp_phase: str) -> bytes:
        return self.send(self.prepare(command, ramp_phase=ramp_phase))

    def last_emitted(self, side: str) -> np.ndarray:
        return self._limiter.last_emitted(side)
