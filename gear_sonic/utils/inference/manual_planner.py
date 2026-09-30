"""Toggle keyboard movement translated into SONIC ZMQ planner commands."""

from __future__ import annotations

from dataclasses import replace
import math

from gear_sonic.utils.teleop.bridge_planner_publisher import PlannerCommand

SLOW_WALK_SPEED = 0.2


class ManualPlanner:
    def __init__(self, heading_yaw: float):
        self._check(heading_yaw)
        self._target = self._last = float(heading_yaw)
        self.active_key = None
        self._movement = (0.0, 0.0, 0.0)

    @staticmethod
    def _check(yaw):
        if not math.isfinite(yaw):
            raise ValueError("heading_yaw must be finite")

    def handle_key(self, key: str, heading_yaw: float) -> None:
        self._check(heading_yaw)
        if key not in {"w", "s", "a", "d", "q", "e", "z"}:
            return
        heading_yaw = float(heading_yaw)
        self._target = self._last = heading_yaw
        if key == "z" or key == self.active_key:
            self.active_key = None
            return
        self.active_key = key
        if key in {"w", "s", "a", "d"}:
            signs = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
            forward, lateral = signs[key]
            c, s = math.cos(heading_yaw), math.sin(heading_yaw)
            self._movement = (forward * c - lateral * s, forward * s + lateral * c, 0.0)

    def command(self, base: PlannerCommand, heading_yaw: float | None) -> PlannerCommand:
        if heading_yaw is None or not math.isfinite(heading_yaw):
            self._target, self.active_key = self._last, None
            return self._idle(base)
        self._last = float(heading_yaw)
        if self.active_key is None:
            return self._idle(base)
        if self.active_key in {"q", "e"}:
            # Keep a small heading target ahead of measured yaw while enabled.
            self._target = self._last + (1 if self.active_key == "q" else -1) * math.radians(5)
            return self._idle(base)
        return replace(
            base, movement=self._movement, facing=self._face(self._target), mode=1, speed=SLOW_WALK_SPEED
        )

    def _idle(self, base):
        return replace(base, movement=(0.0, 0.0, 0.0), facing=self._face(self._target), mode=0, speed=0.0)

    @staticmethod
    def _face(yaw):
        return (math.cos(yaw), math.sin(yaw), 0.0)
