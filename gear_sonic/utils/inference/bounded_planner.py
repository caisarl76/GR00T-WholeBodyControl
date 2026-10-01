"""Small, feedback-paced PLANNER motions preserving measured arms and hands."""

from dataclasses import replace
import math
from numbers import Real

import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import feedback_payload_heading_yaw

from .harness_profile import HarnessLimits
from .standing_reset import StandingReset


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def positive(value, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError("Motion exceeds finite positive bounds")


class BoundedPlanner:
    def __init__(self, start_feedback, started_at, limits=None):
        self.limits = limits or HarnessLimits()
        self.yaw = feedback_payload_heading_yaw(start_feedback)
        if self.yaw is None:
            raise ValueError("Measured heading required")
        self.command = StandingReset(
            start_feedback,
            start_feedback.get("left_hand_q_measured", start_feedback.get("left_hand_q")),
            start_feedback.get("right_hand_q_measured", start_feedback.get("right_hand_q")),
        ).command
        self.started_at = self.last_at = started_at
        self.finished, self.kind, self.dwell, self.last_index = False, None, None, None

    def begin_walk(self, direction, duration_s, speed_mps):
        positive(duration_s, self.limits.walk_max_duration_s)
        positive(speed_mps, self.limits.walk_max_speed_mps)
        directions = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
        if direction not in directions:
            raise ValueError("Unsupported walking direction")
        x, y = directions[direction]
        self.movement = (
            math.cos(self.yaw) * x - math.sin(self.yaw) * y,
            math.sin(self.yaw) * x + math.cos(self.yaw) * y,
            0.0,
        )
        self.kind, self.deadline, self.speed = "walk", self.started_at + duration_s, speed_mps

    def begin_turn(self, angle_rad, rate_rps):
        if isinstance(angle_rad, bool) or not isinstance(angle_rad, Real) or not math.isfinite(angle_rad):
            raise ValueError("Finite angle required")
        positive(abs(angle_rad), self.limits.turn_max_angle_rad)
        positive(rate_rps, self.limits.turn_max_rate_rps)
        self.kind, self.goal, self.rate = "turn", wrap(self.yaw + angle_rad), rate_rps
        self.deadline = self.started_at + self.limits.turn_deadline_s

    def advance(self, feedback, now):
        if self.kind is None or feedback is None or not math.isfinite(now) or now < self.last_at:
            raise ValueError("Fresh feedback and monotonic time required")
        measured = feedback_payload_heading_yaw(feedback)
        if measured is None:
            raise ValueError("Measured heading required")
        dt = np.clip(now - self.last_at, 0.0, 0.05)
        self.last_at = now
        if self.finished:
            return self.command
        if self.kind == "walk":
            self.finished = now >= self.deadline
            self.command = replace(
                self.command,
                mode=0 if self.finished else 1,
                movement=(0.0, 0.0, 0.0) if self.finished else self.movement,
                speed=0.0 if self.finished else self.speed,
            )
            return self.command
        if now > self.deadline:
            raise ValueError("Turn deadline exceeded")
        delta = float(np.clip(wrap(self.goal - self.yaw), -self.rate * dt, self.rate * dt))
        lead = wrap(self.yaw - measured)
        room = max(0.0, self.limits.turn_lead_rad - np.sign(delta) * lead)
        self.yaw = wrap(self.yaw + np.sign(delta) * min(abs(delta), room))
        index = feedback.get("index")
        if type(index) is int and (self.last_index is None or index > self.last_index):
            self.last_index = index
            if abs(wrap(self.goal - measured)) <= self.limits.turn_tolerance_rad:
                self.dwell = now if self.dwell is None else self.dwell
                self.finished = now - self.dwell >= self.limits.settle_dwell_s
            else:
                self.dwell = None
        self.command = replace(
            self.command,
            mode=0,
            movement=(0.0, 0.0, 0.0),
            speed=0.0,
            facing=(math.cos(self.yaw), math.sin(self.yaw), 0.0),
        )
        return self.command
