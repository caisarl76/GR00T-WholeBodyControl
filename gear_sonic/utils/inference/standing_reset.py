"""Feedback-paced return to SONIC's existing straight standing planner pose."""

from dataclasses import replace

import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_UPPER_BODY_JOINT_INDICES,
    UpperBodyFrame,
    anchor_frame_planner_heading,
    build_stop_standing_command,
    feedback_payload_heading_yaw,
)


def _vector(value, size):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Standing reset requires {size} finite joint values")
    return array.copy()


def _measured_joints(state):
    body = _vector(state.get("body_q_measured", state.get("body_q")), 29)
    return np.concatenate(
        [
            body[G1_UPPER_BODY_JOINT_INDICES],
            _vector(state.get("left_hand_q_measured", state.get("left_hand_q")), 7),
            _vector(state.get("right_hand_q_measured", state.get("right_hand_q")), 7),
        ]
    )


class StandingReset:
    """Hold first, then approach standing at <=0.5 rad/s per joint.

    Limit the commanded lead over measured joints to 0.15 rad. In particular,
    waiting for SONIC's planner to initialize cannot advance a timed ramp all
    the way to its endpoint before the robot has started following it.
    Missing/invalid feedback holds the last command; it never advances the ramp.
    """

    def __init__(self, state, left_hand_target, right_hand_target):
        self._position = _measured_joints(state)
        yaw = feedback_payload_heading_yaw(state)
        if yaw is None:
            raise ValueError("Standing reset requires a valid measured heading")
        self._yaw = yaw
        frame = anchor_frame_planner_heading(UpperBodyFrame(self._position[:17], np.zeros(17)), yaw)
        self._standing = build_stop_standing_command(frame, hand_preset="open")
        self._target = np.concatenate(
            [
                self._standing.upper_body_position,
                _vector(left_hand_target, 7),
                _vector(right_hand_target, 7),
            ]
        )

    @property
    def target(self):
        return self._target.copy()

    def is_settled(self, state, joint_tolerance, yaw_tolerance_rad):
        try:
            measured = _measured_joints(state)
            yaw = feedback_payload_heading_yaw(state)
            if yaw is None:
                return False
            yaw_error = np.arctan2(np.sin(yaw - self._yaw), np.cos(yaw - self._yaw))
            return bool(
                np.max(np.abs(measured - self._target)) <= joint_tolerance and abs(yaw_error) <= yaw_tolerance_rad
            )
        except (TypeError, ValueError):
            return False

    @property
    def command(self):
        return replace(
            self._standing,
            upper_body_position=self._position[:17].tolist(),
            left_hand_position=self._position[17:24].tolist(),
            right_hand_position=self._position[24:].tolist(),
        )

    def advance(self, state, dt):
        if state is None:
            return self.command
        try:
            measured = _measured_joints(state)
        except (TypeError, ValueError):
            return self.command
        step = 0.5 * np.clip(dt, 0.0, 0.05)
        delta = np.clip(self._target - self._position, -step, step)
        # Only move toward the target, without getting ahead of feedback.
        room = np.maximum(0.0, 0.15 - np.sign(delta) * (self._position - measured))
        self._position += np.sign(delta) * np.minimum(np.abs(delta), room)
        return self.command
