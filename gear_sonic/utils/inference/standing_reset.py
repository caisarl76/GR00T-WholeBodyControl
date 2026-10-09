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
    body = _vector(state.get("body_q_measured_motor", state.get("body_q_measured", state.get("body_q"))), 29)
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
    Harness resets may compensate a steady upper-body tracking offset after
    reaching the nominal reference. Trim is capped at 0.15 rad; the same rate
    and measured-lead caps apply. The physical settling target never changes.
    """

    def __init__(
        self, state, left_hand_target, right_hand_target, *,
        compensate_tracking_bias=False, joint_tolerance_rad=0.05,
        right_arm_target=None,
    ):
        self._position = _measured_joints(state)
        self._compensate_tracking_bias = compensate_tracking_bias
        if not np.isfinite(joint_tolerance_rad) or joint_tolerance_rad <= 0:
            raise ValueError("Standing reset requires a positive finite joint tolerance")
        self._tracking_deadband = joint_tolerance_rad * 0.5
        self._tracking_trim = np.zeros(17)
        self._nominal_reached = False
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
        if right_arm_target is not None:
            # Upper-body motor order: waist 3, left arm 7, right arm 7.
            self._target[:10] = self._position[:10]
            self._target[10:17] = _vector(right_arm_target, 7)

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
        dt = np.clip(dt, 0.0, 0.05)
        step = 0.5 * dt
        reference = self._target.copy()
        if self._compensate_tracking_bias:
            self._nominal_reached |= bool(np.max(np.abs(self._position[:17] - self._target[:17])) < 1e-6)
            if self._nominal_reached:
                error = self._target[:17] - measured[:17]
                error = np.where(np.abs(error) > self._tracking_deadband, error, 0.0)
                self._tracking_trim = np.clip(self._tracking_trim + dt * error, -0.15, 0.15)
                reference[:17] += self._tracking_trim
        delta = np.clip(reference - self._position, -step, step)
        # Only move toward the target, without getting ahead of feedback.
        room = np.maximum(0.0, 0.15 - np.sign(delta) * (self._position - measured))
        self._position += np.sign(delta) * np.minimum(np.abs(delta), room)
        return self.command
