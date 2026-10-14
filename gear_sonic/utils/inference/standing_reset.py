"""Feedback-paced return to SONIC's existing straight standing planner pose."""

from dataclasses import replace
import numbers

import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_UPPER_BODY_JOINT_INDICES,
    UpperBodyFrame,
    anchor_frame_planner_heading,
    build_stop_standing_command,
    feedback_payload_heading_yaw,
)

# Planner order interleaves the arms; body motor indices 22..28 are the right arm.
_RIGHT_ARM_PLANNER_INDICES = [
    G1_UPPER_BODY_JOINT_INDICES.index(motor_index) for motor_index in range(22, 29)
]


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

    Handover returns can first raise to a reviewed right-arm waypoint, retaining
    entry fingers until fresh upper-body and heading measurements settle. The
    final target then opens the hand. Bounds do not certify obstacle clearance.
    """

    def __init__(
        self, state, left_hand_target, right_hand_target, *,
        compensate_tracking_bias=False, joint_tolerance_rad=0.05,
        right_arm_target=None, right_arm_waypoint=None,
        yaw_tolerance_rad=np.pi / 36, settle_dwell_s=0.5,
    ):
        self._position = _measured_joints(state)
        from .harness_profile import RIGHT_ARM_BOUNDS

        self._right_arm_bounds = np.asarray(list(RIGHT_ARM_BOUNDS.values())).T
        low, high = self._right_arm_bounds
        measured_arm = self._position[_RIGHT_ARM_PLANNER_INDICES]
        if np.any(measured_arm < low) or np.any(measured_arm > high):
            raise ValueError("Measured right-arm exceeds G1 joint limits")
        self._compensate_tracking_bias = compensate_tracking_bias
        if not np.isfinite(joint_tolerance_rad) or joint_tolerance_rad <= 0:
            raise ValueError("Standing reset requires a positive finite joint tolerance")
        self._tracking_deadband = joint_tolerance_rad * 0.5
        self._joint_tolerance = joint_tolerance_rad
        if (not np.isfinite(yaw_tolerance_rad) or yaw_tolerance_rad <= 0
                or not np.isfinite(settle_dwell_s) or settle_dwell_s <= 0):
            raise ValueError("Ready reset requires positive finite heading tolerance and dwell")
        self._yaw_tolerance = yaw_tolerance_rad
        self._settle_dwell = settle_dwell_s
        self._lift_dwell = None
        self._last_lift_index = None
        self.stage = 1
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
            self._target[:17] = self._position[:17]
            self._target[_RIGHT_ARM_PLANNER_INDICES] = _vector(right_arm_target, 7)
            if (np.any(self._target[_RIGHT_ARM_PLANNER_INDICES] < low)
                    or np.any(self._target[_RIGHT_ARM_PLANNER_INDICES] > high)):
                raise ValueError("Ready target exceeds G1 right-arm joint limits")
        self._final_target = self._target.copy()
        if right_arm_waypoint is not None:
            if right_arm_target is None:
                raise ValueError("Raised waypoint requires a final right-arm target")
            waypoint = _vector(right_arm_waypoint, 7)
            low, high = self._right_arm_bounds
            if np.any(waypoint < low) or np.any(waypoint > high):
                raise ValueError("Raised waypoint exceeds G1 right-arm joint limits")
            self._target[_RIGHT_ARM_PLANNER_INDICES] = waypoint
            # Retain entry fingers while near the desk; open only after raising.
            self._target[24:] = self._position[24:]
            self.stage = 0

    @property
    def target(self):
        return self._final_target.copy()

    def is_settled(self, state, joint_tolerance, yaw_tolerance_rad):
        if self.stage == 0:
            return False
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
        if (not isinstance(state, dict) or isinstance(dt, bool)
                or not isinstance(dt, numbers.Real) or not np.isfinite(dt) or dt <= 0):
            self._lift_dwell = None
            return self.command
        try:
            measured = _measured_joints(state)
        except (TypeError, ValueError):
            self._lift_dwell = None
            return self.command
        if self.stage == 0:
            index = state.get("index")
            if (isinstance(index, bool) or not isinstance(index, numbers.Integral) or index < 0
                    or (self._last_lift_index is not None and index < self._last_lift_index)):
                self._lift_dwell = None
                return self.command
            if index == self._last_lift_index:
                return self.command
            self._last_lift_index = index
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
        if self._right_arm_bounds is not None:
            reference[_RIGHT_ARM_PLANNER_INDICES] = np.clip(
                reference[_RIGHT_ARM_PLANNER_INDICES], *self._right_arm_bounds
            )
        # A measured joint can fall back after reaching a reference. Retract
        # toward the feasible interval rather than freezing an excessive lead.
        lower, upper = measured - 0.15, measured + 0.15
        lower[_RIGHT_ARM_PLANNER_INDICES] = np.maximum(lower[_RIGHT_ARM_PLANNER_INDICES], self._right_arm_bounds[0])
        upper[_RIGHT_ARM_PLANNER_INDICES] = np.minimum(upper[_RIGHT_ARM_PLANNER_INDICES], self._right_arm_bounds[1])
        if np.any(lower > upper):
            raise ValueError("Joint feedback cannot satisfy G1 joint and lead bounds")
        bounded_reference = np.clip(reference, lower, upper)
        candidate = self._position + np.clip(bounded_reference - self._position, -step, step)
        if np.max(np.abs(candidate - measured)) > 0.15 + 1e-8:
            raise ValueError("Joint feedback moved beyond the rate-limited lead bound")
        arm = candidate[_RIGHT_ARM_PLANNER_INDICES]
        if np.any(arm < self._right_arm_bounds[0]) or np.any(arm > self._right_arm_bounds[1]):
            raise ValueError("Joint feedback cannot satisfy G1 joint and lead bounds")
        self._position = candidate
        if self.stage == 0:
            yaw = feedback_payload_heading_yaw(state)
            yaw_error = None if yaw is None else np.arctan2(np.sin(yaw - self._yaw), np.cos(yaw - self._yaw))
            # Intermediate settling covers the preserved upper body and heading.
            # Final settling still requires all arm and finger measurements.
            raised = (yaw_error is not None and abs(yaw_error) <= self._yaw_tolerance
                      and np.max(np.abs(measured[:17] - self._target[:17])) <= self._joint_tolerance)
            if raised:
                self._lift_dwell = 0.0 if self._lift_dwell is None else self._lift_dwell + dt
            else:
                self._lift_dwell = None
            if self._lift_dwell is not None and self._lift_dwell >= self._settle_dwell:
                self.stage = 1
                self._target = self._final_target.copy()
                self._tracking_trim[:] = 0
                self._nominal_reached = False
        return self.command
