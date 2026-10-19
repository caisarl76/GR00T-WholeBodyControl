"""Right-shoulder-yaw gravity feedforward for handover ready returns.

SONIC offsets most arm gravity in its actions but not right shoulder yaw. On the
raised waypoint that joint's sag uses the whole 0.15 rad reference lead, so the
raise from a low start can stall. In simulation, adding only that joint's static
gravity torque settled the raise. Adding it on every arm joint stacked on SONIC's
learned offsets and ran away.

The torque is computed from measured joints, at zero velocity, on the fixed-base
model with the pelvis upright. Pelvis tilt is ignored. This is unvalidated on
hardware.
"""
import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

from .ready_clearance import _finite

_YAW_PLANNER_INDEX = G1_UPPER_BODY_JOINT_INDICES.index(24)  # Motor 24: right shoulder yaw.


def right_yaw_gravity_tau(robot_model):
    """Return f(state) -> 17 planner-order torques in Nm, or None for invalid feedback.

    Only right shoulder yaw is nonzero. Use a model instance of its own:
    the gravity computation overwrites its kinematics cache.
    """
    dof = robot_model.dof_index("right_shoulder_yaw_joint")

    def tau(state):
        try:
            q = robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=_finite(state["body_q_measured_motor"], 29),
                left_hand_actuated_joint_values=_finite(state["left_hand_q_measured"], 7),
                right_hand_actuated_joint_values=_finite(state["right_hand_q_measured"], 7),
            )
        except (KeyError, TypeError, ValueError):
            return None
        # Measured joints can sit past URDF limits; clipping would misplace the arm.
        out = np.zeros(17)
        out[_YAW_PLANNER_INDEX] = robot_model.compute_gravity_compensation_torques(q, auto_clip=False)[dof]
        return out

    return tau
