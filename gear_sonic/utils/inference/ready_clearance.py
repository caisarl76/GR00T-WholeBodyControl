"""Decide whether the handover ready return can skip its raised waypoint.

The raise exists to lift a low hand over the desk edge. From an offer pose the
hand is already beside or above the desk, and raising can drive it into the
desk instead. Skip the raise when either:
- the right wrist and hand links all start at or above the desk top, so there is
  nothing to lift over. At a close desk, simulation showed the raise lowering the
  fingers onto it while the direct return cleared; or
- forward kinematics shows the right elbow, wrist and hand staying at least
  DESK_MARGIN_M from the desk along the whole direct StandingReset reference:
  right arm to its target and right hand opening, each joint at 0.5 rad/s,
  everything else held.

Frames are the fixed-base model's pelvis frame with neutral orientation. The
desk is the region x >= front, z <= top, unbounded sideways, behind and below.
Each link is bounded by the world box around its collision-geometry box, so
the gap is a lower bound. It assumes ideal tracking and does not certify the
measured path, people or calibration of the configured desk.
"""
from itertools import product

import numpy as np

DESK_MARGIN_M = 0.03
_RATE_RAD_S = 0.5  # StandingReset reference rate.
_STEP_S = 0.02

# Body-frame boxes (min, max) around each link's collision geometry in
# g1_29dof_with_hand.xml, rounded outward to 0.1 mm; test_ready_clearance checks them.
RIGHT_ARM_LINK_BOXES = {
    "right_elbow_link": ((-0.0298, -0.0348, -0.0455), (0.1001, 0.0313, 0.0298)),
    "right_wrist_roll_link": ((-0.0001, -0.03, -0.03), (0.056, 0.0305, 0.03)),
    "right_wrist_pitch_link": ((-0.0286, -0.0285, -0.0285), (0.0746, 0.0285, 0.0285)),
    "right_wrist_yaw_link": ((-0.018, -0.03, -0.0439), (0.1283, 0.0239, 0.0439)),
    "right_hand_thumb_0_link": ((-0.0116, -0.0001, -0.0095), (0.0101, 0.0284, 0.0095)),
    "right_hand_thumb_1_link": ((-0.011, 0.017, -0.01), (0.0091, 0.047, 0.01)),
    "right_hand_thumb_2_link": ((-0.009, -0.0071, -0.0129), (0.0088, 0.052, 0.0132)),
    "right_hand_middle_0_link": ((-0.0071, -0.0127, -0.013), (0.0549, 0.0147, 0.0129)),
    "right_hand_middle_1_link": ((-0.0071, -0.009, -0.0132), (0.052, 0.0088, 0.0129)),
    "right_hand_index_0_link": ((-0.0071, -0.0127, -0.013), (0.0549, 0.0147, 0.0129)),
    "right_hand_index_1_link": ((-0.0071, -0.009, -0.0132), (0.052, 0.0088, 0.0129)),
}
_CORNERS = {name: np.array(list(product(*zip(low, high)))) for name, (low, high) in RIGHT_ARM_LINK_BOXES.items()}
_HAND_LINKS = [i for i, name in enumerate(RIGHT_ARM_LINK_BOXES) if name != "right_elbow_link"]


def right_arm_link_boxes(robot_model):
    """Return f(body_q, right_hand_q) -> world boxes of the right arm links.

    Use a model instance of its own: forward kinematics overwrites its cache.
    """
    def boxes(body_q, right_hand_q):
        q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q, right_hand_actuated_joint_values=right_hand_q)
        robot_model.cache_forward_kinematics(q)
        out = []
        for name, corners in _CORNERS.items():
            placement = robot_model.frame_placement(name)
            world = corners @ placement.rotation.T + placement.translation
            out.append((world.min(0), world.max(0)))
        return out

    return boxes


def desk_gap(boxes, desk_top, desk_front):
    """Smallest separation between the boxes and the desk region; negative overlaps."""
    gaps = []
    for low, high in boxes:
        ahead, above = desk_front - high[0], low[2] - desk_top
        gaps.append(np.hypot(ahead, above) if ahead > 0 and above > 0 else max(ahead, above))
    return float(min(gaps))


def _finite(value, size):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Requires {size} finite joint values")
    return array


def direct_return_desk_gap(link_boxes, state, right_arm_target, desk_top, desk_front):
    """Smallest desk gap along the direct ready-return reference from the measured state."""
    body = _finite(state["body_q_measured_motor"], 29)
    hand = _finite(state["right_hand_q_measured"], 7)
    target = _finite(right_arm_target, 7)
    move = np.r_[target - body[22:29], -hand]
    gap, t = np.inf, 0.0
    while True:
        step = np.clip(move, -_RATE_RAD_S * t, _RATE_RAD_S * t)
        sample = body.copy()
        sample[22:29] += step[:7]
        gap = min(gap, desk_gap(link_boxes(sample, hand + step[7:]), desk_top, desk_front))
        if np.all(np.abs(step) >= np.abs(move)):
            return gap
        t += _STEP_S


def hand_above_desk_top(link_boxes, state, desk_top):
    """Whether every right wrist and hand link box starts at or above the desk top."""
    boxes = link_boxes(_finite(state["body_q_measured_motor"], 29), _finite(state["right_hand_q_measured"], 7))
    return all(boxes[i][0][2] >= desk_top for i in _HAND_LINKS)


def ready_waypoint(handover, state, link_boxes):
    """The profile's raised waypoint, or None when the direct path keeps the desk margin."""
    if handover.desk_top_above_pelvis_m is None:
        return handover.right_arm_waypoint
    try:
        if hand_above_desk_top(link_boxes, state, handover.desk_top_above_pelvis_m):
            return None
        gap = direct_return_desk_gap(link_boxes, state, handover.right_arm_target,
                                     handover.desk_top_above_pelvis_m, handover.desk_front_from_pelvis_m)
    except (KeyError, TypeError, ValueError):
        return handover.right_arm_waypoint  # Unknown entry geometry: keep the raise.
    return None if gap >= DESK_MARGIN_M else handover.right_arm_waypoint

