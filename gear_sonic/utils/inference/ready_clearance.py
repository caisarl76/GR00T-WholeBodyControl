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

Geometry is in the level pelvis frame: pelvis origin, z against gravity, x along
the pelvis heading. Pelvis roll and pitch come from the measured base_quat. The
desk is the region x >= front, z <= top, unbounded sideways, behind and below.
The desk top is configured above the floor. The pelvis height is estimated at
each decision from the measured legs, assuming both soles flat on the floor,
because SONIC's knee bend changes it by centimetres. Each link is bounded by the
level-frame box around its collision-geometry box, so the gap is a lower bound.
Measured joints are not clipped to URDF limits. The check assumes ideal tracking
and holds the measured tilt along the path. It does not certify the measured
path, people or calibration of the configured desk.
"""
from itertools import product

import numpy as np

DESK_MARGIN_M = 0.03
# Foot collision box bottom below the ankle-roll frame in g1_29dof_with_hand.xml.
ANKLE_TO_SOLE_M = 0.035
FEET_LEVEL_M = 0.02  # Larger ankle height difference: a foot is not flat on the floor.
PELVIS_HEIGHT_RANGE_M = (0.65, 0.85)
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
    """Return f(body_q, right_hand_q, tilt=I) -> level-frame boxes of the right arm links.

    Use a model instance of its own: forward kinematics overwrites its cache.
    """
    def boxes(body_q, right_hand_q, tilt=np.eye(3)):
        q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q, right_hand_actuated_joint_values=right_hand_q)
        robot_model.cache_forward_kinematics(q, auto_clip=False)
        out = []
        for name, corners in _CORNERS.items():
            placement = robot_model.frame_placement(name)
            level = (corners @ placement.rotation.T + placement.translation) @ tilt.T
            out.append((level.min(0), level.max(0)))
        return out

    return boxes


def pelvis_height_estimator(robot_model):
    """Return f(body_q, tilt) -> pelvis height above the floor; ValueError if implausible.

    Use a model instance of its own: forward kinematics overwrites its cache.
    """
    def height(body_q, tilt):
        q = robot_model.get_configuration_from_actuated_joints(body_actuated_joint_values=body_q)
        robot_model.cache_forward_kinematics(q, auto_clip=False)
        ankles = [(tilt @ robot_model.frame_placement(f"{side}_ankle_roll_link").translation)[2]
                  for side in ("left", "right")]
        estimate = ANKLE_TO_SOLE_M - min(ankles)
        if abs(ankles[0] - ankles[1]) > FEET_LEVEL_M:
            raise ValueError("Feet are not level; pelvis height unknown")
        if not PELVIS_HEIGHT_RANGE_M[0] <= estimate <= PELVIS_HEIGHT_RANGE_M[1]:
            raise ValueError(f"Implausible pelvis height {estimate:.3f} m")
        return estimate

    return height


def level_tilt(base_quat):
    """Pelvis roll and pitch from a w, x, y, z quaternion, heading removed."""
    w, x, y, z = _finite(base_quat, 4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not 0.9 < norm < 1.1:
        raise ValueError("base_quat is not a unit quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rotation


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
        raise ValueError(f"Requires {size} finite values")
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


def ready_decision(handover, state, link_boxes, pelvis_height):
    """Return (waypoint or None, reason) for the ready return from the measured state."""
    if handover.desk_top_above_floor_m is None:
        return handover.right_arm_waypoint, "no desk configured"
    try:
        tilt = level_tilt(state["base_quat"])
        height = pelvis_height(_finite(state["body_q_measured_motor"], 29), tilt)
        top = handover.desk_top_above_floor_m - height
        level_boxes = lambda body_q, hand_q: link_boxes(body_q, hand_q, tilt)  # noqa: E731
        where = f"pelvis {height:.3f} m, desk top {top:+.3f} m"
        if hand_above_desk_top(level_boxes, state, top):
            return None, f"hand above desk top ({where})"
        gap = direct_return_desk_gap(level_boxes, state, handover.right_arm_target, top,
                                     handover.desk_front_from_pelvis_m)
    except (KeyError, TypeError, ValueError) as error:
        return handover.right_arm_waypoint, f"unknown geometry: {error}"  # Keep the raise.
    if gap >= DESK_MARGIN_M:
        return None, f"direct path clears by {gap * 1000:.0f} mm ({where})"
    return handover.right_arm_waypoint, f"direct path {gap * 1000:.0f} mm from desk ({where})"


def ready_waypoint(handover, state, link_boxes, pelvis_height):
    """The profile's raised waypoint, or None when the return can skip it."""
    return ready_decision(handover, state, link_boxes, pelvis_height)[0]

