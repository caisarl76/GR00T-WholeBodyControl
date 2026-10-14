"""Measured right Dex3 finger pose; not bottle contact or handover success."""

import math
from collections.abc import Mapping, Set
from numbers import Real

OPEN_MAX_RAD = 0.05
CLOSED_FINGER_MIN_RAD = 0.35
CLOSED_THUMB_MIN_RAD = 0.25
HAND_FRAME_MAX_SKEW_S = 0.10
HAND_STATES = {'open', 'closed', 'partial', 'unknown'}
RIGHT_HAND_JOINT_NAMES = tuple(f'right_hand_{name}_joint' for name in (
    'thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1', 'middle_0', 'middle_1'))
# Native G1 XML bounds, with the existing open tolerance for measured noise.
RIGHT_HAND_LIMITS_RAD = ((-1.0472, 1.0472), (-1.0472, 0.724312), (-2.0944, 0.),
                         (-0.191986, 1.8326), (0., 2.0944),
                         (-0.191986, 1.8326), (0., 2.0944))


def classify_right_hand(joints):
    """Thumb-first [thumb0, thumb1, thumb2, index0, index1, middle0, middle1].

    Open uses the existing native tolerance. Closed thresholds are provisional
    bottle-grasp rules checked against source 15/30/16, not contact sensing.
    """
    if isinstance(joints, (Mapping, Set, str, bytes)):
        return 'unknown'
    try:
        q = tuple(joints)
        if len(q) != 7 or any(not isinstance(v, Real) or isinstance(v, bool)
                             or not math.isfinite(v) for v in q):
            return 'unknown'
    except (TypeError, ValueError, OverflowError):
        return 'unknown'
    if any(not low - OPEN_MAX_RAD <= v <= high + OPEN_MAX_RAD
           for v, (low, high) in zip(q, RIGHT_HAND_LIMITS_RAD)):
        return 'unknown'
    if all(-OPEN_MAX_RAD <= v <= OPEN_MAX_RAD for v in q):
        return 'open'
    if (all(v >= CLOSED_FINGER_MIN_RAD for v in q[3:])
            and q[1] <= -CLOSED_THUMB_MIN_RAD and q[2] <= -CLOSED_THUMB_MIN_RAD):
        return 'closed'
    return 'partial'


def hand_state_for_frame(frame, now, max_age_s):
    """Ignore feedback that cannot be associated with this camera capture."""
    state = frame.right_hand_state
    at, index = frame.right_hand_sample_at, frame.right_hand_sample_index
    if (type(state) is not str or state not in HAND_STATES or state == 'unknown'
            or type(at) not in (int, float) or at < 0 or at > now or not math.isfinite(at)
            or type(index) is not int or index < 0
            or not 0 <= now - at <= max_age_s
            or abs(at - frame.received_at) > HAND_FRAME_MAX_SKEW_S):
        return 'unknown'
    return state
