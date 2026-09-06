"""Published G1 Dex3 ACT absolute targets to stationary SONIC v1 references.

This adapter is for the recorded-observation simulation integration fixture.
It does not establish task success or visual-domain compatibility.
"""

from __future__ import annotations

import numpy as np

from .timeline import resample_trajectory, trajectory_velocities

ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
ARM_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_SUFFIXES)
LEFT_HAND_NAMES = tuple(
    f"left_hand_{joint}_joint"
    for joint in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
)
RIGHT_HAND_NAMES = tuple(
    f"right_hand_{joint}_joint"
    for joint in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
)
ACTION_NAMES = ARM_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES
BODY_NAMES = (
    tuple(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    )
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + ARM_NAMES
)
# Native policy_parameters.hpp: Isaac reference slot -> hardware/MuJoCo body slot.
ISAAC_FROM_MOTOR = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
# Native policy_parameters.hpp default_angles, in hardware order.
NOMINAL_BODY = np.array([
    -.312, 0, 0, .669, -.363, 0, -.312, 0, 0, .669, -.363, 0,
    0, 0, 0, .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0,
])


def compose_standing_reference(reset: dict) -> dict:
    """Bootstrap real SONIC inference after unhang/reset, before ACT.

    Confirm the controller's initialized target matches the pinned nominal
    stance. Actual hanging or falling measurements must never become this
    standing target. Hand rows remain in the measured simulator order.
    """
    target = np.asarray(reset["body_command_target"][:29], dtype=float)
    if target.shape != (29,) or not np.allclose(target, NOMINAL_BODY, atol=1e-6, rtol=0):
        raise ValueError("native initialized target does not match the nominal standing pose")
    hands = [np.asarray(reset[f"{side}_hand_q"], dtype=float) for side in ("left", "right")]
    if any(hand.shape != (7,) or not np.all(np.isfinite(hand)) for hand in hands):
        raise ValueError("invalid measured hands for standing bootstrap")
    frames = 1046  # 20 seconds and the native 46-frame preview padding.
    return {
        "joint_pos": np.tile(NOMINAL_BODY[ISAAC_FROM_MOTOR], (frames, 1)).astype(np.float32),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1)),
        "frame_index": np.arange(frames, dtype=np.int64),
        "left_hand_joints": np.tile(hands[0], (frames, 1)).astype(np.float32),
        "right_hand_joints": np.tile(hands[1], (frames, 1)).astype(np.float32),
    }


def compose_reference(actions, initial: dict, contract: dict) -> tuple[dict, dict]:
    raw = np.asarray(actions)
    if (
        raw.ndim != 2 or raw.shape[1] != 28 or not 100 <= len(raw) <= 600
        or len(raw) % 100 or raw.dtype.kind not in "fi" or not np.all(np.isfinite(raw))
    ):
        raise ValueError("expected finite ACT action chunks [N,28], N=100,...,600")
    names = {group: [row["name"] for row in contract[group]] for group in ("body", "left", "right")}
    if (
        tuple(names["body"]) != BODY_NAMES
        or set(names["left"]) != set(LEFT_HAND_NAMES)
        or set(names["right"]) != set(RIGHT_HAND_NAMES)
    ):
        raise ValueError("simulator joint names do not match the G1 Dex3 contract")
    if len(names["left"]) != 7 or len(names["right"]) != 7:
        raise ValueError("expected seven unique joints per hand")
    body = np.asarray(initial["body_q"], dtype=float)
    left = np.asarray(initial["left_hand_q"], dtype=float)
    right = np.asarray(initial["right_hand_q"], dtype=float)
    root = np.asarray(initial["floating_base_pose"], dtype=float)
    if (
        body.shape != (29,)
        or left.shape != (7,)
        or right.shape != (7,)
        or root.shape != (7,)
        or not np.all(np.isfinite(np.r_[body, left, right, root]))
    ):
        raise ValueError("invalid measured initial state")
    measured = dict(zip(names["body"] + names["left"] + names["right"], np.r_[body, left, right]))
    limits = {row["name"]: row["range"] for group in contract.values() for row in group}
    bounds = np.asarray([limits[name] for name in ACTION_NAMES], dtype=float)
    if bounds.shape != (28, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("invalid robot joint limits")
    initial_action = np.asarray([measured[name] for name in ACTION_NAMES])
    if np.any(initial_action < bounds[:, 0] - 1e-4) or np.any(initial_action > bounds[:, 1] + 1e-4):
        raise ValueError("initial state outside model joint limits")
    clipped = np.clip(raw, bounds[:, 0], bounds[:, 1])
    # Two-second entry from measured pose, then a 30 Hz policy chunk. The
    # trailing settle interval lets the rate limiter approach the final target.
    source_t = np.r_[0.0, 2.0 + np.arange(len(raw)) / 30.0]
    source_q = np.vstack([initial_action, clipped])
    dense_t = np.arange(0.0, source_t[-1] + 3.0 + 1e-9, 0.02)
    desired = resample_trajectory(source_t, source_q, dense_t, tail_policy="hold")
    filtered = np.empty_like(desired)
    filtered[0] = initial_action
    max_step = np.r_[np.full(14, 1.0), np.full(14, 2.0)] * 0.02
    for i in range(1, len(filtered)):
        filtered[i] = filtered[i - 1] + np.clip(desired[i] - filtered[i - 1], -max_step, max_step)
    # Native v1.1 reads offsets 0,5,...45. Pad body AND hands with the same
    # constant terminal row; zero velocities are derived from the final path.
    filtered = np.vstack([filtered, np.repeat(filtered[-1:], 46, axis=0)])
    times = np.arange(len(filtered)) * 0.02
    body_motor = np.tile(body, (len(filtered), 1))
    body_motor[:, 15:] = filtered[:, :14]
    body_isaac = body_motor[:, ISAAC_FROM_MOTOR]
    quat = root[3:7]
    norm = np.linalg.norm(quat)
    if not 0.999 <= norm <= 1.001:
        raise ValueError("root orientation must be a unit wxyz quaternion")
    payload = {
        "joint_pos": body_isaac.astype(np.float32),
        "joint_vel": trajectory_velocities(times, body_isaac, held_from=dense_t[-1]).astype(np.float32),
        "body_quat": np.tile(quat / norm, (len(filtered), 1)).astype(np.float32),
        "frame_index": np.arange(len(filtered), dtype=np.int64),
        "left_hand_joints": filtered[:, [ACTION_NAMES.index(name) for name in names["left"]]].astype(np.float32),
        "right_hand_joints": filtered[:, [ACTION_NAMES.index(name) for name in names["right"]]].astype(np.float32),
    }
    report = {
        "source_hz": 30,
        "source_action_frames": len(raw),
        "reference_hz": 50,
        "source_action_names": ACTION_NAMES,
        "hand_runtime_names": {"left": names["left"], "right": names["right"]},
        "reference_frames": len(filtered),
        "terminal_padding_frames": 46,
        "joint_limit_clipped_values": int(np.count_nonzero(raw != clipped)),
        "max_arm_speed_rad_s": 1.0,
        "max_hand_speed_rad_s": 2.0,
        "initial_transition_seconds": 2.0,
        "lower_body_source": "measured initial simulation pose",
    }
    return payload, report


def leased_payload(reference: dict, *, session_id: int, chunk_id: int, deadline_ns: int) -> dict:
    if (
        any(type(value) is not int for value in (session_id, chunk_id, deadline_ns))
        or session_id <= 0
        or chunk_id < 0
        or deadline_ns <= 0
    ):
        raise ValueError("invalid adapter lease metadata")
    return {
        **reference,
        "adapter_session_id": np.array([session_id], dtype=np.int64),
        "adapter_chunk_id": np.array([chunk_id], dtype=np.int64),
        "adapter_deadline_ns": np.array([deadline_ns], dtype=np.int64),
    }
