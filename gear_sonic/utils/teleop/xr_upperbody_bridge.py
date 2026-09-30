"""Upper-body trajectory bridge helpers for GEAR-SONIC ZMQ manager.

The bridge intentionally publishes only ZMQ planner/command messages. It never
opens Unitree DDS channels and never writes body ``LowCmd``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np

from gear_sonic.utils.teleop.bridge_planner_publisher import (
    BridgePlannerPublisher,
    PlannerCommand,
    PlannerPreparationError,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

UPPER_BODY_DOF = 17
XR_G1_29_ARM_DOF = 14
HAND_DOF = 7
DUAL_HAND_DOF = 14
G1_UPPER_BODY_JOINT_INDICES = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
G1_CALIB_FULL_UPPER_BODY = np.zeros(UPPER_BODY_DOF, dtype=np.float32)
G1_STANDING_UPPER_BODY = np.asarray(
    [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, -0.2, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STOP_UPPER_BODY_PRESETS = {
    "straight": G1_STANDING_UPPER_BODY,
    "deploy-standing": G1_STANDING_UPPER_BODY,
    "calib-full": G1_CALIB_FULL_UPPER_BODY,
}
STOP_RELEASE_RAMP_PHASES = {"out", "hold"}
DEX3_OPEN_HAND = np.zeros(HAND_DOF, dtype=np.float32)
DEX3_LEFT_STOP_HAND_TUCKED = np.asarray(
    [0.0, 0.7, 1.2, -0.3, -0.3, -0.3, -0.3],
    dtype=np.float32,
)
DEX3_RIGHT_STOP_HAND_TUCKED = np.asarray(
    [0.0, -0.7, -1.2, 0.3, 0.3, 0.3, 0.3],
    dtype=np.float32,
)


@dataclass(frozen=True)
class UpperBodyFrame:
    """One upper-body target frame for planner-mode ZMQ streaming."""

    upper_body_position: np.ndarray
    upper_body_velocity: np.ndarray
    timestamp: float | None = None
    left_hand_joints: np.ndarray | None = None
    right_hand_joints: np.ndarray | None = None
    mode: int = 0
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (1.0, 0.0, 0.0)
    speed: float = -1.0
    height: float = -1.0
    velocity_provided: bool = False
    toggle_data_collection: bool = False
    toggle_data_abort: bool = False
    stop: bool = False
    ramp_phase: str = "track"
    ramp_alpha: float = 1.0


@dataclass(frozen=True)
class BridgeConfig:
    """Runtime safety defaults for the upper-body bridge."""

    max_abs_joint: float = 3.14
    max_joint_step: float = 0.03


@dataclass
class StopReleaseSchedule:
    """Timing state for stop-release ramp and exact final-pose hold frames."""

    frame: UpperBodyFrame
    release_started_at: float
    release_until: float
    final_hold_until: float
    last_publish_time: float = 0.0
    final_frame_sent: bool = False
    active: bool = True

    @classmethod
    def start(
        cls,
        frame: UpperBodyFrame,
        *,
        now: float,
        release_s: float,
        final_hold_s: float,
    ) -> "StopReleaseSchedule":
        release_duration = max(float(release_s), 0.0)
        final_hold_duration = max(float(final_hold_s), 0.0)
        release_until = now + release_duration
        return cls(
            frame=frame,
            release_started_at=now,
            release_until=release_until,
            final_hold_until=release_until + final_hold_duration,
        )

    def next_alpha(self, *, now: float, hz: float) -> float | None:
        """Return the next stop-release alpha to publish, or ``None``."""

        if not self.active:
            return None
        period = 1.0 / max(float(hz), 1e-6)
        if self.last_publish_time > 0.0 and now - self.last_publish_time < period:
            return None

        release_duration = self.release_until - self.release_started_at
        if release_duration > 0.0 and now < self.release_until:
            alpha = 1.0 - max(0.0, self.release_until - now) / release_duration
        elif not self.final_frame_sent:
            alpha = 1.0
            self.final_frame_sent = True
        elif now < self.final_hold_until:
            alpha = 1.0
        else:
            self.active = False
            return None

        self.last_publish_time = now
        return float(np.clip(alpha, 0.0, 1.0))


@dataclass(frozen=True)
class FeedbackPrime:
    """Deploy feedback used to seed bridge startup state."""

    upper_body: np.ndarray | None = None
    heading_yaw: float | None = None


class UpperBodyFilter:
    """Clip joint targets and limit per-frame target jumps."""

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._last_position: np.ndarray | None = None
        self._entry_ramp_origin: np.ndarray | None = None
        self._entry_ramp_phase: str | None = None

    def reset(self) -> None:
        self._last_position = None
        self._clear_entry_ramp()

    def seed(self, position: Sequence[float]) -> None:
        """Initialize rate limiting from the robot's current measured position."""

        self._last_position = np.clip(
            np.asarray(position, dtype=np.float32).reshape(UPPER_BODY_DOF),
            -self._config.max_abs_joint,
            self._config.max_abs_joint,
        )
        self._clear_entry_ramp()

    def snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        """Capture mutable filter state before planner preparation."""

        return (
            None if self._last_position is None else self._last_position.copy(),
            None if self._entry_ramp_origin is None else self._entry_ramp_origin.copy(),
            self._entry_ramp_phase,
        )

    def restore(
        self,
        snapshot: tuple[np.ndarray | None, np.ndarray | None, str | None],
    ) -> None:
        """Restore a snapshot after a planner frame is rejected."""

        last_position, entry_ramp_origin, entry_ramp_phase = snapshot
        self._last_position = None if last_position is None else last_position.copy()
        self._entry_ramp_origin = None if entry_ramp_origin is None else entry_ramp_origin.copy()
        self._entry_ramp_phase = entry_ramp_phase

    def _clear_entry_ramp(self) -> None:
        self._entry_ramp_origin = None
        self._entry_ramp_phase = None

    def _limit_from_last(self, position: np.ndarray) -> np.ndarray:
        clipped = np.clip(
            np.asarray(position, dtype=np.float32),
            -self._config.max_abs_joint,
            self._config.max_abs_joint,
        )
        if self._last_position is None:
            self._last_position = clipped
            return clipped.copy()

        delta = np.clip(
            clipped - self._last_position,
            -self._config.max_joint_step,
            self._config.max_joint_step,
        )
        limited = self._last_position + delta
        limited = np.clip(limited, -self._config.max_abs_joint, self._config.max_abs_joint)
        self._last_position = limited
        return limited.copy()

    def current_position(self) -> np.ndarray | None:
        """Return the last upper-body command emitted by this filter."""

        if self._last_position is None:
            return None
        return self._last_position.copy()

    def apply(self, position: np.ndarray) -> np.ndarray:
        self._clear_entry_ramp()
        return self._limit_from_last(position)

    def apply_ramped(self, position: np.ndarray, *, ramp_phase: str, ramp_alpha: float) -> np.ndarray:
        clipped = np.clip(
            np.asarray(position, dtype=np.float32),
            -self._config.max_abs_joint,
            self._config.max_abs_joint,
        )
        if ramp_phase in {"out", "hold", "pause"}:
            # These phases are already explicitly ramped by the XR source.
            # Applying the bridge's fixed per-frame limiter here can leave the
            # streamed target behind the source ramp; when the source exits, the
            # downstream controller can then jump to its idle/CALIB posture.
            self._clear_entry_ramp()
            self._last_position = clipped
            return clipped.copy()
        if ramp_phase in {"start", "in", "resume"} and self._last_position is not None:
            alpha = float(np.clip(ramp_alpha, 0.0, 1.0))
            entry_phase = "resume" if ramp_phase == "resume" else "start"
            if self._entry_ramp_origin is None or self._entry_ramp_phase != entry_phase:
                self._entry_ramp_origin = self._last_position.copy()
                self._entry_ramp_phase = entry_phase
            ramped = (1.0 - alpha) * self._entry_ramp_origin + alpha * clipped
            return self._limit_from_last(ramped)
        return self.apply(clipped)


def _upper_body_from_feedback_payload(payload: dict[str, Any]) -> np.ndarray | None:
    body_q = payload.get("body_q_measured", payload.get("body_q"))
    if body_q is None:
        return None
    body_q_array = np.asarray(body_q, dtype=np.float32).reshape(-1)
    if body_q_array.shape[0] < max(G1_UPPER_BODY_JOINT_INDICES) + 1:
        return None
    return body_q_array[G1_UPPER_BODY_JOINT_INDICES].astype(np.float32)


def _heading_yaw_from_quat_wxyz(quat: Sequence[float] | np.ndarray) -> float | None:
    quat_array = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat_array.shape != (4,) or not np.all(np.isfinite(quat_array)):
        return None
    norm = float(np.linalg.norm(quat_array))
    if norm <= 1e-9:
        return None
    w, x, y, z = quat_array / norm
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def feedback_payload_heading_yaw(payload: dict[str, Any]) -> float | None:
    """Extract current robot yaw from deploy feedback quaternions."""

    for key in ("base_quat", "base_quat_measured"):
        quat = payload.get(key)
        if quat is None:
            continue
        yaw = _heading_yaw_from_quat_wxyz(quat)
        if yaw is not None:
            return yaw
    return None


def _feedback_prime_from_payload(payload: dict[str, Any]) -> FeedbackPrime:
    return FeedbackPrime(
        upper_body=_upper_body_from_feedback_payload(payload),
        heading_yaw=feedback_payload_heading_yaw(payload),
    )


def _facing_from_yaw(yaw: float) -> tuple[float, float, float]:
    return (float(math.cos(yaw)), float(math.sin(yaw)), 0.0)


def _rotate_xy(values: Sequence[float], yaw: float) -> tuple[float, float, float]:
    x, y, z = np.asarray(values, dtype=np.float64).reshape(3)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (float(c * x - s * y), float(s * x + c * y), float(z))


def _normalize_xy_or_heading(values: Sequence[float], fallback_yaw: float) -> tuple[float, float, float]:
    x, y, z = np.asarray(values, dtype=np.float64).reshape(3)
    norm = float(math.hypot(x, y))
    if norm <= 1e-9:
        return _facing_from_yaw(fallback_yaw)
    return (float(x / norm), float(y / norm), float(z))


def anchor_frame_planner_heading(frame: UpperBodyFrame, heading_yaw: float) -> UpperBodyFrame:
    """Rotate Unitree-local planner vectors into the robot's startup heading frame."""

    rotated_facing = _rotate_xy(frame.facing, heading_yaw)
    return replace(
        frame,
        movement=_rotate_xy(frame.movement, heading_yaw),
        facing=_normalize_xy_or_heading(rotated_facing, heading_yaw),
    )


def build_heading_hold_planner_message(
    heading_yaw: float | None,
    *,
    upper_body_position: Sequence[float] | np.ndarray | None = None,
) -> bytes:
    """Build an idle planner command that preserves startup heading and arms."""

    upper_body = None
    upper_body_velocity = None
    if upper_body_position is not None:
        upper_body = np.asarray(upper_body_position, dtype=np.float32).reshape(UPPER_BODY_DOF)
        upper_body_velocity = np.zeros(UPPER_BODY_DOF, dtype=np.float32)
    yaw = 0.0 if heading_yaw is None else float(heading_yaw)

    return build_planner_message(
        0,
        (0.0, 0.0, 0.0),
        _facing_from_yaw(yaw),
        speed=0.0,
        height=-1.0,
        upper_body_position=None if upper_body is None else upper_body.tolist(),
        upper_body_velocity=None if upper_body_velocity is None else upper_body_velocity.tolist(),
    )


def build_start_control_messages(
    *,
    heading_yaw: float | None,
    upper_body_position: Sequence[float] | np.ndarray | None,
) -> list[bytes]:
    """Build startup messages in safe order: hold target first, then start."""

    messages: list[bytes] = []
    if heading_yaw is not None or upper_body_position is not None:
        messages.append(
            build_heading_hold_planner_message(
                heading_yaw,
                upper_body_position=upper_body_position,
            )
        )
    messages.append(build_command_message(start=True, stop=False, planner=True))
    return messages


def _read_feedback_prime(
    *,
    context: Any,
    host: str,
    port: int,
    topic: str,
    timeout_s: float,
) -> FeedbackPrime:
    if timeout_s <= 0.0:
        return FeedbackPrime()
    try:
        import msgpack
        import zmq
    except ImportError:
        return FeedbackPrime()

    sub = context.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    sub.setsockopt(zmq.RCVTIMEO, min(100, max(1, int(timeout_s * 1000))))
    sub.connect(f"tcp://{host}:{port}")
    try:
        latest = _read_feedback_prime_from_sub(
            sub=sub,
            msgpack_module=msgpack,
            zmq_module=zmq,
            topic=topic,
            timeout_s=timeout_s,
        )
    finally:
        sub.close(linger=0)
    return latest


def _read_feedback_prime_from_sub(
    *,
    sub: Any,
    msgpack_module: Any,
    zmq_module: Any,
    topic: str,
    timeout_s: float,
) -> FeedbackPrime:
    deadline = time.monotonic() + timeout_s
    latest = FeedbackPrime()
    topic_bytes = topic.encode()
    while time.monotonic() < deadline:
        try:
            raw = sub.recv()
        except zmq_module.Again:
            continue
        payload_bytes = raw[len(topic) :] if raw.startswith(topic_bytes) else raw
        try:
            payload = msgpack_module.unpackb(payload_bytes, raw=False)
        except Exception:
            continue
        if isinstance(payload, dict):
            prime = _feedback_prime_from_payload(payload)
            latest = FeedbackPrime(
                upper_body=prime.upper_body if prime.upper_body is not None else latest.upper_body,
                heading_yaw=prime.heading_yaw if prime.heading_yaw is not None else latest.heading_yaw,
            )
    return latest


def _as_vector(value: Any, *, name: str, size: int, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(size, default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _optional_vector(value: Any, *, name: str, size: int) -> np.ndarray | None:
    if value is None:
        return None
    return _as_vector(value, name=name, size=size)


def _as_xyz(value: Any, *, name: str, default: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(default if value is None else value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return (float(array[0]), float(array[1]), float(array[2]))


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite number")
    return scalar


def _debug_range(values: Sequence[float] | np.ndarray | None) -> str:
    if values is None:
        return "None"
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return "empty"
    return f"min={array.min():+.3f} max={array.max():+.3f} mean={array.mean():+.3f}"


def _debug_pose(debug_payload: dict[str, Any], key: str) -> Any:
    pose = debug_payload.get(key)
    if not isinstance(pose, dict):
        return None
    return {
        "pos": pose.get("pos"),
        "x_axis": pose.get("x_axis"),
        "y_axis": pose.get("y_axis"),
        "z_axis": pose.get("z_axis"),
    }


def xr_g1_29_arm_to_sonic_upper_body(
    dual_arm_position: Sequence[float],
    waist: Sequence[float] | None = None,
) -> np.ndarray:
    """Map Unitree ``xr_teleoperate`` G1_29 dual-arm order to SONIC upper body.

    Unitree ``xr_teleoperate`` G1_29 arm order is left arm first, then right arm:
    ``[L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw, L_elbow,
    L_wrist_roll, L_wrist_pitch, L_wrist_yaw, R_shoulder_pitch, ...,
    R_wrist_yaw]``.

    SONIC's ZMQ manager expects 17 upper-body joints in IsaacLab upper-body
    order: waist yaw/roll/pitch, then interleaved left/right shoulder, elbow,
    and wrist joints.
    """

    arm = _as_vector(
        dual_arm_position,
        name="dual_arm_position",
        size=XR_G1_29_ARM_DOF,
    )
    waist_array = _as_vector(
        waist,
        name="waist",
        size=3,
        default=0.0,
    )
    upper = np.zeros(UPPER_BODY_DOF, dtype=np.float32)
    upper[0:3] = waist_array
    upper[3:] = arm[
        [
            0,  # left shoulder pitch
            7,  # right shoulder pitch
            1,  # left shoulder roll
            8,  # right shoulder roll
            2,  # left shoulder yaw
            9,  # right shoulder yaw
            3,  # left elbow
            10,  # right elbow
            4,  # left wrist roll
            11,  # right wrist roll
            5,  # left wrist pitch
            12,  # right wrist pitch
            6,  # left wrist yaw
            13,  # right wrist yaw
        ]
    ]
    return upper


def split_xr_dex3_dual_hand_joints(
    dual_hand_joints: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Split Unitree XR Dex3 dual-hand action order into left/right commands.

    ``xr_teleoperate`` records Dex3 actions as left hand 7 values followed by
    right hand 7 values. Each side is already in Unitree motor command index
    order, which is the same order used by GEAR-SONIC's ``Dex3Hands`` command
    path.
    """

    dual = _as_vector(dual_hand_joints, name="dual_hand_joints", size=DUAL_HAND_DOF)
    return dual[:HAND_DOF].copy(), dual[HAND_DOF:].copy()


def frame_from_mapping(data: dict[str, Any]) -> UpperBodyFrame:
    """Validate a mapping and convert it into an ``UpperBodyFrame``."""

    upper_body_position = data.get("upper_body_position")
    if upper_body_position is None and data.get("dual_arm_position") is not None:
        upper_body_position = xr_g1_29_arm_to_sonic_upper_body(
            data.get("dual_arm_position"),
            waist=data.get("waist"),
        )

    upper_body_velocity = data.get("upper_body_velocity")
    if upper_body_velocity is None and data.get("dual_arm_velocity") is not None:
        upper_body_velocity = xr_g1_29_arm_to_sonic_upper_body(
            data.get("dual_arm_velocity"),
            waist=data.get("waist_velocity", [0.0, 0.0, 0.0]),
        )

    left_hand_joints = data.get("left_hand_joints")
    right_hand_joints = data.get("right_hand_joints")
    if (left_hand_joints is None or right_hand_joints is None) and data.get("dual_hand_joints") is not None:
        split_left, split_right = split_xr_dex3_dual_hand_joints(data["dual_hand_joints"])
        if left_hand_joints is None:
            left_hand_joints = split_left
        if right_hand_joints is None:
            right_hand_joints = split_right

    return UpperBodyFrame(
        timestamp=(
            None if data.get("timestamp") is None else _as_finite_float(data["timestamp"], name="timestamp")
        ),
        upper_body_position=_as_vector(upper_body_position, name="upper_body_position", size=UPPER_BODY_DOF),
        upper_body_velocity=_as_vector(
            upper_body_velocity,
            name="upper_body_velocity",
            size=UPPER_BODY_DOF,
            default=0.0,
        ),
        velocity_provided=upper_body_velocity is not None,
        left_hand_joints=_optional_vector(left_hand_joints, name="left_hand_joints", size=HAND_DOF),
        right_hand_joints=_optional_vector(right_hand_joints, name="right_hand_joints", size=HAND_DOF),
        mode=int(data.get("mode", 0)),
        movement=_as_xyz(data.get("movement"), name="movement", default=(0.0, 0.0, 0.0)),
        facing=_as_xyz(data.get("facing"), name="facing", default=(1.0, 0.0, 0.0)),
        speed=_as_finite_float(data.get("speed", -1.0), name="speed"),
        height=_as_finite_float(data.get("height", -1.0), name="height"),
        toggle_data_collection=bool(data.get("toggle_data_collection", False)),
        toggle_data_abort=bool(data.get("toggle_data_abort", False)),
        stop=bool(data.get("stop", False)),
        ramp_phase=str(data.get("ramp_phase", "track")),
        ramp_alpha=_as_finite_float(data.get("ramp_alpha", 1.0), name="ramp_alpha"),
    )


def load_jsonl_frames(path: Path) -> list[UpperBodyFrame]:
    frames: list[UpperBodyFrame] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frames.append(frame_from_mapping(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not frames:
        raise ValueError(f"{path} did not contain any frames")
    return frames


def _npz_optional(data: np.lib.npyio.NpzFile, name: str, index: int) -> np.ndarray | None:
    if name not in data:
        return None
    return data[name][index]


def load_npz_frames(path: Path) -> list[UpperBodyFrame]:
    with np.load(path) as data:
        if "upper_body_position" in data:
            positions = np.asarray(data["upper_body_position"])
            expected_width = UPPER_BODY_DOF
        elif "dual_arm_position" in data:
            positions = np.asarray(data["dual_arm_position"])
            expected_width = XR_G1_29_ARM_DOF
        else:
            raise ValueError(f"{path} missing upper_body_position or dual_arm_position")
        if positions.ndim != 2 or positions.shape[1] != expected_width:
            raise ValueError(f"position data must have shape (N, {expected_width}), got {positions.shape}")
        count = positions.shape[0]
        velocities = data["upper_body_velocity"] if "upper_body_velocity" in data else None
        dual_arm_velocities = data["dual_arm_velocity"] if "dual_arm_velocity" in data else None
        timestamps = data["timestamp"] if "timestamp" in data else None

        frames: list[UpperBodyFrame] = []
        for i in range(count):
            if "upper_body_position" in data:
                frame_data: dict[str, Any] = {"upper_body_position": positions[i]}
            else:
                frame_data = {"dual_arm_position": positions[i]}
            if velocities is not None:
                frame_data["upper_body_velocity"] = velocities[i]
            if dual_arm_velocities is not None:
                frame_data["dual_arm_velocity"] = dual_arm_velocities[i]
            if timestamps is not None:
                frame_data["timestamp"] = float(timestamps[i])
            for name in ("left_hand_joints", "right_hand_joints"):
                optional = _npz_optional(data, name, i)
                if optional is not None:
                    frame_data[name] = optional
            optional_dual_hand = _npz_optional(data, "dual_hand_joints", i)
            if optional_dual_hand is not None:
                frame_data["dual_hand_joints"] = optional_dual_hand
            for name in ("mode", "speed", "height"):
                if name in data:
                    frame_data[name] = data[name][i]
            for name in ("movement", "facing"):
                if name in data:
                    frame_data[name] = data[name][i]
            frames.append(frame_from_mapping(frame_data))
    return frames


def load_frames(source: str | Path | Iterable[dict[str, Any]], fmt: str = "auto") -> list[UpperBodyFrame]:
    """Load upper-body frames from JSONL, NPZ, or an iterable of mappings."""

    if not isinstance(source, (str, Path)):
        frames = [frame_from_mapping(item) for item in source]
        if not frames:
            raise ValueError("source did not contain any frames")
        return frames

    path = Path(source)
    chosen = fmt
    if chosen == "auto":
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            chosen = "jsonl"
        elif suffix == ".npz":
            chosen = "npz"
        else:
            raise ValueError(f"cannot infer input format from {path}")
    if chosen == "jsonl":
        return load_jsonl_frames(path)
    if chosen == "npz":
        return load_npz_frames(path)
    raise ValueError(f"unsupported input format: {fmt}")


def normalize_live_source_payload(payload: dict[str, Any]) -> UpperBodyFrame:
    """Normalize one live JSON payload from GenONAI/xr_teleoperate."""

    return frame_from_mapping(payload)


def synthesize_missing_velocities(
    frames: Sequence[UpperBodyFrame],
    *,
    default_hz: float,
) -> list[UpperBodyFrame]:
    """Fill missing upper-body velocities from adjacent positions."""

    if not frames:
        return []
    synthesized: list[UpperBodyFrame] = []
    previous = frames[0]
    synthesized.append(previous)
    for frame in frames[1:]:
        if frame.velocity_provided:
            synthesized.append(frame)
            previous = frame
            continue

        if previous.timestamp is not None and frame.timestamp is not None:
            dt = max(frame.timestamp - previous.timestamp, 1e-6)
        else:
            dt = 1.0 / default_hz
        velocity = (frame.upper_body_position - previous.upper_body_position) / dt
        synthesized.append(copy.copy(frame))
        object.__setattr__(synthesized[-1], "upper_body_velocity", velocity.astype(np.float32))
        previous = frame
    return synthesized


def collect_range_warnings(
    frames: Sequence[UpperBodyFrame],
    *,
    arm_warn_rad: float = 6.5,
    hand_warn_rad: float = 3.2,
) -> list[str]:
    warnings: list[str] = []
    if not frames:
        return warnings
    upper = np.stack([frame.upper_body_position for frame in frames])
    max_upper = float(np.max(np.abs(upper)))
    if max_upper > arm_warn_rad:
        warnings.append(
            f"upper_body_position max abs {max_upper:.3f} rad exceeds {arm_warn_rad:.3f}; check units/order"
        )

    for name in ("left_hand_joints", "right_hand_joints"):
        values = [getattr(frame, name) for frame in frames if getattr(frame, name) is not None]
        if not values:
            continue
        max_hand = float(np.max(np.abs(np.stack(values))))
        if max_hand > hand_warn_rad:
            warnings.append(f"{name} max abs {max_hand:.3f} rad exceeds {hand_warn_rad:.3f}; check units/order")
    return warnings


def _range_line(name: str, values: np.ndarray | None) -> str:
    if values is None:
        return f"{name}: absent"
    return f"{name} range: [{float(np.min(values)):.4f}, {float(np.max(values)):.4f}]"


def build_inspection_report(
    frames: Sequence[UpperBodyFrame],
    *,
    source: str,
    arm_warn_rad: float = 6.5,
    hand_warn_rad: float = 3.2,
) -> str:
    """Return a human-readable schema/range report for trajectory preflight."""

    if not frames:
        return f"XR upper-body bridge inspect: {source}\nframes: 0"
    upper = np.stack([frame.upper_body_position for frame in frames])
    left_values = [frame.left_hand_joints for frame in frames if frame.left_hand_joints is not None]
    right_values = [frame.right_hand_joints for frame in frames if frame.right_hand_joints is not None]
    first = np.array2string(frames[0].upper_body_position, precision=4, separator=", ")
    lines = [
        f"XR upper-body bridge inspect: {source}",
        f"frames: {len(frames)}",
        _range_line("upper_body_position", upper),
        _range_line("left_hand_joints", np.stack(left_values) if left_values else None),
        _range_line("right_hand_joints", np.stack(right_values) if right_values else None),
        f"first upper_body_position[17]: {first}",
    ]
    warnings = collect_range_warnings(frames, arm_warn_rad=arm_warn_rad, hand_warn_rad=hand_warn_rad)
    if warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def build_frame_planner_command(
    frame: UpperBodyFrame,
    config: BridgeConfig,
    filt: UpperBodyFilter | None = None,
    stop_hand_preset: str | None = None,
) -> PlannerCommand:
    """Convert one frame to a planner command before hand limiting/encoding."""

    filtered_position = (
        filt.apply_ramped(
            frame.upper_body_position,
            ramp_phase=frame.ramp_phase,
            ramp_alpha=frame.ramp_alpha,
        )
        if filt is not None
        else np.clip(frame.upper_body_position, -config.max_abs_joint, config.max_abs_joint)
    )
    left_hand_position = None if frame.left_hand_joints is None else frame.left_hand_joints.tolist()
    right_hand_position = None if frame.right_hand_joints is None else frame.right_hand_joints.tolist()
    if (
        stop_hand_preset is not None
        and stop_hand_preset != "none"
        and (frame.stop or frame.ramp_phase in {"out", "hold"})
    ):
        left_hand_position, right_hand_position = _stop_hand_targets(
            frame,
            float(np.clip(frame.ramp_alpha, 0.0, 1.0)),
            stop_hand_preset,
        )
    return PlannerCommand(
        mode=frame.mode,
        movement=frame.movement,
        facing=frame.facing,
        speed=frame.speed,
        height=frame.height,
        upper_body_position=filtered_position.tolist(),
        upper_body_velocity=frame.upper_body_velocity.tolist(),
        left_hand_position=left_hand_position,
        right_hand_position=right_hand_position,
    )


def build_frame_planner_message(*args, **kwargs) -> bytes:
    """Compatibility wrapper returning the encoded planner message."""

    return build_frame_planner_command(*args, **kwargs).encode()


def _blend_hand_target(
    source: np.ndarray | None,
    target: np.ndarray,
    alpha: float,
) -> list[float]:
    start = target if source is None else np.asarray(source, dtype=np.float32)
    hand = (1.0 - alpha) * start + alpha * target
    return hand.astype(np.float32).tolist()


def _stop_hand_targets(
    frame: UpperBodyFrame,
    alpha: float,
    hand_preset: str,
) -> tuple[list[float] | None, list[float] | None]:
    if hand_preset == "none":
        return None, None
    if hand_preset == "source":
        return (
            None if frame.left_hand_joints is None else frame.left_hand_joints.tolist(),
            None if frame.right_hand_joints is None else frame.right_hand_joints.tolist(),
        )
    if hand_preset == "open":
        return (
            _blend_hand_target(frame.left_hand_joints, DEX3_OPEN_HAND, alpha),
            _blend_hand_target(frame.right_hand_joints, DEX3_OPEN_HAND, alpha),
        )
    if hand_preset == "tucked-thumb":
        return (
            _blend_hand_target(frame.left_hand_joints, DEX3_LEFT_STOP_HAND_TUCKED, alpha),
            _blend_hand_target(frame.right_hand_joints, DEX3_RIGHT_STOP_HAND_TUCKED, alpha),
        )
    raise ValueError(f"unsupported stop hand preset: {hand_preset}")


def _stop_upper_body_target(upper_body_preset: str) -> np.ndarray:
    try:
        return STOP_UPPER_BODY_PRESETS[upper_body_preset]
    except KeyError as exc:
        raise ValueError(f"unsupported stop upper-body preset: {upper_body_preset}") from exc


def build_stop_standing_command(
    frame: UpperBodyFrame,
    alpha: float = 1.0,
    hand_preset: str = "tucked-thumb",
    upper_body_preset: str = "straight",
) -> PlannerCommand:
    """Build an idle planner command ramped to the configured exit pose."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    start = np.asarray(frame.upper_body_position, dtype=np.float32)
    target = _stop_upper_body_target(upper_body_preset)
    upper_body_position = (1.0 - alpha) * start + alpha * target
    left_hand_position, right_hand_position = _stop_hand_targets(frame, alpha, hand_preset)

    return PlannerCommand(
        mode=0,
        movement=(0.0, 0.0, 0.0),
        facing=frame.facing,
        speed=0.0,
        height=-1.0,
        upper_body_position=upper_body_position.tolist(),
        upper_body_velocity=(0.0,) * UPPER_BODY_DOF,
        left_hand_position=left_hand_position,
        right_hand_position=right_hand_position,
    )


def build_stop_standing_message(*args, **kwargs) -> bytes:
    """Compatibility wrapper returning the encoded stop planner message."""

    return build_stop_standing_command(*args, **kwargs).encode()


def frame_requests_stop_release(frame: UpperBodyFrame) -> bool:
    return bool(frame.stop) or frame.ramp_phase in STOP_RELEASE_RAMP_PHASES


def frame_with_current_upper_body_position(
    frame: UpperBodyFrame,
    filt: UpperBodyFilter,
) -> UpperBodyFrame:
    """Snapshot the filtered command that was just sent downstream."""

    current_position = filt.current_position()
    if current_position is None:
        return frame
    return replace(frame, upper_body_position=current_position)


def start_stop_release_schedule(
    frame: UpperBodyFrame,
    *,
    last_tracking_frame: UpperBodyFrame | None,
    now: float,
    release_s: float,
    final_hold_s: float,
) -> StopReleaseSchedule:
    return StopReleaseSchedule.start(
        last_tracking_frame if last_tracking_frame is not None else frame,
        now=now,
        release_s=release_s,
        final_hold_s=final_hold_s,
    )


def build_due_stop_release_commands(
    schedule: StopReleaseSchedule,
    *,
    now: float,
    hz: float,
    stream_mode: int,
    hand_preset: str,
    upper_body_preset: str,
) -> tuple[PlannerCommand, bytes, str] | None:
    alpha = schedule.next_alpha(now=now, hz=hz)
    if alpha is None:
        return None
    phase = "final-hold" if schedule.final_frame_sent else "stop-release"
    return (
        build_stop_standing_command(
            schedule.frame,
            alpha=alpha,
            hand_preset=hand_preset,
            upper_body_preset=upper_body_preset,
        ),
        build_manager_state_message(stream_mode=stream_mode),
        phase,
    )


def build_due_stop_release_messages(*args, **kwargs) -> tuple[bytes, bytes] | None:
    """Compatibility wrapper returning encoded stop-release messages."""

    commands = build_due_stop_release_commands(*args, **kwargs)
    if commands is None:
        return None
    planner_command, manager_message, _phase = commands
    return planner_command.encode(), manager_message


def build_manager_state_message(
    *,
    stream_mode: int = 5,
    toggle_data_collection: bool = False,
    toggle_data_abort: bool = False,
) -> bytes:
    """Build manager-state message consumed by ``run_data_exporter.py``.

    ``stream_mode=5`` matches ``PLANNER_VR_3PT`` in the PICO manager and tells
    the exporter to use the latest planner message for planner/hand teleop
    fields. The bridge still sends explicit 17-DOF upper-body targets on the
    planner topic; this manager state is metadata for recording.
    """

    return pack_pose_message(
        {
            "stream_mode": np.array([stream_mode], dtype=np.int32),
            "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
            "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
        },
        topic="manager_state",
    )


def _send_loop(args: argparse.Namespace, frames: list[UpperBodyFrame]) -> int:
    config = BridgeConfig(max_abs_joint=args.max_abs_joint, max_joint_step=args.max_joint_step)
    filt = UpperBodyFilter(config)
    dt = 1.0 / args.hz
    published = 0

    socket = None
    context = None
    if not args.dry_run:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://{args.bind_host}:{args.port}")
        time.sleep(args.pub_warmup_s)
        socket.send(build_command_message(start=args.start_control, stop=False, planner=True))
    planner_publisher = None if socket is None else BridgePlannerPublisher(socket)

    try:
        finished = False
        while not finished:
            for frame in frames:
                command = build_frame_planner_command(frame, config, filt)
                if planner_publisher is not None:
                    prepared = planner_publisher.prepare(
                        command,
                        ramp_phase=frame.ramp_phase,
                    )
                    planner_publisher.send(prepared)
                    socket.send(build_manager_state_message(stream_mode=args.stream_mode))
                else:
                    command.encode()
                published += 1
                if args.once:
                    finished = True
                    break
                time.sleep(dt)
            if not args.loop:
                break
    finally:
        if socket is not None and args.send_stop_on_exit:
            socket.send(build_command_message(start=False, stop=True, planner=True))
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()

    mode = "Dry run published" if args.dry_run else "Published"
    print(f"{mode} {published} frame{'s' if published != 1 else ''}")
    return 0


def _send_zmq_json_loop(args: argparse.Namespace) -> int:
    import zmq

    config = BridgeConfig(max_abs_joint=args.max_abs_joint, max_joint_step=args.max_joint_step)
    filt = UpperBodyFilter(config)
    context = zmq.Context.instance()
    sub = context.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, args.source_topic)
    recv_timeout_s = min(args.source_timeout_s, 1.0 / args.hz)
    sub.setsockopt(zmq.RCVTIMEO, max(1, int(recv_timeout_s * 1000)))
    sub.connect(f"tcp://{args.source_host}:{args.source_port}")

    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://{args.bind_host}:{args.port}")
    time.sleep(args.pub_warmup_s)
    planner_publisher = BridgePlannerPublisher(pub)

    planner_heading_yaw: float | None = None
    late_feedback_sub = None
    late_feedback_msgpack = None
    late_feedback_poll_s = min(max(recv_timeout_s, 0.001), 0.005)
    feedback_prime = FeedbackPrime()
    if not args.no_feedback_prime:
        feedback_prime = _read_feedback_prime(
            context=context,
            host=args.feedback_host,
            port=args.feedback_port,
            topic=args.feedback_topic,
            timeout_s=args.feedback_prime_timeout_s,
        )
        if feedback_prime.upper_body is not None:
            filt.seed(feedback_prime.upper_body)
            print(
                "[xr_upperbody_bridge] seeded upper-body ramp from feedback "
                f"({_debug_range(feedback_prime.upper_body)})"
            )
        else:
            print(
                "[xr_upperbody_bridge] feedback seed unavailable; first upper-body "
                "target cannot be rate-limited from measured robot pose"
            )
        if args.anchor_planner_heading and feedback_prime.heading_yaw is not None:
            planner_heading_yaw = feedback_prime.heading_yaw
            print(f"[xr_upperbody_bridge] anchored planner heading to feedback yaw {planner_heading_yaw:+.3f} rad")
        elif args.anchor_planner_heading:
            print("[xr_upperbody_bridge] feedback heading unavailable; planner facing will use source coordinates")
    late_feedback_pending = (
        args.start_control and not args.allow_unseeded_start_control and feedback_prime.upper_body is None
    )
    if late_feedback_pending and args.no_feedback_prime:
        print(
            "[xr_upperbody_bridge] ERROR: --start-control requires measured "
            "upper-body feedback before live upper-body planner frames, but "
            "--no-feedback-prime disables feedback seeding. Pass "
            "--allow-unseeded-start-control only for debugging.",
            flush=True,
        )
        sub.close(linger=0)
        pub.close(linger=0)
        context.term()
        return 2
    if late_feedback_pending:
        try:
            import msgpack
        except ImportError:
            print(
                "[xr_upperbody_bridge] ERROR: msgpack is required to wait for "
                "late deploy feedback before live upper-body planner frames.",
                flush=True,
            )
            sub.close(linger=0)
            pub.close(linger=0)
            context.term()
            return 2
        late_feedback_msgpack = msgpack
        late_feedback_sub = context.socket(zmq.SUB)
        late_feedback_sub.setsockopt_string(zmq.SUBSCRIBE, args.feedback_topic)
        late_feedback_sub.setsockopt(zmq.RCVTIMEO, max(1, int(late_feedback_poll_s * 1000)))
        late_feedback_sub.connect(f"tcp://{args.feedback_host}:{args.feedback_port}")
        print(
            "[xr_upperbody_bridge] feedback seed unavailable before start; "
            "sending command-only start and holding live upper-body planner "
            "frames until measured g1_debug feedback arrives",
            flush=True,
        )

    def send_start_command() -> None:
        for message in build_start_control_messages(
            heading_yaw=planner_heading_yaw,
            upper_body_position=filt.current_position(),
        ):
            pub.send(message)

    def poll_late_feedback() -> bool:
        nonlocal late_feedback_pending, planner_heading_yaw
        if not late_feedback_pending:
            return True
        if late_feedback_sub is None or late_feedback_msgpack is None:
            return False
        prime = _read_feedback_prime_from_sub(
            sub=late_feedback_sub,
            msgpack_module=late_feedback_msgpack,
            zmq_module=zmq,
            topic=args.feedback_topic,
            timeout_s=late_feedback_poll_s,
        )
        if args.anchor_planner_heading and planner_heading_yaw is None and prime.heading_yaw is not None:
            planner_heading_yaw = prime.heading_yaw
            print(
                "[xr_upperbody_bridge] anchored planner heading to late feedback yaw "
                f"{planner_heading_yaw:+.3f} rad"
            )
        if prime.upper_body is None:
            return False
        filt.seed(prime.upper_body)
        late_feedback_pending = False
        print(
            f"[xr_upperbody_bridge] seeded upper-body ramp from late feedback ({_debug_range(prime.upper_body)})"
        )
        return True

    start_command_deadline = 0.0
    last_start_command_time = 0.0
    if args.start_control:
        start_command_deadline = time.monotonic() + max(args.start_command_repeat_s, 0.0)
        send_start_command()
        last_start_command_time = time.monotonic()
        print(
            "[xr_upperbody_bridge] sent start=true command; "
            f"repeating for {max(args.start_command_repeat_s, 0.0):.1f}s"
        )

    previous_frame: UpperBodyFrame | None = None
    last_tracking_frame: UpperBodyFrame | None = None
    stop_release: StopReleaseSchedule | None = None
    stop_release_completed = False
    published = 0
    last_warn_time = 0.0
    last_debug_time = 0.0
    last_invalid_frame_warn_time = -math.inf

    def warn_invalid_frame(context: str, exc: Exception) -> None:
        nonlocal last_invalid_frame_warn_time
        now = time.monotonic()
        if now - last_invalid_frame_warn_time >= 1.0:
            print(f"[xr_upperbody_bridge] {context} skipped: {exc}", flush=True)
            last_invalid_frame_warn_time = now

    try:
        while True:
            if args.start_control and time.monotonic() < start_command_deadline:
                now = time.monotonic()
                if now - last_start_command_time >= args.start_command_interval_s:
                    send_start_command()
                    last_start_command_time = now
            if late_feedback_pending:
                poll_late_feedback()
            try:
                raw = sub.recv()
            except zmq.Again:
                now = time.monotonic()
                if stop_release is not None:
                    commands = build_due_stop_release_commands(
                        stop_release,
                        now=now,
                        hz=args.hz,
                        stream_mode=args.stream_mode,
                        hand_preset=args.stop_hand_preset,
                        upper_body_preset=args.stop_upper_body_preset,
                    )
                    if commands is not None:
                        planner_command, manager_state_message, stop_phase = commands
                        try:
                            prepared = planner_publisher.prepare(
                                planner_command,
                                ramp_phase=stop_phase,
                            )
                        except PlannerPreparationError as exc:
                            warn_invalid_frame("invalid stop planner command", exc)
                            continue
                        planner_publisher.send(prepared)
                        pub.send(manager_state_message)
                        published += 1
                    if args.once:
                        break
                    if stop_release.active:
                        continue
                    stop_release = None
                    stop_release_completed = True
                if now - last_warn_time > 1.0:
                    if late_feedback_pending:
                        print(
                            "[xr_upperbody_bridge] waiting for measured g1_debug "
                            "feedback before live upper-body planner frames..."
                        )
                    else:
                        print("[xr_upperbody_bridge] waiting for live XR JSON source...")
                    last_warn_time = now
                if args.once and not late_feedback_pending:
                    break
                continue

            if late_feedback_pending and not poll_late_feedback():
                now = time.monotonic()
                if now - last_warn_time > 1.0:
                    print(
                        "[xr_upperbody_bridge] holding live XR frame until "
                        "measured g1_debug feedback seeds the upper-body ramp..."
                    )
                    last_warn_time = now
                continue

            try:
                payload_bytes = (
                    raw[len(args.source_topic) :] if raw.startswith(args.source_topic.encode()) else raw
                )
                payload = json.loads(payload_bytes.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("live payload must be a JSON object")
                frame = normalize_live_source_payload(payload)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                warn_invalid_frame("invalid live frame", exc)
                continue
            if planner_heading_yaw is not None:
                frame = anchor_frame_planner_heading(frame, planner_heading_yaw)
            if not frame.velocity_provided and previous_frame is not None:
                frame = synthesize_missing_velocities([previous_frame, frame], default_hz=args.hz)[-1]

            now = time.monotonic()
            stop_release_enabled = args.stop_release_s > 0.0 or args.stop_final_hold_s > 0.0
            if frame_requests_stop_release(frame) and stop_release_enabled:
                if stop_release is None and not stop_release_completed:
                    stop_release = start_stop_release_schedule(
                        frame,
                        last_tracking_frame=last_tracking_frame,
                        now=now,
                        release_s=args.stop_release_s,
                        final_hold_s=args.stop_final_hold_s,
                    )
                    if args.debug_live:
                        print(
                            "[xr_upperbody_bridge] exit ramp frame received; "
                            f"publishing bridge-managed {args.stop_release_s:.2f}s "
                            f"release ramp and {args.stop_final_hold_s:.2f}s exact final hold",
                            flush=True,
                        )
                if stop_release is not None:
                    commands = build_due_stop_release_commands(
                        stop_release,
                        now=now,
                        hz=args.hz,
                        stream_mode=args.stream_mode,
                        hand_preset=args.stop_hand_preset,
                        upper_body_preset=args.stop_upper_body_preset,
                    )
                    if commands is not None:
                        planner_command, manager_state_message, stop_phase = commands
                        try:
                            prepared = planner_publisher.prepare(
                                planner_command,
                                ramp_phase=stop_phase,
                            )
                        except PlannerPreparationError as exc:
                            warn_invalid_frame("invalid stop planner command", exc)
                            continue
                        planner_publisher.send(prepared)
                        previous_frame = frame
                        pub.send(manager_state_message)
                        published += 1
                    if not stop_release.active:
                        stop_release = None
                        stop_release_completed = True
                    if args.once:
                        break
                    continue
                if args.once:
                    break
                continue

            stop_release_completed = False
            filter_snapshot = filt.snapshot()
            planner_command = build_frame_planner_command(
                frame,
                config,
                filt,
                stop_hand_preset=args.stop_hand_preset,
            )
            try:
                prepared = planner_publisher.prepare(
                    planner_command,
                    ramp_phase=frame.ramp_phase,
                )
            except PlannerPreparationError as exc:
                filt.restore(filter_snapshot)
                warn_invalid_frame("invalid planner frame", exc)
                continue

            try:
                planner_publisher.send(prepared)
            except Exception:
                filt.restore(filter_snapshot)
                raise
            previous_frame = frame
            last_tracking_frame = frame_with_current_upper_body_position(frame, filt)
            pub.send(
                build_manager_state_message(
                    stream_mode=args.stream_mode,
                    toggle_data_collection=frame.toggle_data_collection,
                    toggle_data_abort=frame.toggle_data_abort,
                )
            )
            published += 1
            if args.debug_live:
                now = time.monotonic()
                if now - last_debug_time >= args.debug_interval:
                    debug_payload = payload.get("debug", {})
                    if not isinstance(debug_payload, dict):
                        debug_payload = {}
                    print(
                        "[xr_upperbody_bridge DEBUG] "
                        f"source_dual_arm({_debug_range(payload.get('dual_arm_position'))}) "
                        f"mapped_upper({_debug_range(frame.upper_body_position)}) "
                        f"left_hand({_debug_range(frame.left_hand_joints)}) "
                        f"right_hand({_debug_range(frame.right_hand_joints)}) "
                        f"mode={frame.mode} movement={frame.movement} facing={frame.facing} "
                        f"speed={frame.speed:+.3f} height={frame.height:+.3f} "
                        f"toggle_data_collection={frame.toggle_data_collection} stop={frame.stop} "
                        f"ramp_phase={frame.ramp_phase} ramp_alpha={frame.ramp_alpha:+.3f} "
                        f"target_left={_debug_pose(debug_payload, 'target_left_wrist_pose')} "
                        f"target_right={_debug_pose(debug_payload, 'target_right_wrist_pose')}",
                        flush=True,
                    )
                    last_debug_time = now
            if args.once:
                break
    finally:
        if args.send_stop_on_exit:
            pub.send(build_command_message(start=False, stop=True, planner=True))
        if late_feedback_sub is not None:
            late_feedback_sub.close(linger=0)
        sub.close(linger=0)
        pub.close(linger=0)
        context.term()

    print(f"Published {published} live frame{'s' if published != 1 else ''}")
    return 0


def run_local_zmq_smoke(
    frames: Sequence[UpperBodyFrame],
    config: BridgeConfig,
    *,
    port: int = 0,
    timeout_s: float = 2.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Publish one frame locally and verify planner + manager_state decode."""

    import zmq

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    sub = context.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "planner")
    sub.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
    sub.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))

    if port == 0:
        endpoint = f"inproc://xr-upperbody-bridge-smoke-{uuid4().hex}"
        pub.bind(endpoint)
    else:
        endpoint = f"tcp://127.0.0.1:{port}"
        pub.bind(endpoint)
    sub.connect(endpoint)
    time.sleep(0.1)

    planner_publisher = BridgePlannerPublisher(pub)
    planner_publisher.publish(
        build_frame_planner_command(frames[0], config),
        ramp_phase=frames[0].ramp_phase,
    )
    pub.send(build_manager_state_message())

    decoded: dict[str, dict[str, np.ndarray]] = {}
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and len(decoded) < 2:
            raw = sub.recv()
            if raw.startswith(b"planner"):
                decoded["planner"] = unpack_bridge_message(raw, topic="planner")
            elif raw.startswith(b"manager_state"):
                decoded["manager_state"] = unpack_bridge_message(raw, topic="manager_state")
    finally:
        pub.close(linger=0)
        sub.close(linger=0)
        context.term()
    return decoded


def unpack_bridge_message(packed_data: bytes, *, topic: str) -> dict[str, np.ndarray]:
    """Decode the bridge's packed ZMQ format for smoke tests."""

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"message does not start with topic {topic!r}")
    offset = len(topic_bytes)
    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx >= 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "bool": bool,
    }
    current_offset = offset + HEADER_SIZE
    result: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        dtype = dtype_map[field["dtype"]]
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("file", "zmq-json"),
        default="file",
        help="Input source: replay a file or subscribe to live JSON from xr_teleoperate",
    )
    parser.add_argument("--input", help="JSONL or NPZ upper-body trajectory file")
    parser.add_argument("--format", choices=("auto", "jsonl", "npz"), default="auto")
    parser.add_argument("--source-host", default="localhost", help="Live JSON ZMQ publisher host")
    parser.add_argument("--source-port", type=int, default=5560, help="Live JSON ZMQ publisher port")
    parser.add_argument("--source-topic", default="xr_teleop", help="Live JSON ZMQ topic prefix")
    parser.add_argument("--source-timeout-s", type=float, default=0.2)
    parser.add_argument("--bind-host", default="*", help="ZMQ PUB bind host")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port")
    parser.add_argument("--hz", type=float, default=50.0, help="Publish frequency")
    parser.add_argument("--loop", action="store_true", help="Loop trajectory until interrupted")
    parser.add_argument("--once", action="store_true", help="Publish only one frame")
    parser.add_argument("--dry-run", action="store_true", help="Build messages without opening ZMQ")
    parser.add_argument("--inspect", action="store_true", help="Print trajectory schema/ranges and exit")
    parser.add_argument(
        "--synthesize-velocities",
        action="store_true",
        help="Compute missing upper-body velocities from adjacent frames",
    )
    parser.add_argument("--start-control", action="store_true", help="Send start=true on command topic")
    parser.add_argument(
        "--start-command-repeat-s",
        type=float,
        default=5.0,
        help="Seconds to repeat start=true after launch to avoid ZMQ PUB/SUB slow-joiner loss.",
    )
    parser.add_argument(
        "--start-command-interval-s",
        type=float,
        default=0.2,
        help="Interval between repeated start=true command messages.",
    )
    parser.add_argument("--send-stop-on-exit", action="store_true", help="Send stop=true before exit")
    parser.add_argument(
        "--stop-release-s",
        type=float,
        default=1.2,
        help=(
            "After a live XR frame with stop=true, keep publishing idle planner "
            "frames that ramp upper_body_position to --stop-upper-body-preset "
            "before the XR source disappears."
        ),
    )
    parser.add_argument(
        "--stop-final-hold-s",
        type=float,
        default=1.0,
        help=(
            "After --stop-release-s reaches the standing target, keep publishing "
            "the exact final planner frame for this many seconds."
        ),
    )
    parser.add_argument(
        "--stop-upper-body-preset",
        choices=tuple(STOP_UPPER_BODY_PRESETS),
        default="straight",
        help=(
            "Upper-body target used by stop-release/final-hold frames. "
            "'straight' and 'deploy-standing' use the GEAR-SONIC final standing "
            "target; 'calib-full' is the all-zero upper-body calibration pose."
        ),
    )
    parser.add_argument(
        "--stop-hand-preset",
        choices=("tucked-thumb", "source", "open", "none"),
        default="tucked-thumb",
        help=(
            "Hand command used during --stop-release-s live stop frames. "
            "'tucked-thumb' keeps the Dex3 thumbs near the palms while the "
            "upper body returns to standing."
        ),
    )
    parser.add_argument(
        "--stream-mode",
        type=int,
        default=5,
        help="manager_state stream mode for data exporter (default: 5, planner/VR teleop)",
    )
    parser.add_argument("--pub-warmup-s", type=float, default=0.2, help="PUB socket warm-up delay")
    parser.add_argument("--max-abs-joint", type=float, default=3.14)
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.03,
        help="Maximum upper-body joint target step per bridge frame in radians.",
    )
    parser.add_argument(
        "--feedback-host",
        default="localhost",
        help="GEAR-SONIC feedback ZMQ host for measured-pose ramp seeding.",
    )
    parser.add_argument("--feedback-port", type=int, default=5557, help="GEAR-SONIC feedback ZMQ port.")
    parser.add_argument("--feedback-topic", default="g1_debug", help="GEAR-SONIC feedback ZMQ topic.")
    parser.add_argument(
        "--feedback-prime-timeout-s",
        type=float,
        default=1.0,
        help="Seconds to wait for measured upper-body feedback before live streaming.",
    )
    parser.add_argument(
        "--anchor-planner-heading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rotate live-source planner movement/facing by the measured startup "
            "base yaw from feedback. This makes Unitree controller yaw local to "
            "the robot's initial heading; use --no-anchor-planner-heading for "
            "sources that already publish absolute world-frame planner vectors."
        ),
    )
    parser.add_argument(
        "--no-feedback-prime",
        action="store_true",
        help="Do not seed the upper-body rate limiter from measured robot feedback.",
    )
    parser.add_argument(
        "--allow-unseeded-start-control",
        action="store_true",
        help=(
            "Allow --start-control even when no measured upper-body feedback was "
            "received. Unsafe for normal robot runs because startup can fall back "
            "to CALIB_FULL/all-zero upper-body targets."
        ),
    )
    parser.add_argument(
        "--debug-live",
        action="store_true",
        help="Print live XR payload and mapped upper-body summaries.",
    )
    parser.add_argument("--debug-interval", type=float, default=0.5, help="Seconds between --debug-live prints")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.stop_release_s < 0:
        parser.error("--stop-release-s must be non-negative")
    if args.stop_final_hold_s < 0:
        parser.error("--stop-final-hold-s must be non-negative")
    if args.source == "zmq-json":
        if args.inspect:
            parser.error("--inspect is only supported with --source file")
        return _send_zmq_json_loop(args)
    if not args.input:
        parser.error("--input is required with --source file")
    frames = load_frames(args.input, args.format)
    if args.synthesize_velocities:
        frames = synthesize_missing_velocities(frames, default_hz=args.hz)
    if args.inspect:
        print(build_inspection_report(frames, source=args.input))
        return 0
    print(f"Loaded {len(frames)} frame{'s' if len(frames) != 1 else ''} from {args.input}")
    return _send_loop(args, frames)
