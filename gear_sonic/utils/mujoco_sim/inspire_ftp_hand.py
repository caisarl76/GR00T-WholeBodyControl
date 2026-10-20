"""Safe six-motor Inspire FTP command reception for MuJoCo simulation."""

from __future__ import annotations

from collections.abc import Sequence
import time

import mujoco
import numpy as np
from numpy.typing import NDArray
import zmq

from gear_sonic.utils.teleop.inspire_ftp import (
    CLOSED_RADIANS,
    OPEN,
    normalized_to_radians,
    radians_to_normalized,
    validate_hand_pair,
)
from gear_sonic.utils.teleop.zmq.zmq_message_decoder import unpack_pose_message

DEFAULT_MAX_SLEW_SPEED = 1.0 / CLOSED_RADIANS
# Backward-compatible import alias. New code should use the symmetric name.
DEFAULT_MAX_OPEN_SPEED = DEFAULT_MAX_SLEW_SPEED
ACTIVE_JOINT_SUFFIXES = (
    "little_1_joint",
    "ring_1_joint",
    "middle_1_joint",
    "index_1_joint",
    "thumb_2_joint",
    "thumb_1_joint",
)


def active_joint_names(side: str) -> tuple[str, ...]:
    if side not in ("left", "right"):
        raise ValueError(f"invalid Inspire hand side: {side!r}")
    return tuple(f"{side}_{suffix}" for suffix in ACTIVE_JOINT_SUFFIXES)


class InspireFtpMujocoPlant:
    """Name-resolved adapter between normalized FTP commands and MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        joint_ids: dict[str, NDArray[np.int64]],
        qpos_addresses: dict[str, NDArray[np.int64]],
        qvel_addresses: dict[str, NDArray[np.int64]],
        actuator_ids: dict[str, NDArray[np.int64]],
    ) -> None:
        self.model = model
        self.data = data
        self.joint_ids = joint_ids
        self.qpos_addresses = qpos_addresses
        self.qvel_addresses = qvel_addresses
        self.actuator_ids = actuator_ids
        self.last_measurement_limit_error_rad = 0.0

    @classmethod
    def resolve(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        left_joint_names: Sequence[str] | None = None,
        right_joint_names: Sequence[str] | None = None,
        left_actuator_names: Sequence[str] | None = None,
        right_actuator_names: Sequence[str] | None = None,
    ) -> InspireFtpMujocoPlant:
        """Resolve every active joint and actuator by exact MuJoCo name."""

        configured_joints = {
            "left": tuple(left_joint_names or active_joint_names("left")),
            "right": tuple(right_joint_names or active_joint_names("right")),
        }
        configured_actuators = {
            "left": tuple(left_actuator_names or configured_joints["left"]),
            "right": tuple(right_actuator_names or configured_joints["right"]),
        }
        for side in ("left", "right"):
            if len(configured_joints[side]) != 6:
                raise ValueError(f"{side} active joint list must contain six names")
            if len(configured_actuators[side]) != 6:
                raise ValueError(f"{side} actuator list must contain six names")

        joint_ids = {}
        qpos_addresses = {}
        qvel_addresses = {}
        actuator_ids = {}
        for side in ("left", "right"):
            resolved_joint_ids = np.array(
                [cls._require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in configured_joints[side]],
                dtype=np.int64,
            )
            resolved_actuator_ids = np.array(
                [
                    cls._require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    for name in configured_actuators[side]
                ],
                dtype=np.int64,
            )
            driven_joint_ids = model.actuator_trnid[resolved_actuator_ids, 0]
            if not np.array_equal(driven_joint_ids, resolved_joint_ids):
                raise ValueError(f"{side} Inspire actuators do not drive configured joints")
            joint_ids[side] = resolved_joint_ids
            qpos_addresses[side] = model.jnt_qposadr[resolved_joint_ids].astype(np.int64, copy=True)
            qvel_addresses[side] = model.jnt_dofadr[resolved_joint_ids].astype(np.int64, copy=True)
            actuator_ids[side] = resolved_actuator_ids

        all_actuator_ids = np.concatenate(tuple(actuator_ids.values()))
        if len(set(all_actuator_ids.tolist())) != 12:
            raise ValueError("Inspire hand actuator names must resolve to twelve distinct IDs")
        return cls(
            model,
            data,
            joint_ids=joint_ids,
            qpos_addresses=qpos_addresses,
            qvel_addresses=qvel_addresses,
            actuator_ids=actuator_ids,
        )

    @staticmethod
    def _require_id(model: mujoco.MjModel, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model does not define required object {name!r}")
        return object_id

    def write_targets(
        self,
        left: Sequence[float] | NDArray[np.floating],
        right: Sequence[float] | NDArray[np.floating],
    ) -> None:
        """Write twelve validated normalized commands as radian position targets."""

        left_command, right_command = validate_hand_pair(left, right)
        self.data.ctrl[self.actuator_ids["left"]] = normalized_to_radians(left_command)
        self.data.ctrl[self.actuator_ids["right"]] = normalized_to_radians(right_command)

    def read_normalized_state(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Read the twelve active joint positions in the normalized FTP convention."""

        raw_left = self.data.qpos[self.qpos_addresses["left"]].copy()
        raw_right = self.data.qpos[self.qpos_addresses["right"]].copy()
        if not np.all(np.isfinite(raw_left)) or not np.all(np.isfinite(raw_right)):
            raise ValueError("Inspire active-joint measurements must be finite")
        violations = np.concatenate(
            (
                np.maximum(-raw_left, 0.0),
                np.maximum(raw_left - CLOSED_RADIANS, 0.0),
                np.maximum(-raw_right, 0.0),
                np.maximum(raw_right - CLOSED_RADIANS, 0.0),
            )
        )
        self.last_measurement_limit_error_rad = float(np.max(violations))
        # MuJoCo joint limits are soft constraints and can transiently exceed
        # their ranges under contact. Command validation remains strict; only
        # measured state is projected onto the normalized hardware endpoints.
        left = radians_to_normalized(np.clip(raw_left, 0.0, CLOSED_RADIANS))
        right = radians_to_normalized(np.clip(raw_right, 0.0, CLOSED_RADIANS))
        return left, right


class InspireCommandState:
    """Latest valid hand pair with bounded tracking and fail-open slew."""

    def __init__(
        self,
        *,
        stale_after_s: float = 0.25,
        max_slew_speed: Sequence[float] | NDArray[np.floating] | None = None,
        max_open_speed: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> None:
        if not np.isfinite(stale_after_s) or stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be finite and positive")
        if max_slew_speed is not None and max_open_speed is not None:
            raise ValueError("specify only one of max_slew_speed and max_open_speed")
        selected_speed = (
            max_slew_speed if max_slew_speed is not None else max_open_speed
        )
        if selected_speed is None:
            selected_speed = DEFAULT_MAX_SLEW_SPEED
        speed = np.asarray(selected_speed, dtype=np.float64)
        if speed.shape != (6,) or not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
            raise ValueError("max_slew_speed must contain six finite positive values")

        self.stale_after_s = float(stale_after_s)
        self.max_slew_speed = speed.copy()
        self.max_open_speed = self.max_slew_speed
        self.last_valid: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self.last_receive_monotonic: float | None = None
        self.output = (OPEN.copy(), OPEN.copy())

    def reset(self) -> None:
        """Forget external commands and restore the fail-open startup state."""

        self.last_valid = None
        self.last_receive_monotonic = None
        self.output = (OPEN.copy(), OPEN.copy())

    def accept(
        self,
        left: Sequence[float] | NDArray[np.floating],
        right: Sequence[float] | NDArray[np.floating],
        *,
        now: float,
    ) -> None:
        """Atomically replace the latest pair after both hands validate."""

        timestamp = float(now)
        if not np.isfinite(timestamp):
            raise ValueError("receive timestamp must be finite")
        validated = validate_hand_pair(left, right)
        self.last_valid = validated
        self.last_receive_monotonic = timestamp

    def advance(self, *, now: float, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the watchdog and return isolated left/right command copies."""

        timestamp = float(now)
        step = float(dt)
        if not np.isfinite(timestamp):
            raise ValueError("current timestamp must be finite")
        if not np.isfinite(step) or step < 0.0:
            raise ValueError("dt must be finite and non-negative")

        fresh = (
            self.last_valid is not None
            and self.last_receive_monotonic is not None
            and timestamp - self.last_receive_monotonic <= self.stale_after_s
        )
        target = self.last_valid if fresh else (OPEN, OPEN)
        maximum_delta = self.max_slew_speed * step
        self.output = tuple(
            current
            + np.clip(goal - current, -maximum_delta, maximum_delta)
            for current, goal in zip(self.output, target)
        )
        return tuple(command.copy() for command in self.output)


class InspireFtpZmqSubscriber:
    """Non-publishing subscriber that drains pose/planner commands to latest."""

    TOPICS = ("pose", "planner")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        *,
        state: InspireCommandState | None = None,
    ) -> None:
        self.state = state or InspireCommandState()
        self._context: zmq.Context | None = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        for topic in self.TOPICS:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.connect(f"tcp://{host}:{port}")

    @classmethod
    def from_socket(
        cls,
        socket,
        *,
        state: InspireCommandState | None = None,
    ) -> InspireFtpZmqSubscriber:
        """Construct around an already configured socket, primarily for tests."""

        instance = cls.__new__(cls)
        instance.state = state or InspireCommandState()
        instance._context = None
        instance._socket = socket
        return instance

    @staticmethod
    def _decode_hand_pair(
        message: bytes,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if message.startswith(b"pose"):
            topic = "pose"
        elif message.startswith(b"planner"):
            topic = "planner"
        else:
            raise ValueError("message is not on a subscribed hand-command topic")
        decoded = unpack_pose_message(message, topic=topic)
        if "left_hand_joints" not in decoded or "right_hand_joints" not in decoded:
            raise ValueError("message must contain both Inspire hand fields")
        return validate_hand_pair(decoded["left_hand_joints"], decoded["right_hand_joints"])

    def poll(self, *, now: float | None = None) -> bool:
        """Drain queued messages and atomically accept only the latest valid pair."""

        latest = None
        while True:
            try:
                message = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                latest = self._decode_hand_pair(message)
            except (TypeError, ValueError):
                continue

        if latest is None:
            return False
        timestamp = time.monotonic() if now is None else now
        self.state.accept(*latest, now=timestamp)
        return True

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self) -> InspireFtpZmqSubscriber:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
