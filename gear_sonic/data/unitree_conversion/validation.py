"""Structural validation for immutable Unitree-to-SONIC episode stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import io
from types import MappingProxyType

import av
import numpy as np


@dataclass(frozen=True)
class FrameFieldSpec:
    """One exact target-frame scalar or fixed-width array field."""

    dtype: np.dtype
    shape: tuple[int, ...]


def _spec(dtype: str, width: int) -> FrameFieldSpec:
    return FrameFieldSpec(np.dtype(dtype), (width,))


TARGET_FRAME_SCHEMA: Mapping[str, FrameFieldSpec | None] = MappingProxyType(
    {
        "observation.state": _spec("float64", 43),
        "observation.eef_state": _spec("float64", 14),
        "action.wbc": _spec("float64", 43),
        "observation.root_orientation": _spec("float64", 4),
        "observation.projected_gravity": _spec("float64", 3),
        "observation.cpp_rotation_offset": _spec("float64", 4),
        "observation.init_base_quat": _spec("float64", 4),
        "teleop.delta_heading": _spec("float64", 1),
        "action.motion_token": _spec("float64", 64),
        "teleop.smpl_joints": _spec("float32", 72),
        "teleop.smpl_pose": _spec("float32", 63),
        "teleop.body_quat_w": _spec("float32", 4),
        "teleop.target_body_orientation": _spec("float32", 6),
        "teleop.left_hand_joints": _spec("float32", 7),
        "teleop.right_hand_joints": _spec("float32", 7),
        "teleop.smpl_frame_index": _spec("int64", 1),
        "teleop.left_wrist_joints": _spec("float32", 3),
        "teleop.right_wrist_joints": _spec("float32", 3),
        "teleop.stream_mode": _spec("int32", 1),
        "teleop.planner_mode": _spec("int32", 1),
        "teleop.planner_movement": _spec("float32", 3),
        "teleop.planner_facing": _spec("float32", 3),
        "teleop.planner_speed": _spec("float32", 1),
        "teleop.planner_height": _spec("float32", 1),
        "teleop.vr_3pt_position": _spec("float32", 9),
        "teleop.vr_3pt_orientation": _spec("float32", 18),
        "timestamp": None,
        "task": None,
    }
)
TARGET_VIDEO_KEYS = frozenset(
    {
        "observation.images.ego_view",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
)
TARGET_VIDEO_SIZE = (640, 480)
_QUATERNION_NORM_TOLERANCE = 1e-5
_EXACT_NEUTRAL_FIELDS: Mapping[str, np.ndarray] = MappingProxyType(
    {
        "teleop.delta_heading": np.zeros(1, dtype=np.float64),
        "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.target_body_orientation": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "teleop.left_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.right_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.stream_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_movement": np.zeros(3, dtype=np.float32),
        "teleop.planner_facing": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "teleop.planner_speed": np.array([-1.0], dtype=np.float32),
        "teleop.planner_height": np.array([-1.0], dtype=np.float32),
        "teleop.vr_3pt_position": np.zeros(9, dtype=np.float32),
        "teleop.vr_3pt_orientation": np.zeros(18, dtype=np.float32),
    }
)


@dataclass(frozen=True)
class ValidationReport:
    """A complete successful structural validation result."""

    success: bool
    row_count: int
    field_names: tuple[str, ...]
    video_frame_counts: Mapping[str, int] = field(default_factory=dict)
    video_dimensions: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success is not True:
            raise ValueError("published validation reports must represent success")
        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("validation row_count must be positive")
        if self.field_names != tuple(TARGET_FRAME_SCHEMA):
            raise ValueError("validation field_names must match the exact target schema")
        counts = dict(sorted(self.video_frame_counts.items()))
        dimensions = dict(sorted(self.video_dimensions.items()))
        if set(counts) != set(dimensions):
            raise ValueError("video validation counts and dimensions must have identical keys")
        if "observation.images.ego_view" not in counts:
            raise ValueError("validation requires observation.images.ego_view")
        if any(type(count) is not int or count != self.row_count for count in counts.values()):
            raise ValueError("every video frame count must equal validation row_count")
        if any(size != TARGET_VIDEO_SIZE for size in dimensions.values()):
            raise ValueError(f"video dimensions must be exactly {TARGET_VIDEO_SIZE}")
        object.__setattr__(self, "video_frame_counts", MappingProxyType(counts))
        object.__setattr__(self, "video_dimensions", MappingProxyType(dimensions))

    def to_dict(self) -> dict[str, object]:
        return {
            "success": True,
            "row_count": self.row_count,
            "field_names": list(self.field_names),
            "video_frame_counts": dict(self.video_frame_counts),
            "video_dimensions": {key: list(value) for key, value in self.video_dimensions.items()},
        }


def validate_target_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject rows that cannot be represented by the exact target schema."""
    if isinstance(rows, (str, bytes)) or len(rows) < 1:
        raise ValueError("stage rows must contain at least one target frame")
    expected_keys = tuple(TARGET_FRAME_SCHEMA)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a target-frame mapping")
        if set(row) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(row))
            extra = sorted(set(row) - set(expected_keys))
            raise ValueError(f"row {index} target keys differ: missing={missing}, extra={extra}")
        for key, spec in TARGET_FRAME_SCHEMA.items():
            value = row[key]
            if key == "task":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"row {index} task must be a nonempty UTF-8 string")
                try:
                    value.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise ValueError(f"row {index} task must be a nonempty UTF-8 string") from error
                continue
            if key == "timestamp":
                if not isinstance(value, np.float32):
                    raise ValueError(f"row {index} timestamp must be a numpy.float32 scalar")
                expected = np.float32(index / 50.0)
                if value.tobytes() != expected.tobytes():
                    raise ValueError(f"row {index} timestamp must be bitwise np.float32(j/50)")
                continue
            assert spec is not None
            if not isinstance(value, np.ndarray):
                raise ValueError(f"row {index} {key} must be a numpy.ndarray")
            if value.dtype != spec.dtype or value.shape != spec.shape:
                raise ValueError(f"row {index} {key} must have dtype {spec.dtype} and shape {spec.shape}")
            if value.dtype.kind in "iuf" and not np.isfinite(value).all():
                raise ValueError(f"row {index} {key} must contain only finite values")
        frame_index = row["teleop.smpl_frame_index"]
        assert isinstance(frame_index, np.ndarray)
        if int(frame_index[0]) != index:
            raise ValueError(f"row {index} teleop.smpl_frame_index must equal its episode-local index")
        for key, expected in _EXACT_NEUTRAL_FIELDS.items():
            if not np.array_equal(row[key], expected):
                raise ValueError(f"row {index} {key} must equal the exact neutral target value")
        for key in (
            "observation.root_orientation",
            "observation.cpp_rotation_offset",
            "observation.init_base_quat",
        ):
            quaternion = row[key]
            assert isinstance(quaternion, np.ndarray)
            norm = float(np.linalg.norm(quaternion))
            if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
                raise ValueError(f"row {index} {key} must be a unit WXYZ quaternion")
        eef_state = row["observation.eef_state"]
        assert isinstance(eef_state, np.ndarray)
        for start in (3, 10):
            if abs(float(np.linalg.norm(eef_state[start : start + 4])) - 1.0) > _QUATERNION_NORM_TOLERANCE:
                raise ValueError(f"row {index} observation.eef_state wrist quaternion must be unit WXYZ")
        root = row["observation.root_orientation"]
        gravity = row["observation.projected_gravity"]
        assert isinstance(root, np.ndarray) and isinstance(gravity, np.ndarray)
        normalized = root / np.linalg.norm(root)
        w = normalized[0]
        vector = normalized[1:]
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cross = np.cross(vector, down)
        expected_gravity = down - 2.0 * w * cross + 2.0 * np.cross(vector, cross)
        if not np.allclose(gravity, expected_gravity, rtol=0.0, atol=1e-10):
            raise ValueError(f"row {index} observation.projected_gravity must match inverse root rotation")
        if index and (
            not np.array_equal(row["observation.cpp_rotation_offset"], rows[0]["observation.cpp_rotation_offset"])
            or not np.array_equal(row["observation.init_base_quat"], rows[0]["observation.init_base_quat"])
        ):
            raise ValueError("episode initial root fields must remain constant across every row")


def probe_video_50fps(data: bytes, *, field_name: str) -> tuple[int, tuple[int, int]]:
    """Decode an owned MP4 blob and return its exact 50 Hz count and size."""
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                raise ValueError(f"{field_name} must contain exactly one video stream")
            stream = streams[0]
            if stream.average_rate is None or Fraction(stream.average_rate) != 50:
                raise ValueError(f"{field_name} must have nominal fps 50")
            count = 0
            dimensions: tuple[int, int] | None = None
            expected_position = 0
            for frame in container.decode(stream):
                current = (int(frame.width), int(frame.height))
                if current != TARGET_VIDEO_SIZE:
                    raise ValueError(f"{field_name} frame size {current} != required {TARGET_VIDEO_SIZE}")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"{field_name} frames require PTS and time_base")
                position = Fraction(frame.pts) * Fraction(frame.time_base) * 50
                if position != expected_position:
                    raise ValueError(f"{field_name} PTS must be consecutive on the episode-local 50 Hz grid")
                if dimensions is None:
                    dimensions = current
                elif dimensions != current:
                    raise ValueError(f"{field_name} changes dimensions within the episode")
                count += 1
                expected_position += 1
    except (av.error.FFmpegError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(field_name):
            raise
        raise ValueError(f"{field_name} must be a readable 50 fps MP4") from error
    if count < 1 or dimensions is None:
        raise ValueError(f"{field_name} must contain at least one video frame")
    return count, dimensions


def validate_stage_payload(
    rows: Sequence[Mapping[str, object]],
    videos: Mapping[str, bytes],
) -> ValidationReport:
    """Validate an entire detached stage payload before publication."""
    validate_target_rows(rows)
    if "observation.images.ego_view" not in videos:
        raise ValueError("stage videos require observation.images.ego_view")
    unknown = sorted(set(videos) - TARGET_VIDEO_KEYS)
    if unknown:
        raise ValueError(f"stage videos contain unsupported target keys: {unknown}")
    counts: dict[str, int] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    for key, data in sorted(videos.items()):
        count, size = probe_video_50fps(data, field_name=f"video {key}")
        if count != len(rows):
            raise ValueError(f"video frame count for {key} is {count}; expected target row count {len(rows)}")
        counts[key] = count
        dimensions[key] = size
    return ValidationReport(
        success=True,
        row_count=len(rows),
        field_names=tuple(TARGET_FRAME_SCHEMA),
        video_frame_counts=counts,
        video_dimensions=dimensions,
    )
