"""Deterministic replay scheduling and acceptance metrics for Unitree conversion.

This module intentionally depends only on NumPy and the Python standard library.
MuJoCo is imported lazily by the two helpers that need it so provenance and metric
tests remain usable in lightweight conversion environments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import numpy as np

COMMAND_HZ = 50
WARMUP_FRAMES = 2 * COMMAND_HZ
HOLD_FRAMES = COMMAND_HZ


@dataclass
class ReplaySample:
    """One simulator-rate observation used by the replay acceptance gates."""

    time_s: float
    root_height_m: float
    root_roll_rad: float
    root_pitch_rad: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    torques: np.ndarray
    contact_force_n: float
    controller_fault: bool = False


@dataclass(frozen=True)
class ReplayActionFrame:
    """A single 50 Hz SONIC action, stored in its wire-level float32 shape."""

    motion_token: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray

    def __post_init__(self) -> None:
        for name, expected_shape in (
            ("motion_token", (64,)),
            ("left_hand", (7,)),
            ("right_hand", (7,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
            if value.dtype != np.float32:
                raise ValueError(f"{name} must have dtype float32, got {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            owned = np.array(value, dtype=np.float32, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


@dataclass(frozen=True)
class ScheduledReplayAction:
    phase: str
    source_frame_index: int
    time_s: float
    frame: ReplayActionFrame


@dataclass(frozen=True)
class SourceEpisodeRef:
    source_episode_id: int
    target_episode_index: int
    episode_length: int
    stage: dict[str, Any]


@dataclass(frozen=True)
class ReplayReport:
    sim_dt: float
    exclude_before_s: float
    robot_mass_kg: float
    effort_limit_source: str
    total_sample_count: int
    included_sample_count: int
    included_start_s: float
    included_end_s: float
    duration_s: float
    root_height_min: float
    root_height_p50: float
    root_height_p95: float
    root_height_p99: float
    root_height_max: float
    root_roll_abs_p50: float
    root_roll_abs_p95: float
    root_roll_abs_p99: float
    root_roll_abs_max: float
    root_pitch_abs_p50: float
    root_pitch_abs_p95: float
    root_pitch_abs_p99: float
    root_pitch_abs_max: float
    joint_limit_overshoot_p50: float
    joint_limit_overshoot_p95: float
    joint_limit_overshoot_p99: float
    joint_limit_overshoot_max: float
    joint_velocity_abs_p50: float
    joint_velocity_abs_p95: float
    joint_velocity_abs_p99: float
    joint_velocity_abs_max: float
    torque_ratio_p50: float
    torque_ratio_p95: float
    torque_ratio_p99: float
    torque_ratio_max: float
    contact_force_p50: float
    contact_force_p95: float
    contact_force_p99: float
    contact_force_max: float
    nonfinite_sample_count: int
    controller_fault_count: int
    accepted: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class CohortReplayReport:
    episode_count: int
    included_sample_count: int
    root_height_min: float
    root_height_max: float
    root_roll_abs_max: float
    root_pitch_abs_max: float
    joint_limit_overshoot_max: float
    joint_velocity_abs_max: float
    torque_ratio_max: float
    contact_force_max: float
    nonfinite_sample_count: int
    controller_fault_count: int
    accepted: bool
    gate_failures: tuple[str, ...]


def _finite_real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its valid range")
    return result


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (math.nan,) * 4
    # Non-finite values are gate failures. Keeping an infinite maximum makes the
    # report fail closed while avoiding NumPy's inf-inf percentile warnings.
    if not np.all(np.isfinite(array)):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return (math.inf,) * 4
        p50, p95, p99 = np.percentile(finite, [50, 95, 99])
        return float(p50), float(p95), float(p99), math.inf
    p50, p95, p99 = np.percentile(array, [50, 95, 99])
    return float(p50), float(p95), float(p99), float(np.max(array))


def _acceptance_failures(
    *,
    root_height_min: float,
    root_height_max: float,
    root_roll_abs_max: float,
    root_pitch_abs_max: float,
    joint_limit_overshoot_max: float,
    joint_velocity_abs_max: float,
    torque_ratio_max: float,
    contact_force_max: float,
    contact_limit_n: float,
    nonfinite_sample_count: int,
    controller_fault_count: int,
) -> tuple[str, ...]:
    def exceeds(value: float, limit: float) -> bool:
        return value > limit and not math.isclose(value, limit, rel_tol=0.0, abs_tol=1e-12)

    failures: list[str] = []
    if (root_height_min < 0.45 and not math.isclose(root_height_min, 0.45, rel_tol=0.0, abs_tol=1e-12)) or exceeds(
        root_height_max, 1.05
    ):
        failures.append("root_height")
    if exceeds(root_roll_abs_max, 0.7):
        failures.append("root_roll")
    if exceeds(root_pitch_abs_max, 0.7):
        failures.append("root_pitch")
    if exceeds(joint_limit_overshoot_max, 0.02):
        failures.append("joint_limit_overshoot")
    if exceeds(joint_velocity_abs_max, 50.0):
        failures.append("joint_velocity")
    if exceeds(torque_ratio_max, 1.05):
        failures.append("torque_ratio")
    if exceeds(contact_force_max, contact_limit_n):
        failures.append("contact_force")
    if nonfinite_sample_count:
        failures.append("nonfinite")
    if controller_fault_count:
        failures.append("controller_fault")
    return tuple(failures)


class ReplayMetricsCollector:
    """Collect simulator-rate metrics and apply the exact inclusive replay gates."""

    def __init__(
        self,
        *,
        sim_dt: float,
        exclude_before_s: float,
        robot_mass_kg: float,
        effort_limits: np.ndarray,
        joint_limits: np.ndarray,
        effort_limit_source: str,
    ) -> None:
        self.sim_dt = _finite_real(sim_dt, "sim_dt", minimum=np.nextafter(0.0, 1.0))
        self.exclude_before_s = _finite_real(exclude_before_s, "exclude_before_s", minimum=0.0)
        self.robot_mass_kg = _finite_real(robot_mass_kg, "robot_mass_kg", minimum=np.nextafter(0.0, 1.0))
        effort = np.asarray(effort_limits, dtype=np.float64)
        limits = np.asarray(joint_limits, dtype=np.float64)
        if effort.ndim != 1 or effort.size == 0 or not np.all(np.isfinite(effort)):
            raise ValueError("effort_limits must be a non-empty finite 1D array")
        if np.any(effort <= 0.0):
            raise ValueError("effort_limits must be strictly positive")
        if limits.shape != (effort.size, 2) or not np.all(np.isfinite(limits)):
            raise ValueError(f"joint_limits must have finite shape ({effort.size}, 2)")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("joint_limits lower bounds must not exceed upper bounds")
        if not isinstance(effort_limit_source, str) or not effort_limit_source.strip():
            raise ValueError("effort_limit_source must be a non-empty string")
        self.effort_limits = effort.copy()
        self.joint_limits = limits.copy()
        self.effort_limits.setflags(write=False)
        self.joint_limits.setflags(write=False)
        self.effort_limit_source = effort_limit_source
        self._last_time_s: float | None = None
        self._total_sample_count = 0
        self._included_times: list[float] = []
        self._root_heights: list[float] = []
        self._root_rolls: list[float] = []
        self._root_pitches: list[float] = []
        self._joint_overshoots: list[float] = []
        self._joint_velocities: list[float] = []
        self._torque_ratios: list[float] = []
        self._contact_forces: list[float] = []
        self._nonfinite_sample_count = 0
        self._controller_fault_count = 0

    def add(self, sample: ReplaySample) -> None:
        if not isinstance(sample, ReplaySample):
            raise TypeError("sample must be a ReplaySample")
        time_s = _finite_real(sample.time_s, "sample time_s", minimum=0.0)
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("sample times must be strictly increasing")
        positions = np.asarray(sample.joint_positions, dtype=np.float64)
        velocities = np.asarray(sample.joint_velocities, dtype=np.float64)
        torques = np.asarray(sample.torques, dtype=np.float64)
        expected = (self.effort_limits.size,)
        for name, value in (
            ("joint_positions", positions),
            ("joint_velocities", velocities),
            ("torques", torques),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")

        self._last_time_s = time_s
        self._total_sample_count += 1
        if time_s < self.exclude_before_s:
            return

        scalars = np.asarray(
            [
                sample.root_height_m,
                sample.root_roll_rad,
                sample.root_pitch_rad,
                sample.contact_force_n,
            ],
            dtype=np.float64,
        )
        finite = bool(
            np.all(np.isfinite(scalars))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(torques))
        )
        if not finite:
            self._nonfinite_sample_count += 1

        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        overshoot = np.maximum.reduce((lo - positions, positions - hi, np.zeros_like(positions)))
        self._included_times.append(time_s)
        self._root_heights.append(float(sample.root_height_m))
        self._root_rolls.append(float(abs(sample.root_roll_rad)))
        self._root_pitches.append(float(abs(sample.root_pitch_rad)))
        self._joint_overshoots.append(float(np.max(overshoot)))
        self._joint_velocities.append(float(np.max(np.abs(velocities))))
        self._torque_ratios.append(float(np.max(np.abs(torques) / self.effort_limits)))
        contact_force = float(sample.contact_force_n)
        self._contact_forces.append(contact_force if contact_force >= 0.0 else math.inf)
        self._controller_fault_count += int(bool(sample.controller_fault))

    def finalize(self) -> ReplayReport:
        if not self._included_times:
            raise ValueError("no replay samples remain after the exclusion interval")

        root_height = np.asarray(self._root_heights, dtype=np.float64)
        finite_heights = root_height[np.isfinite(root_height)]
        if finite_heights.size == root_height.size:
            root_height_min = float(np.min(root_height))
            root_height_max = float(np.max(root_height))
        else:
            root_height_min = -math.inf
            root_height_max = math.inf
        rh50, rh95, rh99, _ = _summarize(self._root_heights)
        rr50, rr95, rr99, rrmax = _summarize(self._root_rolls)
        rp50, rp95, rp99, rpmax = _summarize(self._root_pitches)
        jo50, jo95, jo99, jomax = _summarize(self._joint_overshoots)
        jv50, jv95, jv99, jvmax = _summarize(self._joint_velocities)
        tr50, tr95, tr99, trmax = _summarize(self._torque_ratios)
        cf50, cf95, cf99, cfmax = _summarize(self._contact_forces)
        failures = _acceptance_failures(
            root_height_min=root_height_min,
            root_height_max=root_height_max,
            root_roll_abs_max=rrmax,
            root_pitch_abs_max=rpmax,
            joint_limit_overshoot_max=jomax,
            joint_velocity_abs_max=jvmax,
            torque_ratio_max=trmax,
            contact_force_max=cfmax,
            contact_limit_n=10.0 * self.robot_mass_kg * 9.81,
            nonfinite_sample_count=self._nonfinite_sample_count,
            controller_fault_count=self._controller_fault_count,
        )
        return ReplayReport(
            sim_dt=self.sim_dt,
            exclude_before_s=self.exclude_before_s,
            robot_mass_kg=self.robot_mass_kg,
            effort_limit_source=self.effort_limit_source,
            total_sample_count=self._total_sample_count,
            included_sample_count=len(self._included_times),
            included_start_s=self._included_times[0],
            included_end_s=self._included_times[-1],
            duration_s=self._last_time_s or 0.0,
            root_height_min=root_height_min,
            root_height_p50=rh50,
            root_height_p95=rh95,
            root_height_p99=rh99,
            root_height_max=root_height_max,
            root_roll_abs_p50=rr50,
            root_roll_abs_p95=rr95,
            root_roll_abs_p99=rr99,
            root_roll_abs_max=rrmax,
            root_pitch_abs_p50=rp50,
            root_pitch_abs_p95=rp95,
            root_pitch_abs_p99=rp99,
            root_pitch_abs_max=rpmax,
            joint_limit_overshoot_p50=jo50,
            joint_limit_overshoot_p95=jo95,
            joint_limit_overshoot_p99=jo99,
            joint_limit_overshoot_max=jomax,
            joint_velocity_abs_p50=jv50,
            joint_velocity_abs_p95=jv95,
            joint_velocity_abs_p99=jv99,
            joint_velocity_abs_max=jvmax,
            torque_ratio_p50=tr50,
            torque_ratio_p95=tr95,
            torque_ratio_p99=tr99,
            torque_ratio_max=trmax,
            contact_force_p50=cf50,
            contact_force_p95=cf95,
            contact_force_p99=cf99,
            contact_force_max=cfmax,
            nonfinite_sample_count=self._nonfinite_sample_count,
            controller_fault_count=self._controller_fault_count,
            accepted=not failures,
            gate_failures=failures,
        )


def aggregate_replay_reports(reports: Sequence[ReplayReport]) -> CohortReplayReport:
    if not reports:
        raise ValueError("at least one replay report is required")
    first = reports[0]
    mass = first.robot_mass_kg
    if any(
        report.sim_dt != first.sim_dt
        or report.robot_mass_kg != mass
        or report.effort_limit_source != first.effort_limit_source
        for report in reports
    ):
        raise ValueError("all reports must use the same replay physics contract")
    values = {
        "root_height_min": min(report.root_height_min for report in reports),
        "root_height_max": max(report.root_height_max for report in reports),
        "root_roll_abs_max": max(report.root_roll_abs_max for report in reports),
        "root_pitch_abs_max": max(report.root_pitch_abs_max for report in reports),
        "joint_limit_overshoot_max": max(report.joint_limit_overshoot_max for report in reports),
        "joint_velocity_abs_max": max(report.joint_velocity_abs_max for report in reports),
        "torque_ratio_max": max(report.torque_ratio_max for report in reports),
        "contact_force_max": max(report.contact_force_max for report in reports),
    }
    nonfinite = sum(report.nonfinite_sample_count for report in reports)
    faults = sum(report.controller_fault_count for report in reports)
    failures = _acceptance_failures(
        **values,
        contact_limit_n=10.0 * mass * 9.81,
        nonfinite_sample_count=nonfinite,
        controller_fault_count=faults,
    )
    return CohortReplayReport(
        episode_count=len(reports),
        included_sample_count=sum(report.included_sample_count for report in reports),
        nonfinite_sample_count=nonfinite,
        controller_fault_count=faults,
        accepted=not failures,
        gate_failures=failures,
        **values,
    )


def max_mujoco_contact_force(
    model: object,
    data: object,
    *,
    mujoco_module: object | None = None,
) -> float:
    """Return the maximum L2 norm of MuJoCo's six-component contact forces."""

    if mujoco_module is None:
        import mujoco as mujoco_module  # type: ignore[no-redef]

    maximum = 0.0
    for contact_index in range(int(getattr(data, "ncon"))):
        force = np.zeros(6, dtype=np.float64)
        mujoco_module.mj_contactForce(model, data, contact_index, force)
        maximum = max(maximum, float(np.linalg.norm(force)))
    return maximum


def _validated_indices(values: np.ndarray, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-empty 1D integer array")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= size) or np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicate or out-of-range indices")
    return result


def collector_from_mujoco_model(
    model: object,
    *,
    actuator_ids: np.ndarray,
    joint_ids: np.ndarray,
    exclude_before_s: float = 1.0,
    mujoco_module: object | None = None,
) -> ReplayMetricsCollector:
    """Build a collector exclusively from the simulator's loaded model limits."""

    if mujoco_module is None:
        import mujoco as mujoco_module  # type: ignore[no-redef]

    force_ranges = np.asarray(getattr(model, "actuator_forcerange"), dtype=np.float64)
    joint_ranges = np.asarray(getattr(model, "jnt_range"), dtype=np.float64)
    if force_ranges.ndim != 2 or force_ranges.shape[1] != 2:
        raise ValueError("loaded model actuator_forcerange must have shape (A, 2)")
    if joint_ranges.ndim != 2 or joint_ranges.shape[1] != 2:
        raise ValueError("loaded model jnt_range must have shape (J, 2)")
    actuators = _validated_indices(actuator_ids, size=force_ranges.shape[0], name="actuator_ids")
    joints = _validated_indices(joint_ids, size=joint_ranges.shape[0], name="joint_ids")
    if actuators.size != joints.size:
        raise ValueError("actuator_ids and joint_ids must have the same length")
    effort_limits = np.max(np.abs(force_ranges[actuators]), axis=1)
    return ReplayMetricsCollector(
        sim_dt=getattr(getattr(model, "opt"), "timestep"),
        exclude_before_s=exclude_before_s,
        robot_mass_kg=mujoco_module.mj_getTotalmass(model),
        effort_limits=effort_limits,
        joint_limits=joint_ranges[joints],
        effort_limit_source="mujoco.actuator_forcerange",
    )


def _manifest_checksum(root: Path, payload: bytes) -> None:
    checksum_path = root / "dataset-checksums.sha256"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("provenance checksum file is missing or unsafe")
    expected: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*") == "source-manifest.json":
            expected.append(fields[0].lower())
    if len(expected) != 1 or len(expected[0]) != 64:
        raise ValueError("provenance checksum for source-manifest.json is ambiguous")
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(expected[0], actual):
        raise ValueError("provenance checksum mismatch for source-manifest.json")


def _manifest_int_list(value: object, name: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"provenance {name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"provenance {name} must contain only integers")
    result = list(value)
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"provenance {name} must contain only positive values")
    return result


def resolve_source_episode_ids(
    dataset_root: str | Path,
    source_episode_ids: Sequence[int],
) -> tuple[SourceEpisodeRef, ...]:
    """Resolve upstream IDs through immutable merge provenance, never local indices."""

    root = Path(dataset_root)
    manifest_path = root / "source-manifest.json"
    if root.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("provenance source-manifest.json is missing or unsafe")
    payload = manifest_path.read_bytes()
    _manifest_checksum(root, payload)
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provenance manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("provenance manifest must be an object")
    source_ids = _manifest_int_list(manifest.get("source_episode_ids"), "source_episode_ids")
    if any(source_id < 0 for source_id in source_ids):
        raise ValueError("provenance source episode IDs must be nonnegative")
    lengths = _manifest_int_list(manifest.get("episode_lengths"), "episode_lengths", positive=True)
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(source_ids) != len(lengths) or len(stages) != len(source_ids):
        raise ValueError("provenance episode arrays have inconsistent lengths")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("provenance source episode IDs contain duplicates")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("source_episode_id") != source_ids[index]:
            raise ValueError("provenance stage order does not match source episode IDs")
    requested = list(source_episode_ids)
    if not requested or any(isinstance(item, bool) or not isinstance(item, int) for item in requested):
        raise ValueError("source episode request must be a non-empty integer sequence")
    if any(source_id < 0 for source_id in requested):
        raise ValueError("source episode IDs must be nonnegative")
    if len(set(requested)) != len(requested):
        raise ValueError("source episode request contains duplicates")
    lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    missing = [source_id for source_id in requested if source_id not in lookup]
    if missing:
        raise ValueError(f"source episode IDs are absent from immutable provenance: {missing}")
    return tuple(
        SourceEpisodeRef(
            source_episode_id=source_id,
            target_episode_index=lookup[source_id],
            episode_length=lengths[lookup[source_id]],
            stage=dict(stages[lookup[source_id]]),
        )
        for source_id in requested
    )


def build_replay_schedule(frames: Sequence[ReplayActionFrame]) -> tuple[ScheduledReplayAction, ...]:
    if not frames:
        raise ValueError("replay requires at least one action frame")
    if any(not isinstance(frame, ReplayActionFrame) for frame in frames):
        raise TypeError("all replay frames must be ReplayActionFrame instances")
    staged: list[tuple[str, int, ReplayActionFrame]] = []
    staged.extend(("warmup", 0, frames[0]) for _ in range(WARMUP_FRAMES))
    staged.extend(("trajectory", index, frame) for index, frame in enumerate(frames))
    staged.extend(("hold", len(frames) - 1, frames[-1]) for _ in range(HOLD_FRAMES))
    return tuple(
        ScheduledReplayAction(
            phase=phase,
            source_frame_index=source_index,
            time_s=slot / COMMAND_HZ,
            frame=frame,
        )
        for slot, (phase, source_index, frame) in enumerate(staged)
    )


def run_replay_schedule(
    frames: Sequence[ReplayActionFrame],
    *,
    collector: ReplayMetricsCollector,
    publish: Callable[[ReplayActionFrame, str, int, float], None],
    step: Callable[[float], ReplaySample],
    pace: Callable[[float], None] = time.sleep,
) -> ReplayReport:
    """Publish at 50 Hz and collect one sample after every simulator step."""

    steps_float = (1.0 / COMMAND_HZ) / collector.sim_dt
    steps_per_command = round(steps_float)
    if steps_per_command <= 0 or not math.isclose(steps_float, steps_per_command, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("sim_dt must divide the 50 Hz command period exactly")
    step_index = 0
    for item in build_replay_schedule(frames):
        publish(item.frame, item.phase, item.source_frame_index, item.time_s)
        for _ in range(steps_per_command):
            step_index += 1
            sample_time_s = step_index * collector.sim_dt
            replay_sample = step(sample_time_s)
            if not isinstance(replay_sample, ReplaySample):
                raise TypeError("step must return ReplaySample")
            # The harness owns the deterministic simulation clock. Refuse a
            # callback that reports a different timeline.
            if not math.isclose(float(replay_sample.time_s), sample_time_s, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("step returned a sample on a different replay timeline")
            collector.add(replay_sample)
            pace(collector.sim_dt)
    return collector.finalize()


def wait_for_readiness(
    process: object,
    *,
    probe: Callable[[], bool],
    timeout_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for a child-owned service using a hard monotonic deadline."""

    timeout = _finite_real(timeout_s, "readiness timeout_s", minimum=np.nextafter(0.0, 1.0))
    deadline = monotonic() + timeout
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"owned replay process exited before readiness with code {return_code}")
        try:
            if probe():
                return
        except OSError:
            pass
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"replay readiness timed out after {timeout:g} seconds")
        sleep(min(0.05, remaining))


def stop_owned_process(process: object) -> None:
    """Stop only the supplied owned process: SIGINT, wait, then terminate."""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


__all__ = [
    "CohortReplayReport",
    "ReplayActionFrame",
    "ReplayMetricsCollector",
    "ReplayReport",
    "ReplaySample",
    "ScheduledReplayAction",
    "SourceEpisodeRef",
    "aggregate_replay_reports",
    "build_replay_schedule",
    "collector_from_mujoco_model",
    "max_mujoco_contact_force",
    "resolve_source_episode_ids",
    "run_replay_schedule",
    "stop_owned_process",
    "wait_for_readiness",
]
