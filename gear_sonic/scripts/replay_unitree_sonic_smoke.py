#!/usr/bin/env python3
"""Replay converted Unitree SONIC episodes through deployment and MuJoCo.

Heavy runtime dependencies (MuJoCo, ZMQ, PyArrow, and the simulator stack) are
loaded only after CLI/provenance validation. Importing this module is therefore
safe in the lightweight dataset-conversion environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
import time
from typing import Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gear_sonic.data.unitree_conversion.replay import (
    ReplayActionFrame,
    ReplayMetricsCollector,
    ReplayReport,
    ReplaySample,
    SourceEpisodeRef,
    aggregate_replay_reports,
    collector_from_mujoco_model,
    max_mujoco_contact_force,
    resolve_source_episode_ids,
    run_replay_schedule,
    stop_owned_process,
    wait_for_readiness,
)

REPLAY_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOY_BINARY = REPOSITORY_ROOT / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
DEFAULT_POLICY_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/policy/release/model_decoder.onnx"
DEFAULT_MOTION_DATA = REPOSITORY_ROOT / "gear_sonic_deploy/reference/example"
DEFAULT_PLANNER_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
DEFAULT_OBSERVATION_CONFIG = REPOSITORY_ROOT / "gear_sonic_deploy/policy/low_latency/observation_config.yaml"
DEFAULT_ENCODER_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/policy/low_latency/model_encoder.onnx"


@dataclass(frozen=True)
class DeploymentConfig:
    binary: Path
    network_interface: str
    policy_file: Path
    motion_data_path: Path
    planner_file: Path
    observation_config: Path
    encoder_file: Path
    zmq_host: str = "127.0.0.1"
    action_port: int = 5556
    state_port: int = 5557

    def __post_init__(self) -> None:
        if not self.network_interface:
            raise ValueError("network_interface must not be empty")
        if not self.zmq_host:
            raise ValueError("zmq_host must not be empty")
        for name in ("action_port", "state_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer TCP port")
        if self.action_port == self.state_port:
            raise ValueError("action_port and state_port must differ")


class ReplayRuntime(Protocol):
    collector: ReplayMetricsCollector

    def readiness_probe(self) -> bool: ...

    def start_control(self) -> None: ...

    def publish(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None: ...

    def step(self, time_s: float) -> ReplaySample: ...

    def stop_control(self) -> None: ...

    def close(self) -> None: ...


class DeadlinePacer:
    """Pace simulator steps against an absolute monotonic timeline."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._monotonic = monotonic
        self._sleep = sleep
        self._start = monotonic()
        self._period_s: float | None = None
        self._step_count = 0

    def __call__(self, period_s: float) -> None:
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("pacing period must be finite and positive")
        if self._period_s is None:
            self._period_s = period_s
        elif period_s != self._period_s:
            raise ValueError("pacing period cannot change during replay")
        self._step_count += 1
        deadline = self._start + self._step_count * period_s
        remaining = deadline - self._monotonic()
        if remaining > 0.0:
            self._sleep(remaining)


def build_deployment_command(config: DeploymentConfig) -> list[str]:
    """Build the deployed low-latency invocation with no alternate input path."""

    return [
        str(config.binary),
        config.network_interface,
        str(config.policy_file),
        str(config.motion_data_path),
        "--planner-file",
        str(config.planner_file),
        "--obs-config",
        str(config.observation_config),
        "--encoder-file",
        str(config.encoder_file),
        "--input-type",
        "zmq_manager",
        "--output-type",
        "zmq",
        "--zmq-host",
        config.zmq_host,
        "--zmq-port",
        str(config.action_port),
        "--zmq-topic",
        "pose",
        "--zmq-out-port",
        str(config.state_port),
        "--zmq-out-topic",
        "g1_debug",
        "--disable-crc-check",
    ]


def frames_from_columns(
    columns: Mapping[str, Sequence[object]],
    *,
    target_episode_index: int,
    expected_length: int,
) -> tuple[ReplayActionFrame, ...]:
    """Validate one target-local Parquet episode and construct wire actions."""

    missing = [name for name in REPLAY_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"replay dataset columns are missing: {missing}")
    if isinstance(target_episode_index, bool) or not isinstance(target_episode_index, int):
        raise ValueError("target_episode_index must be an integer")
    if isinstance(expected_length, bool) or not isinstance(expected_length, int) or expected_length <= 0:
        raise ValueError("expected_length must be a positive integer")
    if any(len(columns[name]) != expected_length for name in REPLAY_COLUMNS):
        raise ValueError("replay columns do not match the immutable episode length")

    episode_indices = np.asarray(columns["episode_index"])
    frame_indices = np.asarray(columns["frame_index"])
    timestamps = np.asarray(columns["timestamp"])
    if episode_indices.dtype.kind not in "iu" or not np.array_equal(
        episode_indices, np.full(expected_length, target_episode_index, dtype=episode_indices.dtype)
    ):
        raise ValueError("episode_index does not match the provenance-resolved target index")
    if frame_indices.dtype.kind not in "iu" or not np.array_equal(
        frame_indices, np.arange(expected_length, dtype=frame_indices.dtype)
    ):
        raise ValueError("frame_index must be the exact contiguous local timeline")
    expected_timestamps = np.asarray(
        [np.float32(index / 50.0) for index in range(expected_length)], dtype=np.float32
    )
    try:
        actual_timestamps = timestamps.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp must contain numeric 50 Hz values") from error
    if not np.all(np.isfinite(actual_timestamps)) or not np.array_equal(actual_timestamps, expected_timestamps):
        raise ValueError("timestamp must equal float32(frame_index / 50) exactly")

    frames: list[ReplayActionFrame] = []
    for index in range(expected_length):
        frames.append(
            ReplayActionFrame(
                motion_token=np.asarray(columns["action.motion_token"][index], dtype=np.float32),
                left_hand=np.asarray(columns["teleop.left_hand_joints"][index], dtype=np.float32),
                right_hand=np.asarray(columns["teleop.right_hand_joints"][index], dtype=np.float32),
            )
        )
    return tuple(frames)


def _arrow_column_values(table: object, name: str) -> list[object]:
    try:
        column = table.column(name)
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        return column.to_pylist()
    except (AttributeError, KeyError) as error:
        raise ValueError(f"replay Parquet is missing or cannot decode column {name}") from error


def load_episode_frames(dataset_root: Path, episode: SourceEpisodeRef) -> tuple[ReplayActionFrame, ...]:
    """Load a provenance-resolved local episode. Heavy readers stay lazy."""

    import pyarrow.parquet as pq

    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    root = dataset_root.resolve(strict=True)
    metadata = Gr00tDatasetMetadata(repo_id="tmp/unitree_sonic_replay", root=root)
    relative = Path(metadata.get_data_file_path(episode.target_episode_index))
    parquet_path = (root / relative).resolve(strict=True)
    try:
        parquet_path.relative_to(root)
    except ValueError as error:
        raise ValueError("resolved replay Parquet escapes the dataset root") from error
    if parquet_path.is_symlink() or not parquet_path.is_file():
        raise ValueError("resolved replay Parquet is missing or unsafe")
    table = pq.read_table(parquet_path, columns=list(REPLAY_COLUMNS))
    columns = {name: _arrow_column_values(table, name) for name in REPLAY_COLUMNS}
    return frames_from_columns(
        columns,
        target_episode_index=episode.target_episode_index,
        expected_length=episode.episode_length,
    )


def quaternion_wxyz_to_roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("root quaternion must be a finite WXYZ vector")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("root quaternion must have nonzero norm")
    w, x, y, z = value / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def select_robot_actuator_joint_ids(
    model: object,
    robot_joint_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair exactly one actuator with each loaded body/hand robot joint."""

    joints = np.asarray(robot_joint_ids)
    if joints.ndim != 1 or joints.size == 0 or joints.dtype.kind not in "iu":
        raise ValueError("robot_joint_ids must be a non-empty 1D integer array")
    joints = joints.astype(np.int64, copy=False)
    if np.any(joints < 0) or np.unique(joints).size != joints.size:
        raise ValueError("robot_joint_ids must contain unique nonnegative IDs")
    transmissions = np.asarray(getattr(model, "actuator_trnid"))
    if transmissions.ndim != 2 or transmissions.shape[1] < 1:
        raise ValueError("loaded model actuator_trnid must have shape (A, >=1)")
    transmission_joints = transmissions[:, 0]
    actuator_ids: list[int] = []
    for joint_id in joints:
        matches = np.flatnonzero(transmission_joints == joint_id)
        if matches.size != 1:
            raise ValueError(f"loaded robot joint {joint_id} must map to exactly one actuator; got {matches.size}")
        actuator_ids.append(int(matches[0]))
    return np.asarray(actuator_ids, dtype=np.int64), joints.copy()


def send_repeated_command(
    payload: bytes,
    *,
    send: Callable[[bytes], None],
    attempts: int = 3,
    interval_s: float = 0.02,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Bound PUB/SUB slow-joiner risk with a fixed number of idempotent sends."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("command payload must be non-empty bytes")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("command attempts must be a positive integer")
    if not math.isfinite(interval_s) or interval_s < 0.0:
        raise ValueError("command interval_s must be finite and nonnegative")
    for attempt in range(attempts):
        send(payload)
        if attempt + 1 < attempts:
            sleep(interval_s)


def execute_owned_replay(
    frames: Sequence[ReplayActionFrame],
    *,
    deployment_command: Sequence[str],
    runtime: ReplayRuntime,
    process_factory: Callable[[Sequence[str]], object] = subprocess.Popen,
    readiness_timeout_s: float = 20.0,
    pace: Callable[[float], None] | None = None,
) -> ReplayReport:
    """Run one episode and clean exactly the runtime and child created for it."""

    if not deployment_command or any(not isinstance(item, str) or not item for item in deployment_command):
        raise ValueError("deployment_command must be a non-empty string sequence")
    process: object | None = None
    control_started = False
    try:
        process = process_factory(list(deployment_command))
        attach = getattr(runtime, "attach_process", None)
        if callable(attach):
            attach(process)
        wait_for_readiness(
            process,
            probe=runtime.readiness_probe,
            timeout_s=readiness_timeout_s,
        )
        runtime.start_control()
        control_started = True
        return run_replay_schedule(
            frames,
            collector=runtime.collector,
            publish=runtime.publish,
            step=runtime.step,
            pace=DeadlinePacer() if pace is None else pace,
        )
    finally:
        try:
            if control_started:
                runtime.stop_control()
        finally:
            try:
                runtime.close()
            finally:
                if process is not None:
                    stop_owned_process(process)


@dataclass(frozen=True)
class RuntimeDependencies:
    """Late-bound heavy runtime components, injectable for failure testing."""

    mujoco: object
    zmq: object
    simulator_factory: Callable[..., object]
    config_factory: Callable[..., object]
    pack_action: Callable[..., bytes]
    build_command: Callable[..., bytes]


def _load_runtime_dependencies() -> RuntimeDependencies:
    import mujoco
    import zmq

    from gear_sonic.scripts.run_vla_inference import pack_latent_action_message
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    return RuntimeDependencies(
        mujoco=mujoco,
        zmq=zmq,
        simulator_factory=BaseSimulator,
        config_factory=SimLoopConfig,
        pack_action=pack_latent_action_message,
        build_command=build_command_message,
    )


class MujocoZmqRuntime:
    """Concrete deterministic headless simulator and protocol-v4 transport."""

    def __init__(
        self,
        *,
        network_interface: str,
        zmq_host: str,
        action_port: int,
        state_port: int,
        _dependencies: RuntimeDependencies | None = None,
    ):
        self._simulator: object | None = None
        self._context: object | None = None
        self._publisher: object | None = None
        self._process: object | None = None
        self._closed = False
        dependencies = _dependencies or _load_runtime_dependencies()
        self._mujoco = dependencies.mujoco
        self._pack = dependencies.pack_action
        self._build_command = dependencies.build_command
        self._host = zmq_host
        self._state_port = state_port
        try:
            random.seed(0)
            np.random.seed(0)
            config = dependencies.config_factory(
                interface=network_interface,
                sim_frequency=200,
                enable_onscreen=False,
                enable_offscreen=False,
                enable_image_publish=False,
                with_hands=True,
                verbose=False,
            )
            wbc_config = config.load_wbc_yaml()
            wbc_config["ENV_NAME"] = "default"
            self._simulator = dependencies.simulator_factory(
                wbc_config,
                env_name="default",
                onscreen=False,
                offscreen=False,
                enable_image_publish=False,
            )

            env = self._simulator.sim_env
            model = env.mj_model
            data = env.mj_data
            # MuJoCo qpos0 supplies the standing pelvis height. The loaded WBC
            # configuration supplies the exact nominal 29-body-joint posture.
            data.qpos[:] = model.qpos0
            body_joint_ids = np.asarray(env.body_joint_index, dtype=np.int64)
            body_qpos_addresses = np.asarray(model.jnt_qposadr, dtype=np.int64)[body_joint_ids]
            nominal = np.asarray(wbc_config["DEFAULT_DOF_ANGLES"], dtype=np.float64)
            if nominal.shape != body_joint_ids.shape:
                raise ValueError("loaded simulator nominal posture does not match its body joints")
            data.qpos[body_qpos_addresses] = nominal
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[:] = 0.0
            data.time = 0.0
            self._mujoco.mj_forward(model, data)

            robot_joint_ids = np.concatenate(
                (
                    np.asarray(env.body_joint_index, dtype=np.int64),
                    np.asarray(env.left_hand_index, dtype=np.int64),
                    np.asarray(env.right_hand_index, dtype=np.int64),
                )
            )
            actuator_ids, joint_ids = select_robot_actuator_joint_ids(model, robot_joint_ids)
            self._actuator_ids = actuator_ids
            self._joint_ids = joint_ids
            self._qpos_addresses = np.asarray(model.jnt_qposadr, dtype=np.int64)[joint_ids]
            self._qvel_addresses = np.asarray(model.jnt_dofadr, dtype=np.int64)[joint_ids]
            self.collector = collector_from_mujoco_model(
                model,
                actuator_ids=actuator_ids,
                joint_ids=joint_ids,
                exclude_before_s=1.0,
                mujoco_module=self._mujoco,
            )

            self._context = dependencies.zmq.Context()
            self._publisher = self._context.socket(dependencies.zmq.PUB)
            self._publisher.setsockopt(dependencies.zmq.LINGER, 0)
            self._publisher.bind(f"tcp://{zmq_host}:{action_port}")
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def attach_process(self, process: object) -> None:
        self._process = process

    def readiness_probe(self) -> bool:
        try:
            with socket.create_connection((self._host, self._state_port), timeout=0.1):
                return True
        except OSError:
            return False

    def _send(self, payload: bytes) -> None:
        if self._process is not None:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(f"owned deployment exited during replay with code {return_code}")
        self._publisher.send(payload)

    def start_control(self) -> None:
        send_repeated_command(
            self._build_command(start=True, stop=False, planner=False),
            send=self._send,
            attempts=3,
            interval_s=0.02,
        )

    def publish(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None:
        del phase, time_s
        self._send(
            self._pack(
                motion_token=frame.motion_token,
                frame_index=np.asarray([source_index], dtype=np.int64),
                left_hand_joints=frame.left_hand,
                right_hand_joints=frame.right_hand,
            )
        )

    def step(self, time_s: float) -> ReplaySample:
        env = self._simulator.sim_env
        env.sim_step()
        data = env.mj_data
        roll, pitch = quaternion_wxyz_to_roll_pitch(np.asarray(data.qpos[3:7]))
        warning_fault = any(int(warning.number) > 0 for warning in data.warning)
        process_fault = self._process is not None and self._process.poll() is not None
        return ReplaySample(
            time_s=time_s,
            root_height_m=float(data.qpos[2]),
            root_roll_rad=roll,
            root_pitch_rad=pitch,
            joint_positions=np.asarray(data.qpos[self._qpos_addresses], dtype=np.float64).copy(),
            joint_velocities=np.asarray(data.qvel[self._qvel_addresses], dtype=np.float64).copy(),
            torques=np.asarray(data.actuator_force[self._actuator_ids], dtype=np.float64).copy(),
            contact_force_n=max_mujoco_contact_force(
                env.mj_model,
                data,
                mujoco_module=self._mujoco,
            ),
            controller_fault=warning_fault or process_fault or bool(getattr(env, "fall", False)),
        )

    def stop_control(self) -> None:
        if self._process is None or self._process.poll() is None:
            self._publisher.send(self._build_command(start=False, stop=True, planner=False))
            time.sleep(0.01)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for resource, operation in (
            (self._publisher, lambda value: value.close(linger=0)),
            (self._context, lambda value: value.term()),
            (self._simulator, lambda value: value.close()),
        ):
            if resource is None:
                continue
            try:
                operation(resource)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def write_json_report(path: str | Path, report: Mapping[str, object]) -> None:
    """Write a canonical JSON report by fsync + atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return "NaN"
            return "Infinity" if value > 0.0 else "-Infinity"
        return value

    payload = (
        json.dumps(json_safe(report), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_file(path: str | Path, argument: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ValueError(f"{argument} is not an existing file: {value}")
    return value


def _existing_directory(path: str | Path, argument: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise ValueError(f"{argument} is not an existing directory: {value}")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--deploy-binary", default=str(DEFAULT_DEPLOY_BINARY))
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--policy-file", default=str(DEFAULT_POLICY_FILE))
    parser.add_argument("--motion-data-path", default=str(DEFAULT_MOTION_DATA))
    parser.add_argument("--planner-file", default=str(DEFAULT_PLANNER_FILE))
    parser.add_argument("--observation-config", default=str(DEFAULT_OBSERVATION_CONFIG))
    parser.add_argument("--encoder-file", default=str(DEFAULT_ENCODER_FILE))
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=5556)
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--readiness-timeout-s", type=float, default=20.0)
    parser.add_argument("--output-report")
    parsed = parser.parse_args(argv)
    if not 1 <= len(parsed.source_episode_ids) <= 5:
        parser.error("smoke replay requires between one and five source episode IDs")
    return parsed


def _run(args: argparse.Namespace) -> int:
    dataset_root = _existing_directory(args.dataset_root, "--dataset-root")
    episodes = resolve_source_episode_ids(dataset_root, args.source_episode_ids)
    config = DeploymentConfig(
        binary=_existing_file(args.deploy_binary, "--deploy-binary"),
        network_interface=args.network_interface,
        policy_file=_existing_file(args.policy_file, "--policy-file"),
        motion_data_path=_existing_directory(args.motion_data_path, "--motion-data-path"),
        planner_file=_existing_file(args.planner_file, "--planner-file"),
        observation_config=_existing_file(args.observation_config, "--observation-config"),
        encoder_file=_existing_file(args.encoder_file, "--encoder-file"),
        zmq_host=args.zmq_host,
        action_port=args.action_port,
        state_port=args.state_port,
    )
    command = build_deployment_command(config)
    episode_reports: list[ReplayReport] = []
    serialized: list[dict[str, object]] = []
    for episode in episodes:
        try:
            frames = load_episode_frames(dataset_root, episode)
            runtime = MujocoZmqRuntime(
                network_interface=config.network_interface,
                zmq_host=config.zmq_host,
                action_port=config.action_port,
                state_port=config.state_port,
            )
            report = execute_owned_replay(
                frames,
                deployment_command=command,
                runtime=runtime,
                readiness_timeout_s=args.readiness_timeout_s,
            )
            episode_reports.append(report)
            serialized.append(
                {
                    "source_episode_id": episode.source_episode_id,
                    "target_episode_index": episode.target_episode_index,
                    "status": "accepted" if report.accepted else "replay_acceptance_error",
                    "report": asdict(report),
                }
            )
        except Exception as error:
            serialized.append(
                {
                    "source_episode_id": episode.source_episode_id,
                    "target_episode_index": episode.target_episode_index,
                    "status": "replay_acceptance_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    cohort = aggregate_replay_reports(episode_reports) if episode_reports else None
    complete = len(episode_reports) == len(episodes)
    accepted = complete and cohort is not None and cohort.accepted
    output_report = (
        Path(args.output_report).expanduser().resolve()
        if args.output_report is not None
        else dataset_root / "unitree-sonic-replay-report.json"
    )
    write_json_report(
        output_report,
        {
            "accepted": accepted,
            "cohort": asdict(cohort) if cohort is not None else None,
            "episode_reports": serialized,
            "protocol": {
                "command_hz": 50,
                "exclude_before_s": 1.0,
                "final_hold_s": 1.0,
                "seed": 0,
                "warmup_s": 2.0,
            },
            "source_episode_ids": [episode.source_episode_id for episode in episodes],
        },
    )
    return 0 if accepted else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(_parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"replay setup failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
