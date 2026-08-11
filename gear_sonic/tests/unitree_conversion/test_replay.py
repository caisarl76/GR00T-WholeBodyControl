from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.replay import (
    ReplayActionFrame,
    ReplayMetricsCollector,
    ReplaySample,
    aggregate_replay_reports,
    build_replay_schedule,
    collector_from_mujoco_model,
    max_mujoco_contact_force,
    resolve_source_episode_ids,
    run_replay_schedule,
    stop_owned_process,
    wait_for_readiness,
)
from gear_sonic.scripts.replay_unitree_sonic_smoke import (
    DeadlinePacer,
    DeploymentConfig,
    MujocoZmqRuntime,
    RuntimeDependencies,
    _parse_args,
    build_deployment_command,
    execute_owned_replay,
    frames_from_columns,
    quaternion_wxyz_to_roll_pitch,
    select_robot_actuator_joint_ids,
    send_repeated_command,
    write_json_report,
)


def sample(
    *,
    time_s: float = 1.0,
    root_height: float = 0.75,
    root_roll: float = 0.0,
    root_pitch: float = 0.0,
    positions: tuple[float, float] = (0.0, 0.0),
    velocities: tuple[float, float] = (0.0, 0.0),
    torques: tuple[float, float] = (0.0, 0.0),
    contact_force: float = 0.0,
    controller_fault: bool = False,
) -> ReplaySample:
    return ReplaySample(
        time_s=time_s,
        root_height_m=root_height,
        root_roll_rad=root_roll,
        root_pitch_rad=root_pitch,
        joint_positions=np.array(positions, dtype=np.float64),
        joint_velocities=np.array(velocities, dtype=np.float64),
        torques=np.array(torques, dtype=np.float64),
        contact_force_n=contact_force,
        controller_fault=controller_fault,
    )


def collector(*, mass: float = 1.0) -> ReplayMetricsCollector:
    return ReplayMetricsCollector(
        sim_dt=0.005,
        exclude_before_s=1.0,
        robot_mass_kg=mass,
        effort_limits=np.array([10.0, 20.0]),
        joint_limits=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        effort_limit_source="loaded_model.actuator_forcerange",
    )


def test_metrics_exclude_only_first_second_of_two_second_warmup() -> None:
    metrics = collector(mass=35.0)
    metrics.add(sample(time_s=0.5, torques=(100.0, 100.0), contact_force=99_999.0))
    metrics.add(sample(time_s=1.0, torques=(5.0, 10.0), contact_force=100.0))

    report = metrics.finalize()

    assert report.included_sample_count == 1
    assert report.torque_ratio_max == 0.5
    assert report.contact_force_max == 100.0


def test_metrics_record_linear_p50_p95_p99_and_max() -> None:
    metrics = collector(mass=100.0)
    for index, value in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
        metrics.add(
            sample(
                time_s=1.0 + index * 0.005,
                root_roll=value / 10.0,
                torques=(value * 10.0, 0.0),
            )
        )

    report = metrics.finalize()

    assert report.root_roll_abs_p50 == pytest.approx(0.2)
    assert report.root_roll_abs_p95 == pytest.approx(0.38)
    assert report.root_roll_abs_p99 == pytest.approx(0.396)
    assert report.root_roll_abs_max == pytest.approx(0.4)
    assert report.torque_ratio_p50 == pytest.approx(2.0)
    assert report.torque_ratio_p95 == pytest.approx(3.8)
    assert report.torque_ratio_p99 == pytest.approx(3.96)
    assert report.torque_ratio_max == pytest.approx(4.0)


def test_all_acceptance_thresholds_are_inclusive() -> None:
    metrics = collector(mass=1.0)
    metrics.add(
        sample(
            time_s=1.0,
            root_height=0.45,
            root_roll=0.7,
            root_pitch=-0.7,
            positions=(-1.02, 2.02),
            velocities=(-50.0, 50.0),
            torques=(-10.5, 21.0),
            contact_force=98.1,
        )
    )
    metrics.add(sample(time_s=1.005, root_height=1.05))

    report = metrics.finalize()

    assert report.accepted is True
    assert report.gate_failures == ()
    assert report.root_height_min == 0.45
    assert report.root_height_max == 1.05


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"root_height": 0.449}, "root_height"),
        ({"root_height": 1.051}, "root_height"),
        ({"root_roll": 0.701}, "root_roll"),
        ({"root_pitch": -0.701}, "root_pitch"),
        ({"positions": (-1.021, 0.0)}, "joint_limit_overshoot"),
        ({"velocities": (50.001, 0.0)}, "joint_velocity"),
        ({"torques": (10.501, 0.0)}, "torque_ratio"),
        ({"contact_force": 98.101}, "contact_force"),
        ({"controller_fault": True}, "controller_fault"),
        ({"root_height": math.nan}, "nonfinite"),
        ({"contact_force": -0.001}, "contact_force"),
    ],
)
def test_each_acceptance_gate_fails_immediately_beyond_boundary(
    overrides: dict[str, object],
    failure: str,
) -> None:
    metrics = collector(mass=1.0)
    metrics.add(sample(**overrides))

    report = metrics.finalize()

    assert report.accepted is False
    assert failure in report.gate_failures


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sim_dt", 0.0, "sim_dt"),
        ("sim_dt", math.inf, "sim_dt"),
        ("exclude_before_s", -0.1, "exclude_before_s"),
        ("robot_mass_kg", 0.0, "robot_mass_kg"),
        ("effort_limits", np.array([10.0, 0.0]), "effort_limits"),
        ("effort_limits", np.array([[10.0, 20.0]]), "effort_limits"),
        ("joint_limits", np.array([[-1.0, 1.0]]), "joint_limits"),
        ("joint_limits", np.array([[1.0, -1.0], [-2.0, 2.0]]), "joint_limits"),
    ],
)
def test_collector_rejects_invalid_physics_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "sim_dt": 0.005,
        "exclude_before_s": 1.0,
        "robot_mass_kg": 35.0,
        "effort_limits": np.array([10.0, 20.0]),
        "joint_limits": np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        "effort_limit_source": "model",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        ReplayMetricsCollector(**kwargs)


def test_collector_rejects_wrong_sample_shapes_and_time_order() -> None:
    metrics = collector()
    bad = sample()
    bad.joint_positions = np.zeros(3)
    with pytest.raises(ValueError, match="joint_positions"):
        metrics.add(bad)

    metrics.add(sample(time_s=1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.add(sample(time_s=1.0))


def test_cohort_uses_extrema_not_mean() -> None:
    reports = []
    for torque, height in ((0.4, 0.8), (1.06, 0.7), (0.3, 0.6)):
        metrics = collector(mass=100.0)
        metrics.add(sample(root_height=height, torques=(10.0 * torque, 0.0)))
        reports.append(metrics.finalize())

    cohort = aggregate_replay_reports(reports)

    assert cohort.torque_ratio_max == 1.06
    assert cohort.root_height_min == 0.6
    assert cohort.accepted is False


@pytest.mark.parametrize("field", ["sim_dt", "effort_limit_source", "robot_mass_kg"])
def test_cohort_rejects_mixed_physics_contracts(field: str) -> None:
    first = collector(mass=100.0)
    first.add(sample())
    second = collector(mass=100.0)
    second.add(sample())
    first_report = first.finalize()
    second_values = dict(second.finalize().__dict__)
    second_values[field] = {
        "sim_dt": 0.01,
        "effort_limit_source": "different.model",
        "robot_mass_kg": 101.0,
    }[field]

    with pytest.raises(ValueError, match="same replay physics contract"):
        aggregate_replay_reports([first_report, type(first_report)(**second_values)])


def test_mujoco_contact_force_uses_l2_of_all_six_components() -> None:
    forces = (np.array([3.0, 4.0, 0.0, 0.0, 0.0, 12.0]), np.ones(6))

    class FakeMujoco:
        @staticmethod
        def mj_contactForce(model: object, data: object, index: int, output: np.ndarray) -> None:
            del model, data
            output[:] = forces[index]

    data = SimpleNamespace(ncon=2)

    assert max_mujoco_contact_force(object(), data, mujoco_module=FakeMujoco) == 13.0


def test_collector_loads_mass_effort_and_joint_limits_from_mujoco_model() -> None:
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.005),
        actuator_forcerange=np.array([[-10.0, 8.0], [-19.0, 20.0]]),
        jnt_range=np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]),
    )

    class FakeMujoco:
        @staticmethod
        def mj_getTotalmass(value: object) -> float:
            assert value is model
            return 35.0

    metrics = collector_from_mujoco_model(
        model,
        actuator_ids=np.array([0, 1]),
        joint_ids=np.array([1, 2]),
        mujoco_module=FakeMujoco,
    )

    assert metrics.robot_mass_kg == 35.0
    np.testing.assert_array_equal(metrics.effort_limits, [10.0, 20.0])
    np.testing.assert_array_equal(metrics.joint_limits, [[-2.0, 2.0], [-3.0, 3.0]])
    assert metrics.effort_limit_source == "mujoco.actuator_forcerange"


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "source-manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "dataset-checksums.sha256").write_text(
        f"{digest}  source-manifest.json\n",
        encoding="utf-8",
    )


def test_source_episode_resolution_never_assumes_target_local_index(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        {
            "source_episode_ids": [10, 20],
            "episode_lengths": [3, 4],
            "stages": [{"source_episode_id": 10}, {"source_episode_id": 20}],
        },
    )

    resolved = resolve_source_episode_ids(tmp_path, [20, 10])

    assert [(item.source_episode_id, item.target_episode_index, item.episode_length) for item in resolved] == [
        (20, 1, 4),
        (10, 0, 3),
    ]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "length", "stage", "missing", "checksum", "negative_manifest", "negative_request"],
)
def test_source_episode_resolution_fails_closed_on_bad_provenance(
    mutation: str,
    tmp_path: Path,
) -> None:
    manifest = {
        "source_episode_ids": [10, 20],
        "episode_lengths": [3, 4],
        "stages": [{"source_episode_id": 10}, {"source_episode_id": 20}],
    }
    requested = [10]
    if mutation == "duplicate":
        manifest["source_episode_ids"] = [10, 10]
    elif mutation == "length":
        manifest["episode_lengths"] = [3]
    elif mutation == "stage":
        manifest["stages"] = [{"source_episode_id": 10}, {"source_episode_id": 99}]
    elif mutation == "missing":
        requested = [99]
    elif mutation == "negative_manifest":
        manifest["source_episode_ids"] = [-1, 20]
        manifest["stages"] = [{"source_episode_id": -1}, {"source_episode_id": 20}]
    elif mutation == "negative_request":
        requested = [-1]
    _write_manifest(tmp_path, manifest)
    if mutation == "checksum":
        (tmp_path / "source-manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance|source episode"):
        resolve_source_episode_ids(tmp_path, requested)


def action_frame(value: float) -> ReplayActionFrame:
    return ReplayActionFrame(
        motion_token=np.full(64, value, dtype=np.float32),
        left_hand=np.full(7, value, dtype=np.float32),
        right_hand=np.full(7, value, dtype=np.float32),
    )


def test_schedule_has_exact_warmup_trajectory_hold_counts_and_timing() -> None:
    schedule = build_replay_schedule([action_frame(0.0), action_frame(1.0), action_frame(2.0)])

    assert len(schedule) == 100 + 3 + 50
    assert [item.phase for item in schedule[:100]] == ["warmup"] * 100
    assert [item.source_frame_index for item in schedule[:100]] == [0] * 100
    assert [item.time_s for item in schedule[:100]] == [index / 50.0 for index in range(100)]
    assert [item.source_frame_index for item in schedule[100:103]] == [0, 1, 2]
    assert [item.time_s for item in schedule[100:103]] == [2.0, 2.02, 2.04]
    assert [item.phase for item in schedule[-50:]] == ["hold"] * 50
    assert [item.source_frame_index for item in schedule[-50:]] == [2] * 50
    assert schedule[-1].time_s == 3.04


def test_schedule_execution_samples_every_sim_step() -> None:
    frames = [action_frame(0.0), action_frame(1.0)]
    metrics = collector(mass=100.0)
    published: list[tuple[str, int, float]] = []
    step_times: list[float] = []

    def publish(frame: ReplayActionFrame, phase: str, source_index: int, time_s: float) -> None:
        del frame
        published.append((phase, source_index, time_s))

    def step(time_s: float) -> ReplaySample:
        step_times.append(time_s)
        return sample(time_s=time_s)

    report = run_replay_schedule(
        frames,
        collector=metrics,
        publish=publish,
        step=step,
        pace=lambda delay: None,
    )

    assert len(published) == 100 + 2 + 50
    assert len(step_times) == (100 + 2 + 50) * 4
    assert step_times[0] == 0.005
    assert step_times[-1] == pytest.approx(3.04)
    assert report.duration_s == pytest.approx(3.04)


class FakeProcess:
    def __init__(self, *, exit_code: int | None = None, timeout_once: bool = False) -> None:
        self.exit_code = exit_code
        self.timeout_once = timeout_once
        self.events: list[object] = []

    def poll(self) -> int | None:
        return self.exit_code

    def send_signal(self, value: int) -> None:
        self.events.append(("signal", value))

    def wait(self, timeout: float) -> int:
        self.events.append(("wait", timeout))
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("owned", timeout)
        self.exit_code = 0
        return 0

    def terminate(self) -> None:
        self.events.append("terminate")
        self.exit_code = -15


def test_owned_process_cleanup_is_sigint_then_five_second_wait() -> None:
    process = FakeProcess()

    stop_owned_process(process)

    assert process.events == [("signal", signal.SIGINT), ("wait", 5.0)]


def test_owned_process_cleanup_terminates_only_after_timeout() -> None:
    process = FakeProcess(timeout_once=True)

    stop_owned_process(process)

    assert process.events == [
        ("signal", signal.SIGINT),
        ("wait", 5.0),
        "terminate",
        ("wait", 5.0),
    ]


def test_readiness_is_bounded_and_detects_early_process_failure() -> None:
    process = FakeProcess()
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(delay: float) -> None:
        clock[0] += delay

    with pytest.raises(TimeoutError, match="readiness"):
        wait_for_readiness(
            process,
            probe=lambda: False,
            timeout_s=0.2,
            monotonic=monotonic,
            sleep=sleep,
        )

    process.exit_code = 7
    with pytest.raises(RuntimeError, match="code 7"):
        wait_for_readiness(
            process,
            probe=lambda: False,
            timeout_s=1.0,
            monotonic=monotonic,
            sleep=sleep,
        )


def test_deployment_command_forces_deployed_zmq_manager_path(tmp_path: Path) -> None:
    config = DeploymentConfig(
        binary=tmp_path / "g1_deploy_onnx_ref",
        network_interface="lo",
        policy_file=tmp_path / "policy.onnx",
        motion_data_path=tmp_path / "motions",
        planner_file=tmp_path / "planner.onnx",
        observation_config=tmp_path / "observation_config.yaml",
        encoder_file=tmp_path / "encoder.onnx",
        zmq_host="127.0.0.1",
        action_port=6001,
        state_port=6002,
    )

    command = build_deployment_command(config)

    assert command[:4] == [
        str(config.binary),
        "lo",
        str(config.policy_file),
        str(config.motion_data_path),
    ]
    assert command[command.index("--input-type") + 1] == "zmq_manager"
    assert command[command.index("--output-type") + 1] == "zmq"
    assert command[command.index("--zmq-port") + 1] == "6001"
    assert command[command.index("--zmq-out-port") + 1] == "6002"


def test_task14_plural_cli_uses_repo_defaults_and_dataset_report() -> None:
    args = _parse_args(
        [
            "--dataset-root",
            "/converted",
            "--source-episode-ids",
            "0",
            "78",
            "155",
            "233",
            "310",
        ]
    )

    assert args.source_episode_ids == [0, 78, 155, 233, 310]
    assert args.deploy_binary.endswith("gear_sonic_deploy/target/release/g1_deploy_onnx_ref")
    assert args.observation_config.endswith("gear_sonic_deploy/policy/low_latency/observation_config.yaml")
    assert args.encoder_file.endswith("gear_sonic_deploy/policy/low_latency/model_encoder.onnx")
    assert args.output_report is None


def test_exact_filename_cli_bootstraps_repo_when_pythonpath_has_only_external_packages() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    external_paths = [
        path for path in sys.path if "site-packages" in path and repository_root not in Path(path).parents
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(external_paths)

    result = subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/replay_unitree_sonic_smoke.py",
            "--help",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--source-episode-ids" in result.stdout


def test_frames_are_loaded_from_target_index_with_exact_local_timeline() -> None:
    columns = {
        "episode_index": [4, 4],
        "frame_index": [0, 1],
        "timestamp": [0.0, 0.02],
        "action.motion_token": [np.zeros(64), np.ones(64)],
        "teleop.left_hand_joints": [np.zeros(7), np.ones(7)],
        "teleop.right_hand_joints": [np.ones(7), np.zeros(7)],
    }

    frames = frames_from_columns(columns, target_episode_index=4, expected_length=2)

    assert len(frames) == 2
    assert all(frame.motion_token.dtype == np.float32 for frame in frames)
    np.testing.assert_array_equal(frames[1].motion_token, np.ones(64, dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_index", [4, 3], "episode_index"),
        ("frame_index", [0, 2], "frame_index"),
        ("timestamp", [0.0, 0.021], "timestamp"),
    ],
)
def test_frame_loading_fails_closed_on_local_timeline_mismatch(
    field: str,
    value: list[float] | list[int],
    message: str,
) -> None:
    columns = {
        "episode_index": [4, 4],
        "frame_index": [0, 1],
        "timestamp": [0.0, 0.02],
        "action.motion_token": [np.zeros(64), np.ones(64)],
        "teleop.left_hand_joints": [np.zeros(7), np.ones(7)],
        "teleop.right_hand_joints": [np.ones(7), np.zeros(7)],
    }
    columns[field] = value

    with pytest.raises(ValueError, match=message):
        frames_from_columns(columns, target_episode_index=4, expected_length=2)


def test_wxyz_roll_pitch_conversion_is_explicit() -> None:
    half = math.sqrt(0.5)

    roll, pitch = quaternion_wxyz_to_roll_pitch(np.array([half, half, 0.0, 0.0]))

    assert roll == pytest.approx(math.pi / 2)
    assert pitch == pytest.approx(0.0)


def test_robot_actuator_selection_ignores_free_base_and_preserves_robot_order() -> None:
    model = SimpleNamespace(
        actuator_trnid=np.array(
            [
                [0, 0],  # free-base/general actuator: not a robot joint
                [3, 0],
                [1, 0],
                [2, 0],
                [-1, 0],
            ]
        )
    )

    actuator_ids, joint_ids = select_robot_actuator_joint_ids(model, np.array([1, 2, 3]))

    np.testing.assert_array_equal(actuator_ids, [2, 3, 1])
    np.testing.assert_array_equal(joint_ids, [1, 2, 3])


def test_start_command_repetition_is_bounded_and_deterministic() -> None:
    payloads: list[bytes] = []
    delays: list[float] = []

    send_repeated_command(
        b"start-v4",
        send=payloads.append,
        attempts=3,
        interval_s=0.02,
        sleep=delays.append,
    )

    assert payloads == [b"start-v4"] * 3
    assert delays == [0.02, 0.02]


def test_deadline_pacer_compensates_step_work_to_keep_exact_frequency() -> None:
    clock = [10.0]
    delays: list[float] = []

    def monotonic() -> float:
        return clock[0]

    def sleep(delay: float) -> None:
        delays.append(delay)
        clock[0] += delay

    pacer = DeadlinePacer(monotonic=monotonic, sleep=sleep)
    clock[0] += 0.001  # first simulator step work
    pacer(0.005)
    clock[0] += 0.002  # second simulator step work
    pacer(0.005)

    assert delays == pytest.approx([0.004, 0.003])
    assert clock[0] == pytest.approx(10.01)


class FakeReplayRuntime:
    def __init__(self, metrics: ReplayMetricsCollector, *, publish_error: Exception | None = None):
        self.collector = metrics
        self.publish_error = publish_error
        self.events: list[object] = []

    def readiness_probe(self) -> bool:
        self.events.append("probe")
        return True

    def start_control(self) -> None:
        self.events.append("start")

    def publish(self, frame: ReplayActionFrame, phase: str, source_index: int, time_s: float) -> None:
        del frame, source_index, time_s
        self.events.append(("publish", phase))
        if self.publish_error is not None:
            raise self.publish_error

    def step(self, time_s: float) -> ReplaySample:
        return sample(time_s=time_s)

    def stop_control(self) -> None:
        self.events.append("stop")

    def close(self) -> None:
        self.events.append("close")


def test_owned_replay_cleans_runtime_and_only_its_child_on_failure() -> None:
    runtime = FakeReplayRuntime(collector(mass=100.0), publish_error=RuntimeError("publish failed"))
    process = FakeProcess()

    with pytest.raises(RuntimeError, match="publish failed"):
        execute_owned_replay(
            [action_frame(0.0)],
            deployment_command=["owned-deploy"],
            runtime=runtime,
            process_factory=lambda command: process,
            readiness_timeout_s=1.0,
            pace=lambda delay: None,
        )

    assert runtime.events == ["probe", "start", ("publish", "warmup"), "stop", "close"]
    assert process.events == [("signal", signal.SIGINT), ("wait", 5.0)]


def test_mujoco_runtime_constructor_cleans_partial_simulator_and_zmq_resources() -> None:
    events: list[object] = []
    model = SimpleNamespace(
        nu=3,
        qpos0=np.zeros(10),
        jnt_qposadr=np.array([0, 7, 8, 9]),
        jnt_dofadr=np.array([0, 6, 7, 8]),
        actuator_trnid=np.array([[1, 0], [2, 0], [3, 0]]),
        actuator_forcerange=np.array([[-10.0, 10.0]] * 3),
        jnt_range=np.array([[-1.0, 1.0]] * 4),
        opt=SimpleNamespace(timestep=0.005),
    )
    data = SimpleNamespace(
        qpos=np.zeros(10),
        qvel=np.zeros(9),
        qacc=np.zeros(9),
        ctrl=np.zeros(3),
        time=1.0,
    )
    env = SimpleNamespace(
        mj_model=model,
        mj_data=data,
        body_joint_index=np.array([1]),
        left_hand_index=np.array([2]),
        right_hand_index=np.array([3]),
    )

    class FakeSimulator:
        sim_env = env

        def close(self) -> None:
            events.append("simulator.close")

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        @staticmethod
        def load_wbc_yaml() -> dict[str, object]:
            return {"DEFAULT_DOF_ANGLES": [0.0]}

    class FakeMujoco:
        @staticmethod
        def mj_forward(fake_model: object, fake_data: object) -> None:
            assert fake_model is model and fake_data is data

        @staticmethod
        def mj_getTotalmass(fake_model: object) -> float:
            assert fake_model is model
            return 35.0

    class FakeSocket:
        def setsockopt(self, option: object, value: object) -> None:
            del option, value

        def bind(self, endpoint: str) -> None:
            assert endpoint == "tcp://127.0.0.1:5556"
            raise RuntimeError("bind failed")

        def close(self, linger: int) -> None:
            events.append(("socket.close", linger))

    class FakeContext:
        @staticmethod
        def socket(socket_type: object) -> FakeSocket:
            assert socket_type == 1
            return FakeSocket()

        def term(self) -> None:
            events.append("context.term")

    fake_zmq = SimpleNamespace(PUB=1, LINGER=2, Context=FakeContext)
    dependencies = RuntimeDependencies(
        mujoco=FakeMujoco,
        zmq=fake_zmq,
        simulator_factory=lambda *args, **kwargs: FakeSimulator(),
        config_factory=FakeConfig,
        pack_action=lambda **kwargs: b"pose",
        build_command=lambda **kwargs: b"command",
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        MujocoZmqRuntime(
            network_interface="lo",
            zmq_host="127.0.0.1",
            action_port=5556,
            state_port=5557,
            _dependencies=dependencies,
        )

    assert events == [("socket.close", 0), "context.term", "simulator.close"]


def test_json_report_is_canonical_atomic_and_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "replay.json"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")

    write_json_report(output, {"z": 1, "a": [2, 3]})

    assert output.read_bytes() == b'{"a":[2,3],"z":1}\n'
    assert not list(output.parent.glob(".replay.json.*.tmp"))


def test_json_report_preserves_nonfinite_gate_evidence_as_explicit_strings(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"

    write_json_report(output, {"max": math.inf, "min": -math.inf, "sample": math.nan})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "max": "Infinity",
        "min": "-Infinity",
        "sample": "NaN",
    }
