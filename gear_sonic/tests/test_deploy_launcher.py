import os
from pathlib import Path
import shutil
import subprocess

SCRIPT = Path(__file__).parents[2] / "gear_sonic_deploy" / "deploy.sh"


def run_launcher(tmp_path: Path, *args: str, lsof_output: str = "") -> subprocess.CompletedProcess[str]:
    deploy = tmp_path / "gear_sonic_deploy"
    scripts = deploy / "scripts"
    bin_dir = tmp_path / "bin"
    deploy.mkdir()
    scripts.mkdir()
    bin_dir.mkdir()
    shutil.copy2(SCRIPT, deploy / "deploy.sh")
    (scripts / "setup_env.sh").write_text("")
    (scripts / "preflight.sh").write_text("exit 0\n")
    (scripts / "install_deps.sh").write_text("exit 0\n")
    for path in (
        deploy / "policy/release/model_decoder.onnx",
        deploy / "policy/release/model_encoder.onnx",
        deploy / "policy/release/observation_config.yaml",
        deploy / "planner/target_vel/V2/planner_sonic.onnx",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (deploy / "reference/example").mkdir(parents=True)

    (bin_dir / "uname").write_text("#!/bin/sh\necho Linux\n")
    (bin_dir / "ip").write_text("#!/bin/sh\nprintf '1: lo: <LOOPBACK>\\n    inet 127.0.0.1/8 scope host lo\\n'\n")
    (bin_dir / "lsof").write_text("#!/bin/sh\nprintf '%s' \"$LSOF_OUTPUT\"\n")
    (bin_dir / "just").write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$JUST_LOG"\n')
    for command in ("uname", "ip", "lsof", "just"):
        (bin_dir / command).chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["JUST_LOG"] = str(tmp_path / "just.log")
    env["LSOF_OUTPUT"] = lsof_output
    return subprocess.run(
        ["bash", str(deploy / "deploy.sh"), *args],
        input="y\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=deploy,
    )


def test_unknown_option_fails_before_launch(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--not-a-real-option", "sim")

    assert result.returncode == 1
    assert "unknown option: --not-a-real-option" in result.stderr
    assert not (tmp_path / "just.log").exists()


def test_disable_dex3_hands_and_sim_crc_reach_just(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--disable-dex3-hands", "--input-type", "zmq_manager", "sim")

    assert result.returncode == 0
    command = (tmp_path / "just.log").read_text()
    assert "--disable-crc-check" in command
    assert "--disable-dex3-hands" in command
    assert "--input-type zmq_manager" in command
    assert "run g1_deploy_onnx_ref lo " in command


def test_default_sim_keeps_dex3_enabled(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--input-type", "zmq_manager", "sim")
    assert result.returncode == 0
    command = (tmp_path / "just.log").read_text()
    assert "--disable-crc-check" in command
    assert "--disable-dex3-hands" not in command


def test_live_pose_real_interface_preserves_crc_and_dex3(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--live-pose-playback", "--input-type", "zmq_manager", "robot_eth")
    assert result.returncode == 0
    command = (tmp_path / "just.log").read_text()
    assert "run g1_deploy_onnx_ref robot_eth " in command
    assert "--live-pose-playback" in command
    assert "--disable-crc-check" not in command
    assert "--disable-dex3-hands" not in command


def test_occupied_zmq_port_fails_before_build(tmp_path: Path) -> None:
    result = run_launcher(
        tmp_path,
        "sim",
        lsof_output="COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\nworker 42\n",
    )

    assert result.returncode == 1
    assert "ZMQ output port 5557 is already in use" in result.stderr
    assert not (tmp_path / "just.log").exists()


def test_dry_run_prints_live_pose_command_even_with_busy_port(tmp_path: Path) -> None:
    result = run_launcher(
        tmp_path, "--dry-run", "--live-pose-playback", "sim", lsof_output="worker 42\n"
    )
    assert result.returncode == 0
    assert "--live-pose-playback" in result.stdout
    assert not (tmp_path / "just.log").exists()
