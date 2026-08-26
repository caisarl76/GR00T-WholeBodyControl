from pathlib import Path
import subprocess

import pytest

from gear_sonic.scripts.launch_data_collection import (
    DataCollectionLaunchConfig,
    _build_deploy_command,
    _build_pico_command,
    _build_sim_command,
    main,
)
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sim_loop_profile_loads_inspire_scene_counts_and_ownership():
    config = SimLoopConfig(
        hand_profile="inspire_ftp", enable_onscreen=False, enable_offscreen=False
    ).load_wbc_yaml()

    assert config["HAND_TYPE"] == "inspire_ftp"
    assert config["ROBOT_SCENE"].endswith("scene_41dof_inspire_ftp.xml")
    assert config["NUM_HAND_MOTORS"] == 6
    assert config["NUM_HAND_JOINTS"] == 12
    assert len(config["LEFT_HAND_JOINT_NAMES"]) == 12
    assert len(config["RIGHT_HAND_JOINT_NAMES"]) == 12
    assert len(config["LEFT_HAND_ACTUATOR_NAMES"]) == 6
    assert len(config["RIGHT_HAND_ACTUATOR_NAMES"]) == 6
    assert config["ENABLE_DEX3_DDS_HANDS"] is False


def test_inspire_launcher_routes_one_profile_to_sim_pico_and_cpp():
    config = DataCollectionLaunchConfig(sim=True, hand_profile="inspire_ftp")

    sim_command = _build_sim_command(config, REPO_ROOT)
    pico_command = _build_pico_command(config, REPO_ROOT)
    deploy_command = _build_deploy_command(config, REPO_ROOT)

    assert "run_sim_loop.py --hand-profile inspire_ftp" in sim_command
    assert "pico_manager_thread_server.py --hand-profile inspire_ftp" in pico_command
    assert "./deploy.sh --disable-dex3-hands" in deploy_command
    assert deploy_command.endswith("sim")


def test_default_launcher_keeps_dex3_flags_implicit():
    config = DataCollectionLaunchConfig(sim=True)

    assert _build_sim_command(config, REPO_ROOT) == (
        f"cd {REPO_ROOT} && source .venv_sim/bin/activate && "
        "python gear_sonic/scripts/run_sim_loop.py --enable-image-publish "
        "--enable-offscreen --camera-port 5555"
    )
    assert _build_pico_command(config, REPO_ROOT) == (
        f"cd {REPO_ROOT} && source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/pico_manager_thread_server.py --manager"
    )
    assert _build_deploy_command(config, REPO_ROOT) == (
        f"cd {REPO_ROOT / 'gear_sonic_deploy'} && ./deploy.sh --input-type zmq_manager --zmq-host localhost sim"
    )


def test_launcher_rejects_real_inspire_before_external_actions():
    with pytest.raises(ValueError, match="simulation-only"):
        main(DataCollectionLaunchConfig(sim=False, hand_profile="inspire_ftp"))


def test_deploy_shell_dry_run_forwards_disable_dex3_flag():
    result = subprocess.run(
        ["bash", "deploy.sh", "--dry-run", "--disable-dex3-hands", "sim"],
        cwd=REPO_ROOT / "gear_sonic_deploy",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "g1_deploy_onnx_ref" in result.stdout
    assert "--disable-crc-check" in result.stdout
    assert "--disable-dex3-hands" in result.stdout
