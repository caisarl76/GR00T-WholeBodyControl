from pathlib import Path
import time

import mujoco
import numpy as np
import pytest
import yaml

from gear_sonic.scripts.verify_inspire_ftp_mujoco import (
    run_contact_cycle,
    run_motor_sweep,
)
import gear_sonic.utils.mujoco_sim.base_sim as base_sim_module
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.inspire_ftp_hand import InspireFtpMujocoPlant
from gear_sonic.utils.teleop.inspire_ftp import CLOSED_RADIANS, OPEN

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml"

ACTIVE_SUFFIXES = (
    "little_1_joint",
    "ring_1_joint",
    "middle_1_joint",
    "index_1_joint",
    "thumb_2_joint",
    "thumb_1_joint",
)

MIMIC_RELATIONS = {
    "index_2_joint": ("index_1_joint", 1.0843),
    "middle_2_joint": ("middle_1_joint", 1.0843),
    "ring_2_joint": ("ring_1_joint", 1.0843),
    "little_2_joint": ("little_1_joint", 1.0843),
    "thumb_3_joint": ("thumb_2_joint", 0.8024),
    "thumb_4_joint": ("thumb_3_joint", 0.9487),
}

PHYSICAL_SUFFIXES = (
    "thumb_1_joint",
    "thumb_2_joint",
    "thumb_3_joint",
    "thumb_4_joint",
    "index_1_joint",
    "index_2_joint",
    "middle_1_joint",
    "middle_2_joint",
    "ring_1_joint",
    "ring_2_joint",
    "little_1_joint",
    "little_2_joint",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@pytest.fixture()
def plant(model):
    return InspireFtpMujocoPlant.resolve(model, mujoco.MjData(model))


def test_inspire_scene_has_expected_joint_actuator_and_equality_counts(model):
    assert model.njnt == 54  # floating base + 29 G1 body + 24 hand
    assert model.nu == 41  # 29 body + 6 active motors per hand
    assert model.neq == 12  # six URDF mimic constraints per hand
    assert model.nexclude == 2  # palm/thumb exclusions for open-hand stability


@pytest.mark.parametrize("side", ("left", "right"))
def test_only_six_named_hand_joints_are_actuated_per_side(model, side):
    expected = {f"{side}_{suffix}" for suffix in ACTIVE_SUFFIXES}
    actual = {
        model.actuator(index).name
        for index in range(model.nu)
        if model.actuator(index).name.startswith(f"{side}_")
        and any(token in model.actuator(index).name for token in ("thumb", "index", "middle", "ring", "little"))
    }

    assert actual == expected


@pytest.mark.parametrize(
    "joint_name",
    (
        "left_index_2_joint",
        "left_middle_2_joint",
        "left_ring_2_joint",
        "left_little_2_joint",
        "left_thumb_3_joint",
        "left_thumb_4_joint",
        "right_index_2_joint",
        "right_middle_2_joint",
        "right_ring_2_joint",
        "right_little_2_joint",
        "right_thumb_3_joint",
        "right_thumb_4_joint",
    ),
)
def test_dependent_joint_has_no_actuator(model, joint_name):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name) == -1


@pytest.mark.parametrize("side", ("left", "right"))
def test_official_urdf_mimic_relations_are_preserved(model, side):
    actual = {}
    for equality_index in range(model.neq):
        equality = model.equality(equality_index)
        if not equality.name.startswith(f"{side}_"):
            continue
        dependent = model.joint(int(model.eq_obj1id[equality_index])).name
        driver = model.joint(int(model.eq_obj2id[equality_index])).name
        actual[dependent] = (driver, model.eq_data[equality_index, 1])

    expected = {
        f"{side}_{dependent}": (f"{side}_{driver}", ratio)
        for dependent, (driver, ratio) in MIMIC_RELATIONS.items()
    }
    assert actual.keys() == expected.keys()
    for dependent, (driver, ratio) in expected.items():
        assert actual[dependent][0] == driver
        assert actual[dependent][1] == pytest.approx(ratio)


def test_named_plant_resolves_six_active_joints_and_actuators_per_side(model, plant):
    assert plant.qpos_addresses["left"].shape == (6,)
    assert plant.qpos_addresses["right"].shape == (6,)
    assert plant.qvel_addresses["left"].shape == (6,)
    assert plant.qvel_addresses["right"].shape == (6,)
    assert plant.actuator_ids["left"].shape == (6,)
    assert plant.actuator_ids["right"].shape == (6,)

    hand_actuator_ids = set(plant.actuator_ids["left"]) | set(plant.actuator_ids["right"])
    body_actuator_ids = set(range(model.nu)) - hand_actuator_ids
    assert len(body_actuator_ids) == 29
    assert body_actuator_ids.isdisjoint(hand_actuator_ids)


def test_named_plant_writes_normalized_open_and_closed_targets(plant):
    plant.write_targets(OPEN, OPEN)
    np.testing.assert_allclose(plant.data.ctrl[plant.actuator_ids["left"]], 0.0)
    np.testing.assert_allclose(plant.data.ctrl[plant.actuator_ids["right"]], 0.0)

    plant.write_targets(np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(plant.data.ctrl[plant.actuator_ids["left"]], CLOSED_RADIANS)
    np.testing.assert_allclose(plant.data.ctrl[plant.actuator_ids["right"]], CLOSED_RADIANS)


def test_named_plant_reads_active_joint_state_in_normalized_contract(plant):
    expected_left = np.linspace(0.0, 1.0, 6)
    expected_right = expected_left[::-1]
    plant.data.qpos[plant.qpos_addresses["left"]] = (1.0 - expected_left) * CLOSED_RADIANS
    plant.data.qpos[plant.qpos_addresses["right"]] = (1.0 - expected_right) * CLOSED_RADIANS

    actual_left, actual_right = plant.read_normalized_state()

    np.testing.assert_allclose(actual_left, expected_left)
    np.testing.assert_allclose(actual_right, expected_right)


def test_named_plant_projects_soft_limit_measurements_to_normalized_endpoints(plant):
    left_radians = np.zeros(6)
    right_radians = CLOSED_RADIANS.copy()
    left_radians[4] = -0.01
    right_radians[5] += 0.02
    plant.data.qpos[plant.qpos_addresses["left"]] = left_radians
    plant.data.qpos[plant.qpos_addresses["right"]] = right_radians

    left, right = plant.read_normalized_state()

    np.testing.assert_allclose(left, OPEN)
    np.testing.assert_allclose(right, np.zeros(6))
    assert plant.last_measurement_limit_error_rad == pytest.approx(0.02)


def test_open_hand_excludes_palm_thumb_mesh_penetration():
    local_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    local_model.opt.gravity[:] = 0.0
    data = mujoco.MjData(local_model)
    plant = InspireFtpMujocoPlant.resolve(local_model, data)
    plant.write_targets(OPEN, OPEN)

    for _ in range(5):
        mujoco.mj_step(local_model, data)

    contact_body_pairs = {
        frozenset(
            (
                local_model.body(local_model.geom_bodyid[contact.geom1]).name,
                local_model.body(local_model.geom_bodyid[contact.geom2]).name,
            )
        )
        for contact in data.contact
    }
    assert frozenset(("left_wrist_yaw_link", "left_thumb_2")) not in contact_body_pairs
    assert frozenset(("right_wrist_yaw_link", "right_thumb_2")) not in contact_body_pairs


class _NoNetworkSubscriber:
    def __init__(self, *, state, **kwargs):
        del kwargs
        self.state = state

    def poll(self, *, now):
        del now
        return False

    def close(self):
        pass


class _BodyOnlyBridge:
    low_cmd = None
    joystick = None

    def PublishLowState(self, observation):
        self.observation = observation


def test_default_env_installs_inspire_plant_with_physical_and_motor_counts(
    monkeypatch,
):
    monkeypatch.setattr(base_sim_module, "InspireFtpZmqSubscriber", _NoNetworkSubscriber)
    config_path = REPO_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(
        {
            "ROBOT_SCENE": "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml",
            "HAND_TYPE": "inspire_ftp",
            "NUM_HAND_MOTORS": 6,
            "NUM_HAND_JOINTS": 12,
            "LEFT_HAND_JOINT_NAMES": [f"left_{suffix}" for suffix in PHYSICAL_SUFFIXES],
            "RIGHT_HAND_JOINT_NAMES": [f"right_{suffix}" for suffix in PHYSICAL_SUFFIXES],
            "LEFT_HAND_ACTUATOR_NAMES": [f"left_{suffix}" for suffix in ACTIVE_SUFFIXES],
            "RIGHT_HAND_ACTUATOR_NAMES": [f"right_{suffix}" for suffix in ACTIVE_SUFFIXES],
            "ENABLE_DEX3_DDS_HANDS": False,
            "enable_waist": True,
        }
    )

    env = DefaultEnv(config)
    try:
        assert env.torques.shape == (29,)
        assert env.left_hand_index.shape == (12,)
        assert env.right_hand_index.shape == (12,)
        assert env.inspire_hand_plant.actuator_ids["left"].shape == (6,)
        assert env.inspire_hand_plant.actuator_ids["right"].shape == (6,)
        np.testing.assert_allclose(env.mj_data.ctrl[env.inspire_hand_plant.actuator_ids["left"]], 0.0)

        observation = env.prepare_obs()
        assert observation["body_q"].shape == (29,)
        assert observation["left_hand_q"].shape == (12,)
        assert observation["right_hand_q"].shape == (12,)
        np.testing.assert_allclose(observation["left_hand_normalized"], OPEN)
        np.testing.assert_allclose(observation["right_hand_normalized"], OPEN)

        env.set_unitree_bridge(_BodyOnlyBridge())
        env.elastic_band = None
        env.check_fall = lambda: None
        env.inspire_hand_subscriber.state.accept(np.zeros(6), np.zeros(6), now=time.monotonic())
        env.sim_step()
        np.testing.assert_allclose(
            env.mj_data.ctrl[env.inspire_hand_plant.actuator_ids["left"]],
            CLOSED_RADIANS,
        )
        np.testing.assert_allclose(
            env.mj_data.ctrl[env.inspire_hand_plant.actuator_ids["right"]],
            CLOSED_RADIANS,
        )
    finally:
        env.close()


def test_headless_motor_sweep_isolated_motion_coupling_and_limits():
    report = run_motor_sweep(SCENE_PATH, duration_s=2.0)

    assert report["passed"] is True
    assert report["model"] == {"njnt": 54, "nu": 41, "neq": 12}
    assert len(report["motors"]) == 12
    assert report["max_constraint_error_rad"] < 2e-3
    assert report["max_joint_limit_error_rad"] <= 1e-6
    for result in report["motors"]:
        assert result["passed"] is True
        assert result["finite"] is True
        assert result["target_progress"] > 0.5
        assert result["max_unrelated_active_rad"] < 1e-3


def test_contact_enabled_smooth_cycle_keeps_mimic_coupling_stiff():
    report = run_contact_cycle(SCENE_PATH, duration_s=15.0, open_hold_s=5.0)

    assert report["passed"] is True
    assert report["contacts_enabled"] is True
    assert report["max_contacts"] > 0
    assert report["finite"] is True
    assert report["max_constraint_error_rad"] < 2e-3
    assert report["max_joint_limit_error_rad"] <= 1e-6
