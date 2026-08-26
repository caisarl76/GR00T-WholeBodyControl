from pathlib import Path

import mujoco
import pytest

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


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def test_inspire_scene_has_expected_joint_actuator_and_equality_counts(model):
    assert model.njnt == 54  # floating base + 29 G1 body + 24 hand
    assert model.nu == 41  # 29 body + 6 active motors per hand
    assert model.neq == 12  # six URDF mimic constraints per hand


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
