from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import pytest
from test_handover_ready import NAMES, RAISED
from test_harness_control import PROFILE
import yaml

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference import ready_clearance as rc
from gear_sonic.utils.inference.harness_profile import load_profile

READY = [-0.10413093119859695, -0.08507601171731949, 0.30275747179985046, -0.21820877492427826,
         1.70130455493927, -0.09610150009393692, -0.07287607342004776]
# 85 cm desk with the 0.787 m standing pelvis height used in simulation; 40 cm near edge (provisional).
DESK = dict(desk_top_above_pelvis_m=0.063, desk_front_from_pelvis_m=0.40)
# Motor-order body joints and thumb-first right-hand joints. The recorded entries are the
# last frames of user-confirmed handovers; standing is the simulator's grounded entry.
ENTRIES = {
    "source15_f431": (
        [0.0949, -0.0092, 0.0929, 0.1725, -0.1877, 0.0185, 0.1102, -0.1394, -0.3321, 0.1994, -0.1798, 0.075,
         0.0195, 0.0465, 0.1339, 0.1406, 0.2432, -0.4265, 0.8792, -0.5346, -0.0642, -0.045, -0.104, -0.0841,
         0.3028, -0.2177, 1.6993, -0.0958, -0.0738],
        [0.0283, -0.0115, -0.0096, -0.0114, 0.0066, -0.0058, 0.007]),
    "source30_f420": (
        [-0.0028, -0.0455, -0.0397, 0.2384, -0.1539, 0.0225, 0.2091, -0.1204, -0.4935, 0.1412, -0.2241, 0.0653,
         0.045, -0.0539, -0.0052, 0.1521, 0.2879, -0.4474, 1.0697, 0.0555, 0.3854, -0.0834, -0.2676, -0.0677,
         -0.0024, 0.0949, 1.6376, -0.2082, -0.1895],
        [0.0248, -0.0115, -0.0095, -0.0093, 0.0066, -0.0067, 0.0068]),
    "grounded_standing": (
        [0.0196, 0.088, 0.1195, 0.1734, -0.1528, -0.0608, 0.0338, -0.0505, -0.2255, 0.1381, -0.1336, 0.0207,
         -0.0185, 0.0208, 0.1353, -0.0187, 0.2829, -0.6058, 0.9811, -0.124, -0.086, 0.1006, -0.0429, -0.2592,
         0.5996, 1.028, 0.2155, -0.0391, -0.2598],
        [-0.0016, 0.024, -1.7142, 1.517, 1.715, 0.9865, 1.6627]),
}
MJCF = Path(rc.__file__).parents[2] / "data/robot_model/model_data/g1/g1_29dof_with_hand.xml"


def feedback(body, hand):
    return {"body_q_measured_motor": list(body), "right_hand_q_measured": list(hand)}


def profile(tmp_path, **desk):
    data = yaml.safe_load(PROFILE.read_text())
    data["skills"] = {"bottle_handover": dict(prompt="test-only handover",
        completion_criteria="Stable offer", completion_action="hold")}
    data["handover"] = dict(skill_id="bottle_handover", checkpoint_verified=True, ready_pose_reviewed=True,
        ready_right_arm_joints=dict(zip(NAMES, READY)), raised_right_arm_joints=dict(zip(NAMES, RAISED)),
        **desk)
    path = tmp_path / "handover.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_profile(path)


@pytest.fixture(scope="module")
def link_boxes():
    return rc.right_arm_link_boxes(instantiate_g1_robot_model(waist_location="lower_and_upper_body"))


@pytest.mark.parametrize("entry", ["source15_f431", "source30_f420"])
def test_recorded_post_handover_entries_return_directly(link_boxes, entry):
    gap = rc.direct_return_desk_gap(link_boxes, feedback(*ENTRIES[entry]), READY,
                                    DESK["desk_top_above_pelvis_m"], DESK["desk_front_from_pelvis_m"])
    assert gap >= rc.DESK_MARGIN_M


def test_grounded_standing_entry_still_raises(link_boxes, tmp_path):
    handover = profile(tmp_path, **DESK).handover
    gap = rc.direct_return_desk_gap(link_boxes, feedback(*ENTRIES["grounded_standing"]), READY,
                                    DESK["desk_top_above_pelvis_m"], DESK["desk_front_from_pelvis_m"])
    assert gap < rc.DESK_MARGIN_M
    assert rc.ready_waypoint(handover, feedback(*ENTRIES["grounded_standing"]), link_boxes) == tuple(RAISED)


def test_direct_return_checks_the_whole_rate_limited_path():
    # Hand height follows |elbow|: both ends are high, but passing elbow 0 dips into the desk.
    def boxes(body, hand):
        height = abs(body[25]) - 0.2
        return [(np.array([0.5, 0.0, height]), np.array([0.6, 0.0, height + 0.01]))]

    body = np.zeros(29)
    body[25] = 0.5
    target = np.zeros(7)
    target[3] = -0.5
    assert rc.direct_return_desk_gap(boxes, feedback(body, np.zeros(7)), target, 0.0, 0.4) < 0
    target[3] = 0.4
    assert rc.direct_return_desk_gap(boxes, feedback(body, np.zeros(7)), target, 0.0, 0.4) >= 0.2 - 1e-9


def test_desk_gap_is_box_to_region_separation():
    top, front = 0.0, 0.4
    above = (np.array([0.5, 0, 0.05]), np.array([0.6, 0, 0.1]))
    beside = (np.array([0.2, 0, -0.3]), np.array([0.35, 0, -0.1]))
    corner = (np.array([0.2, 0, 0.03]), np.array([0.36, 0, 0.1]))
    inside = (np.array([0.5, 0, -0.1]), np.array([0.6, 0, -0.05]))
    assert rc.desk_gap([above], top, front) == pytest.approx(0.05)
    assert rc.desk_gap([beside], top, front) == pytest.approx(0.05)
    assert rc.desk_gap([corner], top, front) == pytest.approx(0.05)
    assert rc.desk_gap([inside], top, front) < 0
    assert rc.desk_gap([above, beside, corner], top, front) == pytest.approx(0.05)


def test_ready_waypoint_skips_raise_only_when_direct_path_clears(link_boxes, tmp_path):
    configured = profile(tmp_path, **DESK).handover
    for entry in ("source15_f431", "source30_f420"):
        assert rc.ready_waypoint(configured, feedback(*ENTRIES[entry]), link_boxes) is None
    unconfigured = profile(tmp_path).handover
    assert rc.ready_waypoint(unconfigured, feedback(*ENTRIES["source30_f420"]), link_boxes) == tuple(RAISED)


@pytest.mark.parametrize("bad", [
    {}, {"body_q_measured_motor": [0.0] * 28, "right_hand_q_measured": [0.0] * 7},
    {"body_q_measured_motor": [float("nan")] * 29, "right_hand_q_measured": [0.0] * 7},
])
def test_ready_waypoint_raises_when_feedback_is_invalid(link_boxes, tmp_path, bad):
    assert rc.ready_waypoint(profile(tmp_path, **DESK).handover, bad, link_boxes) == tuple(RAISED)


@pytest.mark.parametrize("desk", [
    dict(desk_top_above_pelvis_m=0.063),
    dict(desk_front_from_pelvis_m=0.40),
    dict(desk_top_above_pelvis_m=float("nan"), desk_front_from_pelvis_m=0.40),
    dict(desk_top_above_pelvis_m=True, desk_front_from_pelvis_m=0.40),
    dict(desk_top_above_pelvis_m=0.063, desk_front_from_pelvis_m=-0.1),
])
def test_profile_requires_both_finite_desk_fields(tmp_path, desk):
    with pytest.raises(ValueError):
        profile(tmp_path, **desk)


def test_profile_desk_fields_are_optional(tmp_path):
    assert profile(tmp_path).handover.desk_top_above_pelvis_m is None
    handover = profile(tmp_path, **DESK).handover
    assert (handover.desk_top_above_pelvis_m, handover.desk_front_from_pelvis_m) == (0.063, 0.40)


def test_link_boxes_contain_model_collision_geometry():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    assert set(rc.RIGHT_ARM_LINK_BOXES) == {
        model.body(b).name for b in range(model.nbody)
        if model.body(b).name.startswith(("right_elbow", "right_wrist", "right_hand"))
        and any(model.geom_bodyid[g] == b and model.geom_contype[g] for g in range(model.ngeom))}
    for name, (low, high) in rc.RIGHT_ARM_LINK_BOXES.items():
        body, points = model.body(name).id, []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or not model.geom_contype[g]:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                local = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            else:
                local = np.array(list(product((-1, 1), repeat=3))) * model.geom_size[g]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            points.append(local @ rotation.reshape(3, 3).T + model.geom_pos[g])
        points = np.concatenate(points)
        # Contains every collision point and is no looser than the 0.1 mm rounding.
        np.testing.assert_array_less(np.asarray(low) - 1e-9, points.min(0) + 1e-12)
        np.testing.assert_array_less(points.max(0) - 1e-12, np.asarray(high) + 1e-9)
        assert np.all(points.min(0) - np.asarray(low) < 2e-4) and np.all(np.asarray(high) - points.max(0) < 2e-4)
