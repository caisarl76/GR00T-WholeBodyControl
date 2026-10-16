import mujoco
import numpy as np
import pytest
from test_handover_ready import RAISED
from test_ready_clearance import ENTRIES, MJCF, READY, profile

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference.ready_feedforward import right_yaw_gravity_tau
from gear_sonic.utils.teleop.bridge_planner_publisher import PlannerCommand
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message


@pytest.fixture(scope="module")
def tau():
    return right_yaw_gravity_tau(instantiate_g1_robot_model(waist_location="lower_and_upper_body"))


def state(arm, body=None, hand=None):
    body = np.zeros(29) if body is None else np.array(body)
    body[22:29] = arm
    return {"body_q_measured_motor": body, "left_hand_q_measured": np.zeros(7),
            "right_hand_q_measured": np.zeros(7) if hand is None else hand}


@pytest.mark.parametrize("name, feedback", [
    ("raised", state(RAISED)), ("ready", state(READY)),
    *((name, state(body[22:29], body, hand)) for name, (body, hand) in ENTRIES.items()),
])
def test_only_right_yaw_matches_mujoco_upright_static_gravity(tau, name, feedback):
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0  # Upright floating base.
    # Hand actuators sit between the arms; the rest are the 29 motors in order.
    motors = [j for j in (model.actuator(i).trnid[0] for i in range(model.nu)) if "hand" not in model.joint(j).name]
    for i, joint in enumerate(motors):
        data.qpos[model.jnt_qposadr[joint]] = feedback["body_q_measured_motor"][i]
    right_hand = [model.joint(n).id for n in (
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint")]
    for joint, value in zip(right_hand, feedback["right_hand_q_measured"]):
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    out = tau(feedback)
    assert np.count_nonzero(out) == 1
    assert out[8] == pytest.approx(data.qfrc_bias[model.jnt_dofadr[model.joint("right_shoulder_yaw_joint").id]],
                                   abs=0.01)


def test_raised_waypoint_needs_most_of_the_lead_budget(tau):
    # Gravity sag at Kp 14.25 against the 0.15 rad reference lead.
    assert -tau(state(RAISED))[8] / 14.2506 > 0.12


@pytest.mark.parametrize("bad", [{}, {"body_q_measured_motor": [0.0] * 28},
                                 {**state(READY), "right_hand_q_measured": [np.nan] * 7}])
def test_invalid_feedback_sends_no_feedforward(tau, bad):
    assert tau(bad) is None


def test_planner_message_carries_feedforward_only_when_set():
    base = PlannerCommand(mode=0, movement=(0, 0, 0), facing=(1, 0, 0), speed=0.0, height=-1.0,
                          upper_body_position=(0.0,) * 17, upper_body_velocity=(0.0,) * 17)
    assert "upper_body_tau" not in unpack_bridge_message(base.encode(), topic="planner")
    torque = np.zeros(17)
    torque[8] = -1.75
    decoded = unpack_bridge_message(
        PlannerCommand(**{**base.__dict__, "upper_body_tau": torque}).encode(), topic="planner")
    np.testing.assert_allclose(decoded["upper_body_tau"], torque)


def test_profile_feedforward_is_opt_in_bool(tmp_path):
    assert not profile(tmp_path).handover.ready_gravity_feedforward
    assert profile(tmp_path, ready_gravity_feedforward=True).handover.ready_gravity_feedforward
    with pytest.raises(ValueError):
        profile(tmp_path, ready_gravity_feedforward=1)
