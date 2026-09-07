import numpy as np
import pytest

from gear_sonic.utils.inference.reference_adapter.g1_act import (
    BODY_NAMES,
    ISAAC_FROM_MOTOR,
    LEFT_HAND_NAMES,
    NOMINAL_BODY,
    RIGHT_HAND_NAMES,
    compose_reference,
    compose_standing_reference,
    leased_payload,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


def fixture():
    contract = {
        group: [{"name": name, "range": [-3.0, 3.0]} for name in names]
        for group, names in (("body", BODY_NAMES), ("left", LEFT_HAND_NAMES), ("right", RIGHT_HAND_NAMES))
    }
    initial = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "floating_base_pose": [0, 0, 0.8, 1, 0, 0, 0],
    }
    return initial, contract


def test_named_joint_mapping_velocity_and_paired_tail():
    initial, contract = fixture()
    # Exercise a simulator whose right hand's internal order differs from ACT.
    contract["right"][3:] = contract["right"][5:] + contract["right"][3:5]
    values = np.arange(28) / 20
    actions = np.tile(values, (100, 1))
    payload, report = compose_reference(actions, initial, contract)
    motor_order = payload["joint_pos"][:, np.argsort(ISAAC_FROM_MOTOR)]
    np.testing.assert_array_equal(motor_order[:, :15], 0)
    np.testing.assert_allclose(motor_order[-1, 15:], values[:14])
    np.testing.assert_allclose(payload["left_hand_joints"][-1], values[14:21])
    np.testing.assert_allclose(payload["right_hand_joints"][-1], values[[21, 22, 23, 26, 27, 24, 25]])
    for key in ("joint_pos", "left_hand_joints", "right_hand_joints"):
        np.testing.assert_allclose(payload[key][-46:], np.tile(payload[key][-1], (46, 1)))
    np.testing.assert_array_equal(payload["joint_vel"][-46:], 0)
    assert np.max(np.abs(payload["joint_vel"])) <= 1.00001
    assert report["terminal_padding_frames"] == 46
    packet = pack_pose_message(leased_payload(payload, session_id=1, chunk_id=0, deadline_ns=123), version=1)
    assert packet.startswith(b'pose{"v":1,')


def test_joint_limits_and_invalid_input():
    initial, contract = fixture()
    payload, report = compose_reference(np.full((100, 28), 4.0), initial, contract)
    assert report["joint_limit_clipped_values"] == 2800
    assert np.max(payload["joint_pos"]) <= 3
    with pytest.raises(ValueError, match="finite ACT"):
        compose_reference(np.full((100, 28), np.nan), initial, contract)
    contract["body"][0]["name"] = "unmapped"
    with pytest.raises(ValueError, match="joint names"):
        compose_reference(np.zeros((100, 28)), initial, contract)


def test_six_contiguous_act_chunks_preserve_twenty_seconds_of_motion():
    initial, contract = fixture()
    actions = np.zeros((600, 28))
    actions[:, 0] = 0.4 * np.sin(np.arange(600) / 30)
    actions[:, 21] = 0.5 * np.cos(np.arange(600) / 30)
    payload, report = compose_reference(actions, initial, contract)
    motor = payload["joint_pos"][:, np.argsort(ISAAC_FROM_MOTOR)]
    assert report["source_action_frames"] == 600
    assert 24 < (len(motor) - 46) / 50 < 26
    np.testing.assert_allclose(motor[100:1099:100, 15], actions[0:600:60, 0], atol=0.012)
    assert np.ptp(motor[:, 15]) > 0.75
    assert np.ptp(payload["right_hand_joints"][:, 0]) > 0.95
    with pytest.raises(ValueError, match="finite ACT"):
        compose_reference(np.zeros((700, 28)), initial, contract)


def test_standing_bootstrap_uses_native_nominal_pose_not_hanging_measurement():
    initial, _ = fixture()
    initial["body_q"][12:15] = .5
    initial["body_command_target"] = NOMINAL_BODY.tolist()
    initial["right_hand_q"] = np.arange(7) / 10
    standing = compose_standing_reference(initial)
    motor = standing["joint_pos"][:, np.argsort(ISAAC_FROM_MOTOR)]
    np.testing.assert_allclose(motor, np.tile(NOMINAL_BODY, (1046, 1)))
    np.testing.assert_array_equal(standing["body_quat"], np.tile([1, 0, 0, 0], (1046, 1)))
    np.testing.assert_allclose(standing["right_hand_joints"][0], initial["right_hand_q"])
    initial["body_command_target"][14] = .5
    with pytest.raises(ValueError, match="nominal standing"):
        compose_standing_reference(initial)
