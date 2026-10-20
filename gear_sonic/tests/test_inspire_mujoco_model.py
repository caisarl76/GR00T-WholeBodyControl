"""Exercise the real Inspire MJCF and sim loop without a viewer or DDS channels."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.mujoco_sim import base_sim, unitree_sdk2py_bridge
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.inspire_ftp_hand import CLOSED_RADIANS, OPEN, InspireFtpMujocoPlant

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml")
    )
    # Isolate actuator behavior from falling and incidental hand contacts.
    model.opt.gravity[:] = 0
    model.opt.timestep = 0.002
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("motor", range(6))
def test_each_motor_moves_independently_with_passive_coupling(model, side, motor):
    data = mujoco.MjData(model)
    plant = InspireFtpMujocoPlant.resolve(model, data)
    target = OPEN.copy()
    target[motor] = 0.65
    left, right = (target, OPEN) if side == "left" else (OPEN, target)
    plant.write_targets(left, right)
    np.testing.assert_allclose(plant.read_normalized_state(), [OPEN, OPEN])
    mujoco.mj_step(model, data, nstep=500)
    np.testing.assert_allclose(plant.read_normalized_state(), [left, right], atol=0.003)

    # Twelve passive-joint equalities stay coupled during physical movement.
    assert model.neq == 12
    for passive, active, coefficients in zip(model.eq_obj1id, model.eq_obj2id, model.eq_data, strict=True):
        passive_q = data.qpos[model.jnt_qposadr[passive]]
        active_q = data.qpos[model.jnt_qposadr[active]]
        assert passive_q == pytest.approx(coefficients[0] + coefficients[1] * active_q, abs=1e-4)

    plant.write_targets(OPEN, OPEN)
    mujoco.mj_step(model, data, nstep=500)
    np.testing.assert_allclose(plant.read_normalized_state(), [OPEN, OPEN], atol=0.003)


def test_targets_do_not_modify_body_or_partial_write_on_invalid_pair(model):
    data = mujoco.MjData(model)
    plant = InspireFtpMujocoPlant.resolve(model, data)
    body_ids = np.setdiff1d(np.arange(model.nu), np.concatenate(list(plant.actuator_ids.values())))
    assert len(body_ids) == 29
    data.ctrl[body_ids] = np.arange(29) * 0.01
    before = data.ctrl.copy()
    with pytest.raises(ValueError):
        plant.write_targets(np.zeros(6), [np.nan] * 6)
    np.testing.assert_array_equal(data.ctrl, before)
    plant.write_targets(np.zeros(6), OPEN)
    np.testing.assert_array_equal(data.ctrl[body_ids], before[body_ids])
    np.testing.assert_allclose(data.ctrl[plant.actuator_ids["left"]], CLOSED_RADIANS)
    assert model.nu == 41


def test_profile_preserves_body_settings_and_dex3_default():
    dex3 = SimLoopConfig().load_wbc_yaml()
    inspire = SimLoopConfig(hand_profile="inspire_ftp").load_wbc_yaml()
    assert dex3["NUM_HAND_MOTORS"] == 7
    assert dex3["ROBOT_SCENE"].endswith("scene_43dof.xml")
    assert inspire["NUM_HAND_MOTORS"] == 6
    assert inspire["NUM_HAND_JOINTS"] == 12
    assert not inspire["ENABLE_DEX3_DDS_HANDS"]
    assert inspire["INSPIRE_HAND_COMMAND_SOURCE"] == "controller"
    optical = SimLoopConfig(hand_profile="inspire_ftp", hand_command_source="optical").load_wbc_yaml()
    assert optical["INSPIRE_HAND_COMMAND_SOURCE"] == "optical"
    for key in ("NUM_MOTORS", "NUM_JOINTS", "MOTOR_KP", "MOTOR_KD", "MOTOR2JOINT", "JOINT2MOTOR"):
        assert inspire[key] == dex3[key]
    for options in ({"interface": "192.168.123.164"}, {"with_hands": False}):
        with pytest.raises(ValueError):
            SimLoopConfig(hand_profile="inspire_ftp", **options).load_wbc_yaml()


def test_default_env_steps_body_and_hand_targets_without_dds(monkeypatch):
    subscriber = Mock(side_effect=AssertionError("optical simulation must not open the controller subscriber"))
    monkeypatch.setattr(base_sim, "InspireFtpZmqSubscriber", subscriber)
    config = SimLoopConfig(hand_profile="inspire_ftp", hand_command_source="optical").load_wbc_yaml()
    env = DefaultEnv(config, onscreen=False, offscreen=False)
    subscriber.assert_not_called()
    assert env.inspire_hand_subscriber is None
    command = SimpleNamespace(motor_cmd=[SimpleNamespace(tau=0.0, kp=0.0, kd=0.0, q=0.0, dq=0.0)] * 29)
    bridge = SimpleNamespace(
        low_cmd=command, num_body_motor=29, use_sensor=False, joystick=None, PublishLowState=Mock()
    )
    env.set_unitree_bridge(bridge)
    target = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    adapter = Mock()
    adapter.step.side_effect = lambda **kwargs: env.inspire_hand_plant.write_targets(target, OPEN)
    env.inspire_hand_adapter = adapter
    env.sim_step()
    adapter.step.assert_called_once()
    np.testing.assert_allclose(
        env.mj_data.ctrl[env.inspire_hand_plant.actuator_ids["left"]],
        (1 - target) * CLOSED_RADIANS,
    )
    np.testing.assert_array_equal(env.mj_data.ctrl[env.body_actuator_index], np.zeros(29))
    observations = env.prepare_obs()
    assert observations["left_hand_q"].shape == (12,)
    assert observations["left_hand_normalized"].shape == (6,)
    np.testing.assert_array_equal(observations["body_q"], env.mj_data.qpos[env.body_qpos_adr])
    assert env.mj_data.time == pytest.approx(config["SIMULATE_DT"])
    env.reset()
    adapter.reset.assert_called_once()
    env.close()
    adapter.close.assert_called_once()


def test_inspire_dds_bridge_only_creates_body_endpoints(monkeypatch):
    publisher, subscriber = Mock(), Mock()
    monkeypatch.setattr(unitree_sdk2py_bridge, "ChannelPublisher", publisher)
    monkeypatch.setattr(unitree_sdk2py_bridge, "ChannelSubscriber", subscriber)
    config = SimLoopConfig(hand_profile="inspire_ftp").load_wbc_yaml()
    bridge = unitree_sdk2py_bridge.UnitreeSdk2Bridge(config)
    assert all(
        "dex3" not in call.args[0] and "inspire" not in call.args[0]
        for call in publisher.call_args_list + subscriber.call_args_list
    )
    assert [call.args[0] for call in subscriber.call_args_list] == ["rt/lowcmd"]
    assert bridge.GetAction()[0].shape == (29,)


@pytest.mark.parametrize("source", ["controller", "optical"])
def test_default_env_selects_one_inspire_transport(monkeypatch, source):
    subscriber = Mock()
    monkeypatch.setattr(base_sim, "InspireFtpZmqSubscriber", subscriber)
    config = SimLoopConfig(hand_profile="inspire_ftp", hand_command_source=source).load_wbc_yaml()
    env = DefaultEnv(config, onscreen=False, offscreen=False)
    try:
        assert env.inspire_hand_adapter is None
        if source == "controller":
            subscriber.assert_called_once()
            assert env.inspire_hand_subscriber is subscriber.return_value
            env.reset()
            subscriber.return_value.state.reset.assert_called_once()
        else:
            subscriber.assert_not_called()
            assert env.inspire_hand_subscriber is None
            env.reset()
    finally:
        env.close()
    if source == "controller":
        subscriber.return_value.close.assert_called_once()


@pytest.mark.parametrize("source", ["controller", "optical"])
def test_base_simulator_installs_optical_adapter_only_when_selected(monkeypatch, source):
    from gear_sonic.utils.mujoco_sim import pico_inspire_sim

    adapter = Mock()
    env = Mock()
    env.viewer = None
    env.image_publish_process = None
    monkeypatch.setattr(pico_inspire_sim, "PicoInspireSim", adapter)
    monkeypatch.setattr(base_sim, "DefaultEnv", Mock(return_value=env))
    monkeypatch.setattr(base_sim, "ChannelFactoryInitialize", Mock())
    monkeypatch.setattr(base_sim, "UnitreeSdk2Bridge", Mock())
    config = SimLoopConfig(hand_profile="inspire_ftp", hand_command_source=source).load_wbc_yaml()
    sim = base_sim.BaseSimulator(config)
    try:
        if source == "optical":
            adapter.assert_called_once_with(
                env.inspire_hand_plant, host="localhost", port=5556, status_port=5563
            )
            assert env.inspire_hand_adapter is adapter.return_value
        else:
            adapter.assert_not_called()
    finally:
        sim.close()


@pytest.mark.parametrize(
    "options",
    [
        {"hand_command_source": "unknown"},
        {"hand_profile": "dex3", "hand_command_source": "optical"},
    ],
)
def test_invalid_hand_command_source_is_rejected(options):
    with pytest.raises(ValueError):
        SimLoopConfig(**options).load_wbc_yaml()
