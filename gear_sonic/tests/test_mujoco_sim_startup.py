"""Exercise startup callbacks without creating DDS channels or a simulator."""

from unittest.mock import Mock

import pytest

from gear_sonic.scripts.run_sim_loop import SimWrapper
from gear_sonic.utils.mujoco_sim import base_sim, simulator_factory, unitree_sdk2py_bridge


@pytest.mark.parametrize(
    ("topic", "command_name"),
    [
        ("rt/lowcmd", "low_cmd"),
        ("rt/dex3/left/cmd", "left_hand_cmd"),
        ("rt/dex3/right/cmd", "right_hand_cmd"),
    ],
)
def test_command_received_during_subscriber_init_is_preserved(monkeypatch, topic, command_name):
    message = object()

    def subscriber(name, message_type):
        def init(handler, queue_length):
            if name == topic:
                handler(message)
                assert handler.__self__.cmd_received()

        return Mock(Init=init)

    monkeypatch.setattr(unitree_sdk2py_bridge, "ChannelPublisher", Mock())
    monkeypatch.setattr(unitree_sdk2py_bridge, "ChannelSubscriber", subscriber)

    bridge = unitree_sdk2py_bridge.UnitreeSdk2Bridge(
        {"ROBOT_TYPE": "g1", "NUM_MOTORS": 29, "NUM_HAND_MOTORS": 7, "USE_SENSOR": False}
    )

    assert getattr(bridge, command_name) is message
    assert getattr(bridge, command_name + "_received")
    assert getattr(bridge, "new_" + command_name)
    assert bridge.cmd_received()
    bridge.reset()
    assert not bridge.cmd_received()
    assert not getattr(bridge, "new_" + command_name)


@pytest.mark.parametrize("interface", [None, "lo"])
def test_dds_initialization_failure_stops_before_bridge_creation(monkeypatch, interface):
    initialize = Mock(side_effect=RuntimeError("DDS domain unavailable"))
    bridge = Mock()
    monkeypatch.setattr(base_sim, "ChannelFactoryInitialize", initialize)
    monkeypatch.setattr(base_sim, "Robot", Mock())
    monkeypatch.setattr(base_sim, "DefaultEnv", Mock())
    monkeypatch.setattr(base_sim, "UnitreeSdk2Bridge", bridge)

    with pytest.raises(RuntimeError, match="DDS domain unavailable"):
        base_sim.BaseSimulator(
            {"SIMULATE_DT": 0.002, "DOMAIN_ID": 1, "INTERFACE": interface, "USE_JOYSTICK": False}
        )

    initialize.assert_called_once_with(*((1, interface) if interface else (1,)))
    bridge.assert_not_called()


def test_sim_wrapper_initializes_dds_once(monkeypatch):
    initialize = Mock()
    monkeypatch.setattr(simulator_factory, "ChannelFactoryInitialize", initialize)
    monkeypatch.setattr(base_sim, "ChannelFactoryInitialize", initialize)
    monkeypatch.setattr(base_sim, "Robot", Mock())
    monkeypatch.setattr(base_sim, "DefaultEnv", Mock())
    monkeypatch.setattr(base_sim, "UnitreeSdk2Bridge", Mock())

    wrapper = SimWrapper(
        robot_model=Mock(),
        env_name="default",
        config={"SIMULATE_DT": 0.002, "DOMAIN_ID": 1, "INTERFACE": "lo", "USE_JOYSTICK": False},
    )

    initialize.assert_called_once_with(1, "lo")
    wrapper.sim.close()
