import gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge as bridge_module
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge


class _Endpoint:
    created = []

    def __init__(self, topic, message_type):
        del message_type
        self.topic = topic
        self.created.append(topic)

    def Init(self, *args):
        del args

    def Write(self, message):
        del message


def test_body_only_bridge_constructs_no_dex3_dds_endpoints(monkeypatch):
    _Endpoint.created = []
    monkeypatch.setattr(bridge_module, "ChannelPublisher", _Endpoint)
    monkeypatch.setattr(bridge_module, "ChannelSubscriber", _Endpoint)

    bridge = UnitreeSdk2Bridge(
        {
            "ROBOT_TYPE": "g1_29dof",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 6,
            "USE_SENSOR": False,
            "ENABLE_DEX3_DDS_HANDS": False,
        }
    )

    assert not any(topic.startswith("rt/dex3/") for topic in _Endpoint.created)
    assert "rt/lowcmd" in _Endpoint.created
    assert bridge.left_hand_cmd is None
    assert bridge.right_hand_cmd is None
    assert bridge.cmd_received() is False

    bridge.LowCmdHandler(bridge.low_cmd)
    action, received, is_new = bridge.GetAction()
    assert action.shape == (29,)
    assert received is True
    assert is_new is True


def test_dex3_bridge_remains_enabled_by_default(monkeypatch):
    _Endpoint.created = []
    monkeypatch.setattr(bridge_module, "ChannelPublisher", _Endpoint)
    monkeypatch.setattr(bridge_module, "ChannelSubscriber", _Endpoint)

    UnitreeSdk2Bridge(
        {
            "ROBOT_TYPE": "g1_29dof",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 7,
            "USE_SENSOR": False,
        }
    )

    assert {
        "rt/dex3/left/state",
        "rt/dex3/right/state",
        "rt/dex3/left/cmd",
        "rt/dex3/right/cmd",
    }.issubset(_Endpoint.created)
