"""Sim disturbance option (PRD section 9 Step 9): a pushed wrench lands in xfrc_applied and in sim_gt.

The conda sonic env cannot import cyclonedds, so the unitree bridge modules are stubbed and the real
``BaseSimulator`` methods run on a bare instance with a real (tiny) MjModel/MjData.
"""

import sys
import time
import types

import msgpack
import mujoco
import numpy as np
import pytest
import zmq

for _n in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.core.channel"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["unitree_sdk2py.core.channel"].ChannelFactoryInitialize = None
_b = types.ModuleType("gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge")
_b.ElasticBand = _b.UnitreeSdk2Bridge = None
sys.modules.setdefault(_b.__name__, _b)

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator  # noqa: E402

XML = '<mujoco><worldbody><body name="pelvis"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'


def _sim(cfg):
    s = BaseSimulator.__new__(BaseSimulator)
    m = mujoco.MjModel.from_xml_string(XML)
    s.config, s.sim_env = cfg, types.SimpleNamespace(mj_model=m, mj_data=mujoco.MjData(m))
    return s


def _free_port():
    with zmq.Context.instance().socket(zmq.PULL) as s:
        return s.bind_to_random_port("tcp://127.0.0.1")


def test_sim_wrench_applied_and_published():
    port = _free_port()
    s = _sim({"disturbance_port": port, "publish_ground_truth": True, "ground_truth_port": _free_port(), "gt_bodies": []})
    s.init_publisher()
    push = zmq.Context.instance().socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{port}")
    xf = s.sim_env.mj_data.xfrc_applied[1]  # body 0 is the world; pelvis is body 1
    s.apply_disturbance()
    assert not xf.any()  # zero at start
    for w in ([1, 2, 3, 0, 0, 0], [0, 50.0, 0, 0, 0, 7.5]):  # two pending messages: the last one wins
        push.send(msgpack.packb({"force": w[:3], "torque": w[3:]}))
    time.sleep(0.2)
    s.apply_disturbance()
    assert list(xf) == [0, 50.0, 0, 0, 0, 7.5]
    s.apply_disturbance()  # nothing new: held
    assert list(xf) == [0, 50.0, 0, 0, 0, 7.5]
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"sim_gt")
    sub.connect(f"tcp://127.0.0.1:{s.config['ground_truth_port']}")
    time.sleep(0.3)
    s.publish_ground_truth()
    msg = msgpack.unpackb(sub.recv()[len(b"sim_gt"):], raw=False)
    assert msg["pelvis_wrench"] == [0, 50.0, 0, 0, 0, 7.5]


def test_gt_has_no_wrench_without_option_and_band_conflict_raises():
    s = _sim({"publish_ground_truth": True, "ground_truth_port": _free_port(), "gt_bodies": []})
    s.init_publisher()
    assert s._dist_sock is None
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"sim_gt")
    sub.connect(f"tcp://127.0.0.1:{s.config['ground_truth_port']}")
    time.sleep(0.3)
    s.publish_ground_truth()
    assert "pelvis_wrench" not in msgpack.unpackb(sub.recv()[len(b"sim_gt"):], raw=False)
    with pytest.raises(ValueError):
        _sim({"disturbance_port": _free_port(), "ENABLE_ELASTIC_BAND": True}).init_publisher()


def test_missing_pelvis_body_raises():
    s = _sim({"disturbance_port": _free_port()})
    s.sim_env.mj_model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="x"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):
        s.init_publisher()
