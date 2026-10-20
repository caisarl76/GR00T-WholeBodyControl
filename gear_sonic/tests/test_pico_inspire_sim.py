"""Production optical packets and measured-only feedback without hardware."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest
import zmq

from gear_sonic.utils.mujoco_sim.pico_inspire_sim import PicoInspireSim
from gear_sonic.utils.teleop.pico_inspire_protocol import HAND_SCHEMA, validate_inspire_status
from gear_sonic.utils.teleop.pico_recording import make_pv
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message


class Socket:
    def __init__(self):
        self.packets = deque()
        self.sent = []

    def recv(self, flags):
        if not self.packets:
            raise zmq.Again()
        return self.packets.popleft()

    def send(self, raw, flags):
        self.sent.append(raw)

    def close(self):
        pass


class Plant:
    def __init__(self):
        self.measured = (np.full(6, 0.97), np.full(6, 0.91))
        self.targets = None

    def write_targets(self, left, right):
        self.targets = (np.array(left), np.array(right))

    def read_normalized_state(self):
        return self.measured


def setup_sim():
    command, status, plant = Socket(), Socket(), Plant()
    sim = PicoInspireSim(plant, command_socket=command, status_socket=status)
    return sim, command, status, plant


def test_optical_packet_drives_real_mujoco_and_reports_measured_feedback():
    import mujoco

    from gear_sonic.utils.mujoco_sim.inspire_ftp_hand import InspireFtpMujocoPlant

    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(
        str(root / "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml")
    )
    model.opt.gravity[:] = 0
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    command, status = Socket(), Socket()
    sim = PicoInspireSim(
        InspireFtpMujocoPlant.resolve(model, data), command_socket=command, status_socket=status
    )
    pv = make_pv(b"a" * 16, 0, 1)
    command.packets.extend([manager(pv), hand(pv, value=0.4)])
    sim.step(now=1)
    np.testing.assert_array_equal(status_fields(status)["left_angle_act"], [1000] * 6)
    mujoco.mj_step(model, data, nstep=1000)
    sim.step(now=2)  # Transport expires, but the physical target stays held.
    fields = status_fields(status)
    np.testing.assert_allclose(fields["left_angle_act"], [400] * 6, atol=3)
    np.testing.assert_allclose(fields["right_angle_act"], [700] * 6, atol=3)
    assert fields["feedback_healthy"][0]
    assert fields["bridge_state"][0] == 1


def manager(pv):
    return pack_pose_message(
        {
            "pv": pv,
            "stream_mode": np.array([int.from_bytes(pv[-4:], "little")], dtype="<i4"),
            "hand_profile": np.array([1], dtype="<i4"),
        },
        "manager_state",
        version=4,
    )


def hand(pv, seq=0, value=0.4):
    fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in HAND_SCHEMA}
    fields["pv"][:] = pv
    fields["message_seq"][0] = seq
    fields["left_command"][:] = value
    fields["right_command"][:] = 0.7
    fields["left_tracking_state"][:] = 3
    fields["right_tracking_state"][:] = 3
    return pack_pose_message(fields, "inspire_hand", version=1)


def status_fields(socket):
    return validate_inspire_status(unpack_pose_message(socket.sent[-1], "inspire_hand_status"))


def test_bootstrap_and_feedback_use_measured_joints_not_targets():
    sim, command, status, plant = setup_sim()
    for tick in range(10):
        sim.step(now=1 + tick * 0.1)
        fields = status_fields(status)
        assert fields["status_seq"][0] == tick
        assert fields["bridge_state"][0] == 1
        assert fields["feedback_healthy"][0]
        np.testing.assert_array_equal(fields["left_angle_act"], [970] * 6)
    pv = make_pv(b"a" * 16, 0, 1)
    command.packets.extend([manager(pv), hand(pv)])
    sim.step(now=2)
    fields = status_fields(status)
    np.testing.assert_allclose(plant.targets[0], 0.4)
    np.testing.assert_allclose(fields["left_applied"], 0.4)
    np.testing.assert_array_equal(fields["left_angle_act"], [970] * 6)
    assert fields["bridge_state"][0] == 2
    assert fields["last_applied_message_seq"][0] == 0


def test_stale_replayed_and_invalid_commands_hold_last_target():
    sim, command, status, plant = setup_sim()
    pv = make_pv(b"a" * 16, 0, 1)
    command.packets.extend([manager(pv), hand(pv, 2)])
    sim.step(now=1)
    command.packets.extend([hand(pv, 2, 0.1), hand(pv, 1, 0.2), hand(pv, 3, float("nan")), b"bad"])
    sim.step(now=1.2)
    np.testing.assert_allclose(plant.targets[0], 0.4)
    assert sim.command_received == 1
    sim.step(now=1.3)
    assert status_fields(status)["bridge_state"][0] == 1
    np.testing.assert_allclose(plant.targets[0], 0.4)
    command.packets.append(hand(pv, 4, 0.1))  # manager expired
    sim.step(now=1.4)
    assert sim.last_sequence == 2


def test_manager_mode_and_session_changes_invalidate_pending_commands():
    sim, command, status, plant = setup_sim()
    pv = make_pv(b"a" * 16, 0, 1)
    stop = make_pv(b"a" * 16, 1, 0)
    command.packets.extend([manager(pv), hand(pv), manager(stop), manager(pv), hand(pv, 1)])
    sim.step(now=1)
    np.testing.assert_array_equal(plant.targets[0], np.ones(6))
    newer = make_pv(b"b" * 16, 0, 1)
    command.packets.extend([manager(newer), hand(newer, 0, 0.6), manager(pv), hand(pv, 10)])
    sim.step(now=1.1)
    np.testing.assert_allclose(plant.targets[0], 0.6)
    np.testing.assert_array_equal(status_fields(status)["accepted_pv"], newer)


def test_bad_measurements_cannot_bootstrap_and_reset_reseeds_session():
    sim, command, status, plant = setup_sim()
    sim.step(now=1)
    old_session = status_fields(status)["pc2_session_id"].tobytes()
    plant.measured = (np.full(6, np.nan), np.ones(6))
    sim.step(now=1.1)
    fields = status_fields(status)
    assert not fields["feedback_healthy"][0]
    assert fields["bridge_state"][0] == 4
    sim.reset()
    assert sim.session != old_session
    assert sim.last_applied_sequence == -1
    np.testing.assert_array_equal(plant.targets[0], np.ones(6))


def test_remote_host_rejected_before_socket_creation():
    with pytest.raises(ValueError, match="loopback"):
        PicoInspireSim(Plant(), host="192.168.123.164")
