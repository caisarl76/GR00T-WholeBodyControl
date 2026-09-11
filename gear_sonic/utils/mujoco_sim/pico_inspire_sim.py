"""Local optical Inspire transport for MuJoCo; stale commands hold position."""

import time
import uuid

import numpy as np
import zmq

from gear_sonic.scripts.run_pico_inspire_bridge import ManagerPacket, decode_bridge_packet
from gear_sonic.utils.teleop.pico_inspire_protocol import STATUS_SCHEMA, validate_inspire_status
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class PicoInspireSim:
    """Consume optical q6 commands and report measured simulator feedback.

    This simulation transport deliberately has no DDS or real-hand output.
    The manager owns optical tracking and slew; transport loss holds the last
    applied target. Feedback is sampled from the plant, never from that target.
    """

    def __init__(
        self, plant, host="127.0.0.1", port=5556, status_port=5563, *, command_socket=None, status_socket=None
    ):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("optical Inspire simulation requires a loopback command host")
        self.plant = plant
        context = zmq.Context.instance()
        self.command_socket = command_socket
        self.status_socket = status_socket
        try:
            if self.command_socket is None:
                self.command_socket = context.socket(zmq.SUB)
                self.command_socket.setsockopt(zmq.LINGER, 0)
                self.command_socket.setsockopt(zmq.RCVHWM, 64)
                self.command_socket.setsockopt(zmq.MAXMSGSIZE, 1_048_576)
                for topic in ("manager_state", "inspire_hand"):
                    self.command_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
                self.command_socket.connect(f"tcp://127.0.0.1:{port}")
            if self.status_socket is None:
                self.status_socket = context.socket(zmq.PUB)
                self.status_socket.setsockopt(zmq.LINGER, 0)
                self.status_socket.setsockopt(zmq.SNDHWM, 10)
                self.status_socket.bind(f"tcp://127.0.0.1:{status_port}")
            self.reset()
            print(
                f"[SIMULATION] Inspire optical commands: 127.0.0.1:{port}; "
                f"measured hand feedback: 127.0.0.1:{status_port}"
            )
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.session = uuid.uuid4().bytes
        self.status_sequence = 0
        self.last_status = None
        self.manager = None
        self.manager_received = None
        self.retired_sessions = set()
        self.last_sequence = -1
        self.last_applied_sequence = -1
        self.command_received = None
        self.target = None
        self.applied = (np.ones(6), np.ones(6))
        self.plant.write_targets(*self.applied)

    def _accept(self, raw, now):
        topic = "manager_state" if raw.startswith(b"manager_state") else "inspire_hand"
        packet = decode_bridge_packet(raw, topic, int(now * 1e9))
        if isinstance(packet, ManagerPacket):
            candidate = packet.provenance
            if candidate.session_id in self.retired_sessions:
                return
            if self.manager is not None:
                if candidate.session_id == self.manager.session_id:
                    if candidate.mode_epoch < self.manager.mode_epoch or (
                        candidate.mode_epoch == self.manager.mode_epoch and candidate != self.manager
                    ):
                        return
                elif len(self.retired_sessions) >= 64:
                    return
                else:
                    self.retired_sessions.add(self.manager.session_id)
            if candidate != self.manager:
                self.last_sequence = self.last_applied_sequence = -1
                self.target = self.command_received = None
            self.manager = candidate
            self.manager_received = now
        elif (
            self.manager is not None
            and packet.provenance == self.manager
            and self.manager.stream_mode in (1, 5)
            and 0 <= now - self.manager_received < 0.25
            and packet.message_seq > self.last_sequence
        ):
            self.last_sequence = packet.message_seq
            self.command_received = now
            self.target = (np.asarray(packet.left), np.asarray(packet.right))

    def step(self, *, now=None, dt=None):
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError("simulation timestamp must be finite and nonnegative")
        # Bound work even if a publisher floods the socket.
        for _ in range(64):
            try:
                raw = self.command_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                self._accept(raw, now)
            except (ValueError, TypeError, KeyError, OverflowError):
                continue
        active = (
            self.target is not None
            and self.manager.stream_mode in (1, 5)
            and 0 <= now - self.manager_received < 0.25
            and 0 <= now - self.command_received < 0.25
        )
        if active:
            self.plant.write_targets(*self.target)
            self.applied = tuple(q.copy() for q in self.target)
            self.last_applied_sequence = self.last_sequence
        if self.last_status is None or now - self.last_status >= 0.1 - 1e-9:
            self._publish_status(now, active)

    def _publish_status(self, now, active):
        fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in STATUS_SCHEMA}
        fields["pc2_session_id"][:] = np.frombuffer(self.session, dtype=np.uint8)
        fields["status_seq"][0] = self.status_sequence
        fields["bridge_state"][0] = 2 if active else 1
        fields["last_applied_message_seq"][0] = self.last_applied_sequence
        if self.manager is not None:
            fields["accepted_pv"][:] = self.manager.to_array()
        try:
            measured = self.plant.read_normalized_state()
            for side, q in zip(("left", "right"), measured, strict=True):
                q = np.asarray(q)
                if q.shape != (6,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                    raise ValueError("invalid measured Inspire state")
                fields[f"{side}_angle_act"][:] = np.rint(q * 1000).astype(np.int32)
            fields["feedback_healthy"][0] = True
        except (ValueError, TypeError):
            fields["fault_code"][0] = 1
            fields["bridge_state"][0] = 4
        for side, applied in zip(("left", "right"), self.applied, strict=True):
            fields[f"{side}_applied"][:] = applied
        validate_inspire_status({**fields, "version": 1, "endian": "le"})
        self.status_socket.send(pack_pose_message(fields, "inspire_hand_status", version=1), zmq.NOBLOCK)
        self.status_sequence += 1
        self.last_status = now

    def close(self):
        for socket in (self.command_socket, self.status_socket):
            if socket is not None:
                socket.close()
