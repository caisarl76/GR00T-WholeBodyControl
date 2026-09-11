"""Manager hand runtime with fake SDK/socket, actual schemas and watchdogs."""

from collections import deque
import json
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.data.pico_hand_features import join_hand_frame
from gear_sonic.tests.test_pico_hand_tracking import snapshot
from gear_sonic.utils.teleop import pico_hand_runtime as runtime
from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, TrackingState
from gear_sonic.utils.teleop.pico_inspire_protocol import HAND_SCHEMA, STATUS_SCHEMA, validate_inspire_hand
from gear_sonic.utils.teleop.pico_recording import make_pv
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message

BASE = 1_000_000_000
PV = make_pv(b"m" * 16, 0, 1)


class Socket:
    def __init__(self):
        self.packets = deque()
        self.sent = []

    def recv(self, *args):
        if not self.packets:
            raise zmq.Again()
        return self.packets.popleft()

    def send(self, data):
        self.sent.append(data)

    def setsockopt(self, *args):
        pass

    def setsockopt_string(self, *args):
        pass

    def connect(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture
def make_runtime(monkeypatch):
    def retargeter(profile, side):
        size = 7 if profile == "dex3" else 6
        return SimpleNamespace(
            lower=np.zeros(size), upper=np.ones(size), retarget=lambda points: np.ones(size) * 0.5
        )

    monkeypatch.setattr(runtime, "HandRetargeter", retargeter)

    def build(profile="dex3"):
        sdk = SimpleNamespace(
            get_left_hand_snapshot=lambda: snapshot(1), get_right_hand_snapshot=lambda: snapshot(1)
        )
        sock = Socket()
        return runtime.PicoHandRuntime(sdk, SimpleNamespace(socket=lambda _: sock), profile, "fake://")

    return build


def dex_feedback(index=1, **changes):
    fields = {"index": index}
    for side in ("left", "right"):
        fields.update(
            {f"{side}_hand_feedback_valid": True, f"{side}_hand_feedback_age_ns": 0, f"{side}_hand_q": [0.2] * 7}
        )
    return b"g1_debug" + msgpack.packb({**fields, **changes}, use_bin_type=True)


def status(seq=0, session=b"p" * 16, healthy=True, state=1):
    fields = {name: np.zeros(size, dtype) for name, dtype, size in STATUS_SCHEMA}
    fields["pc2_session_id"][:] = np.frombuffer(session, np.uint8)
    fields["status_seq"][0] = seq
    fields["bridge_state"][0] = state
    fields["last_applied_message_seq"][0] = -1
    fields["feedback_healthy"][0] = healthy
    fields["left_angle_act"][:] = 750
    fields["right_angle_act"][:] = 250
    return pack_pose_message(fields, "inspire_hand_status", 1)


def body(now=BASE):
    poses = np.zeros((24, 7))
    poses[22:24, :3] = 100  # Arm/end-effector points must not become hand wrist anchors.
    return {"timestamp_monotonic": now * 1e-9, "body_poses_np": poses}


def test_dex_feedback_uses_dds_positions_not_visualization_commands(make_runtime):
    r = make_runtime()
    r.socket.packets.append(
        dex_feedback(
            left_hand_q_measured=[2.0] * 7,
            right_hand_q_measured=[2.0] * 7,
        )
    )
    r.poll_feedback(BASE)
    assert r.ready(BASE)
    np.testing.assert_allclose(r.measured(BASE), [[0.2] * 7] * 2)
    r.socket.packets.append(dex_feedback(2, left_hand_q=None, left_hand_q_measured=[0.2] * 7))
    r.poll_feedback(BASE)
    assert not r.ready(BASE)


def test_input_diagnostics_distinguish_missing_hand_from_wrong_schema(make_runtime):
    r = make_runtime()
    diagnostic = {"packets": 0, "hand_json": ""}
    r.sdk.get_hand_packet_diagnostics = lambda: diagnostic
    assert r.input_diagnostics() == ["No XR packets received"]
    diagnostic["packets"] = 20
    assert "no Hand section" in r.input_diagnostics()[0]
    diagnostic["hand_json"] = "{}"
    assert all("no hand object" in line for line in r.input_diagnostics())
    diagnostic["hand_json"] = json.dumps({"leftHand": {"isActive": True, "scale": 1, "HandJointLocations": []}})
    assert "isActive=True" in r.input_diagnostics()[0]
    assert "joints=0" in r.input_diagnostics()[0]
    diagnostic["hand_json"] = "null"
    assert "must be an object" in r.input_diagnostics()[0]


def test_dex_measured_limit_roundoff_is_clamped_but_large_excursions_rejected(make_runtime):
    r = make_runtime()
    r.socket.packets.append(dex_feedback(left_hand_q=[-3e-5] * 7, right_hand_q=[1 + 3e-5] * 7))
    r.poll_feedback(BASE)
    assert r.ready(BASE)
    r.step(body(), BASE, enabled=False)
    np.testing.assert_array_equal(r.commands(), [np.zeros(7), np.ones(7)])
    r.socket.packets.append(dex_feedback(2, left_hand_q=[-0.0047] * 7))
    r.poll_feedback(BASE)
    assert not r.ready(BASE)


def test_feedback_blockers_distinguish_transport_schema_dds_and_joint_limits(make_runtime):
    r = make_runtime()
    assert "No g1_debug received" in r.feedback_blockers(BASE)[0]
    r.socket.packets.append(b"g1_debug" + msgpack.packb({"index": 1}))
    r.poll_feedback(BASE)
    blockers = r.feedback_blockers(BASE)
    assert len(blockers) == 2
    assert "missing left_hand_feedback_valid" in blockers[0]
    assert "rebuild deployment" in blockers[0]
    r.socket.packets.append(dex_feedback(2, left_hand_feedback_valid=False, left_hand_feedback_age_ns=-1))
    r.poll_feedback(BASE)
    assert "left: DDS feedback valid=False, age_ns=-1" in r.feedback_blockers(BASE)[0]
    assert len(r.feedback_blockers(BASE)) == 1
    r.socket.packets.append(dex_feedback(3, left_hand_q=[2.0] * 7))
    r.poll_feedback(BASE)
    assert "outside command limits" in r.feedback_blockers(BASE)[0]
    r.socket.packets.append(dex_feedback(4))
    r.poll_feedback(BASE)
    assert r.feedback_blockers(BASE) == []
    assert r.ready(BASE)
    assert "transport stale" in r.feedback_blockers(BASE + 100_000_000)[0]


def test_feedback_blockers_report_decode_failure(make_runtime):
    r = make_runtime()
    r.socket.packets.append(b"g1_debug\xc1")
    r.poll_feedback(BASE)
    assert r.feedback_blockers(BASE)[0].startswith("Invalid g1_debug:")


def test_feedback_blockers_explain_inspire_readmission(make_runtime):
    r = make_runtime("inspire_ftp")
    r.socket.packets.append(status())
    r.poll_feedback(BASE)
    assert "PC2 feedback not admitted; healthy_streak=1" in r.feedback_blockers(BASE)[0]


def test_dex3_combines_local_and_dds_age_per_side(make_runtime):
    r = make_runtime()
    r.socket.packets.append(dex_feedback(left_hand_feedback_age_ns=80_000_000))
    r.poll_feedback(BASE)
    assert r.ready(BASE)
    assert r.measured(BASE + 19_999_999)[0] is not None
    assert r.measured(BASE + 20_000_000)[0] is None
    assert r.measured(BASE + 20_000_000)[1] is not None
    assert r.measured(BASE + 100_000_000) == [None, None]
    r.socket.packets.append(dex_feedback(2, left_hand_feedback_age_ns=101_000_000))
    r.poll_feedback(BASE + 101_000_000)
    assert r.measured(BASE + 101_000_000)[0] is None  # New publish cannot freshen cached DDS.
    assert r.measured(BASE + 101_000_000)[1] is not None


def test_zero_fallback_old_schema_and_duplicate_index_never_ready(make_runtime):
    r = make_runtime()
    r.socket.packets.append(
        b"g1_debug" + msgpack.packb({"index": 1, "left_hand_q": [0.0] * 7, "right_hand_q": [0.0] * 7})
    )
    r.poll_feedback(BASE)
    assert not r.ready(BASE)
    r.socket.packets.append(dex_feedback(2))
    r.poll_feedback(BASE + 1)
    r.socket.packets.append(dex_feedback(2))
    r.poll_feedback(BASE + 100_000_001)
    assert not r.ready(BASE + 100_000_001)
    assert r.received_ns == BASE + 1


@pytest.mark.parametrize("value", [[], None, {}, {"index": 1.5}, {"index": True}, {"index": -1}, {"index": "3"}])
def test_bad_dex3_packet_does_not_escape(value, make_runtime):
    r = make_runtime()
    r.socket.packets.append(b"g1_debug" + msgpack.packb(value))
    r.poll_feedback(BASE)
    assert not r.ready(BASE)


def test_sdk_and_measured_errors_leave_other_side_running(make_runtime):
    r = make_runtime()

    def broken():
        raise TypeError("malformed XR sample")

    r.sdk.get_left_hand_snapshot = broken
    r.socket.packets.append(dex_feedback(left_hand_q="bad"))
    r.step(body(), BASE, enabled=True)
    assert r.outputs[0].reason == HandReason.FEEDBACK_UNAVAILABLE
    assert r.outputs[1].reason == HandReason.OK
    assert r.outputs[1].command is not None
    for tick in range(1, 6):
        now = BASE + tick * 20_000_000
        r.sdk.get_right_hand_snapshot = lambda tick=tick: snapshot(tick + 1)
        r.socket.packets.append(dex_feedback(tick + 1, left_hand_q="bad"))
        r.step(body(now), now, enabled=True)
    assert r.outputs[1].state == TrackingState.RECOVERING
    assert r.outputs[0].command is None


@pytest.mark.parametrize(
    "bad_body",
    [
        None,
        [],
        {"timestamp_monotonic": float("inf")},
        {"timestamp_monotonic": 1.0, "body_poses_np": [[1], [1, 2]]},
    ],
)
def test_malformed_body_wrist_rejects_without_crashing(bad_body, make_runtime):
    r = make_runtime()
    r.socket.packets.append(dex_feedback())
    r.step(bad_body, BASE, enabled=True)
    assert all(o.reason == HandReason.BODY_WRIST_UNAVAILABLE for o in r.outputs)


def test_inspire_ten_advancing_statuses_restart_retirement_and_gap(make_runtime):
    r = make_runtime("inspire_ftp")
    for seq in range(10):
        now = BASE + seq * 100_000_000
        r.socket.packets.append(status(seq))
        r.poll_feedback(now)
        assert r.ready(now) == (seq == 9)
    np.testing.assert_allclose(r.measured(now), [[0.75] * 6, [0.25] * 6])
    r.socket.packets.append(status(9))
    r.poll_feedback(now + 100_000_000)
    assert r.received_ns == now
    assert not r.ready(now + 500_000_000)
    r.socket.packets.append(status(10))
    r.poll_feedback(now + 500_000_000)
    assert r.healthy_streak == 1
    assert not r.ready(now + 500_000_000)
    for seq in range(9):
        r.socket.packets.append(status(seq, b"q" * 16))
        r.poll_feedback(now + 600_000_000 + seq * 100_000_000)
        assert not r.ready(now + 600_000_000 + seq * 100_000_000)
    r.socket.packets.append(status(99, b"p" * 16))
    r.poll_feedback(now + 1_450_000_000)
    assert r.healthy_streak == 9  # Retired process cannot contribute to re-admission.
    r.socket.packets.append(status(9, b"q" * 16))
    r.poll_feedback(now + 1_500_000_000)
    assert r.ready(now + 1_500_000_000)


@pytest.mark.parametrize("state", [0, 1, 2, 3, 4, 5])
def test_only_ready_or_active_inspire_feedback_can_arm(state, make_runtime):
    r = make_runtime("inspire_ftp")
    for seq in range(10):
        now = BASE + seq * 100_000_000
        r.socket.packets.append(status(seq, state=state))
        r.poll_feedback(now)
    assert r.ready(now) == (state in (1, 2))


def test_diagnostics_joins_exact_body_generation_and_inspire_schema(make_runtime):
    r = make_runtime()
    for tick in range(6):
        now = BASE + tick * 20_000_000
        r.sdk.get_left_hand_snapshot = r.sdk.get_right_hand_snapshot = lambda tick=tick: snapshot(tick + 1)
        r.socket.packets.append(dex_feedback(tick + 1))
        r.step(body(now), now, enabled=True)
    diag = r.diagnostics(PV, 7, body_sent=True)
    diag["received_ns"] = now
    packet = {"pv": PV, "sample_generation": np.array([7], np.int64), "received_ns": now}
    for side, command in zip(("left", "right"), r.commands()):
        packet[f"{side}_hand_joints"] = command
    joined = join_hand_frame(packet, diag, "dex3", now, b"m" * 16, 1)
    np.testing.assert_array_equal(joined["teleop.left_hand_joints"], r.commands()[0])
    assert joined["teleop.hand_input"][0] == 0
    packet["sample_generation"][0] = 8
    with pytest.raises(ValueError, match="generation"):
        join_hand_frame(packet, diag, "dex3", now, b"m" * 16, 1)

    r = make_runtime("inspire_ftp")
    for tick in range(10):
        now = BASE + tick * 100_000_000
        r.socket.packets.append(status(tick))
        r.step(body(now), now, enabled=True)
    fields = r.inspire_command(PV, 9)
    assert tuple(fields) == tuple(row[0] for row in HAND_SCHEMA)
    decoded = unpack_pose_message(pack_pose_message(fields, "inspire_hand", 1), "inspire_hand")
    validate_inspire_hand(decoded)
    assert fields["message_seq"][0] == 0
    assert r.inspire_command(PV, 10)["message_seq"][0] == 1
    assert r.inspire_command(make_pv(b"m" * 16, 1, 1), 11)["message_seq"][0] == 0


@pytest.mark.parametrize("left_source,right_source", [(0, 1), (1, 0)])
def test_diagnostics_transmit_independent_optical_clock_sources(make_runtime, left_source, right_source):
    r = make_runtime()
    r.sdk.get_left_hand_snapshot = lambda: {**snapshot(1), "timestamp_source": left_source}
    r.sdk.get_right_hand_snapshot = lambda: {**snapshot(1), "timestamp_source": right_source}
    r.socket.packets.append(dex_feedback())
    r.step(body(), BASE, enabled=True)
    fields = r.diagnostics(PV, 1, body_sent=True)
    decoded = unpack_pose_message(pack_pose_message(fields, "hand_diagnostics", 4), "hand_diagnostics")
    for side, source in (("left", left_source), ("right", right_source)):
        assert decoded[f"{side}_timestamp_source"].dtype == np.int32
        assert decoded[f"{side}_timestamp_source"].item() == source


def test_diagnostics_mark_uninitialized_clock_unavailable(make_runtime):
    r = make_runtime()
    r.sdk.get_left_hand_snapshot = r.sdk.get_right_hand_snapshot = lambda: None
    r.socket.packets.append(dex_feedback())
    r.step(body(), BASE, enabled=True)
    fields = r.diagnostics(PV, 1, body_sent=True)
    assert fields["left_timestamp_source"].item() == -1
    assert fields["right_timestamp_source"].item() == -1
