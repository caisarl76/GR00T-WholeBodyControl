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
            side=side, lower=np.zeros(size), upper=np.ones(size), retarget=lambda points: np.ones(size) * 0.5
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
    r.socket.packets.append(dex_feedback(2, left_hand_q=[-0.0201] * 7))
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


@pytest.mark.parametrize("value", [-0.0007002827478572726, -0.000917662, -0.001, -0.02])
def test_right_index_near_zero_admission_preserves_raw_feedback_and_slew(make_runtime, value):
    r = make_runtime()
    measured = np.zeros(7)
    measured[5] = value
    r.socket.packets.append(dex_feedback(right_hand_q=measured.tolist()))
    r.poll_feedback(BASE)
    assert r.ready(BASE)
    assert r.feedback_blockers(BASE) == []
    sample = snapshot()
    r.sdk.get_right_hand_snapshot = lambda: sample
    previous = None
    for frame in range(6):
        now = BASE + frame * 20_000_000
        sample["source_timestamp_ns"] = sample["binding_generation"] = frame + 1
        r.socket.packets.append(dex_feedback(frame + 2, right_hand_q=measured.tolist()))
        r.step(body(now), now, enabled=True)
        out = r.outputs[1]
        assert out.reason == HandReason.OK
        assert out.state == (TrackingState.WAITING if frame < 4 else TrackingState.RECOVERING)
        assert out.command[5] >= 0
        if frame <= 4:
            assert out.command[5] == 0
        if previous is not None:
            assert np.max(np.abs(out.command - previous)) <= 2.0 * 0.02 + 1e-7
        previous = out.command
        assert r.measured_inputs[1][5] == value
        assert r.measured(now)[1][5] == value
    assert r.measured(BASE + 200_000_000) == [None, None]
    assert not r.ready(BASE + 200_000_000)


@pytest.mark.parametrize("side,joint,value", [
    ("right", 5, -0.020000001), ("right", 5, -0.05),
    ("left", 5, -0.0201), ("right", 4, -0.0201), ("right", 5, 1.0201),
    ("right", 5, float("nan")),
])
def test_near_zero_allowance_does_not_hide_other_feedback_errors(make_runtime, side, joint, value):
    r = make_runtime()
    measured = np.zeros(7)
    measured[joint] = value
    r.socket.packets.append(dex_feedback(**{f"{side}_hand_q": measured.tolist()}))
    r.poll_feedback(BASE)
    assert not r.ready(BASE)
    assert any(message.startswith(side + ":") for message in r.feedback_blockers(BASE))
    r.step(body(), BASE, enabled=True)
    assert r.outputs[("left", "right").index(side)].reason == HandReason.FEEDBACK_UNAVAILABLE


@pytest.mark.parametrize("measured", [
    [-0.8329587578773499, -0.5226656794548035, -0.9470111131668091,
     1.3914557695388794, 1.7438240051269531, 1.3232935667037964, 1.7463444471359253],
    [0.0014184658648446202, -0.013307612389326096, -1.742186188697815,
     1.5447142124176025, 1.742560625076294, 1.5710546970367432, 1.7458572387695312],
])
def test_pose_reentry_with_captured_right_index_stop_feedback(make_runtime, measured):
    """Sep14 optical MuJoCo run: A+X was recognized but ready() blocked re-entry."""
    r = make_runtime()
    right = r.trackers[1]
    right.lower[:] = [-1.04719755, -1.04719755, -1.74532925, 0, 0, 0, 0]
    right.upper[:] = [1.04719755, 0.72431163, 0, 1.57079632, 1.74532925, 1.57079632, 1.74532925]
    right.retargeter.retarget = lambda points: (right.lower + right.upper) / 2
    measured = np.array(measured)
    initial = np.clip(measured, right.lower, right.upper)
    sample = snapshot()
    r.sdk.get_left_hand_snapshot = r.sdk.get_right_hand_snapshot = lambda: sample
    for frame in range(12):
        now = BASE + frame * 20_000_000
        sample["source_timestamp_ns"] = sample["binding_generation"] = frame + 1
        q = initial if frame < 6 else measured
        r.socket.packets.append(dex_feedback(frame + 1, right_hand_q=q.tolist()))
        r.step(body(now), now, enabled=frame < 6)
    assert right.state == TrackingState.HOLDING
    held = initial.astype(np.float32)
    # This is the manager's PLANNER -> POSE admission gate, with fresh DDS data.
    assert r.ready(now), r.feedback_blockers(now)
    for frame in range(12, 18):
        now = BASE + frame * 20_000_000
        sample["source_timestamp_ns"] = sample["binding_generation"] = frame + 1
        r.socket.packets.append(dex_feedback(frame + 1, right_hand_q=measured.tolist()))
        r.step(body(now), now, enabled=True)
        out = r.outputs[1]
        assert out.reason == HandReason.OK
        if frame < 16:
            assert out.state == TrackingState.WAITING
            np.testing.assert_array_equal(out.command, held)
        assert np.all(out.command >= right.lower - 1e-7)
        assert np.all(out.command <= right.upper + 1e-7)
        assert np.max(np.abs(out.command - held)) <= 2.0 * 0.02 + 1e-7
        np.testing.assert_array_equal(r.measured_inputs[1], measured)
        held = out.command.copy()
    assert out.state in (TrackingState.RECOVERING, TrackingState.TRACKING)


@pytest.mark.parametrize("side", [0, 1])
def test_native_planner_fist_can_seed_bounded_pose_entry(make_runtime, side):
    r = make_runtime()
    tracker = r.trackers[side]
    right_lower = np.array([-1.04719755, -1.04719755, -1.74532925, 0, 0, 0, 0])
    right_upper = np.array([1.04719755, 0.72431163, 0, 1.57079632, 1.74532925, 1.57079632, 1.74532925])
    tracker.lower[:] = right_lower if side else -right_upper
    tracker.upper[:] = right_upper if side else -right_lower
    fist = np.array([0, 0, -1.75, 1.57, 1.75, 1.57, 1.75]) * (1 if side else -1)
    key = f"{('left', 'right')[side]}_hand_q"
    r.socket.packets.append(dex_feedback(**{key: fist.tolist()}))
    r.poll_feedback(BASE)
    assert r.ready(BASE), r.feedback_blockers(BASE)
    r.step(body(), BASE, enabled=True)
    np.testing.assert_allclose(r.outputs[side].command, np.clip(fist, tracker.lower, tracker.upper))
    np.testing.assert_array_equal(r.measured_inputs[side], fist)
    assert tracker._bounded(fist) is None  # Native fist is not an optical target.
    fist[6] += 0.03 * (1 if side else -1)
    r.socket.packets.append(dex_feedback(2, **{key: fist.tolist()}))
    r.poll_feedback(BASE)
    assert not r.ready(BASE)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("joint", [3, 5])
@pytest.mark.parametrize("excursion,accepted", [(0.00025833, True), (0.0127, True), (0.02, True), (0.02000001, False)])
def test_fist_knuckle_soft_stop_feedback_is_measured_only(make_runtime, side, joint, excursion, accepted):
    r = make_runtime()
    tracker = r.trackers[("left", "right").index(side)]
    sign = 1 if side == "right" else -1
    if sign > 0:
        tracker.upper[joint] = 1.57079632
    else:
        tracker.lower[joint] = -1.57079632
    measured = np.full(7, 0.5)
    measured[joint] = sign * (1.57079632 + excursion)
    r.socket.packets.append(dex_feedback(**{f"{side}_hand_q": measured.tolist()}))
    r.poll_feedback(BASE)
    assert r.ready(BASE) == accepted
    assert tracker._bounded(measured) is None  # Never admit an out-of-range optical target.
    r.step(body(), BASE, enabled=True)
    np.testing.assert_array_equal(r.measured_inputs[("left", "right").index(side)], measured)
    if accepted:
        np.testing.assert_allclose(r.commands()[("left", "right").index(side)][joint], sign * 1.57079632)
        assert not r.ready(BASE + 100_000_000)
    else:
        assert r.outputs[("left", "right").index(side)].reason == HandReason.FEEDBACK_UNAVAILABLE


def test_pose_reentry_reseeds_from_planner_feedback_without_resetting_source_clock(make_runtime):
    r = make_runtime()
    sample = snapshot()
    r.sdk.get_left_hand_snapshot = r.sdk.get_right_hand_snapshot = lambda: sample
    for frame in range(20):
        now = BASE + frame * 20_000_000
        sample["source_timestamp_ns"] = sample["binding_generation"] = frame + 1
        q = [0.2] * 7 if frame < 10 else [0.9] * 7
        r.socket.packets.append(dex_feedback(frame + 1, left_hand_q=q, right_hand_q=q))
        if frame == 15:
            r.poll_feedback(now)
            assert r.ready(now)
            # An unrelated newer packet must not replace the admitted snapshot
            # between the manager's entry gate and its first POSE hand command.
            r.socket.packets.append(dex_feedback(100, left_hand_feedback_valid=False))
        r.step(body(now), now, enabled=frame < 10 or frame >= 15, poll_feedback=frame != 15)
        if frame == 15:
            r.socket.packets.clear()
        if frame >= 15:
            np.testing.assert_allclose(r.outputs[1].command, 0.9)
            assert r.trackers[1].source_timestamp_ns == frame + 1
            assert r.outputs[1].source_epoch == 0
            assert r.outputs[1].state == (TrackingState.WAITING if frame < 19 else TrackingState.RECOVERING)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("joint", range(7))
@pytest.mark.parametrize("value,bounded", [(-0.02, 0.0), (1.02, 1.0)])
def test_uniform_dex3_feedback_tolerance_preserves_raw_and_target_limits(make_runtime, side, joint, value, bounded):
    r = make_runtime()
    index = ("left", "right").index(side)
    q = np.full(7, 0.5)
    q[joint] = value
    r.socket.packets.append(dex_feedback(**{f"{side}_hand_q": q.tolist()}))
    r.poll_feedback(BASE)
    assert r.ready(BASE)
    assert r.trackers[index]._bounded(q) is None
    r.step(body(), BASE, enabled=True)
    assert r.commands()[index][joint] == bounded
    np.testing.assert_array_equal(r.measured_inputs[index], q)
    assert not r.ready(BASE + 100_000_000)
    q[joint] += -1e-6 if value < 0 else 1e-6
    assert r.trackers[index]._bounded(q, measured=True) is None


def test_offline_replay_matches_runtime_across_planner_fist_reentry(make_runtime):
    from gear_sonic.scripts.replay_pico_hands import replay
    from gear_sonic.tests.test_pico_hand_replay import FakeRetargeter, synthetic_capture
    from gear_sonic.utils.teleop.pico_hand_log import snapshot_at

    arrays = synthetic_capture(25)
    arrays["tick_ns"] += BASE
    arrays["enabled"] = np.ones(25, dtype=bool)
    arrays["enabled"][10:15] = False
    arrays["measured"][:10] = 0.2
    arrays["measured"][10:] = 0.9
    r = make_runtime()
    commands, states = [], []
    for frame, now in enumerate(arrays["tick_ns"]):
        r.sdk.get_left_hand_snapshot = lambda: snapshot_at(arrays, frame, 0)
        r.sdk.get_right_hand_snapshot = lambda: snapshot_at(arrays, frame, 1)
        r.socket.packets.append(dex_feedback(
            frame + 1, left_hand_q=arrays["measured"][frame, 0].tolist(),
            right_hand_q=arrays["measured"][frame, 1].tolist(),
        ))
        r.step(body(now), int(now), enabled=bool(arrays["enabled"][frame]))
        commands.append(np.stack(r.commands()))
        states.append([out.state for out in r.outputs])
    result, report = replay(arrays, retargeters=[FakeRetargeter(0.5), FakeRetargeter(0.5)])
    np.testing.assert_array_equal(result["emitted"], commands)
    np.testing.assert_array_equal(result["state"], states)
    np.testing.assert_allclose(result["emitted"][15], 0.9)
    assert report["replay_safety_checks_pass"]
