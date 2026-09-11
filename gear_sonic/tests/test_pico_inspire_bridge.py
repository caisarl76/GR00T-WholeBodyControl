"""No-hardware coverage for strict optical intake and the reviewed PC2 safety FSM."""

from collections import deque
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
import zmq

from gear_sonic.scripts import run_pico_inspire_bridge as b
from gear_sonic.utils.teleop.inspire_ftp_real_bridge import (
    BridgeState,
    InspireFtpSafetyController,
    StreamProvenance,
    ValidatedHandState,
)
from gear_sonic.utils.teleop.pico_inspire_protocol import HAND_SCHEMA, validate_inspire_status
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message

BASE = 1_000_000_000
PERIOD = 100_000_000
PV = StreamProvenance(b"a" * 16, 0, 1)
OPEN = (1000,) * 6


def feedback(controller, now, angles=OPEN):
    for side in ("left", "right"):
        controller.accept_hand_state(ValidatedHandState(side, angles, (0,) * 6, now))


def ready():
    c = InspireFtpSafetyController()
    c.arm_publish_from_feedback(
        ValidatedHandState("left", OPEN, (0,) * 6, BASE), ValidatedHandState("right", OPEN, (0,) * 6, BASE), BASE
    )
    for tick in range(10):
        now = BASE + tick * PERIOD
        feedback(c, now)
        c.accept_manager(PV, now)
        c.tick(now)
    assert c.state is BridgeState.READY and c.open_confirm_ticks == 10
    return c


def command(c, tick, seq, *, pv=PV, left=(0.8,) * 6, right=(1.0,) * 6):
    now = BASE + tick * PERIOD
    feedback(c, now)
    c.accept_manager(pv, now - 2)
    accepted = c.accept_pair(left, right, pv, "inspire_hand", now - 1, seq, repr((seq, left, right)).encode())
    decision = c.tick(now)
    if decision.publish:
        c.record_write_result("left", decision.left_counts, True)
        c.record_write_result("right", decision.right_counts, True)
    return accepted, decision


def hand_fields(seq=0, pv=PV):
    fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in HAND_SCHEMA}
    fields["pv"][:] = pv.to_array()
    fields["message_seq"][0] = seq
    fields["left_command"][:] = 0.8
    fields["right_command"][:] = 0.95
    fields["left_tracking_state"][0] = 1  # optical HOLDING remains transport-fresh
    fields["right_tracking_state"][0] = 3
    return fields


def manager_wire(pv=PV, profile=1, version=4):
    return pack_pose_message(
        {
            "pv": pv.to_array(),
            "stream_mode": np.array([pv.stream_mode], dtype="<i4"),
            "hand_profile": np.array([profile], dtype="<i4"),
        },
        topic="manager_state",
        version=version,
    )


def hand_wire(seq=0, pv=PV):
    return pack_pose_message(hand_fields(seq, pv), topic="inspire_hand", version=1)


class Socket:
    def __init__(self, packets=()):
        self.packets = deque(packets)
        self.sent = []
        self.closed = False

    def recv(self, flags):
        if not self.packets:
            raise zmq.Again()
        return self.packets.popleft()

    def send(self, raw, flags):
        self.sent.append(raw)

    def close(self):
        self.closed = True


class Clock:
    def __init__(self, now=BASE):
        self.now = now

    def __call__(self):
        self.now += 1
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


def test_q6_no_projection_and_fixed_envelope_slew():
    c = ready()
    accepted, decision = command(c, 10, 0, left=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), right=(1.0,) * 6)
    assert accepted and decision.state is BridgeState.ACTIVE
    assert decision.left_counts == (995,) * 5 + (1000,)
    assert decision.right_counts == OPEN
    for tick in range(11, 55):
        _, decision = command(c, tick, tick - 10, left=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert decision.left_counts == (800,) * 5 + (1000,)
    assert decision.right_counts == OPEN


def test_duplicate_does_not_refresh_source_and_conflict_latches():
    c = ready()
    command(c, 10, 0)
    received = c.latest_pair.received_ns
    assert not c.accept_pair(
        (0.8,) * 6, (1.0,) * 6, PV, "inspire_hand", received + 1, 0, repr((0, (0.8,) * 6, (1.0,) * 6)).encode()
    )
    assert c.latest_pair.received_ns == received and c.fault_reason is None
    assert not c.accept_pair((0.9,) * 6, (1.0,) * 6, PV, "inspire_hand", received + 2, 0, b"changed")
    assert "conflict" in c.fault_reason


def test_forward_gaps_count_and_regression_latches():
    c = ready()
    command(c, 10, 0)
    assert command(c, 11, 4)[0]
    assert c.sequence_gaps == 3
    assert not command(c, 12, 3)[0]
    assert c.state is BridgeState.FAULT_LATCHED


@pytest.mark.parametrize("stale_source", ["manager", "q6"])
def test_transport_timeout_opens_both_at_250ms(stale_source):
    c = ready()
    command(c, 10, 0, right=(0.8,) * 6)
    receipt = BASE + 10 * PERIOD + PERIOD // 2
    c.accept_manager(PV, receipt - 1)
    c.accept_pair((0.8,) * 6, (0.8,) * 6, PV, "inspire_hand", receipt, 1, b"later")
    if stale_source == "q6":
        assert c._pair_is_fresh(receipt + 249_999_999)
        assert not c._pair_is_fresh(receipt + 250_000_000)
    for tick in (11, 12, 13):
        now = BASE + tick * PERIOD
        feedback(c, now)
        if stale_source == "q6":
            c.accept_manager(PV, now - 1)
        elif tick == 12:
            c.accept_pair((0.8,) * 6, (0.8,) * 6, PV, "inspire_hand", now - 1, 2, b"fresh q6")
        c.tick(now)
    assert c.state is BridgeState.OPENING
    assert c.latest_pair is None


def test_new_manager_requires_ten_open_feedback_ticks_then_later_q6():
    c = ready()
    command(c, 10, 0)
    pv2 = StreamProvenance(b"b" * 16, 0, 1)
    assert not command(c, 11, 0, pv=pv2)[0]
    assert c.state is BridgeState.OPENING and c.open_confirm_ticks == 0
    for tick in range(12, 22):
        assert not command(c, tick, tick - 11, pv=pv2)[0]
    assert c.state is BridgeState.READY
    assert command(c, 22, 11, pv=pv2)[0]
    assert c.state is BridgeState.ACTIVE
    assert not c.accept_manager(PV, BASE + 23 * PERIOD)
    assert "retired" in c.fault_reason


@pytest.mark.parametrize("candidate", [StreamProvenance(b"a" * 16, 0, 5), StreamProvenance(b"a" * 16, 0, 1)])
def test_epoch_same_epoch_mode_change_or_rollback_faults(candidate):
    c = ready()
    c.accept_manager(StreamProvenance(b"a" * 16, 1, 1), BASE + 10 * PERIOD)
    assert not c.accept_manager(candidate, BASE + 10 * PERIOD + 1)
    assert c.fault_reason


def test_more_than_64_retired_sessions_is_terminal():
    c = InspireFtpSafetyController()
    for n in range(65):
        pv = StreamProvenance((n + 1).to_bytes(16, "little"), 0, 1)
        assert c.accept_manager(pv, BASE + n * PERIOD)
        c.tick(BASE + n * PERIOD)
    assert c.retired_session_count == 64
    assert not c.accept_manager(StreamProvenance(b"z" * 16, 0, 1), BASE + 66 * PERIOD)
    assert "retired" in c.fault_reason


def test_raw_wire_requires_earlier_manager_and_ignores_q7_topics():
    c = ready()
    unseen = StreamProvenance(b"b" * 16, 0, 1)
    clock = Clock(BASE + 10 * PERIOD)
    result = b.drain_zmq_packets(Socket([hand_wire(0, unseen)]), c, monotonic_ns=clock)
    assert result.dispatched == 1 and c.manager_provenance == PV and c.latest_pair is None
    result = b.drain_zmq_packets(Socket([manager_wire(), hand_wire(0)]), c, monotonic_ns=clock)
    assert result.dispatched == 2 and c.latest_pair.left == tuple(float(x) for x in hand_fields()["left_command"])
    assert c.latest_pair.right == tuple(float(x) for x in hand_fields()["right_command"])
    assert c.fault_reason is None
    with pytest.raises(ValueError):
        b.decode_bridge_packet(pack_pose_message({"pv": PV.to_array()}, "pose", 3), "pose", clock())


@pytest.mark.parametrize("wire", [manager_wire(profile=0), manager_wire(version=3), hand_wire() + b"extra"])
def test_wrong_profile_version_or_wire_size_rejected(wire):
    c = ready()
    assert b.drain_zmq_packets(Socket([wire]), c, monotonic_ns=Clock(BASE + PERIOD * 10)).rejected == 1
    assert c.latest_pair is None


@pytest.mark.parametrize("mutation", ["dtype", "order", "nan", "range"])
def test_malformed_q6_is_rejected(mutation):
    fields = hand_fields()
    if mutation == "dtype":
        fields["left_command"] = fields["left_command"].astype("f8")
    elif mutation == "order":
        fields = dict(reversed(list(fields.items())))
    elif mutation == "nan":
        fields["left_command"][0] = np.nan
    else:
        fields["left_command"][0] = 1.1
    with pytest.raises(ValueError):
        b.decode_bridge_packet(pack_pose_message(fields, "inspire_hand", 1), "inspire_hand", BASE)


def test_feedback_timeout_write_failure_and_overrun_are_latched():
    c = ready()
    command(c, 10, 0)
    c.record_write_result("left", c.last_successful_left, False)
    c.tick(BASE + 11 * PERIOD)
    assert c.state is BridgeState.FAULT_LATCHED and "write failed" in c.fault_reason
    c = ready()
    c.tick(BASE + 11 * PERIOD)
    assert c.state is BridgeState.FAULT_LATCHED and "overrun" in c.fault_reason
    c = ready()
    for tick in range(10, 15):
        c.tick(BASE + tick * PERIOD)
    assert c.state is BridgeState.FAULT_LATCHED and "timeout" in c.fault_reason


def test_status_roundtrip_is_exact_and_applied_values_are_successful_write_baselines():
    c = ready()
    command(c, 10, 0)
    c.last_applied_message_seq = 0
    sock = Socket()
    status = b.InspireStatusPublisher(socket=sock)
    for tick in range(2):
        status.publish(c, BASE + 10 * PERIOD + tick)
        decoded = unpack_pose_message(sock.sent[-1], "inspire_hand_status")
        validate_inspire_status(decoded)
        assert decoded["status_seq"][0] == tick
        assert decoded["bridge_state"][0] == 2
        np.testing.assert_array_equal(decoded["left_applied"], np.array(c.last_successful_left, dtype="f4") / 1000)
        np.testing.assert_array_equal(decoded["accepted_pv"], PV.to_array())
        assert decoded["feedback_healthy"][0]
    decoded = status.fields(c, BASE + 15 * PERIOD)
    assert not decoded["feedback_healthy"][0]


def ownership():
    tracker = b.PublicationOwnershipTracker()
    for side in ("left", "right"):
        tracker.observe(
            [b.PublicationLifecycleSample("state-" + side, "driver", b.STATE_TOPICS[side], True, BASE)]
        )
        tracker.observe(
            [b.PublicationLifecycleSample("owned-" + side, "bridge", b.COMMAND_TOPICS[side], True, BASE)]
        )
        tracker.register_owned(side, "owned-" + side)
    tracker.seal_ownership()
    return tracker


def test_dds_foreign_writer_or_driver_disappearance_is_sticky():
    tracker = ownership()
    assert tracker.health_fault() is None
    tracker.observe(
        [
            b.PublicationLifecycleSample("foreign", "other", b.COMMAND_TOPICS["left"], True, BASE + 1),
            b.PublicationLifecycleSample("foreign", "other", b.COMMAND_TOPICS["left"], False, BASE + 2),
        ]
    )
    assert tracker.health_fault() is not None and tracker.evidence().foreign_history
    tracker = ownership()
    tracker.observe(
        [b.PublicationLifecycleSample("state-left", "driver", b.STATE_TOPICS["left"], False, BASE + 1)]
    )
    assert tracker.health_fault() is not None


class sequence:
    def __init__(self, subtype, max_length):
        self.subtype, self.max_length = subtype, max_length


class State:
    __annotations__ = {
        name: Annotated[Sequence[scalar], sequence(scalar, 6)]
        for name, scalar in [("angle_act", Annotated[int, "int16"]), ("err", Annotated[int, "uint8"])]
    }

    def __init__(self):
        self.angle_act, self.err = list(OPEN), [0] * 6


def test_callback_snapshot_is_immutable_and_overflow_faults():
    inbox = b.InboundInbox(capacity=1)
    source = State()
    inbox.capture_state("left", source, BASE)
    source.angle_act[0] = 0
    inbox.capture_state("right", State(), BASE)
    events, overflow = inbox.drain()
    assert overflow and events[0].snapshot.angle_act == OPEN


def test_monitor_runtime_creates_no_dds_writers_and_publishes_status_every_tick():
    clock, sock, status_sock = Clock(), Socket(), Socket()
    status = b.InspireStatusPublisher(socket=status_sock)
    subscribers = []

    class Subscriber:
        def Init(self, callback, queue):
            self.callback = callback

        def Close(self):
            pass

    def subscribe(topic, kind):
        sub = Subscriber()
        subscribers.append(sub)
        return sub

    def poll(*, monotonic_ns):
        for sub in subscribers:
            sub.callback(State())
        return ()

    def forbidden(*args):
        pytest.fail("monitor mode created a command resource")

    dds = SimpleNamespace(
        state_type=State,
        subscriber_factory=subscribe,
        drain_publications=poll,
        publisher_factory=forbidden,
        message_factory=forbidden,
    )
    deps = b.RuntimeDependencies(
        monotonic_ns=clock,
        sleep=clock.sleep,
        zmq_module=zmq,
        zmq_socket_factory=lambda host, port: sock,
        status_publisher_factory=lambda port: status,
        dds_loader=lambda interface: dds,
        signal_source_factory=lambda: SimpleNamespace(drain=lambda: 0, close=lambda: None),
        logger=lambda record: None,
        max_ticks=3,
    )
    args = b.parse_args([])
    assert not args.publish and args.status_port == 5563
    result = b.run_bridge(args, deps)
    assert result.exit_code == 0 and result.ticks == 3
    assert len(subscribers) == 2 and len(status_sock.sent) == 3
    for raw in status_sock.sent:
        fields = unpack_pose_message(raw, "inspire_hand_status")
        assert fields["bridge_state"][0] == 0 and fields["last_applied_message_seq"][0] == -1
    assert sock.closed and status_sock.closed


def test_publish_loop_uses_only_bounded_q6_and_status_tracks_successful_pair():
    c, clock = ready(), Clock(BASE + 10 * PERIOD)
    status_socket = Socket()
    writes = {"left": [], "right": []}

    class Writer:
        def __init__(self, side):
            self.side = side

        def Write(self, message):
            writes[self.side].append(tuple(message.angle_set))
            return True

    resources = b.BridgeRuntimeResources(
        controller=c,
        inbox=b.InboundInbox(),
        zmq_socket=Socket([manager_wire(), hand_wire(7)]),
        discovery=SimpleNamespace(drain_publications=lambda **kwargs: ()),
        ownership=ownership(),
        signal_source=SimpleNamespace(drain=lambda: 0),
        left_writer=Writer("left"),
        right_writer=Writer("right"),
        left_message=SimpleNamespace(),
        right_message=SimpleNamespace(),
        status_publisher=b.InspireStatusPublisher(socket=status_socket),
    )
    deps = b.RuntimeDependencies(
        monotonic_ns=clock, sleep=clock.sleep, zmq_module=zmq, logger=lambda record: None, max_ticks=3
    )
    result = b.run_control_loop(resources, deps)
    assert result.exit_code == 0 and result.ticks == 3
    assert len(writes["left"]) == len(writes["right"]) == 3
    for side in ("left", "right"):
        previous = OPEN
        for counts in writes[side]:
            assert all(800 <= value <= 1000 for value in counts)
            assert max(abs(a - z) for a, z in zip(counts, previous)) <= 5
            previous = counts
    assert len(status_socket.sent) == 3
    for raw in status_socket.sent:
        fields = unpack_pose_message(raw, "inspire_hand_status")
        assert fields["last_applied_message_seq"][0] == 7
        assert fields["bridge_state"][0] == 2
    assert not c.fault_reason  # ingress timing jitter must not skip a 10Hz decision


def test_foreign_writer_is_latched_before_closing_command_in_runtime():
    c, clock = ready(), Clock(BASE + 10 * PERIOD)
    writes = []
    writer = SimpleNamespace(Write=lambda message: writes.append(tuple(message.angle_set)) or True)
    foreign = b.PublicationLifecycleSample("foreign", "other", b.COMMAND_TOPICS["left"], True, clock())
    resources = b.BridgeRuntimeResources(
        controller=c,
        inbox=b.InboundInbox(),
        zmq_socket=Socket([manager_wire(), hand_wire()]),
        discovery=SimpleNamespace(drain_publications=lambda **kwargs: (foreign,)),
        ownership=ownership(),
        signal_source=SimpleNamespace(drain=lambda: 0),
        left_writer=writer,
        right_writer=writer,
        left_message=SimpleNamespace(),
        right_message=SimpleNamespace(),
    )
    result = b.run_control_loop(
        resources,
        b.RuntimeDependencies(
            monotonic_ns=clock, sleep=clock.sleep, zmq_module=zmq, logger=lambda record: None, max_ticks=1
        ),
    )
    assert result.ticks == 1 and c.state is BridgeState.FAULT_LATCHED
    assert writes == [OPEN, OPEN]
