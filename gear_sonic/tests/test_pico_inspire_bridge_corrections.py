"""Hardware-free regressions for the native optical physical-runtime port."""

from types import SimpleNamespace

import pytest
import zmq

from gear_sonic.scripts import run_pico_inspire_bridge as b
from gear_sonic.tests.test_pico_inspire_bridge import (
    BASE,
    OPEN,
    PERIOD,
    PV,
    Clock,
    Socket,
    command,
    feedback,
    hand_wire,
    manager_wire,
    ownership,
    ready,
)
from gear_sonic.utils.teleop.inspire_ftp_real_bridge import (
    BridgeState,
    InspireFtpSafetyController,
    ValidatedHandState,
)


def test_separate_driver_participants_and_metadata_less_disposal():
    tracker = b.PublicationOwnershipTracker()
    for side in ("left", "right"):
        tracker.observe(
            [b.PublicationLifecycleSample("state-" + side, "driver-" + side, b.STATE_TOPICS[side], True, BASE)]
        )
    assert tracker.evidence().state_discovery_healthy
    tracker = ownership()
    tracker.observe([b.PublicationLifecycleSample("owned-left", "bridge", "None", False, BASE + 1)])
    assert dict(tracker.evidence().command_alive)["left"] == ()
    assert tracker.evidence().owned_disruption == (("left", "owned-left"),)


def test_fault_keeps_first_offending_feedback_after_healthy_update():
    c = ready()
    command(c, 10, 0)
    bad = ValidatedHandState("left", (987,) * 6, (0, 0, 9, 0, 0, 0), BASE + 11 * PERIOD)
    c.accept_hand_state(bad)
    feedback(c, BASE + 11 * PERIOD + 1)
    assert c.fault_feedback == bad
    record = b.build_status_record(c, ownership().evidence(), BASE + 11 * PERIOD + 2, None, None)
    assert record["fault_feedback"]["errors"] == list(bad.err)


def test_200ms_scheduling_guard_keeps_active_commands_and_faults_above_guard():
    c = ready()
    command(c, 10, 0)
    now = BASE + 10 * PERIOD + 150_000_000
    feedback(c, now)
    assert c.tick(now).state is BridgeState.ACTIVE
    feedback(c, now + 200_000_001)
    assert c.tick(now + 200_000_001).state is BridgeState.FAULT_LATCHED


def test_initialization_is_measured_bounded_and_requires_new_source():
    c = InspireFtpSafetyController(initialize_hands=True)
    start = (900,) * 6
    c.arm_publish_from_feedback(
        ValidatedHandState("left", start, (0,) * 6, BASE), ValidatedHandState("right", start, (0,) * 6, BASE), BASE
    )
    assert not c.accept_manager(PV, BASE)
    for tick in range(80):
        now = BASE + tick * PERIOD
        measured = tuple(max(900, x - 13) for x in c.last_successful_left)
        feedback(c, now, measured)
        decision = c.tick(now)
        if decision.publish:
            for side, counts in [("left", decision.left_counts), ("right", decision.right_counts)]:
                old = getattr(c, "last_successful_" + side)
                assert all(a <= z <= a + 5 for a, z in zip(old, counts))
                c.record_write_result(side, counts, True)
        if c.state is BridgeState.READY:
            break
    assert c.open_only_completed and not c.open_only
    assert c.manager_provenance is None and c.latest_pair is None
    assert c.last_applied_message_seq == -1


def test_initialization_stops_when_measured_feedback_does_not_follow():
    c = InspireFtpSafetyController(initialize_hands=True)
    start = (200,) * 6
    c.arm_publish_from_feedback(
        ValidatedHandState("left", start, (0,) * 6, BASE), ValidatedHandState("right", start, (0,) * 6, BASE), BASE
    )
    for tick in range(260):
        now = BASE + tick * PERIOD
        feedback(c, now, start)
        d = c.tick(now)
        if d.exit_now:
            break
        if d.publish:
            for side, values in [("left", d.left_counts), ("right", d.right_counts)]:
                assert max(values) <= 250
                c.record_write_result(side, values, True)
    assert d.exit_now and not d.publish and not c.open_only_completed
    assert "timed out" in c.fault_reason


def test_full_range_and_active_slew_are_explicit_opt_ins():
    args = b.parse_args(["--publish", "--initialize-hands", "--full-position-range", "--no-active-slew-limit"])
    assert args.full_position_range and args.initialize_hands and args.no_active_slew_limit
    c = InspireFtpSafetyController(full_position_range=True, no_active_slew_limit=True)
    c.arm_publish_from_feedback(
        ValidatedHandState("left", OPEN, (0,) * 6, BASE), ValidatedHandState("right", OPEN, (0,) * 6, BASE), BASE
    )
    for tick in range(10):
        feedback(c, BASE + tick * PERIOD)
        c.accept_manager(PV, BASE + tick * PERIOD)
        c.tick(BASE + tick * PERIOD)
    _, d = command(c, 10, 0, left=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert d.left_counts == (0, 200, 400, 600, 800, 1000)
    c.request_shutdown(1)
    d = c.tick(BASE + 11 * PERIOD)
    assert d.left_counts == (5, 205, 405, 605, 805, 1000)
    assert c.shutdown_open_attempt_limit == 200


def test_writes_paced_from_pair_completion_and_failed_pair_not_acknowledged():
    c, clock = ready(), Clock(BASE + 10 * PERIOD)
    starts, completes = [], []

    def write(message):
        starts.append(clock.now)
        clock.now += 20_000_000
        completes.append(clock.now)
        return len(starts) != 2

    writer = SimpleNamespace(Write=write)
    resources = b.BridgeRuntimeResources(
        controller=c,
        inbox=b.InboundInbox(),
        zmq_socket=Socket([manager_wire(), hand_wire(7)]),
        discovery=SimpleNamespace(drain_publications=lambda **kwargs: ()),
        ownership=ownership(),
        signal_source=SimpleNamespace(drain=lambda: 0),
        left_writer=writer,
        right_writer=writer,
        left_message=SimpleNamespace(),
        right_message=SimpleNamespace(),
    )
    b.run_control_loop(
        resources,
        b.RuntimeDependencies(
            monotonic_ns=clock, sleep=clock.sleep, zmq_module=zmq, logger=lambda record: None, max_ticks=2
        ),
    )
    assert starts[2] - completes[1] >= PERIOD
    assert c.last_applied_message_seq == -1


def test_budget_exhaustion_never_dispatches_or_replays_old_partial_batch():
    c = ready()
    sock = Socket([manager_wire()] + [hand_wire(i) for i in range(10)])
    stats = b.drain_zmq_packets(sock, c, monotonic_ns=Clock(BASE + 10 * PERIOD), max_packets=4)
    assert stats.packet_budget_exhausted and stats.dispatched == 0
    assert c.latest_pair is None and c.last_applied_message_seq == -1


@pytest.mark.parametrize("mode, admitted", [(2, True), (3, True), (0, False), (4, False)])
def test_planner_and_frozen_q6_holds_admitted_but_off_pause_denied(mode, admitted):
    c = InspireFtpSafetyController()
    pv = type(PV)(PV.session_id, 0, mode)
    c.arm_publish_from_feedback(
        ValidatedHandState("left", OPEN, (0,) * 6, BASE), ValidatedHandState("right", OPEN, (0,) * 6, BASE), BASE
    )
    for tick in range(10):
        now = BASE + tick * PERIOD
        feedback(c, now)
        c.accept_manager(pv, now)
        c.tick(now)
    assert command(c, 10, 0, pv=pv)[0] is admitted


def test_shutdown_failed_open_hold_writes_do_not_claim_success():
    c = ready()
    c.request_shutdown(1)
    for tick in range(100):
        d = c.tick(BASE + (10 + tick) * PERIOD)
        if d.exit_now:
            break
        if d.publish:
            c.record_write_result("left", d.left_counts, False)
            c.record_write_result("right", d.right_counts, True)
    assert d.exit_now and c.shutdown_hold_ticks < 20


def test_emergency_full_range_opens_boundedly_and_failed_hold_is_unconfirmed():
    writes = []
    writer = SimpleNamespace(Write=lambda message: writes.append(tuple(message.angle_set)) or True)
    clock = Clock()
    assert b._emergency_open_existing_writers(
        writer,
        writer,
        SimpleNamespace(),
        SimpleNamespace(),
        (0,) * 6,
        (0,) * 6,
        monotonic_ns=clock,
        sleep=clock.sleep,
    )
    assert writes[0] == (5,) * 6 and writes[-1] == OPEN
    failing = SimpleNamespace(Write=lambda message: False)
    assert not b._emergency_open_existing_writers(
        failing, failing, SimpleNamespace(), SimpleNamespace(), OPEN, OPEN, monotonic_ns=clock, sleep=clock.sleep
    )


@pytest.mark.parametrize("fail_left", [False, True])
def test_complete_runtime_initialization_defers_source_and_disposes_writers(fail_left):
    from gear_sonic.tests.test_pico_inspire_bridge import State

    clock, logs, subscribers, writers, connections = Clock(), [], {}, {}, []
    lifecycle = {}
    source = Socket()
    status = b.InspireStatusPublisher(socket=Socket())

    def subscribe(topic, kind):
        side = next(side for side, name in b.STATE_TOPICS.items() if name == topic)
        sub = SimpleNamespace(Init=lambda callback, depth: setattr(sub, "callback", callback), Close=lambda: None)
        subscribers[side] = sub
        return sub

    def publish(topic, kind):
        side = next(side for side, name in b.COMMAND_TOPICS.items() if name == topic)
        writer = SimpleNamespace(messages=[], closed=False, participant_key="bridge", Init=lambda: None)

        def write(message):
            writer.messages.append(tuple(message.angle_set))
            return not (fail_left and side == "left")

        writer.Write = write
        writer.Close = lambda: setattr(writer, "closed", True)
        writers[side] = writer
        return writer

    def poll(**kwargs):
        samples = []
        for side, sub in subscribers.items():
            state = State()
            writer = writers.get(side)
            state.angle_act = list(writer.messages[-1] if writer and writer.messages else (0,) * 6)
            sub.callback(state)
            samples.append(
                b.PublicationLifecycleSample(
                    "state-" + side, "driver-" + side, b.STATE_TOPICS[side], True, clock()
                )
            )
        for side, writer in writers.items():
            alive = not writer.closed
            if lifecycle.get(side) != alive:
                lifecycle[side] = alive
                samples.append(
                    b.PublicationLifecycleSample(
                        "owned-" + side, "bridge", b.COMMAND_TOPICS[side] if alive else "None", alive, clock()
                    )
                )
        return samples

    def connect(host, port):
        assert all(writer.messages[-20:] == [OPEN] * 20 for writer in writers.values())
        connections.append(clock.now)
        return source

    dds = SimpleNamespace(
        state_type=State,
        command_type=object,
        subscriber_factory=subscribe,
        publisher_factory=publish,
        message_factory=SimpleNamespace,
        drain_publications=poll,
        publisher_participant_key=lambda writer: writer.participant_key,
    )
    deps = b.RuntimeDependencies(
        monotonic_ns=clock,
        sleep=clock.sleep,
        zmq_module=zmq,
        logger=logs.append,
        max_ticks=235,
        dds_loader=lambda interface: dds,
        zmq_socket_factory=connect,
        status_publisher_factory=lambda port: status,
        signal_source_factory=lambda: SimpleNamespace(drain=lambda: 0, close=lambda: None),
    )
    result = b.run_bridge(b.parse_args(["--publish", "--initialize-hands"]), deps)
    assert all(writer.closed for writer in writers.values())
    assert any(r.get("phase") == "final_zero" and r["result"] == "healthy" for r in logs)
    if fail_left:
        assert result.exit_code == 1 and not result.emergency_attempted
        assert not connections and len(writers["left"].messages) == 1
        assert not writers["right"].messages
    else:
        assert result.exit_code == 0, result
        assert len(connections) == 1 and source.closed
        assert writers["left"].messages[0] == (5,) * 6
        assert any(r.get("event") == "hand_initialized" for r in logs)


def test_status_carries_cached_measurement_age_and_initialization_uses_existing_wire_state():
    c = InspireFtpSafetyController(initialize_hands=True)
    c.arm_publish_from_feedback(
        ValidatedHandState("left", (500,) * 6, (0,) * 6, BASE),
        ValidatedHandState("right", (500,) * 6, (0,) * 6, BASE),
        BASE,
    )
    publisher = b.InspireStatusPublisher(socket=Socket())
    for age in (1, 100_000_000, 500_000_000):
        fields = publisher.fields(c, BASE + age)
        assert fields["left_feedback_age_ns"][0] == fields["right_feedback_age_ns"][0] == age
        assert fields["bridge_state"][0] == 3
        assert bool(fields["feedback_healthy"][0]) is (age < 500_000_000)


def test_initialization_tick_limit_is_not_success():
    c, clock = InspireFtpSafetyController(initialize_hands=True), Clock()
    feedback(c, BASE, (200,) * 6)
    c.arm_publish_from_feedback(*c.hand_states, BASE)
    writer = SimpleNamespace(Write=lambda message: True)
    resources = b.BridgeRuntimeResources(
        controller=c,
        inbox=b.InboundInbox(),
        zmq_socket=None,
        discovery=SimpleNamespace(drain_publications=lambda **kwargs: ()),
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
    assert result.exit_code == 1 and result.opening_unconfirmed


def test_closed_hand_pose_planner_pose_switch_preserves_applied_pose():
    c = ready()
    for tick in range(10, 35):
        command(c, tick, tick - 10, left=(0.8,) * 6, right=(0.8,) * 6)
    for tick, mode in [(35, 2), (36, 1), (37, 3), (38, 5)]:
        pv = type(PV)(PV.session_id, tick - 34, mode)
        applied = c.last_successful_left
        accepted, decision = command(
            c,
            tick,
            0,
            pv=pv,
            left=tuple(v / 1000 for v in applied),
            right=tuple(v / 1000 for v in c.last_successful_right),
        )
        assert accepted and decision.state is BridgeState.ACTIVE
        assert decision.left_counts == applied
        assert c.latest_pair.provenance == pv
        assert c.last_applied_message_seq == -1  # only runtime commits completed writes


def test_authorized_switch_waits_at_applied_counts_without_ack_and_times_out():
    c = ready()
    command(c, 10, 0, left=(0.8,) * 6)
    c.last_applied_message_seq = 0
    applied = c.last_successful_left
    for tick, epoch in [(11, 1), (12, 2), (13, 3)]:
        now = BASE + tick * PERIOD
        feedback(c, now)
        c.accept_manager(type(PV)(PV.session_id, epoch, 2), now - 1)
        d = c.tick(now)
        if tick < 13:
            assert d.state is BridgeState.ACTIVE and d.left_counts == applied
            assert c.latest_pair is None and c.last_applied_message_seq == -1
        else:
            assert d.state is BridgeState.OPENING and d.left_counts == OPEN


@pytest.mark.parametrize("mode, session", [(0, PV.session_id), (4, PV.session_id), (2, b"b" * 16)])
def test_unsafe_active_mode_or_session_transition_still_opens(mode, session):
    c = ready()
    command(c, 10, 0)
    pv = type(PV)(session, 1, mode)
    accepted, d = command(c, 11, 0, pv=pv)
    assert not accepted and d.state is BridgeState.OPENING


def test_pending_off_then_authorized_transition_cannot_bypass_opening():
    c = ready()
    command(c, 10, 0)
    now = BASE + 11 * PERIOD
    c.accept_manager(type(PV)(PV.session_id, 1, 0), now - 3)
    c.accept_manager(type(PV)(PV.session_id, 2, 2), now - 2)
    assert not c.accept_pair(
        (0.8,) * 6, (1.0,) * 6, type(PV)(PV.session_id, 2, 2), "inspire_hand", now - 1, 0, b"new"
    )
    feedback(c, now)
    assert c.tick(now).state is BridgeState.OPENING


def test_runtime_handoff_ack_waits_for_matching_new_command():
    c = ready()
    command(c, 10, 0)
    c.last_applied_message_seq = 0
    held = c.last_successful_left
    pv = type(PV)(PV.session_id, 1, 2)
    clock, writes, status_socket = Clock(BASE + 11 * PERIOD), [], Socket()
    writer = SimpleNamespace(Write=lambda message: writes.append(tuple(message.angle_set)) or True)
    sock = Socket([manager_wire(pv)])
    resources = b.BridgeRuntimeResources(
        controller=c,
        inbox=b.InboundInbox(),
        zmq_socket=sock,
        discovery=SimpleNamespace(drain_publications=lambda **kwargs: ()),
        ownership=ownership(),
        signal_source=SimpleNamespace(drain=lambda: 0),
        left_writer=writer,
        right_writer=writer,
        left_message=SimpleNamespace(),
        right_message=SimpleNamespace(),
        status_publisher=b.InspireStatusPublisher(socket=status_socket),
    )
    deps = b.RuntimeDependencies(
        monotonic_ns=clock, sleep=clock.sleep, zmq_module=zmq, logger=lambda record: None, max_ticks=1
    )
    assert b.run_control_loop(resources, deps).exit_code == 0
    assert c.state is BridgeState.ACTIVE and writes[0] == held
    assert c.latest_pair is None and c.last_applied_message_seq == -1
    clock.now += PERIOD
    sock.packets.extend([manager_wire(pv), hand_wire(4, pv)])
    assert b.run_control_loop(resources, deps).exit_code == 0
    assert c.latest_pair.provenance == pv and c.last_applied_message_seq == 4


def test_old_provenance_command_cannot_acknowledge_new_mode():
    c = ready()
    command(c, 10, 0)
    pv = type(PV)(PV.session_id, 1, 2)
    now = BASE + 11 * PERIOD
    c.accept_manager(pv, now - 2)
    feedback(c, now)
    assert c.tick(now).state is BridgeState.ACTIVE
    assert not c.accept_pair((0.8,) * 6, (1.0,) * 6, PV, "inspire_hand", now + 1, 1, b"old")
    assert c.last_applied_message_seq == -1 and c.latest_pair is None
