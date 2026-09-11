"""Protocol tests exercise restarts, loss, ordering and save ownership."""

import numpy as np
import pytest

from gear_sonic.utils.teleop.pico_recording import (
    MAX_SEQUENCE,
    STALE_NS,
    ManagerRecording,
    RecorderProtocol,
    RecordingCommand as C,
    RecordingError as E,
    RecordingState as S,
    make_pv,
    parse_pv,
)

SID = b"m" * 16


def pair():
    manager, recorder = ManagerRecording(SID, "dex3"), RecorderProtocol("dex3")
    assert recorder.receive(manager.fields(0, 1), 0) is None
    assert manager.receive_status(recorder.status_fields(0), 0)
    return manager, recorder


def command_packet(command, seq, *, session=SID, mode=1, epoch=0):
    fields = ManagerRecording(session, "dex3").fields(epoch, mode)
    fields["recording_command"] = np.array([command], np.int32)
    fields["recording_command_seq"] = np.array([seq], np.int64)
    return fields


def test_pv_strict_and_roundtrip():
    assert parse_pv(make_pv(SID, 12, 5)) == (SID, 12, 5)
    for args in [(bytes(16), 0, 1), (SID, -1, 1), (SID, 0, 6), (SID, MAX_SEQUENCE + 1, 1)]:
        with pytest.raises(ValueError):
            make_pv(*args)
    with pytest.raises(ValueError):
        parse_pv(np.zeros(28, np.int32))


def test_start_stop_ack_and_owned_save():
    m, r = pair()
    assert m.enqueue(C.START, 1, tracking=True)
    assert not m.may_exit_tracking(1)
    assert r.receive(m.fields(0, 1), 1) == "start"
    assert m.receive_status(r.status_fields(0), 2)
    assert m.pending_count == 0 and m.capture_active
    assert m.enqueue(C.STOP_AND_SAVE, 3, tracking=True)
    assert r.receive(m.fields(0, 1), 3) == "save"
    assert not r.capture_active
    assert m.receive_status(r.status_fields(0), 4)
    assert not m.may_exit_tracking(4)
    assert r.check_timeout(10 * STALE_NS, STALE_NS) is None
    r.finish_save(True)
    assert m.receive_status(r.status_fields(1), 5)
    assert m.may_exit_tracking(5)


def test_old_idle_ack_cannot_clear_start_latch():
    m, r = pair()
    assert m.enqueue(C.START, 1, tracking=True)
    assert m.receive_status(r.status_fields(0), 2)
    assert not m.may_exit_tracking(2)
    assert m.recording_may_be_active


def test_exporter_restart_drops_pending_and_restarts_at_one():
    m, r = pair()
    assert m.enqueue(C.START, 1, tracking=True)
    assert r.receive(m.fields(0, 1), 1) == "start"
    assert m.receive_status(r.status_fields(0), 2)
    assert m.enqueue(C.STOP_AND_SAVE, 3, tracking=True)
    restarted = RecorderProtocol("dex3")
    assert m.receive_status(restarted.status_fields(0), 4)
    assert m.pending_count == 0
    assert not m.enqueue(C.START, 4, tracking=True)
    handshake = m.fields(0, 1)
    assert handshake["recording_command_seq"].item() == 0
    restarted.receive(handshake, 5)
    assert m.receive_status(restarted.status_fields(0), 6)
    assert m.enqueue(C.START, 7, tracking=True)
    assert m.fields(0, 1)["recording_command_seq"].item() == 1
    assert not m.receive_status(r.status_fields(0), 8)


def test_manager_restart_cannot_steal_recording_and_times_out():
    m, r = pair()
    m.enqueue(C.START, 1, tracking=True)
    assert r.receive(m.fields(0, 1), 1) == "start"
    other = ManagerRecording(b"n" * 16, "dex3")
    assert r.receive(other.fields(0, 1), STALE_NS - 1) is None
    assert r.manager_session_id == SID
    assert r.check_timeout(STALE_NS + 1, 0) == "abort"
    r.finish_save(True)
    r.receive(other.fields(0, 1), STALE_NS + 2)
    assert r.manager_session_id == b"n" * 16
    assert r.receive(m.fields(0, 1), STALE_NS + 3) is None
    assert r.manager_session_id == b"n" * 16


def test_duplicate_gap_conflict_and_sequence_regression():
    _, r = pair()
    start = command_packet(C.START, 1)
    assert r.receive(command_packet(C.STOP_AND_SAVE, 2), 1) is None
    assert r.last_applied_command_seq == 0
    assert r.receive(start, 2) == "start"
    assert r.receive(start, 3) is None
    assert r.receive(command_packet(C.STOP_AND_SAVE, 1), 4) is None
    assert r.state == S.ERROR and not r.capture_active
    assert r.error_code == E.COMMAND_CONFLICT


def test_timeout_boundaries_and_save_failure():
    m, r = pair()
    m.enqueue(C.START, 0, tracking=True)
    r.receive(m.fields(0, 1), 0)
    assert r.check_timeout(STALE_NS - 1, STALE_NS - 1) is None
    assert r.check_timeout(STALE_NS, 0) == "abort"
    r.finish_save(False)
    assert r.state == S.ERROR and r.error_code == E.SAVE_FAILED
    assert m.receive_status(r.status_fields(0), STALE_NS)
    assert m.may_exit_tracking(STALE_NS)
    assert not m.enqueue(C.START, STALE_NS, tracking=True)


def test_fifo_overflow_stale_and_legacy_pulse():
    m, _ = pair()
    assert not m.enqueue(C.START, STALE_NS, tracking=True)
    for _ in range(16):
        assert m.enqueue(C.TOGGLE, 1, tracking=True)
    assert not m.enqueue(C.TOGGLE, 1, tracking=True)
    assert m.error_code == E.RECORDER_QUEUE_FULL
    assert m.fields(0, 1)["toggle_data_collection"].item()
    assert not m.fields(0, 1)["toggle_data_collection"].item()
    assert not m.may_exit_tracking(STALE_NS)


@pytest.mark.parametrize(
    "name,bad",
    [
        ("recording_command_seq", np.array([1], np.int32)),
        ("recording_command", np.array([1, 1], np.int32)),
        ("pv", np.zeros(28, np.float32)),
        ("hand_profile", np.array([1], np.int32)),
    ],
)
def test_invalid_command_is_atomic_and_does_not_refresh(name, bad):
    _, r = pair()
    fields = command_packet(C.START, 1)
    fields[name] = bad
    assert r.receive(fields, 100) is None
    assert r.state == S.IDLE and r.last_applied_command_seq == 0
    assert r._received_ns == 0


def test_status_duplicates_wrong_shape_and_wrong_session_rejected():
    m, r = pair()
    status = r.status_fields(0)
    assert m.receive_status(status, 1)
    assert not m.receive_status(status, 2)
    wrong = r.status_fields(0)
    wrong["capture_active"] = np.array([0], np.int32)
    assert not m.receive_status(wrong, 3)
    wrong = r.status_fields(0)
    wrong["adopted_manager_session_id"] = np.frombuffer(b"x" * 16, np.uint8)
    assert not m.receive_status(wrong, 4)
    assert m._received_ns == 1


def test_mode_epoch_and_authorized_modes():
    _, r = pair()
    assert r.receive(command_packet(C.START, 1, mode=2, epoch=1), 1) is None
    assert r.state == S.IDLE and r.last_applied_command_seq == 1
    assert r.receive(command_packet(C.START, 2, epoch=0), 2) is None
    assert r.receive(command_packet(C.START, 2, mode=1, epoch=1), 3) is None
    assert r.receive(command_packet(C.START, 2, mode=1, epoch=2), 4) == "start"


def test_sequence_overflow_and_retired_session_cap():
    m, r = pair()
    m._next_seq = MAX_SEQUENCE + 1
    assert not m.enqueue(C.START, 1, tracking=True)
    assert m.error_code == E.SEQUENCE_OVERFLOW
    for i in range(64):
        session = (i + 1).to_bytes(16, "little")
        r.receive(command_packet(C.NONE, 0, session=session), i + 1)
    assert len(r._retired) == 64
    r.receive(command_packet(C.NONE, 0, session=b"z" * 16), 65)
    assert r.state == S.ERROR and r.error_code == E.SESSION_LIMIT


def test_regressing_ack_and_future_ack_cannot_discard_fifo():
    m, r = pair()
    m.enqueue(C.START, 1, tracking=True)
    r.receive(m.fields(0, 1), 1)
    assert m.receive_status(r.status_fields(0), 2)
    m.enqueue(C.STOP_AND_SAVE, 3, tracking=True)
    for ack in (0, 3):
        status = r.status_fields(0)
        status["last_applied_command_seq"] = np.array([ack], np.int64)
        assert not m.receive_status(status, 4)
    assert m.pending_count == 1
    assert m.recording_may_be_active


def test_abort_legacy_pulse_and_discard_completion():
    m, r = pair()
    m.enqueue(C.START, 1, tracking=True)
    r.receive(m.fields(0, 1), 1)
    m.receive_status(r.status_fields(0), 2)
    assert m.enqueue(C.ABORT, 3, tracking=True)
    fields = m.fields(0, 1)
    assert fields["toggle_data_abort"].item()
    assert not m.fields(0, 1)["toggle_data_abort"].item()
    assert r.receive(fields, 3) == "abort"
    assert r.state == S.SAVING and not r.capture_active
    assert r.receive(fields, 4) is None
    r.finish_save(True)
    assert r.state == S.IDLE


def test_feedback_loss_discards_despite_live_manager_heartbeats():
    m, r = pair()
    m.enqueue(C.START, 1, tracking=True)
    r.receive(m.fields(0, 1), 1)
    m.receive_status(r.status_fields(0), 2)
    r.receive(m.fields(0, 1), STALE_NS)
    assert r.check_timeout(STALE_NS, STALE_NS) == "abort"


def test_pending_start_can_queue_shutdown_save_before_recording_ack():
    m, r = pair()
    assert m.enqueue(C.START, 1, tracking=True)
    assert m.recording_state == S.IDLE
    assert m.enqueue(C.STOP_AND_SAVE, 2, tracking=True)
    assert m.pending_count == 2
    assert r.receive(m.fields(0, 1), 3) == "start"
    assert m.receive_status(r.status_fields(0), 4)
    assert r.receive(m.fields(0, 1), 5) == "save"
    assert m.receive_status(r.status_fields(0), 6)
    assert m.pending_count == 0
    r.finish_save(True)
    assert m.receive_status(r.status_fields(1), 7)
    assert m.may_exit_tracking(7)


@pytest.mark.parametrize("command,event", [(C.ABORT, "abort"), (C.STOP_AND_SAVE, "save")])
def test_adopted_recording_can_stop_when_manager_enters_off(command, event):
    m, r = pair()
    assert m.enqueue(C.START, 1, tracking=True)
    assert r.receive(m.fields(0, 1), 1) == "start"
    assert m.receive_status(r.status_fields(0), 2)
    assert m.enqueue(command, 3, tracking=True)
    assert r.receive(m.fields(1, 0), 3) == event
    assert r.state == S.SAVING and not r.capture_active
    assert r.receive(m.fields(1, 0), 4) is None
