"""Pure manager/recorder protocol state; callers own clocks, transport and saving.

Call ``fields`` once per published manager tick, ``receive_status`` for each
status, and ``enqueue`` for operator events. Recorder ``receive`` returns
``start``, ``save`` or ``abort``; save/abort detach capture immediately and the
caller reports I/O completion with ``finish_save``. All times are monotonic ns.
"""

from collections import deque
from enum import IntEnum
import struct
import uuid

import numpy as np

STALE_NS = 500_000_000
MAX_SEQUENCE = 2**63 - 1
PROFILES = {"dex3": 0, "inspire_ftp": 1}
AUTHORIZED_MODES = {1, 5}


class RecordingCommand(IntEnum):
    NONE = 0
    START = 1
    STOP_AND_SAVE = 2
    TOGGLE = 3
    ABORT = 4


class RecordingState(IntEnum):
    IDLE = 0
    RECORDING = 1
    SAVING = 2
    ERROR = 3


class RecordingError(IntEnum):
    NONE = 0
    RECORDER_QUEUE_FULL = 1
    SEQUENCE_OVERFLOW = 2
    COMMAND_CONFLICT = 3
    SESSION_LIMIT = 4
    SAVE_FAILED = 5
    PROFILE_MISMATCH = 6


def _session(value, *, zero=False):
    if not isinstance(value, bytes) or len(value) != 16 or (not zero and not any(value)):
        raise ValueError("session must be 16 bytes and nonzero")
    return value


def _array(fields, name, dtype, shape):
    value = fields.get(name)
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"invalid {name} dtype/shape")
    return value


def _scalar(fields, name, dtype):
    return _array(fields, name, dtype, (1,)).item()


def _one(value, dtype):
    return np.array([value], dtype=dtype)


def make_pv(session_id: bytes, mode_epoch: int, stream_mode: int) -> np.ndarray:
    """Build the shared strict 28-byte session/mode provenance field."""
    _session(session_id)
    if type(mode_epoch) is not int or not 0 <= mode_epoch <= MAX_SEQUENCE:
        raise ValueError("invalid mode epoch")
    if type(stream_mode) is not int or stream_mode not in range(6):
        raise ValueError("invalid stream mode")
    return np.frombuffer(struct.pack("<16sqi", session_id, mode_epoch, stream_mode), dtype=np.uint8).copy()


def parse_pv(value: np.ndarray) -> tuple[bytes, int, int]:
    """Validate and decode provenance; zero sessions are never commands."""
    raw = _array({"pv": value}, "pv", np.uint8, (28,))
    session, epoch, mode = struct.unpack("<16sqi", raw.tobytes())
    make_pv(session, epoch, mode)
    return session, epoch, mode


class ManagerRecording:
    def __init__(self, session_id: bytes, profile: str):
        self.session_id = _session(session_id)
        self.profile = PROFILES[profile]
        self.recorder_session_id = None
        self._retired = set()
        self._status_seq = -1
        self._last_ack = -1
        self._received_ns = None
        self._adopted = False
        self._next_seq = 1
        self._queue = deque()
        self._published = set()
        self._active_seq = None
        self.recording_may_be_active = False
        self.recording_state = RecordingState.IDLE
        self.capture_active = False
        self.error_code = RecordingError.NONE

    @property
    def pending_count(self):
        return len(self._queue)

    def status_fresh(self, now_ns):
        return self._received_ns is not None and 0 <= now_ns - self._received_ns < STALE_NS

    def may_exit_tracking(self, now_ns):
        if not self.recording_may_be_active:
            return True
        return (
            self.status_fresh(now_ns)
            and self._adopted
            and not self.capture_active
            and self.recording_state in (RecordingState.IDLE, RecordingState.ERROR)
            and self._active_seq is None
        )

    def fields(self, mode_epoch: int, stream_mode: int):
        command, seq = self._queue[0] if self._adopted and self._queue else (RecordingCommand.NONE, 0)
        first_publication = seq not in self._published
        pulse = command == RecordingCommand.TOGGLE and first_publication
        if seq:
            self._published.add(seq)
        return {
            "pv": make_pv(self.session_id, mode_epoch, stream_mode),
            "stream_mode": _one(stream_mode, np.int32),
            "hand_profile": _one(self.profile, np.int32),
            "recording_protocol_version": _one(1, np.int32),
            "recording_command": _one(command, np.int32),
            "recording_command_seq": _one(seq, np.int64),
            "toggle_data_collection": _one(pulse, np.bool_),
            "toggle_data_abort": _one(command == RecordingCommand.ABORT and first_publication, np.bool_),
        }

    def receive_status(self, fields, now_ns):
        try:
            recorder = _session(_array(fields, "recorder_session_id", np.uint8, (16,)).tobytes())
            adopted = _session(_array(fields, "adopted_manager_session_id", np.uint8, (16,)).tobytes(), zero=True)
            seq = _scalar(fields, "status_seq", np.int64)
            ack = _scalar(fields, "last_applied_command_seq", np.int64)
            state = RecordingState(_scalar(fields, "recording_state", np.int32))
            active = _scalar(fields, "capture_active", np.bool_)
            episode = _scalar(fields, "episode_index", np.int64)
            error = _scalar(fields, "error_code", np.int32)
            profile = _scalar(fields, "hand_profile", np.int32)
            if seq < 0 or ack < -1 or episode < 0 or error < 0 or profile != self.profile:
                return False
            if active != (state == RecordingState.RECORDING):
                return False
            if recorder in self._retired or (recorder == self.recorder_session_id and seq <= self._status_seq):
                return False
            if adopted not in (bytes(16), self.session_id):
                return False
            if adopted == bytes(16) and ack != -1:
                return False
            if adopted == self.session_id and ack < 0:
                return False
        except (ValueError, TypeError, KeyError):
            return False
        changed = recorder != self.recorder_session_id
        # A restarted exporter must first answer the explicit NONE/0 handshake.
        if changed and adopted == self.session_id and ack != 0:
            return False
        if not changed and adopted == self.session_id and ack >= self._next_seq:
            return False
        if not changed and ack < self._last_ack:
            return False
        if changed:
            if self.recorder_session_id is not None:
                if len(self._retired) >= 64:
                    self.error_code = RecordingError.SESSION_LIMIT
                    return False
                self._retired.add(self.recorder_session_id)
            self.recorder_session_id = recorder
            self._queue.clear()
            self._published.clear()
            self._next_seq = 1
            self._active_seq = None
            self._adopted = False
        self._status_seq = seq
        self._last_ack = ack
        self._received_ns = now_ns
        self.recording_state, self.capture_active = state, active
        if adopted == self.session_id:
            if not self._adopted and state == RecordingState.IDLE and ack == 0:
                self._adopted = True
            if self._adopted:
                while self._queue and self._queue[0][1] <= ack:
                    _, removed = self._queue.popleft()
                    self._published.discard(removed)
                if (
                    (self._active_seq is None or ack >= self._active_seq)
                    and not active
                    and state in (RecordingState.IDLE, RecordingState.ERROR)
                ):
                    self._active_seq = None
                    self.recording_may_be_active = False
        if active:
            self.recording_may_be_active = True
        return True

    def enqueue(self, command, now_ns, *, tracking: bool):
        try:
            command = RecordingCommand(command)
        except (ValueError, TypeError):
            return False
        if (
            command == RecordingCommand.NONE
            or not tracking
            or not self._adopted
            or not self.status_fresh(now_ns)
            or self.error_code in (RecordingError.SEQUENCE_OVERFLOW, RecordingError.SESSION_LIMIT)
        ):
            return False
        if self.recording_state in (RecordingState.SAVING, RecordingState.ERROR):
            return False
        if command == RecordingCommand.START and self.recording_state != RecordingState.IDLE:
            return False
        if (
            command in (RecordingCommand.STOP_AND_SAVE, RecordingCommand.ABORT)
            and self.recording_state != RecordingState.RECORDING
            and not self.recording_may_be_active
        ):
            return False
        if len(self._queue) >= 16:
            self.error_code = RecordingError.RECORDER_QUEUE_FULL
            return False
        if self._next_seq > MAX_SEQUENCE:
            self.error_code = RecordingError.SEQUENCE_OVERFLOW
            return False
        seq = self._next_seq
        self._next_seq += 1
        self._queue.append((command, seq))
        if command in (RecordingCommand.START, RecordingCommand.TOGGLE):
            self.recording_may_be_active = True
            self._active_seq = seq
        return True


class RecorderProtocol:
    def __init__(self, profile: str):
        self.profile = PROFILES[profile]
        self.session_id = uuid.uuid4().bytes
        self.manager_session_id = None
        self._retired = set()
        self._mode_epoch = -1
        self._mode = None
        self._received_ns = None
        self._last_command = RecordingCommand.NONE
        self.last_applied_command_seq = -1
        self._status_seq = 0
        self.state = RecordingState.IDLE
        self.error_code = RecordingError.NONE

    @property
    def capture_active(self):
        return self.state == RecordingState.RECORDING

    def _error(self, code):
        self.state = RecordingState.ERROR
        self.error_code = code

    def receive(self, fields, now_ns):
        try:
            session, epoch, mode = parse_pv(fields.get("pv"))
            profile = _scalar(fields, "hand_profile", np.int32)
            version = _scalar(fields, "recording_protocol_version", np.int32)
            command = RecordingCommand(_scalar(fields, "recording_command", np.int32))
            seq = _scalar(fields, "recording_command_seq", np.int64)
            declared_mode = _scalar(fields, "stream_mode", np.int32)
            if version != 1 or profile != self.profile or declared_mode != mode or seq < 0:
                return None
            if (command == RecordingCommand.NONE) != (seq == 0):
                return None
        except (ValueError, TypeError, KeyError):
            return None
        if session in self._retired:
            return None
        if session != self.manager_session_id:
            if command != RecordingCommand.NONE or self.state != RecordingState.IDLE:
                return None
            if self.manager_session_id is not None:
                if len(self._retired) >= 64:
                    self._error(RecordingError.SESSION_LIMIT)
                    return None
                self._retired.add(self.manager_session_id)
            self.manager_session_id = session
            self.last_applied_command_seq = 0
            self._last_command = RecordingCommand.NONE
            self._mode_epoch = epoch
            self._mode = mode
        if epoch < self._mode_epoch or (epoch == self._mode_epoch and mode != self._mode):
            return None
        # Only current, structurally valid adopted provenance refreshes heartbeat.
        self._mode_epoch, self._mode = epoch, mode
        self._received_ns = now_ns
        if command == RecordingCommand.NONE:
            return None
        if seq == self.last_applied_command_seq:
            if command != self._last_command:
                self._error(RecordingError.COMMAND_CONFLICT)
            return None
        if seq != self.last_applied_command_seq + 1:
            return None
        # Consume unauthorized requests as no-ops so a refusal cannot wedge FIFO.
        self.last_applied_command_seq, self._last_command = seq, command
        if self.state in (RecordingState.SAVING, RecordingState.ERROR):
            return None
        if mode not in AUTHORIZED_MODES and command in (RecordingCommand.START, RecordingCommand.TOGGLE):
            return None
        if command == RecordingCommand.TOGGLE:
            command = (
                RecordingCommand.START if self.state == RecordingState.IDLE else RecordingCommand.STOP_AND_SAVE
            )
        if command == RecordingCommand.START and self.state == RecordingState.IDLE:
            self.state = RecordingState.RECORDING
            return "start"
        if self.state == RecordingState.RECORDING:
            if command == RecordingCommand.STOP_AND_SAVE:
                self.state = RecordingState.SAVING
                return "save"
            if command == RecordingCommand.ABORT:
                self.state = RecordingState.SAVING
                return "abort"
        return None

    def finish_save(self, success: bool):
        if self.state != RecordingState.SAVING:
            return
        if success:
            self.state = RecordingState.IDLE
        else:
            self._error(RecordingError.SAVE_FAILED)

    def check_timeout(self, now_ns, feedback_age_ns):
        if self.capture_active and (
            self._received_ns is None
            or not 0 <= now_ns - self._received_ns < STALE_NS
            or feedback_age_ns is None
            or not 0 <= feedback_age_ns < STALE_NS
        ):
            self.state = RecordingState.SAVING
            return "abort"
        return None

    def status_fields(self, episode_index: int):
        if self._status_seq > MAX_SEQUENCE:
            self._error(RecordingError.SEQUENCE_OVERFLOW)
            raise OverflowError("recording status sequence exhausted")
        if type(episode_index) is not int or not 0 <= episode_index <= MAX_SEQUENCE:
            raise ValueError("invalid episode index")
        result = {
            "recorder_session_id": np.frombuffer(self.session_id, dtype=np.uint8).copy(),
            "status_seq": _one(self._status_seq, np.int64),
            "adopted_manager_session_id": np.frombuffer(
                self.manager_session_id or bytes(16), dtype=np.uint8
            ).copy(),
            "last_applied_command_seq": _one(self.last_applied_command_seq, np.int64),
            "recording_state": _one(self.state, np.int32),
            "capture_active": _one(self.capture_active, np.bool_),
            "episode_index": _one(episode_index, np.int64),
            "error_code": _one(self.error_code, np.int32),
            "hand_profile": _one(self.profile, np.int32),
        }
        self._status_seq += 1
        return result
