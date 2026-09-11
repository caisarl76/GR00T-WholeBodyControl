# Safety/discovery behavior imported from reviewed Inspire PC2 bridge d6368d8.
# Only command eligibility changes: optical q6 replaces q7 gesture projection.
"""Dependency-light wire primitives for the Inspire FTP real-hand bridge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import enum
import math
import struct
from typing import Annotated, get_args, get_origin

import numpy as np

PROVENANCE_FIELD = "pv"
PROVENANCE_SIZE = 28
MAX_MODE_EPOCH = 2**63 - 1
VALID_STREAM_MODES = frozenset(range(6))
_PROVENANCE_STRUCT = struct.Struct("<16sqi")

CONTROL_PERIOD_NS = 100_000_000
MAX_TICK_INTERVAL_NS = 110_000_000
SOURCE_STALE_NS = 250_000_000
STATE_STALE_NS = 500_000_000
MIN_COMMAND = 800
MAX_COMMAND = 1000
MAX_STEP = 5
OPEN_FEEDBACK_MIN = 950
OPEN_CONFIRM_TICKS = 10
# More than sixteen distinct mode/session changes inside one 10 Hz control period
# indicates a control-plane flood, while this cap still tolerates a large burst.
MAX_PENDING_MANAGER_TRANSITIONS = 16
# Retired sessions are never evicted: faulting at 64 keeps replay protection
# deterministic without permitting unbounded process-lifetime growth.
MAX_RETIRED_SESSIONS = 64
MAX_REJECTED_PAIRS = 2**31 - 1

_SIDES = frozenset(("left", "right"))
_OPEN_COUNTS = (MAX_COMMAND,) * 6
_AUTHORIZED_TOPICS = {1: "inspire_hand", 5: "inspire_hand"}

SixInts = tuple[int, int, int, int, int, int]
SixFloats = tuple[float, float, float, float, float, float]


class BridgeState(enum.Enum):
    """Safety states for the real Inspire FTP hand bridge."""

    MONITORING = enum.auto()
    READY = enum.auto()
    ACTIVE = enum.auto()
    OPENING = enum.auto()
    FAULT_LATCHED = enum.auto()
    SHUTDOWN_OPENING = enum.auto()


@dataclass(frozen=True, slots=True)
class ValidatedHandState:
    """One strictly validated hand feedback sample."""

    side: str
    angle_act: SixInts
    err: SixInts
    received_ns: int


@dataclass(frozen=True, slots=True)
class HandPair:
    """One atomically validated and provenance-bound hand target pair."""

    left: SixFloats
    right: SixFloats
    provenance: StreamProvenance
    topic: str
    received_ns: int
    message_seq: int = -1

    def __post_init__(self) -> None:
        _validate_pair_side(self.left, "left hand pair")
        _validate_pair_side(self.right, "right hand pair")


@dataclass(frozen=True, slots=True)
class StateSnapshotEvent:
    """Immutable callback-to-control-loop hand-state event."""

    snapshot: ValidatedHandState


@dataclass(frozen=True, slots=True)
class CallbackFaultEvent:
    """Immutable callback-to-control-loop validation failure."""

    side: str
    reason: str
    received_ns: int


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """One bounded command attempt selected by the safety controller."""

    state: BridgeState
    left_counts: SixInts
    right_counts: SixInts
    publish: bool
    fault_reason: str | None
    exit_now: bool


@dataclass(frozen=True, slots=True)
class _ManagerEvent:
    provenance: StreamProvenance
    received_ns: int


@dataclass(frozen=True, slots=True)
class StreamProvenance:
    """Immutable compact identity for a PICO stream generation."""

    session_id: bytes
    mode_epoch: int
    stream_mode: int

    def __post_init__(self) -> None:
        if type(self.session_id) is not bytes or len(self.session_id) != 16:
            raise ValueError("publisher session must be exactly 16 bytes")
        if type(self.mode_epoch) is not int or not 0 <= self.mode_epoch <= MAX_MODE_EPOCH:
            raise ValueError("mode epoch must be a nonnegative signed int64")
        if type(self.stream_mode) is not int or self.stream_mode not in VALID_STREAM_MODES:
            raise ValueError("stream mode must be an integer in [0, 5]")

    def to_bytes(self) -> bytes:
        """Return the canonical 28-byte little-endian wire representation."""

        return _PROVENANCE_STRUCT.pack(self.session_id, self.mode_epoch, self.stream_mode)

    def to_array(self) -> np.ndarray:
        """Return an owned native uint8 array suitable for packed ZMQ fields."""

        return np.frombuffer(self.to_bytes(), dtype=np.uint8).copy()

    @classmethod
    def from_bytes(cls, payload: bytes) -> StreamProvenance:
        """Decode one exact canonical provenance payload."""

        if type(payload) is not bytes or len(payload) != PROVENANCE_SIZE:
            raise ValueError("pv must contain exactly 28 bytes")
        return cls(*_PROVENANCE_STRUCT.unpack(payload))

    @classmethod
    def from_array(cls, value: np.ndarray) -> StreamProvenance:
        """Decode a native uint8 provenance field without coercion."""

        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != (PROVENANCE_SIZE,):
            raise ValueError("pv must have dtype uint8 and shape (28,)")
        return cls.from_bytes(value.tobytes())


def _provenance_corruption_reason(
    candidate: StreamProvenance,
    current: StreamProvenance | None,
    retired_sessions: set[bytes],
    source: str,
) -> str | None:
    if candidate.session_id in retired_sessions:
        if source == "manager":
            return "retired publisher session replay"
        return "retired pair provenance session replay"
    if current is None or candidate.session_id != current.session_id:
        return None
    if candidate.mode_epoch < current.mode_epoch:
        if source == "manager":
            return "manager mode epoch regression"
        return "pair provenance epoch regression"
    if candidate.mode_epoch == current.mode_epoch and candidate.stream_mode != current.stream_mode:
        if source == "manager":
            return "manager same-epoch mode change"
        return "pair provenance same-epoch mode change"
    return None


def _is_exact_bounded_sequence(annotation: object, scalar_name: str) -> bool:
    """Recognize the generated CycloneDDS ``sequence<T, 6>`` contract."""

    if get_origin(annotation) is not Annotated:
        return False
    annotated_args = get_args(annotation)
    if len(annotated_args) != 2:
        return False
    sequence_type, metadata = annotated_args
    if get_origin(sequence_type) is not Sequence:
        return False
    sequence_args = get_args(sequence_type)
    if len(sequence_args) != 1:
        return False
    scalar = sequence_args[0]
    if get_origin(scalar) is not Annotated or get_args(scalar) != (int, scalar_name):
        return False
    return (
        type(metadata).__name__ == "sequence"
        and getattr(metadata, "subtype", None) == scalar
        and getattr(metadata, "max_length", None) == 6
    )


def _strict_six_ints(values: object, label: str, upper: int) -> SixInts:
    try:
        copied = tuple(values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an exact six-element sequence") from exc
    if len(copied) != 6:
        raise ValueError(f"{label} must contain exactly six values")
    if any(type(value) is not int for value in copied):
        raise ValueError(f"{label} values must be built-in int")
    if any(value < 0 or value > upper for value in copied):
        raise ValueError(f"{label} values must lie within [0,{upper}]")
    return copied  # type: ignore[return-value]


def validate_state_fields(message: object, side: str, received_ns: int) -> ValidatedHandState:
    """Validate an FTP feedback sample without narrowing or coercing values."""

    if side not in _SIDES:
        raise ValueError(f"invalid hand side: {side!r}")
    if type(received_ns) is not int or received_ns < 0:
        raise ValueError("received_ns must be a nonnegative built-in int")
    annotations = getattr(type(message), "__annotations__", {})
    if not _is_exact_bounded_sequence(annotations.get("angle_act"), "int16"):
        raise ValueError("generated message angle_act contract must be sequence<int16,6>")
    if not _is_exact_bounded_sequence(annotations.get("err"), "uint8"):
        raise ValueError("generated message err contract must be sequence<uint8,6>")
    try:
        angle_values = message.angle_act
        error_values = message.err
    except AttributeError as exc:
        raise ValueError("generated state message is missing required fields") from exc
    return ValidatedHandState(
        side=side,
        angle_act=_strict_six_ints(angle_values, "angle_act", 1000),
        err=_strict_six_ints(error_values, "err", 255),
        received_ns=received_ns,
    )


def _canonical_snapshot(snapshot: ValidatedHandState, expected_side: str) -> ValidatedHandState:
    if (
        expected_side not in _SIDES
        or not isinstance(snapshot, ValidatedHandState)
        or snapshot.side != expected_side
    ):
        raise ValueError(f"expected validated {expected_side} hand state")
    if type(snapshot.received_ns) is not int or snapshot.received_ns < 0:
        raise ValueError("received_ns must be a nonnegative built-in int")
    return ValidatedHandState(
        side=expected_side,
        angle_act=_strict_six_ints(snapshot.angle_act, "angle_act", 1000),
        err=_strict_six_ints(snapshot.err, "err", 255),
        received_ns=snapshot.received_ns,
    )


def _validate_pair_side(values: object, label: str) -> SixFloats:
    """Require the immutable, copied normalized q6 tuple produced by bridge decoding."""

    if type(values) is tuple and len(values) == 6 and all(type(value) is float for value in values):
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"{label} values must be finite")
        return values  # type: ignore[return-value]
    raise ValueError(f"{label} must be a tuple of six built-in floats")


def _to_enveloped_counts(values: SixFloats) -> tuple[SixInts, int]:
    limited = sum(value < 0.8 for value in values)
    counts = tuple(max(MIN_COMMAND, min(MAX_COMMAND, round(value * 1000))) for value in values)
    return counts, limited  # type: ignore[return-value]


def _bounded_counts(target: SixInts, baseline: SixInts) -> SixInts:
    return tuple(
        max(MIN_COMMAND, min(MAX_COMMAND, current + max(-MAX_STEP, min(MAX_STEP, goal - current))))
        for goal, current in zip(target, baseline, strict=True)
    )  # type: ignore[return-value]


class InspireFtpSafetyController:
    """Pure fail-open state machine for two real Inspire FTP hands."""

    def __init__(self) -> None:
        self._state = BridgeState.MONITORING
        self._armed = False
        self._manager_provenance: StreamProvenance | None = None
        self._manager_received_ns: int | None = None
        self._pending_manager_events: list[_ManagerEvent] = []
        self._pending_manager_transition_count = 0
        self._retired_sessions: set[bytes] = set()
        self._provenance_generation = 0
        self._manager_changed = False
        self._latest_pair: HandPair | None = None
        self._latest_left_target: SixInts = _OPEN_COUNTS
        self._latest_right_target: SixInts = _OPEN_COUNTS
        self._latest_left_limited = False
        self._latest_right_limited = False
        self._left_state: ValidatedHandState | None = None
        self._right_state: ValidatedHandState | None = None
        self._last_successful_left: SixInts | None = None
        self._last_successful_right: SixInts | None = None
        self._open_confirm_ticks = 0
        self._ready_after_ns: int | None = None
        self._fault_reason: str | None = None
        self._pending_fault: str | None = None
        self._signal_count = 0
        self._last_control_cycle_ns: int | None = None
        self._maximum_tick_interval_ns = 0
        self._loop_overrun_count = 0
        self._left_write_failures = 0
        self._right_write_failures = 0
        self._rejected_pairs = 0
        self._envelope_limited_pairs = 0
        self._envelope_limited_values = 0
        self._shutdown_open_attempts = 0
        self._shutdown_hold_ticks = 0
        self._sequence_pv = None
        self._last_seen_sequence = -1
        self._last_seen_fingerprint = None
        self.sequence_gaps = 0
        self.last_applied_message_seq = -1

    @property
    def state(self) -> BridgeState:
        return self._state

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def manager_provenance(self) -> StreamProvenance | None:
        return self._manager_provenance

    @property
    def manager_received_ns(self) -> int | None:
        return self._manager_received_ns

    @property
    def provenance_generation(self) -> int:
        return self._provenance_generation

    @property
    def pending_manager_event_count(self) -> int:
        return len(self._pending_manager_events)

    @property
    def pending_manager_transition_count(self) -> int:
        return self._pending_manager_transition_count

    @property
    def retired_session_count(self) -> int:
        return len(self._retired_sessions)

    @property
    def latest_pair(self) -> HandPair | None:
        return self._latest_pair

    @property
    def latest_targets(self) -> tuple[SixInts, SixInts]:
        return self._latest_left_target, self._latest_right_target

    @property
    def latest_envelope_limited(self) -> tuple[bool, bool]:
        return self._latest_left_limited, self._latest_right_limited

    @property
    def last_successful_left(self) -> SixInts | None:
        return self._last_successful_left

    @property
    def last_successful_right(self) -> SixInts | None:
        return self._last_successful_right

    @property
    def hand_states(self) -> tuple[ValidatedHandState | None, ValidatedHandState | None]:
        """Return the latest immutable left/right feedback snapshots."""

        return self._left_state, self._right_state

    @property
    def open_confirm_ticks(self) -> int:
        return self._open_confirm_ticks

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason or self._pending_fault

    @property
    def rejected_pairs(self) -> int:
        return self._rejected_pairs

    @property
    def envelope_limited_pairs(self) -> int:
        return self._envelope_limited_pairs

    @property
    def envelope_limited_values(self) -> int:
        return self._envelope_limited_values

    @property
    def shutdown_open_attempts(self) -> int:
        return self._shutdown_open_attempts

    @property
    def shutdown_hold_ticks(self) -> int:
        return self._shutdown_hold_ticks

    @property
    def maximum_tick_interval_ns(self) -> int:
        return self._maximum_tick_interval_ns

    @property
    def loop_overrun_count(self) -> int:
        return self._loop_overrun_count

    @property
    def write_failure_counts(self) -> tuple[int, int]:
        return self._left_write_failures, self._right_write_failures

    def manager_age_ns(self, now_ns: int) -> int | None:
        """Return the latest manager age, or ``None`` before first receipt."""

        return self._age_ns(now_ns, self._manager_received_ns)

    def source_age_ns(self, now_ns: int) -> int | None:
        """Return the latest accepted pair age, or ``None`` when absent."""

        received_ns = None if self._latest_pair is None else self._latest_pair.received_ns
        return self._age_ns(now_ns, received_ns)

    def hand_state_ages_ns(self, now_ns: int) -> tuple[int | None, int | None]:
        """Return left/right feedback ages without mutating freshness state."""

        left_ns = None if self._left_state is None else self._left_state.received_ns
        right_ns = None if self._right_state is None else self._right_state.received_ns
        return self._age_ns(now_ns, left_ns), self._age_ns(now_ns, right_ns)

    def accept_manager(self, provenance: StreamProvenance, received_ns: int) -> bool:
        """Validate and stage a manager tuple for the next eligible control tick."""

        if not isinstance(provenance, StreamProvenance):
            self._set_pending_fault("malformed manager provenance")
            return False
        if type(received_ns) is not int or received_ns < 0:
            self._set_pending_fault("malformed manager receive timestamp")
            return False
        event = _ManagerEvent(provenance, received_ns)
        if self._pending_manager_events:
            last = self._pending_manager_events[-1]
            if provenance == last.provenance:
                if received_ns < last.received_ns:
                    return False
                self._pending_manager_events[-1] = event
                return True
        elif provenance == self._manager_provenance:
            if self._manager_received_ns is not None and received_ns < self._manager_received_ns:
                return False
            self._pending_manager_events.append(event)
            return True
        current, current_received_ns, retired_sessions, _ = self._projected_manager()
        corruption = _provenance_corruption_reason(provenance, current, retired_sessions, "manager")
        if corruption is not None:
            self._set_pending_fault(corruption)
            return False
        if current_received_ns is not None and received_ns < current_received_ns:
            return False
        is_transition = provenance != current
        retires_current_session = (
            current is not None
            and provenance.session_id != current.session_id
            and current.session_id not in retired_sessions
        )
        if retires_current_session and len(retired_sessions) >= MAX_RETIRED_SESSIONS:
            self._set_pending_fault("retired session history overflow")
            return False
        if is_transition and self._pending_manager_transition_count >= MAX_PENDING_MANAGER_TRANSITIONS:
            self._set_pending_fault("pending manager transition overflow")
            return False
        if (
            is_transition
            and self._pending_manager_transition_count == 0
            and len(self._pending_manager_events) == 1
            and self._pending_manager_events[0].provenance == self._manager_provenance
        ):
            self._pending_manager_events.clear()
        self._pending_manager_events.append(event)
        self._pending_manager_transition_count += int(is_transition)
        return True

    def accept_pair(
        self,
        left: object,
        right: object,
        provenance: StreamProvenance,
        topic: str,
        received_ns: int,
        message_seq: int,
        fingerprint: bytes,
    ) -> bool:
        """Atomically accept one exact, authorized, normalized hand pair."""

        try:
            left_values = _validate_pair_side(left, "left hand pair")
            right_values = _validate_pair_side(right, "right hand pair")
        except ValueError:
            self._reject_pair()
            return False
        if not isinstance(provenance, StreamProvenance):
            self._set_pending_fault("malformed pair provenance")
            self._reject_pair()
            return False
        manager, manager_received_ns, retired_sessions, manager_change_pending = self._projected_manager()
        corruption = _provenance_corruption_reason(provenance, manager, retired_sessions, "pair")
        if corruption is not None:
            self._set_pending_fault(corruption)
            self._reject_pair()
            return False
        if manager is None or provenance != manager:
            self._reject_pair()
            return False
        if type(message_seq) is not int or message_seq < 0 or type(fingerprint) is not bytes:
            self._set_pending_fault("invalid optical message sequence")
            return False
        if self._sequence_pv != provenance:
            self._sequence_pv = provenance
            self._last_seen_sequence = -1
            self._last_seen_fingerprint = None
        if message_seq <= self._last_seen_sequence:
            if message_seq < self._last_seen_sequence or fingerprint != self._last_seen_fingerprint:
                self._set_pending_fault("optical message sequence regression or conflict")
            self._reject_pair()
            return False
        if self._last_seen_sequence >= 0:
            self.sequence_gaps += message_seq - self._last_seen_sequence - 1
        self._last_seen_sequence = message_seq
        self._last_seen_fingerprint = fingerprint
        authorized = (
            self._state in (BridgeState.READY, BridgeState.ACTIVE)
            and self._fault_reason is None
            and self._pending_fault is None
            and self._signal_count == 0
            and not self._manager_changed
            and not manager_change_pending
            and manager is not None
            and provenance == manager
            and type(received_ns) is int
            and (self._latest_pair is None or received_ns >= self._latest_pair.received_ns)
            and manager_received_ns is not None
            and received_ns > manager_received_ns
            and received_ns - manager_received_ns < SOURCE_STALE_NS
            and self._topic_authorized(manager.stream_mode, topic)
        )
        if self._state is BridgeState.READY:
            authorized = (
                authorized
                and self._open_confirm_ticks >= OPEN_CONFIRM_TICKS
                and self._ready_after_ns is not None
                and received_ns > self._ready_after_ns
            )
        if not authorized:
            self._reject_pair()
            return False
        # Everything above is side-effect-free.  Convert and stage both hands before
        # publishing any accepted-pair state so malformed pairs cannot refresh freshness.
        left_counts, left_limited = _to_enveloped_counts(left_values)
        right_counts, right_limited = _to_enveloped_counts(right_values)
        self._latest_pair = HandPair(left_values, right_values, provenance, topic, received_ns, message_seq)
        self._latest_left_target = left_counts
        self._latest_right_target = right_counts
        self._latest_left_limited = left_limited > 0
        self._latest_right_limited = right_limited > 0
        limited_values = left_limited + right_limited
        self._envelope_limited_values += limited_values
        self._envelope_limited_pairs += int(limited_values > 0)
        return True

    def _topic_authorized(self, stream_mode: int, topic: str) -> bool:
        return _AUTHORIZED_TOPICS.get(stream_mode) == topic

    def _reject_pair(self) -> None:
        self._rejected_pairs = min(MAX_REJECTED_PAIRS, self._rejected_pairs + 1)

    def accept_hand_state(self, snapshot: ValidatedHandState) -> bool:
        """Accept one already-validated feedback snapshot for its named hand."""

        try:
            canonical = _canonical_snapshot(snapshot, snapshot.side)
        except (AttributeError, ValueError):
            self._set_pending_fault("malformed hand feedback")
            return False
        unhealthy = any(canonical.err)
        if unhealthy:
            self._set_pending_fault(f"{canonical.side} hand reported nonzero error")
        current = self._left_state if canonical.side == "left" else self._right_state
        if current is not None and canonical.received_ns < current.received_ns:
            return False
        if canonical.side == "left":
            self._left_state = canonical
        else:
            self._right_state = canonical
        return not unhealthy

    def arm_publish_from_feedback(
        self,
        left_state: ValidatedHandState,
        right_state: ValidatedHandState,
        now_ns: int,
    ) -> None:
        """Enter READY from preflight using fresh, open measured baselines."""

        if self._state is not BridgeState.MONITORING:
            raise RuntimeError("publish controller can only be armed once from monitoring")
        if self._fault_reason is not None or self._pending_fault is not None:
            raise RuntimeError("a faulted controller cannot be armed")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative built-in int")
        canonical_left = _canonical_snapshot(left_state, "left")
        canonical_right = _canonical_snapshot(right_state, "right")
        for snapshot in (canonical_left, canonical_right):
            age = now_ns - snapshot.received_ns
            if age < 0 or age >= STATE_STALE_NS:
                raise ValueError(f"{snapshot.side} hand feedback is stale")
            if any(snapshot.err):
                raise ValueError(f"{snapshot.side} hand feedback is unhealthy")
            if any(value < OPEN_FEEDBACK_MIN for value in snapshot.angle_act):
                raise ValueError(f"{snapshot.side} hand is not open")
        self._left_state = canonical_left
        self._right_state = canonical_right
        self._last_successful_left = canonical_left.angle_act
        self._last_successful_right = canonical_right.angle_act
        self._armed = True
        self._state = BridgeState.READY
        self._open_confirm_ticks = 0
        self._ready_after_ns = None
        self._last_control_cycle_ns = None

    def require_publish_feedback_healthy(self, now_ns: int) -> None:
        """Revalidate current feedback before creating or retaining writers."""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative built-in int")
        for side, snapshot in (("left", self._left_state), ("right", self._right_state)):
            if snapshot is None:
                raise ValueError(f"{side} hand feedback is missing")
            age = now_ns - snapshot.received_ns
            if age < 0 or age >= STATE_STALE_NS:
                raise ValueError(f"{side} hand feedback is stale")
            if any(snapshot.err):
                raise ValueError(f"{side} hand feedback is unhealthy")
            if any(value < OPEN_FEEDBACK_MIN for value in snapshot.angle_act):
                raise ValueError(f"{side} hand is not open")

    def latch_fault(self, reason: str) -> None:
        """Queue an irreversible safety fault for the next control tick."""

        if type(reason) is not str or not reason:
            raise ValueError("fault reason must be a nonempty string")
        self._set_pending_fault(reason)

    def request_shutdown(self, signal_count: int | None = None) -> None:
        """Record one signal, or monotonically raise the observed signal count."""

        if signal_count is None:
            self._signal_count += 1
        elif type(signal_count) is not int or signal_count < 1:
            raise ValueError("signal_count must be a positive built-in int")
        else:
            self._signal_count = max(self._signal_count, signal_count)

    def tick(self, now_ns: int) -> ControlDecision:
        """Advance the deterministic FSM and return at most one bounded attempt."""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative built-in int")
        timing_fault, cycle_eligible = self._observe_cycle_timing(now_ns)
        if self._signal_count >= 2:
            self._discard_manager_events()
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False, exit_now=True)
        if self._signal_count == 1 and self._state is not BridgeState.SHUTDOWN_OPENING:
            self._state = BridgeState.SHUTDOWN_OPENING
            self._clear_pair_targets()
        if self._state is BridgeState.SHUTDOWN_OPENING:
            self._discard_manager_events()
            if not cycle_eligible:
                return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False)
            self._last_control_cycle_ns = now_ns
            return self._shutdown_decision()

        if self._fault_reason is not None:
            self._state = BridgeState.FAULT_LATCHED
            self._clear_pair_targets()
            self._discard_manager_events()
        else:
            newly_detected = self._new_fault(now_ns, timing_fault)
            if newly_detected is not None:
                self._fault_reason = newly_detected
                self._pending_fault = None
                self._state = BridgeState.FAULT_LATCHED
                self._clear_pair_targets()
                self._discard_manager_events()
            else:
                if not cycle_eligible:
                    return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False)
                self._apply_manager_events()
                self._apply_normal_transitions(now_ns)

        if not cycle_eligible:
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False)
        self._last_control_cycle_ns = now_ns

        if self._state is BridgeState.MONITORING:
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False)
        if self._state is BridgeState.FAULT_LATCHED and not self._armed:
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False, exit_now=True)
        if self._state is BridgeState.ACTIVE:
            left_target = self._latest_left_target
            right_target = self._latest_right_target
        else:
            left_target = right_target = _OPEN_COUNTS
        return self._decision(left_target, right_target, publish=True)

    def record_write_result(self, side: str, attempted: object, succeeded: bool) -> None:
        """Commit one successful hand write; a failure leaves that baseline intact."""

        if side not in _SIDES:
            raise ValueError(f"invalid hand side: {side!r}")
        values = _strict_six_ints(attempted, "attempted command", MAX_COMMAND)
        if any(value < MIN_COMMAND for value in values):
            raise ValueError("attempted command lies below the safety envelope")
        if type(succeeded) is not bool:
            raise ValueError("succeeded must be bool")
        baseline = self._last_successful_left if side == "left" else self._last_successful_right
        if baseline is None:
            raise RuntimeError("controller has not been armed")
        if any(abs(value - previous) > MAX_STEP for value, previous in zip(values, baseline, strict=True)):
            raise ValueError("attempted command exceeds maximum per-tick step")
        if succeeded:
            if side == "left":
                self._last_successful_left = values
            else:
                self._last_successful_right = values
        else:
            if side == "left":
                self._left_write_failures += 1
            else:
                self._right_write_failures += 1
            self._set_pending_fault(f"{side} hand command write failed")

    def _set_pending_fault(self, reason: str) -> None:
        if self._fault_reason is None and self._pending_fault is None:
            self._pending_fault = reason

    def _projected_manager(
        self,
    ) -> tuple[StreamProvenance | None, int | None, set[bytes], bool]:
        current = self._manager_provenance
        received_ns = self._manager_received_ns
        retired_sessions = set(self._retired_sessions)
        provenance_change_pending = False
        for event in self._pending_manager_events:
            if current is not None and event.provenance.session_id != current.session_id:
                retired_sessions.add(current.session_id)
            if event.provenance != current:
                provenance_change_pending = True
            current = event.provenance
            received_ns = event.received_ns
        return current, received_ns, retired_sessions, provenance_change_pending

    def _apply_manager_events(self) -> None:
        manager_changed = False
        for event in self._pending_manager_events:
            current = self._manager_provenance
            changed = event.provenance != current
            if current is not None and event.provenance.session_id != current.session_id:
                self._retired_sessions.add(current.session_id)
            self._manager_provenance = event.provenance
            self._manager_received_ns = event.received_ns
            if changed:
                self.last_applied_message_seq = -1
                self._invalidate_authorization(increment_generation=True)
                manager_changed = manager_changed or current is not None
        self._manager_changed = self._manager_changed or manager_changed
        self._pending_manager_events.clear()
        self._pending_manager_transition_count = 0

    def _discard_manager_events(self) -> None:
        self._pending_manager_events.clear()
        self._pending_manager_transition_count = 0

    def _invalidate_authorization(self, *, increment_generation: bool) -> None:
        self._clear_pair_targets()
        self._open_confirm_ticks = 0
        self._ready_after_ns = None
        if increment_generation:
            self._provenance_generation += 1

    def _observe_cycle_timing(self, now_ns: int) -> tuple[str | None, bool]:
        if self._last_control_cycle_ns is None:
            return None, True
        interval_ns = now_ns - self._last_control_cycle_ns
        if interval_ns < 0:
            return "control tick timestamp regression", False
        self._maximum_tick_interval_ns = max(self._maximum_tick_interval_ns, interval_ns)
        if interval_ns > MAX_TICK_INTERVAL_NS:
            self._loop_overrun_count += 1
            return "control tick interval overrun", True
        return None, interval_ns >= CONTROL_PERIOD_NS

    def _new_fault(self, now_ns: int, timing_fault: str | None) -> str | None:
        if self._pending_fault is not None:
            return self._pending_fault
        if timing_fault is not None:
            return timing_fault
        if any(event.received_ns > now_ns for event in self._pending_manager_events):
            return "manager timestamp is in the future"
        if self._latest_pair is not None and self._latest_pair.received_ns > now_ns:
            return "source timestamp is in the future"
        if self._state in (BridgeState.READY, BridgeState.ACTIVE, BridgeState.OPENING):
            for side, snapshot in (("left", self._left_state), ("right", self._right_state)):
                if snapshot is not None and now_ns < snapshot.received_ns:
                    return f"{side} hand state timestamp is in the future"
                if snapshot is None or now_ns - snapshot.received_ns >= STATE_STALE_NS:
                    return f"{side} hand state timeout"
                if any(snapshot.err):
                    return f"{side} hand reported nonzero error"
        return None

    def _manager_is_fresh(self, now_ns: int) -> bool:
        return (
            self._manager_provenance is not None
            and self._manager_received_ns is not None
            and 0 <= now_ns - self._manager_received_ns < SOURCE_STALE_NS
        )

    def _manager_is_authorized(self) -> bool:
        if self._manager_provenance is None:
            return False
        return self._manager_provenance.stream_mode in _AUTHORIZED_TOPICS

    def _pair_is_fresh(self, now_ns: int) -> bool:
        return self._latest_pair is not None and 0 <= now_ns - self._latest_pair.received_ns < SOURCE_STALE_NS

    def _feedback_is_open(self, now_ns: int) -> bool:
        return all(
            snapshot is not None
            and 0 <= now_ns - snapshot.received_ns < STATE_STALE_NS
            and not any(snapshot.err)
            and all(value >= OPEN_FEEDBACK_MIN for value in snapshot.angle_act)
            for snapshot in (self._left_state, self._right_state)
        )

    def _update_open_confirmation(self, now_ns: int) -> None:
        if self._feedback_is_open(now_ns):
            self._open_confirm_ticks = min(OPEN_CONFIRM_TICKS, self._open_confirm_ticks + 1)
        else:
            self._open_confirm_ticks = 0
            self._ready_after_ns = None
        if self._open_confirm_ticks == OPEN_CONFIRM_TICKS and self._ready_after_ns is None:
            self._ready_after_ns = now_ns

    def _apply_normal_transitions(self, now_ns: int) -> None:
        if self._state is BridgeState.READY:
            self._update_open_confirmation(now_ns)
            if self._manager_changed:
                self._manager_changed = False
            if self._open_confirm_ticks < OPEN_CONFIRM_TICKS:
                self._clear_pair_targets()
                return
            if not self._manager_is_fresh(now_ns) or not self._manager_is_authorized():
                self._clear_pair_targets()
                return
            if self._latest_pair is not None and self._pair_is_fresh(now_ns):
                self._state = BridgeState.ACTIVE
        elif self._state is BridgeState.ACTIVE:
            if self._manager_changed:
                self._manager_changed = False
                self._clear_pair_targets()
                self._state = BridgeState.OPENING
            elif (
                not self._manager_is_fresh(now_ns)
                or not self._manager_is_authorized()
                or not self._pair_is_fresh(now_ns)
            ):
                self._invalidate_authorization(increment_generation=True)
                self._state = BridgeState.OPENING
        elif self._state is BridgeState.OPENING:
            self._manager_changed = False
            self._update_open_confirmation(now_ns)
            if self._open_confirm_ticks >= OPEN_CONFIRM_TICKS:
                self._state = BridgeState.READY
                self._latest_pair = None
                self._ready_after_ns = now_ns

    @staticmethod
    def _age_ns(now_ns: int, received_ns: int | None) -> int | None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative built-in int")
        return None if received_ns is None else now_ns - received_ns

    def _clear_pair_targets(self) -> None:
        self._latest_pair = None
        self._latest_left_target = _OPEN_COUNTS
        self._latest_right_target = _OPEN_COUNTS
        self._latest_left_limited = False
        self._latest_right_limited = False

    def _decision(
        self,
        left_target: SixInts,
        right_target: SixInts,
        *,
        publish: bool,
        exit_now: bool = False,
    ) -> ControlDecision:
        left = self._last_successful_left or _OPEN_COUNTS
        right = self._last_successful_right or _OPEN_COUNTS
        return ControlDecision(
            state=self._state,
            left_counts=_bounded_counts(left_target, left),
            right_counts=_bounded_counts(right_target, right),
            publish=publish and self._armed,
            fault_reason=self._fault_reason or self._pending_fault,
            exit_now=exit_now,
        )

    def _shutdown_decision(self) -> ControlDecision:
        if not self._armed:
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False, exit_now=True)
        left = self._last_successful_left or _OPEN_COUNTS
        right = self._last_successful_right or _OPEN_COUNTS
        both_open = left == _OPEN_COUNTS and right == _OPEN_COUNTS
        if both_open:
            if self._shutdown_hold_ticks >= 20:
                return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False, exit_now=True)
            self._shutdown_hold_ticks += 1
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=True)
        if self._shutdown_open_attempts >= 40:
            return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=False, exit_now=True)
        self._shutdown_open_attempts += 1
        return self._decision(_OPEN_COUNTS, _OPEN_COUNTS, publish=True)
