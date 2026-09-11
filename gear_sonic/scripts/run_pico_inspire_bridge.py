# DDS ownership/discovery and fail-open runtime from reviewed bridge d6368d8.
# Optical input uses only manager_state v4 followed by inspire_hand v1 q6.
"""Monitor-first PC2 Inspire FTP bridge with fail-closed DDS ownership proof.

Vendor DDS support is loaded lazily, and runtime effects are injected for
deterministic tests. Callbacks only enqueue immutable feedback; the owning main
thread performs discovery, safety decisions, command writes, and cleanup.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
import uuid

# Direct script launches must use this checkout, even with another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gear_sonic.utils.teleop.inspire_ftp_real_bridge import (
    CONTROL_PERIOD_NS,
    MAX_TICK_INTERVAL_NS,
    MIN_COMMAND,
    OPEN_FEEDBACK_MIN,
    STATE_STALE_NS,
    BridgeState,
    CallbackFaultEvent,
    HandPair,
    InspireFtpSafetyController,
    StateSnapshotEvent,
    StreamProvenance,
    validate_state_fields,
)
from gear_sonic.utils.teleop.pico_inspire_protocol import (
    HAND_SCHEMA,
    STATUS_SCHEMA,
    validate_inspire_hand,
    validate_inspire_status,
)
from gear_sonic.utils.teleop.pico_recording import parse_pv
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message

MAX_ZMQ_PACKETS_PER_TICK = 64
MAX_ZMQ_DRAIN_NS = 5_000_000
# Fixed defense-in-depth cap, comfortably above current production pose frames.
MAX_ZMQ_MESSAGE_BYTES = 1_048_576
DEFAULT_INBOX_CAPACITY = 256
ZMQ_RCVHWM = 64
DISCOVERY_WINDOW_NS = 2_000_000_000
DISCOVERY_POLL_SECONDS = 0.1
DEFAULT_DISCOVERY_TIMEOUT_NS = 5_000_000_000
MAX_DISCOVERY_EVIDENCE = 256
MAX_DISCOVERY_IDENTITIES = 64
MAX_DISCOVERY_STRING = 256

STATE_TOPICS: Mapping[str, str] = {
    "left": "rt/inspire_hand/state/l",
    "right": "rt/inspire_hand/state/r",
}
COMMAND_TOPICS: Mapping[str, str] = {
    "left": "rt/inspire_hand/ctrl/l",
    "right": "rt/inspire_hand/ctrl/r",
}

SOURCE_PROTOCOLS: Mapping[str, Mapping[str, int]] = {
    "pico": {"manager_state": 4, "inspire_hand": 1},
}


@dataclass(frozen=True, slots=True)
class ManagerPacket:
    """One strictly decoded manager-state packet."""

    provenance: StreamProvenance
    received_ns: int
    stream_mode: int
    toggle_data_collection: bool | None = None
    toggle_data_abort: bool | None = None


@dataclass(frozen=True, slots=True)
class DrainStats:
    """Bounded accounting for one ZMQ drain attempt; retains no wire data."""

    received: int
    dispatched: int
    rejected: int
    oversized: int
    packet_budget_exhausted: bool
    time_budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class _DecodedManager:
    provenance: StreamProvenance
    stream_mode: int
    toggle_data_collection: bool | None
    toggle_data_abort: bool | None


@dataclass(frozen=True, slots=True)
class _DecodedPair:
    left: tuple[float, ...]
    right: tuple[float, ...]
    provenance: StreamProvenance
    topic: str
    message_seq: int
    fingerprint: bytes


BridgePacket = ManagerPacket | HandPair
DecodedBridgePayload = _DecodedManager | _DecodedPair
InboxEvent = StateSnapshotEvent | CallbackFaultEvent


class DiscoveryFailure(RuntimeError):
    """Fail-closed DDS discovery or ownership error."""


class _PreControlSignal(RuntimeError):
    """Signal observed before the main control loop owns signal counting."""

    def __init__(self, reason: str, count: int) -> None:
        super().__init__(reason)
        self.count = count


@dataclass(frozen=True, slots=True)
class PublicationLifecycleSample:
    """Dependency-free copy of one DDS publication lifecycle observation."""

    key: str
    participant_key: str
    topic_name: str
    alive: bool
    observed_ns: int

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key or len(self.key) > MAX_DISCOVERY_STRING:
            raise ValueError("publication key must be a nonempty string")
        if (
            type(self.participant_key) is not str
            or not self.participant_key
            or len(self.participant_key) > MAX_DISCOVERY_STRING
        ):
            raise ValueError("participant key must be a nonempty string")
        if type(self.topic_name) is not str or not self.topic_name or len(self.topic_name) > MAX_DISCOVERY_STRING:
            raise ValueError("topic name must be a nonempty string")
        if type(self.alive) is not bool:
            raise ValueError("publication alive must be bool")
        _validate_received_ns(self.observed_ns)


@dataclass(frozen=True, slots=True)
class PublicationIdentityEvidence:
    """Durable bounded lifecycle summary for one relevant DDS identity."""

    key: str
    participant_key: str
    topic_name: str
    side: str
    first_observed_ns: int
    last_observed_ns: int
    ever_alive: bool
    ever_disposed: bool
    classification: str


@dataclass(frozen=True, slots=True)
class PublicationOwnershipEvidence:
    """Bounded immutable snapshot of DDS discovery and ownership evidence."""

    state_alive: tuple[tuple[str, tuple[str, ...]], ...]
    command_alive: tuple[tuple[str, tuple[str, ...]], ...]
    owned: tuple[tuple[str, str], ...]
    owned_disruption: tuple[tuple[str, str], ...]
    foreign_history: tuple[tuple[str, str], ...]
    foreign_overflow: bool
    identity_overflow: bool
    identity_mismatch: bool
    state_identity_fault: bool
    state_discovery_healthy: bool
    ownership_healthy: bool
    identities: tuple[PublicationIdentityEvidence, ...]
    lifecycle: tuple[PublicationLifecycleSample, ...]


@dataclass(frozen=True, slots=True)
class FinalZeroWriterClosure:
    """Ownership snapshot and the exact continuous empty-writer window."""

    ownership: PublicationOwnershipEvidence
    zero_since_ns: int
    observation_end_ns: int


class PublicationOwnershipTracker:
    """Pure exact-sole-writer tracker with sticky foreign GUID history."""

    def __init__(self) -> None:
        self._alive_by_topic: dict[str, set[str]] = {}
        self._owned: dict[str, str] = {}
        self._owned_disrupted: set[tuple[str, str]] = set()
        self._foreign: set[tuple[str, str]] = set()
        self._unsealed_command_history: set[tuple[str, str]] = set()
        self._foreign_overflow = False
        self._identity_overflow = False
        self._identity_mismatch = False
        self._state_identity_fault = False
        self._state_identity: dict[str, tuple[str, str]] = {}
        self._identities: dict[tuple[str, str], PublicationIdentityEvidence] = {}
        self._lifecycle: deque[PublicationLifecycleSample] = deque(maxlen=MAX_DISCOVERY_EVIDENCE)
        self._command_event_count = 0
        self._command_alive_event_count = 0
        self._state_disruption_count = 0
        self._sealed = False

    @property
    def command_event_count(self) -> int:
        return self._command_event_count

    @property
    def command_alive_event_count(self) -> int:
        return self._command_alive_event_count

    @property
    def state_disruption_count(self) -> int:
        return self._state_disruption_count

    def observe(self, samples: Iterable[PublicationLifecycleSample]) -> None:
        """Apply lifecycle values in order without importing DDS types."""

        for sample in samples:
            if not isinstance(sample, PublicationLifecycleSample):
                raise TypeError("discovery samples must be PublicationLifecycleSample values")
            self._lifecycle.append(sample)
            state_side = self._side_for_topic(STATE_TOPICS, sample.topic_name)
            side = self._side_for_topic(COMMAND_TOPICS, sample.topic_name)
            if state_side is None and side is None:
                continue
            relevant_side = state_side if state_side is not None else side
            assert relevant_side is not None
            self._summarize_identity(sample, relevant_side, state_side is not None)
            if state_side is not None and not sample.alive:
                self._state_disruption_count += 1
            alive = self._alive_by_topic.setdefault(sample.topic_name, set())
            if sample.alive:
                if sample.key in alive or len(alive) < MAX_DISCOVERY_IDENTITIES:
                    alive.add(sample.key)
                else:
                    self._identity_overflow = True
            else:
                alive.discard(sample.key)
            if side is not None:
                self._command_event_count += 1
                if sample.alive:
                    self._command_alive_event_count += 1
                if self._sealed:
                    if self._owned.get(side) == sample.key:
                        if not sample.alive:
                            self._owned_disrupted.add((side, sample.key))
                    else:
                        self._record_foreign(side, sample.key)
                elif len(self._unsealed_command_history) < MAX_DISCOVERY_IDENTITIES:
                    self._unsealed_command_history.add((side, sample.key))
                elif (side, sample.key) not in self._unsealed_command_history:
                    self._foreign_overflow = True

    def register_owned(self, side: str, key: str) -> None:
        if side not in COMMAND_TOPICS:
            raise ValueError(f"invalid hand side: {side!r}")
        if type(key) is not str or not key:
            raise ValueError("owned publication key must be a nonempty string")
        if side in self._owned:
            raise DiscoveryFailure(f"{side} owned writer already registered")
        self._owned[side] = key

    def seal_ownership(self) -> None:
        if set(self._owned) != set(COMMAND_TOPICS):
            raise DiscoveryFailure("both owned command writers must be registered")
        for side, key in self._unsealed_command_history:
            if self._owned.get(side) != key:
                self._record_foreign(side, key)
        for side, key in self._owned.items():
            identity = self._identities.get((COMMAND_TOPICS[side], key))
            if identity is not None and identity.ever_disposed:
                self._owned_disrupted.add((side, key))
        self._sealed = True

    def command_writers_empty(self) -> bool:
        return all(not self._alive_by_topic.get(topic, set()) for topic in COMMAND_TOPICS.values())

    def evidence(self) -> PublicationOwnershipEvidence:
        state_alive = tuple(
            (side, tuple(sorted(self._alive_by_topic.get(topic, set())))) for side, topic in STATE_TOPICS.items()
        )
        command_alive = tuple(
            (side, tuple(sorted(self._alive_by_topic.get(topic, set())))) for side, topic in COMMAND_TOPICS.items()
        )
        owned = tuple((side, self._owned[side]) for side in COMMAND_TOPICS if side in self._owned)
        state_participants = []
        for side, keys in state_alive:
            if len(keys) == 1:
                identity = self._identities.get((STATE_TOPICS[side], keys[0]))
                if identity is not None:
                    state_participants.append(identity.participant_key)
        state_healthy = (
            all(len(keys) == 1 for _, keys in state_alive)
            and len(state_participants) == len(STATE_TOPICS)
            and len(set(state_participants)) == 1
            and not self._state_identity_fault
            and not self._identity_mismatch
        )
        ownership_healthy = (
            self._sealed
            and state_healthy
            and not self._foreign
            and not self._foreign_overflow
            and not self._identity_overflow
            and not self._owned_disrupted
            and all(tuple(keys) == (self._owned.get(side),) for side, keys in command_alive)
        )
        identities = tuple(self._classified_identity(identity) for _, identity in sorted(self._identities.items()))
        return PublicationOwnershipEvidence(
            state_alive=state_alive,
            command_alive=command_alive,
            owned=owned,
            owned_disruption=tuple(sorted(self._owned_disrupted)),
            foreign_history=tuple(sorted(self._foreign)),
            foreign_overflow=self._foreign_overflow,
            identity_overflow=self._identity_overflow,
            identity_mismatch=self._identity_mismatch,
            state_identity_fault=self._state_identity_fault,
            state_discovery_healthy=state_healthy,
            ownership_healthy=ownership_healthy,
            identities=identities,
            lifecycle=tuple(self._lifecycle),
        )

    def health_fault(self) -> str | None:
        evidence = self.evidence()
        if not evidence.state_discovery_healthy:
            return "expected left/right state writer discovery missing"
        if evidence.identity_overflow:
            return "DDS publication identity history overflow"
        if evidence.foreign_overflow:
            return "foreign command writer history overflow"
        if evidence.foreign_history:
            return "foreign command writer lifecycle observed"
        if evidence.owned_disruption:
            return "owned command writer lifecycle disrupted"
        if not self._sealed:
            return "owned command writers are not registered"
        for side, keys in evidence.command_alive:
            owned = self._owned[side]
            if owned not in keys:
                return f"missing owned {side} command writer"
            if keys != (owned,):
                return f"{side} command topic does not have exact sole ownership"
        return None

    @staticmethod
    def _side_for_topic(topics: Mapping[str, str], topic_name: str) -> str | None:
        return next((side for side, topic in topics.items() if topic == topic_name), None)

    def _record_foreign(self, side: str, key: str) -> None:
        if (side, key) in self._foreign:
            return
        if len(self._foreign) >= MAX_DISCOVERY_IDENTITIES:
            self._foreign_overflow = True
            return
        self._foreign.add((side, key))

    def participant_key(self, topic_name: str, key: str) -> str | None:
        identity = self._identities.get((topic_name, key))
        return None if identity is None else identity.participant_key

    def _summarize_identity(
        self,
        sample: PublicationLifecycleSample,
        side: str,
        is_state: bool,
    ) -> None:
        identity_key = (sample.topic_name, sample.key)
        existing = self._identities.get(identity_key)
        if existing is None:
            if len(self._identities) >= MAX_DISCOVERY_IDENTITIES:
                self._identity_overflow = True
                return
            existing = PublicationIdentityEvidence(
                key=sample.key,
                participant_key=sample.participant_key,
                topic_name=sample.topic_name,
                side=side,
                first_observed_ns=sample.observed_ns,
                last_observed_ns=sample.observed_ns,
                ever_alive=sample.alive,
                ever_disposed=not sample.alive,
                classification="state" if is_state else "unclassified",
            )
        else:
            if existing.participant_key != sample.participant_key:
                self._identity_mismatch = True
            existing = replace(
                existing,
                last_observed_ns=max(existing.last_observed_ns, sample.observed_ns),
                ever_alive=existing.ever_alive or sample.alive,
                ever_disposed=existing.ever_disposed or not sample.alive,
            )
        self._identities[identity_key] = existing
        if is_state:
            expected = self._state_identity.get(side)
            observed = (sample.key, sample.participant_key)
            if expected is None:
                self._state_identity[side] = observed
            elif expected != observed:
                self._state_identity_fault = True

    def _classified_identity(self, identity: PublicationIdentityEvidence) -> PublicationIdentityEvidence:
        if identity.classification == "state":
            return identity
        if self._owned.get(identity.side) == identity.key:
            classification = "owned"
        elif (identity.side, identity.key) in self._foreign:
            classification = "foreign"
        else:
            classification = "unclassified"
        return replace(identity, classification=classification)


def _poll_discovery(
    tracker: PublicationOwnershipTracker,
    poll: Callable[[], Iterable[PublicationLifecycleSample]],
) -> None:
    try:
        tracker.observe(poll())
    except _PreControlSignal:
        raise
    except DiscoveryFailure:
        raise
    except Exception as error:
        raise DiscoveryFailure(f"DDS discovery exception: {error}") from error


def require_zero_writer_preflight(
    tracker: PublicationOwnershipTracker,
    poll: Callable[[], Iterable[PublicationLifecycleSample]],
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    timeout_ns: int = DEFAULT_DISCOVERY_TIMEOUT_NS,
) -> PublicationOwnershipEvidence:
    """Prove state discovery and no command lifecycle for one full window."""

    start_ns = monotonic_ns()
    initial_events = tracker.command_event_count
    if initial_events:
        raise DiscoveryFailure("command writer lifecycle observed before writer creation")
    zero_since_ns: int | None = None
    disruption_count = tracker.state_disruption_count
    while True:
        _poll_discovery(tracker, poll)
        if tracker.command_event_count != initial_events:
            raise DiscoveryFailure("command writer lifecycle observed before writer creation")
        now_ns = monotonic_ns()
        evidence = tracker.evidence()
        if evidence.identity_overflow or evidence.foreign_overflow:
            raise DiscoveryFailure("DDS discovery identity history overflow")
        disrupted = tracker.state_disruption_count != disruption_count
        disruption_count = tracker.state_disruption_count
        if now_ns - start_ns >= timeout_ns:
            raise DiscoveryFailure("DDS discovery preflight timeout")
        if evidence.state_discovery_healthy:
            zero_since_ns = now_ns if zero_since_ns is None or disrupted else zero_since_ns
            if now_ns - zero_since_ns >= DISCOVERY_WINDOW_NS:
                return evidence
        else:
            zero_since_ns = None
        sleep(DISCOVERY_POLL_SECONDS)


def identify_bridge_writers(
    tracker: PublicationOwnershipTracker,
    before: PublicationOwnershipEvidence,
    *,
    expected_participant_key: str,
) -> PublicationOwnershipEvidence:
    """Register exactly one newly alive command writer GUID per hand."""

    previous = dict(before.command_alive)
    current = dict(tracker.evidence().command_alive)
    candidates = {side: tuple(sorted(set(current[side]) - set(previous[side]))) for side in COMMAND_TOPICS}
    if any(len(keys) != 1 for keys in candidates.values()):
        raise DiscoveryFailure("expected exactly one new alive command writer per side")
    if type(expected_participant_key) is not str or not expected_participant_key:
        raise DiscoveryFailure("created publisher participant identity unavailable")
    participants = {
        side: tracker.participant_key(COMMAND_TOPICS[side], keys[0]) for side, keys in candidates.items()
    }
    if (
        any(participant != expected_participant_key for participant in participants.values())
        or len(set(participants.values())) != 1
    ):
        raise DiscoveryFailure("new command writer participant identity mismatch")
    for side, keys in candidates.items():
        tracker.register_owned(side, keys[0])
    tracker.seal_ownership()
    fault = tracker.health_fault()
    if fault is not None:
        raise DiscoveryFailure(fault)
    return tracker.evidence()


def require_bridge_writer_registration(
    tracker: PublicationOwnershipTracker,
    before: PublicationOwnershipEvidence,
    poll: Callable[[], Iterable[PublicationLifecycleSample]],
    *,
    expected_participant_key: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    timeout_ns: int = DEFAULT_DISCOVERY_TIMEOUT_NS,
) -> PublicationOwnershipEvidence:
    """Wait boundedly for exactly the two newly created bridge GUIDs."""

    started_ns = monotonic_ns()
    previous = dict(before.command_alive)
    while True:
        _poll_discovery(tracker, poll)
        current = dict(tracker.evidence().command_alive)
        counts = [len(set(current[side]) - set(previous[side])) for side in COMMAND_TOPICS]
        if any(count > 1 for count in counts):
            raise DiscoveryFailure("expected exactly one new alive command writer per side")
        now_ns = monotonic_ns()
        if now_ns - started_ns >= timeout_ns:
            raise DiscoveryFailure("DDS bridge writer discovery timeout")
        if counts == [1, 1]:
            return identify_bridge_writers(
                tracker,
                before,
                expected_participant_key=expected_participant_key,
            )
        sleep(DISCOVERY_POLL_SECONDS)


def require_final_zero_writers(
    tracker: PublicationOwnershipTracker,
    poll: Callable[[], Iterable[PublicationLifecycleSample]],
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    timeout_ns: int = DEFAULT_DISCOVERY_TIMEOUT_NS,
) -> FinalZeroWriterClosure:
    """Require a continuous two-second zero-command-writer closure window."""

    started_ns = monotonic_ns()
    zero_since_ns: int | None = None
    command_event_count = tracker.command_event_count
    while True:
        _poll_discovery(tracker, poll)
        now_ns = monotonic_ns()
        if now_ns - started_ns >= timeout_ns:
            raise DiscoveryFailure("DDS final writer closure timeout")
        disrupted = tracker.command_event_count != command_event_count
        command_event_count = tracker.command_event_count
        if tracker.command_writers_empty():
            zero_since_ns = now_ns if zero_since_ns is None or disrupted else zero_since_ns
            if now_ns - zero_since_ns >= DISCOVERY_WINDOW_NS:
                return FinalZeroWriterClosure(tracker.evidence(), zero_since_ns, now_ns)
        else:
            zero_since_ns = None
        sleep(DISCOVERY_POLL_SECONDS)


@dataclass(frozen=True, slots=True)
class Pc2DdsBindings:
    """Lazily loaded PC2 DDS factories and built-in discovery reader."""

    participant: Any
    publication_reader: Any
    alive_state: Any
    command_type: Any
    state_type: Any
    message_factory: Callable[[], Any]
    publisher_factory: Any
    subscriber_factory: Any

    def publisher_participant_key(self, publisher: Any) -> str:
        """Return the created Unitree publisher participant identity or fail closed."""

        participant_key = getattr(publisher, "participant_key", None)
        if participant_key is None:
            channel = getattr(publisher, "_ChannelPublisher__channel", None)
            participant = getattr(channel, "_Channel__participant", None)
            participant_key = getattr(participant, "guid", None)
        try:
            return _normalize_dds_identity(participant_key, "publisher participant identity")
        except (TypeError, ValueError) as error:
            raise DiscoveryFailure("created publisher participant identity unavailable") from error

    def drain_publications(
        self,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> tuple[PublicationLifecycleSample, ...]:
        """Copy at most 256 built-in DDS samples into dependency-free values."""

        drained = tuple(self.publication_reader.take(256))
        if len(drained) >= 256:
            raise DiscoveryFailure("DDS publication discovery drain saturated at bounded batch cap")
        copied = []
        for sample in drained:
            observed_ns = monotonic_ns()
            copied.append(
                PublicationLifecycleSample(
                    key=_normalize_dds_identity(sample.key, "publication key"),
                    participant_key=_normalize_dds_identity(sample.participant_key, "participant key"),
                    topic_name=str(sample.topic_name),
                    alive=sample.sample_info.instance_state == self.alive_state,
                    observed_ns=observed_ns,
                )
            )
        return tuple(copied)


def _normalize_dds_identity(value: Any, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} is unavailable")
    normalized = str(value)
    if not normalized or len(normalized) > MAX_DISCOVERY_STRING:
        raise ValueError(f"{label} is invalid")
    return normalized


def load_pc2_dds(network_interface: str) -> Pc2DdsBindings:
    """Load and initialize the installed PC2 DDS contract on explicit request."""

    if type(network_interface) is not str or not network_interface:
        raise ValueError("network interface must be a nonempty string")
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    from cyclonedds.core import InstanceState
    from cyclonedds.domain import DomainParticipant
    from inspire_sdkpy import inspire_dds, inspire_hand_defaut
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )

    ChannelFactoryInitialize(0, network_interface)
    participant = DomainParticipant(0)
    reader = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    return Pc2DdsBindings(
        participant=participant,
        publication_reader=reader,
        alive_state=InstanceState.Alive,
        command_type=inspire_dds.inspire_hand_ctrl,
        state_type=inspire_dds.inspire_hand_state,
        message_factory=inspire_hand_defaut.get_inspire_hand_ctrl,
        publisher_factory=ChannelPublisher,
        subscriber_factory=ChannelSubscriber,
    )


def _validate_received_ns(received_ns: int) -> None:
    if type(received_ns) is not int or received_ns < 0:
        raise ValueError("received_ns must be a nonnegative built-in int")


def _optional_toggle(fields: dict[str, Any], name: str) -> bool | None:
    if name not in fields:
        return None
    value = fields[name]
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("?") or value.shape != (1,):
        raise ValueError(f"manager {name} must have dtype bool and shape [1]")
    return bool(value[0])


def _source_topics(source_profile: str) -> Mapping[str, int]:
    try:
        return SOURCE_PROTOCOLS[source_profile]
    except KeyError as exc:
        raise ValueError("source_profile must be 'pico' (optical q6)") from exc


def _decode_bridge_payload(raw: bytes, topic: str, *, source_profile: str = "pico") -> DecodedBridgePayload:
    """Validate and copy one packet without assigning an event timestamp."""

    expected_version = _source_topics(source_profile).get(topic)
    if expected_version is None:
        raise ValueError(f"unsupported bridge topic: {topic!r}")
    fields = unpack_pose_message(raw, topic=topic)
    version = fields.get("version", 0)
    if type(version) is not int or version != expected_version:
        raise ValueError(f"{topic} packet version must be exactly {expected_version}")
    provenance_field = fields.get("pv")
    if provenance_field is None:
        raise ValueError("bridge packet requires exactly one pv field")
    parse_pv(provenance_field)
    provenance = StreamProvenance.from_array(provenance_field)

    if topic == "manager_state":
        profile = fields.get("hand_profile")
        if (
            not isinstance(profile, np.ndarray)
            or profile.dtype != np.dtype("<i4")
            or profile.shape != (1,)
            or profile[0] != 1
        ):
            raise ValueError("manager must declare Inspire optical hand_profile=1")
        legacy_mode = fields.get("stream_mode")
        if (
            not isinstance(legacy_mode, np.ndarray)
            or legacy_mode.dtype != np.dtype("<i4")
            or legacy_mode.shape != (1,)
        ):
            raise ValueError("manager stream_mode must have dtype int32 and shape [1]")
        stream_mode = int(legacy_mode[0])
        if stream_mode != provenance.stream_mode:
            raise ValueError("manager stream mode must match pv")
        return _DecodedManager(
            provenance=provenance,
            stream_mode=stream_mode,
            toggle_data_collection=_optional_toggle(fields, "toggle_data_collection"),
            toggle_data_abort=_optional_toggle(fields, "toggle_data_abort"),
        )

    validate_inspire_hand(fields)
    fingerprint = sha256(b"".join(fields[name].tobytes() for name, _, _ in HAND_SCHEMA)).digest()
    return _DecodedPair(
        tuple(float(x) for x in fields["left_command"]),
        tuple(float(x) for x in fields["right_command"]),
        provenance,
        topic,
        int(fields["message_seq"][0]),
        fingerprint,
    )


def _stamp_bridge_payload(decoded: DecodedBridgePayload, received_ns: int) -> BridgePacket:
    _validate_received_ns(received_ns)
    if isinstance(decoded, _DecodedManager):
        return ManagerPacket(
            provenance=decoded.provenance,
            received_ns=received_ns,
            stream_mode=decoded.stream_mode,
            toggle_data_collection=decoded.toggle_data_collection,
            toggle_data_abort=decoded.toggle_data_abort,
        )
    return HandPair(
        decoded.left,
        decoded.right,
        decoded.provenance,
        decoded.topic,
        received_ns,
        decoded.message_seq,
    )  # type: ignore[arg-type]


def decode_bridge_packet(
    raw: bytes, topic: str, received_ns: int, *, source_profile: str = "pico"
) -> BridgePacket:
    """Strictly decode and timestamp one already topic-identified packet."""

    decoded = _decode_bridge_payload(raw, topic, source_profile=source_profile)
    return _stamp_bridge_payload(decoded, received_ns)


def create_zmq_subscriber(
    host: str,
    port: int,
    *,
    context: Any | None = None,
    zmq_module: Any | None = None,
    source_profile: str = "pico",
) -> Any:
    """Create the bridge's single, bounded three-topic SUB socket."""

    if type(host) is not str or not host:
        raise ValueError("host must be a nonempty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1,65535]")
    if zmq_module is None:
        import zmq as zmq_module

    if context is None:
        context = zmq_module.Context.instance()
    socket = context.socket(zmq_module.SUB)
    try:
        socket.setsockopt(zmq_module.RCVHWM, ZMQ_RCVHWM)
        socket.setsockopt(zmq_module.LINGER, 0)
        socket.setsockopt(zmq_module.MAXMSGSIZE, MAX_ZMQ_MESSAGE_BYTES)
        for topic in _source_topics(source_profile):
            socket.setsockopt_string(zmq_module.SUBSCRIBE, topic)
        socket.connect(f"tcp://{host}:{port}")
    except BaseException:
        try:
            socket.close()
        except Exception:
            pass
        raise
    return socket


def _packet_topic(raw: bytes, *, source_profile: str = "pico") -> str:
    if type(raw) is not bytes:
        raise TypeError("ZMQ packet must be bytes")
    matches = [topic for topic in _source_topics(source_profile) if raw.startswith(topic.encode("ascii"))]
    if len(matches) != 1:
        raise ValueError("packet does not have one exact authorized topic prefix")
    return matches[0]


def drain_zmq_packets(
    socket: Any,
    controller: Any,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    zmq_module: Any | None = None,
    max_packets: int = MAX_ZMQ_PACKETS_PER_TICK,
    max_drain_ns: int = MAX_ZMQ_DRAIN_NS,
) -> DrainStats:
    """Receive and dispatch one bounded, ordered nonblocking ZMQ batch."""

    if type(max_packets) is not int or max_packets <= 0:
        raise ValueError("max_packets must be a positive built-in int")
    if type(max_drain_ns) is not int or max_drain_ns <= 0:
        raise ValueError("max_drain_ns must be a positive built-in int")
    if zmq_module is None:
        import zmq as zmq_module
    source_profile = getattr(controller, "source_profile", None)
    if source_profile not in SOURCE_PROTOCOLS:
        raise ValueError("controller must expose a valid source_profile")

    start_ns = monotonic_ns()
    current_ns = start_ns
    received_count = 0
    dispatched_count = 0
    rejected_count = 0
    oversized_count = 0
    time_budget_exhausted = False
    while received_count < max_packets and not time_budget_exhausted:
        try:
            raw = socket.recv(zmq_module.NOBLOCK)
        except zmq_module.Again:
            current_ns = monotonic_ns()
            time_budget_exhausted = current_ns - start_ns >= max_drain_ns
            break
        received_count += 1
        if len(raw) > MAX_ZMQ_MESSAGE_BYTES:
            rejected_count += 1
            oversized_count += 1
            current_ns = monotonic_ns()
            time_budget_exhausted = current_ns - start_ns >= max_drain_ns
            continue
        try:
            topic = _packet_topic(raw, source_profile=source_profile)
            decoded = _decode_bridge_payload(raw, topic, source_profile=source_profile)
        except (TypeError, ValueError):
            rejected_count += 1
            current_ns = monotonic_ns()
            time_budget_exhausted = current_ns - start_ns >= max_drain_ns
            continue
        received_ns = monotonic_ns()
        packet = _stamp_bridge_payload(decoded, received_ns)
        if isinstance(packet, ManagerPacket):
            controller.accept_manager(packet.provenance, packet.received_ns)
        else:
            controller.accept_pair(
                packet.left,
                packet.right,
                packet.provenance,
                packet.topic,
                packet.received_ns,
                packet.message_seq,
                decoded.fingerprint,
            )
        dispatched_count += 1
        current_ns = monotonic_ns()
        time_budget_exhausted = current_ns - start_ns >= max_drain_ns
    return DrainStats(
        received=received_count,
        dispatched=dispatched_count,
        rejected=rejected_count,
        oversized=oversized_count,
        packet_budget_exhausted=received_count >= max_packets,
        time_budget_exhausted=time_budget_exhausted,
    )


class InspireStatusPublisher:
    """Publish exact q6 feedback/status at each 10 Hz control tick.

    fault_code 0 means no latched fault; 1 means a bridge safety fault.
    Detailed stable safety reasons remain in the JSONL evidence stream.
    """

    def __init__(self, port=5563, *, context=None, zmq_module=None, socket=None):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("status port must be in [1,65535]")
        if zmq_module is None:
            import zmq as zmq_module
        self._zmq = zmq_module
        self.session_id = uuid.uuid4().bytes
        self.sequence = 0
        self.socket = socket
        if self.socket is None:
            context = context or zmq_module.Context.instance()
            self.socket = context.socket(zmq_module.PUB)
            try:
                self.socket.setsockopt(zmq_module.LINGER, 0)
                self.socket.setsockopt(zmq_module.SNDHWM, 10)
                self.socket.bind(f"tcp://*:{port}")
            except BaseException:
                self.socket.close()
                raise

    def fields(self, controller, now_ns):
        fields = {name: np.zeros(size, dtype=dtype) for name, dtype, size in STATUS_SCHEMA}
        fields["pc2_session_id"][:] = np.frombuffer(self.session_id, dtype=np.uint8)
        fields["status_seq"][0] = self.sequence
        if controller.manager_provenance is not None:
            fields["accepted_pv"][:] = controller.manager_provenance.to_array()
        fields["bridge_state"][0] = list(BridgeState).index(controller.state)
        fields["last_applied_message_seq"][0] = controller.last_applied_message_seq
        healthy = True
        for side, snapshot, applied in zip(
            ("left", "right"),
            controller.hand_states,
            (controller.last_successful_left, controller.last_successful_right),
            strict=True,
        ):
            if snapshot is None:
                healthy = False
            else:
                fields[f"{side}_angle_act"][:] = snapshot.angle_act
                fields[f"{side}_err"][:] = snapshot.err
                healthy &= 0 <= now_ns - snapshot.received_ns < STATE_STALE_NS and not any(snapshot.err)
            if applied is not None:
                fields[f"{side}_applied"][:] = np.asarray(applied, dtype=np.float32) / 1000
        fields["feedback_healthy"][0] = healthy
        fields["fault_code"][0] = int(controller.fault_reason is not None)
        validate_inspire_status({**fields, "version": 1, "endian": "le"})
        return fields

    def publish(self, controller, now_ns):
        if self.sequence >= 2**63 - 1:
            raise RuntimeError("PC2 status sequence exhausted")
        payload = pack_pose_message(self.fields(controller, now_ns), topic="inspire_hand_status", version=1)
        self.socket.send(payload, self._zmq.NOBLOCK)
        self.sequence += 1

    def close(self):
        self.socket.close()


class InboundInbox:
    """Single-lock FIFO for immutable callback snapshots and faults."""

    __slots__ = ("_capacity", "_events", "_lock", "_overflow_pending")

    def __init__(self, capacity: int = DEFAULT_INBOX_CAPACITY) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive built-in int")
        self._capacity = capacity
        self._events: deque[InboxEvent] = deque()
        self._lock = threading.Lock()
        self._overflow_pending = False

    def _append_locked(self, event: InboxEvent) -> None:
        if len(self._events) >= self._capacity:
            self._overflow_pending = True
            return
        self._events.append(event)

    def capture_state(self, side: str, message: object, received_ns: int) -> None:
        """Validate, copy, and enqueue one callback sample under the inbox lock."""

        with self._lock:
            try:
                snapshot = validate_state_fields(message, side, received_ns)
                event: InboxEvent = StateSnapshotEvent(snapshot)
            except Exception as error:
                reason = str(error) or type(error).__name__
                event = CallbackFaultEvent(side, reason, received_ns)
            self._append_locked(event)

    def capture_fault(self, side: str, reason: str, received_ns: int) -> None:
        """Enqueue a callback-boundary fault without touching any controller state."""

        with self._lock:
            self._append_locked(CallbackFaultEvent(side, reason, received_ns))

    def drain(self) -> tuple[tuple[InboxEvent, ...], bool]:
        """Atomically remove every retained event and clear sticky overflow."""

        with self._lock:
            events = tuple(self._events)
            overflow = self._overflow_pending
            self._events.clear()
            self._overflow_pending = False
        return events, overflow


def create_dds_state_handler(
    side: str,
    inbox: InboundInbox,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> Callable[[object], None]:
    """Return a callback that can mutate only its inbox."""

    if side not in ("left", "right"):
        raise ValueError(f"invalid hand side: {side!r}")
    if not isinstance(inbox, InboundInbox):
        raise TypeError("inbox must be an InboundInbox")

    def handle(message: object) -> None:
        try:
            received_ns = monotonic_ns()
        except Exception as error:
            inbox.capture_fault(side, str(error) or type(error).__name__, 0)
            return
        try:
            inbox.capture_state(side, message, received_ns)
        except Exception as error:
            inbox.capture_fault(side, str(error) or type(error).__name__, received_ns)

    return handle


def drain_controller_events(inbox: InboundInbox, controller: Any) -> int:
    """Apply one callback batch to the controller from its owning main thread."""

    events, overflow = inbox.drain()
    if overflow:
        controller.latch_fault("inbound callback inbox overflow")
    for event in events:
        if isinstance(event, StateSnapshotEvent):
            controller.accept_hand_state(event.snapshot)
        else:
            controller.latch_fault(f"{event.side} callback fault: {event.reason}")
    return len(events)


def _require_controller_not_faulted(controller: Any) -> None:
    if controller.fault_reason is not None:
        raise DiscoveryFailure(_bounded_text(controller.fault_reason))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed, monitor-first PC2 bridge command line."""

    parser = argparse.ArgumentParser(
        description="Monitor or safely publish Inspire FTP hand commands",
        allow_abbrev=False,
    )
    parser.add_argument("--host", default="192.168.0.62")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--network-interface", default="enP8p1s0")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--source-profile", choices=tuple(SOURCE_PROTOCOLS), default="pico")
    parser.add_argument("--status-port", type=int, default=5563)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--evidence-log", default="")
    return parser.parse_args(argv)


def _json_logger(record: Mapping[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


class JsonlEvidenceSink:
    """Append bounded runtime evidence one JSON record at a time."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "a", encoding="utf-8")

    def __call__(self, record: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class WakeupSignalPipe:
    """Async-safe empty handlers backed by a nonblocking wakeup pipe."""

    def __init__(self) -> None:
        self._read_fd, self._write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        previous_wakeup_fd: int | None = None
        previous_handlers: dict[int, Any] = {}
        try:
            previous_wakeup_fd = signal.set_wakeup_fd(self._write_fd)
            previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
            for signum in previous_handlers:
                signal.signal(signum, self._empty_handler)
        except BaseException:
            for signum, handler in previous_handlers.items():
                try:
                    signal.signal(signum, handler)
                except Exception:
                    pass
            if previous_wakeup_fd is not None:
                try:
                    signal.set_wakeup_fd(previous_wakeup_fd)
                except Exception:
                    pass
            os.close(self._read_fd)
            os.close(self._write_fd)
            raise
        self._previous_wakeup_fd = previous_wakeup_fd
        self._previous_handlers = previous_handlers
        self._closed = False

    @staticmethod
    def _empty_handler(signum: int, frame: object) -> None:
        del signum, frame

    def drain(self) -> int:
        count = 0
        while True:
            try:
                chunk = os.read(self._read_fd, 256)
            except BlockingIOError:
                break
            if not chunk:
                break
            count += len(chunk)
            if len(chunk) < 256:
                break
        return count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            signal.set_wakeup_fd(self._previous_wakeup_fd)
        except Exception:
            pass
        for signum, handler in self._previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                pass
        try:
            os.close(self._read_fd)
        finally:
            os.close(self._write_fd)


@dataclass(slots=True)
class BridgeRuntimeResources:
    """All mutable transport objects owned exclusively by the main loop."""

    controller: Any
    inbox: InboundInbox
    zmq_socket: Any
    discovery: Any
    ownership: PublicationOwnershipTracker
    signal_source: Any
    left_writer: Any | None = None
    right_writer: Any | None = None
    left_message: Any | None = None
    right_message: Any | None = None
    subscribers: tuple[Any, ...] = ()
    zmq_rejected: int = 0
    zmq_oversized: int = 0
    zmq_backlog_discarded: int = 0
    signal_count: int = 0
    status_publisher: Any | None = None


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Fully injectable runtime effects for deterministic transport tests."""

    monotonic_ns: Callable[[], int] = time.monotonic_ns
    sleep: Callable[[float], None] = time.sleep
    zmq_module: Any | None = None
    zmq_context: Any | None = None
    zmq_socket_factory: Callable[[str, int], Any] | None = None
    status_publisher_factory: Callable[[int], Any] | None = None
    dds_loader: Callable[[str], Any] = load_pc2_dds
    signal_source_factory: Callable[[], Any] = WakeupSignalPipe
    controller_factory: Callable[[], Any] = InspireFtpSafetyController
    logger: Callable[[Mapping[str, Any]], None] = _json_logger
    evidence_sink: Callable[[Mapping[str, Any]], None] | None = None
    max_ticks: int | None = None
    discovery_timeout_ns: int = DEFAULT_DISCOVERY_TIMEOUT_NS

    def replaced(self, **changes: Any) -> RuntimeDependencies:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    exit_code: int
    exit_reason: str
    ticks: int
    forced_exit: bool = False
    opening_confirmed: bool = False
    opening_unconfirmed: bool = False
    measured_feedback_confirmed: bool = False
    emergency_attempted: bool = False
    emergency_completed: bool = False


def _measured_open_feedback_confirmed(controller: Any, now_ns: int) -> bool:
    try:
        states = tuple(controller.hand_states)
        return len(states) == 2 and all(
            snapshot is not None
            and 0 <= now_ns - snapshot.received_ns < STATE_STALE_NS
            and len(snapshot.err) == 6
            and len(snapshot.angle_act) == 6
            and not any(snapshot.err)
            and all(value >= OPEN_FEEDBACK_MIN for value in snapshot.angle_act)
            for snapshot in states
        )
    except Exception:
        return False


def _terminal_record(result: RuntimeResult, now_ns: int) -> dict[str, Any]:
    return {
        "event": "terminal",
        "timestamp_ns": now_ns,
        "exit_code": result.exit_code,
        "reason": _bounded_text(result.exit_reason),
        "ticks": result.ticks,
        "forced_exit": result.forced_exit,
        "command_open_completed": result.opening_confirmed,
        "opening_unconfirmed": result.opening_unconfirmed,
        "measured_feedback_confirmed": result.measured_feedback_confirmed,
        "emergency": {
            "attempted": result.emergency_attempted,
            "completed": result.emergency_completed,
        },
    }


def _emit_terminal_result(
    dependencies: RuntimeDependencies,
    result: RuntimeResult,
    now_ns: int,
) -> RuntimeResult:
    record = _terminal_record(result, now_ns)
    if dependencies.evidence_sink is not None:
        try:
            dependencies.evidence_sink(record)
        except Exception as error:
            evidence_failure = f"terminal evidence failure: {_bounded_text(error)}"
            failed = replace(
                result,
                exit_code=1,
                exit_reason=_bounded_text(f"{result.exit_reason}; {evidence_failure}"),
            )
            try:
                dependencies.logger(_terminal_record(failed, now_ns))
            except Exception:
                pass
            return failed
    try:
        dependencies.logger(record)
    except Exception:
        pass
    return result


def write_hand_command(writer: Any, message: Any, counts: object) -> bool:
    """Write one exact FTP command, containing transport exceptions per side."""

    copied = tuple(counts)  # type: ignore[arg-type]
    if len(copied) != 6 or any(type(value) is not int for value in copied):
        raise ValueError("FTP command must contain exactly six built-in ints")
    try:
        message.mode = 0b0001
        message.angle_set = list(copied)
        return writer.Write(message) is True
    except Exception:
        return False


def _emergency_open_existing_writers(
    left_writer: Any,
    right_writer: Any,
    left_message: Any,
    right_message: Any,
    last_left: tuple[int, ...] | None,
    last_right: tuple[int, ...] | None,
    *,
    monotonic_ns: Callable[[], int],
    sleep: Callable[[float], None],
    abort: Callable[[], bool] | None = None,
) -> bool:
    """Boundedly fail open using only already-created transport resources."""

    if any(item is None for item in (left_writer, right_writer, left_message, right_message)):
        return False
    if last_left is None or last_right is None:
        return False
    left = tuple(last_left)
    right = tuple(last_right)
    if (
        len(left) != 6
        or len(right) != 6
        or any(type(value) is not int or not MIN_COMMAND <= value <= 1000 for value in left + right)
    ):
        return False
    open_counts = (1000,) * 6
    next_deadline_ns = monotonic_ns() + CONTROL_PERIOD_NS

    def wait_for_deadline() -> None:
        nonlocal next_deadline_ns
        now_ns = monotonic_ns()
        if now_ns < next_deadline_ns:
            sleep((next_deadline_ns - now_ns) / 1_000_000_000)
        next_deadline_ns += CONTROL_PERIOD_NS
        now_ns = monotonic_ns()
        if next_deadline_ns < now_ns:
            skipped = (now_ns - next_deadline_ns) // CONTROL_PERIOD_NS + 1
            next_deadline_ns += skipped * CONTROL_PERIOD_NS

    for _ in range(40):
        if left == open_counts and right == open_counts:
            break
        if abort is not None and abort():
            return False
        wait_for_deadline()
        if abort is not None and abort():
            return False
        attempted_left = tuple(min(1000, value + 5) for value in left)
        attempted_right = tuple(min(1000, value + 5) for value in right)
        if write_hand_command(left_writer, left_message, attempted_left):
            left = attempted_left
        if abort is not None and abort():
            return False
        if write_hand_command(right_writer, right_message, attempted_right):
            right = attempted_right
    if left != open_counts or right != open_counts:
        return False
    for _ in range(20):
        if abort is not None and abort():
            return False
        wait_for_deadline()
        if abort is not None and abort():
            return False
        write_hand_command(left_writer, left_message, open_counts)
        if abort is not None and abort():
            return False
        write_hand_command(right_writer, right_message, open_counts)
    return True


def _discard_zmq_backlog(
    socket: Any,
    *,
    monotonic_ns: Callable[[], int],
    zmq_module: Any,
) -> int:
    if zmq_module is None:
        import zmq as zmq_module

    started_ns = monotonic_ns()
    discarded = 0
    while discarded < MAX_ZMQ_PACKETS_PER_TICK:
        if monotonic_ns() - started_ns >= MAX_ZMQ_DRAIN_NS:
            break
        try:
            socket.recv(zmq_module.NOBLOCK)
        except zmq_module.Again:
            break
        discarded += 1
    return discarded


def _bounded_text(value: object) -> str:
    return str(value)[:256]


def _state_record(snapshot: Any | None, age_ns: int | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "angles": list(snapshot.angle_act),
        "errors": list(snapshot.err),
        "age_ns": age_ns,
    }


def build_status_record(
    controller: Any,
    ownership: PublicationOwnershipEvidence,
    now_ns: int,
    attempted_left: tuple[int, ...] | None,
    attempted_right: tuple[int, ...] | None,
    *,
    zmq_rejected: int = 0,
    zmq_oversized: int = 0,
    zmq_backlog_discarded: int = 0,
) -> dict[str, Any]:
    """Build one bounded JSON-safe status/evidence record."""

    manager = controller.manager_provenance
    pair = controller.latest_pair
    source_profile = getattr(controller, "source_profile", "pico")
    left_state, right_state = controller.hand_states
    left_age, right_age = controller.hand_state_ages_ns(now_ns)
    latest_left, latest_right = controller.latest_targets
    latest_limited = getattr(controller, "latest_envelope_limited", (False, False))
    raw_source_targets = (
        None
        if pair is None
        else {"left": [float(value) for value in pair.left], "right": [float(value) for value in pair.right]}
    )
    effective_normalized_targets = {
        "left": [value / 1000.0 for value in latest_left],
        "right": [value / 1000.0 for value in latest_right],
    }
    lifecycle = [
        {
            "key": event.key,
            "participant_key": event.participant_key,
            "topic": event.topic_name,
            "alive": event.alive,
            "observed_ns": event.observed_ns,
        }
        for event in ownership.lifecycle
    ]
    return {
        "event": "status",
        "timestamp_ns": now_ns,
        "fsm": controller.state.name,
        "manager": None
        if manager is None
        else {
            "session": manager.session_id.hex(),
            "epoch": manager.mode_epoch,
            "mode": manager.stream_mode,
            "age_ns": controller.manager_age_ns(now_ns),
        },
        "provenance_generation": controller.provenance_generation,
        "source": None
        if pair is None
        else {
            "profile": _bounded_text(source_profile),
            "topic": _bounded_text(pair.topic),
            "version": _source_topics(source_profile).get(pair.topic),
            "age_ns": controller.source_age_ns(now_ns),
        },
        "raw_source_targets": raw_source_targets,
        "envelope_targets": {
            "left": list(latest_left),
            "right": list(latest_right),
            "limited": {"left": bool(latest_limited[0]), "right": bool(latest_limited[1])},
        },
        "effective_normalized_targets": effective_normalized_targets,
        "attempted_ftp": {
            "left": None if attempted_left is None else list(attempted_left),
            "right": None if attempted_right is None else list(attempted_right),
        },
        "last_successful_ftp": {
            "left": None if controller.last_successful_left is None else list(controller.last_successful_left),
            "right": None if controller.last_successful_right is None else list(controller.last_successful_right),
        },
        "measured": {
            "left": _state_record(left_state, left_age),
            "right": _state_record(right_state, right_age),
        },
        "ownership": {
            "owned": [list(item) for item in ownership.owned],
            "owned_disruption": [list(item) for item in ownership.owned_disruption],
            "foreign": [list(item) for item in ownership.foreign_history],
            "foreign_overflow": ownership.foreign_overflow,
            "identity_overflow": ownership.identity_overflow,
            "identity_mismatch": ownership.identity_mismatch,
            "state_identity_fault": ownership.state_identity_fault,
            "state_alive": [[side, list(keys)] for side, keys in ownership.state_alive],
            "command_alive": [[side, list(keys)] for side, keys in ownership.command_alive],
            "healthy": ownership.ownership_healthy,
            "identities": [
                {
                    "key": item.key,
                    "participant_key": item.participant_key,
                    "topic": item.topic_name,
                    "side": item.side,
                    "first_observed_ns": item.first_observed_ns,
                    "last_observed_ns": item.last_observed_ns,
                    "ever_alive": item.ever_alive,
                    "ever_disposed": item.ever_disposed,
                    "classification": item.classification,
                }
                for item in ownership.identities
            ],
            "lifecycle": lifecycle,
        },
        "maximum_tick_gap_ns": controller.maximum_tick_interval_ns,
        "counters": {
            "overruns": controller.loop_overrun_count,
            "write_failures": list(controller.write_failure_counts),
            "rejected_pairs": controller.rejected_pairs,
            "sequence_gaps": controller.sequence_gaps,
            "zmq_rejected": zmq_rejected,
            "zmq_oversized": zmq_oversized,
            "zmq_backlog_discarded": zmq_backlog_discarded,
            "envelope_limited_pairs": controller.envelope_limited_pairs,
            "envelope_limited_values": controller.envelope_limited_values,
        },
        "fault": None if controller.fault_reason is None else _bounded_text(controller.fault_reason),
    }


def _emit_record(dependencies: RuntimeDependencies, record: Mapping[str, Any]) -> None:
    dependencies.logger(record)
    if dependencies.evidence_sink is not None:
        dependencies.evidence_sink(record)


def _discovery_phase_record(
    phase: str,
    evidence: PublicationOwnershipEvidence,
    observation_start_ns: int,
    observation_end_ns: int,
    result: str,
    *,
    zero_since_ns: int | None = None,
) -> dict[str, Any]:
    record = {
        "event": "discovery",
        "phase": phase,
        "timestamp_ns": observation_end_ns,
        "observation_start_ns": observation_start_ns,
        "observation_end_ns": observation_end_ns,
        "duration_ns": max(0, observation_end_ns - observation_start_ns),
        "result": _bounded_text(result),
        "state_alive": [[side, list(keys)] for side, keys in evidence.state_alive],
        "command_alive": [[side, list(keys)] for side, keys in evidence.command_alive],
        "owned": [list(item) for item in evidence.owned],
        "owned_disruption": [list(item) for item in evidence.owned_disruption],
        "foreign": [list(item) for item in evidence.foreign_history],
        "foreign_overflow": evidence.foreign_overflow,
        "identity_overflow": evidence.identity_overflow,
        "identity_mismatch": evidence.identity_mismatch,
        "state_identity_fault": evidence.state_identity_fault,
        "identities": [
            {
                "key": item.key,
                "participant_key": item.participant_key,
                "topic": item.topic_name,
                "side": item.side,
                "first_observed_ns": item.first_observed_ns,
                "last_observed_ns": item.last_observed_ns,
                "ever_alive": item.ever_alive,
                "ever_disposed": item.ever_disposed,
                "classification": item.classification,
            }
            for item in evidence.identities
        ],
        "lifecycle": [
            {
                "key": item.key,
                "participant_key": item.participant_key,
                "topic": item.topic_name,
                "alive": item.alive,
                "observed_ns": item.observed_ns,
            }
            for item in evidence.lifecycle
        ],
    }
    if zero_since_ns is not None:
        record["zero_since_ns"] = zero_since_ns
    return record


def _physical_stop_record(
    fault: str,
    evidence: PublicationOwnershipEvidence,
    now_ns: int,
) -> dict[str, Any]:
    return {
        "event": "physical_stop_required",
        "timestamp_ns": now_ns,
        "reason": _bounded_text(fault),
        "guidance": (
            "POWER OFF both Inspire hand actuators immediately; do not rely on a software stop, "
            "and verify both command topics have zero writers before restart."
        ),
        "ownership": {
            "owned": [list(item) for item in evidence.owned],
            "alive": [[side, list(keys)] for side, keys in evidence.command_alive],
            "foreign": [list(item) for item in evidence.foreign_history],
            "owned_disruption": [list(item) for item in evidence.owned_disruption],
        },
    }


def _command_ownership_requires_physical_stop(evidence: PublicationOwnershipEvidence) -> bool:
    owned = dict(evidence.owned)
    return bool(
        evidence.foreign_history
        or evidence.foreign_overflow
        or evidence.owned_disruption
        or any(side in owned and keys != (owned[side],) for side, keys in evidence.command_alive)
    )


def run_control_loop(
    resources: BridgeRuntimeResources,
    dependencies: RuntimeDependencies,
) -> RuntimeResult:
    """Run the deadline-scheduled, single-owner 10 Hz bridge loop."""

    if dependencies.max_ticks is not None and dependencies.max_ticks < 0:
        raise ValueError("max_ticks must be nonnegative")
    next_deadline_ns = dependencies.monotonic_ns()
    last_tick_start_ns: int | None = None
    last_status_ns: int | None = None
    previous_state = resources.controller.state
    ticks = 0
    attempted_left: tuple[int, ...] | None = None
    attempted_right: tuple[int, ...] | None = None
    runtime_fault: str | None = None
    physical_stop_emitted = False

    def drain_signals() -> int:
        new_signals = resources.signal_source.drain()
        if type(new_signals) is not int or new_signals < 0:
            raise RuntimeError("signal source returned an invalid count")
        if new_signals:
            resources.signal_count += new_signals
            resources.controller.request_shutdown(resources.signal_count)
        return new_signals

    def forced_result(completed_ticks: int) -> RuntimeResult:
        result_now_ns = dependencies.monotonic_ns()
        command_resources_exist = resources.left_writer is not None or resources.right_writer is not None
        return RuntimeResult(
            0,
            "second signal",
            completed_ticks,
            forced_exit=True,
            opening_unconfirmed=command_resources_exist,
            measured_feedback_confirmed=_measured_open_feedback_confirmed(
                resources.controller,
                result_now_ns,
            ),
        )

    while dependencies.max_ticks is None or ticks < dependencies.max_ticks:
        now_ns = dependencies.monotonic_ns()
        if now_ns < next_deadline_ns:
            dependencies.sleep((next_deadline_ns - now_ns) / 1_000_000_000)
        tick_start_ns = dependencies.monotonic_ns()
        overrun = last_tick_start_ns is not None and tick_start_ns - last_tick_start_ns > MAX_TICK_INTERVAL_NS
        if overrun:
            resources.controller.latch_fault("control tick interval overrun")

        drain_signals()
        if resources.signal_count >= 2:
            return forced_result(ticks)

        drain_controller_events(resources.inbox, resources.controller)
        drain_signals()
        if resources.signal_count >= 2:
            return forced_result(ticks)
        if overrun:
            resources.zmq_backlog_discarded += _discard_zmq_backlog(
                resources.zmq_socket,
                monotonic_ns=dependencies.monotonic_ns,
                zmq_module=dependencies.zmq_module,
            )
        else:
            drain_stats = drain_zmq_packets(
                resources.zmq_socket,
                resources.controller,
                monotonic_ns=dependencies.monotonic_ns,
                zmq_module=dependencies.zmq_module,
            )
            resources.zmq_rejected += drain_stats.rejected
            resources.zmq_oversized += drain_stats.oversized
        drain_signals()
        if resources.signal_count >= 2:
            return forced_result(ticks)
        try:
            resources.ownership.observe(
                resources.discovery.drain_publications(monotonic_ns=dependencies.monotonic_ns)
            )
            evidence = resources.ownership.evidence()
            if len(evidence.owned) == 2:
                ownership_fault = resources.ownership.health_fault()
                if ownership_fault is not None:
                    resources.controller.latch_fault(ownership_fault)
                    if not physical_stop_emitted and _command_ownership_requires_physical_stop(evidence):
                        physical_stop_emitted = True
                        _emit_record(
                            dependencies,
                            _physical_stop_record(
                                ownership_fault,
                                evidence,
                                dependencies.monotonic_ns(),
                            ),
                        )
        except Exception as error:
            runtime_fault = f"DDS discovery exception: {error}"
            resources.controller.latch_fault(runtime_fault)
            evidence = resources.ownership.evidence()

        drain_signals()
        if resources.signal_count >= 2:
            return forced_result(ticks)
        controller_now_ns = dependencies.monotonic_ns()
        try:
            decision = resources.controller.tick(controller_now_ns)
        except Exception as error:
            runtime_fault = f"control loop exception: {error}"
            resources.controller.latch_fault(runtime_fault)
            raise RuntimeError(runtime_fault) from error

        if (
            decision is not None
            and resources.signal_count == 1
            and decision.state is not BridgeState.SHUTDOWN_OPENING
        ):
            decision = None

        if decision is not None:
            if decision.state is not previous_state:
                _emit_record(
                    dependencies,
                    {
                        "event": "transition",
                        "timestamp_ns": controller_now_ns,
                        "from": previous_state.name,
                        "to": decision.state.name,
                        "fault": decision.fault_reason,
                    },
                )
                previous_state = decision.state
            if decision.publish:
                if decision.state is not BridgeState.ACTIVE:
                    resources.controller.last_applied_message_seq = -1
                attempted_left = tuple(decision.left_counts)
                attempted_right = tuple(decision.right_counts)
                new_signals = drain_signals()
                if resources.signal_count >= 2:
                    return forced_result(ticks + 1)
                if new_signals:
                    ticks += 1
                    last_tick_start_ns = tick_start_ns
                    next_deadline_ns += CONTROL_PERIOD_NS
                    now_ns = dependencies.monotonic_ns()
                    if next_deadline_ns < now_ns:
                        skipped = (now_ns - next_deadline_ns) // CONTROL_PERIOD_NS + 1
                        next_deadline_ns += skipped * CONTROL_PERIOD_NS
                    continue
                try:
                    left_ok = write_hand_command(resources.left_writer, resources.left_message, attempted_left)
                except Exception as error:
                    runtime_fault = f"left command exception: {error}"
                    left_ok = False
                resources.controller.record_write_result("left", attempted_left, left_ok)
                drain_signals()
                if resources.signal_count >= 2:
                    return forced_result(ticks + 1)
                try:
                    right_ok = write_hand_command(resources.right_writer, resources.right_message, attempted_right)
                except Exception as error:
                    runtime_fault = f"right command exception: {error}"
                    right_ok = False
                resources.controller.record_write_result("right", attempted_right, right_ok)
                if left_ok and right_ok and decision.state is BridgeState.ACTIVE:
                    resources.controller.last_applied_message_seq = resources.controller.latest_pair.message_seq
                drain_signals()
                if resources.signal_count >= 2:
                    return forced_result(ticks + 1)
            if decision.exit_now:
                if resources.controller.state is BridgeState.SHUTDOWN_OPENING:
                    if resources.left_writer is None and resources.right_writer is None:
                        return RuntimeResult(0, "monitor shutdown", ticks + 1)
                    opening_confirmed = (
                        resources.controller.last_successful_left == (1000,) * 6
                        and resources.controller.last_successful_right == (1000,) * 6
                        and resources.controller.shutdown_hold_ticks >= 20
                    )
                    if opening_confirmed:
                        return RuntimeResult(
                            0,
                            "graceful open hold complete",
                            ticks + 1,
                            opening_confirmed=True,
                            measured_feedback_confirmed=_measured_open_feedback_confirmed(
                                resources.controller,
                                controller_now_ns,
                            ),
                        )
                    exhausted = resources.controller.shutdown_open_attempts >= 40
                    return RuntimeResult(
                        1,
                        "shutdown opening exhausted" if exhausted else "shutdown opening unconfirmed",
                        ticks + 1,
                        opening_unconfirmed=True,
                        measured_feedback_confirmed=_measured_open_feedback_confirmed(
                            resources.controller,
                            controller_now_ns,
                        ),
                    )
                return RuntimeResult(1 if runtime_fault else 0, runtime_fault or "controller exit", ticks + 1)

        if resources.status_publisher is not None:
            resources.status_publisher.publish(resources.controller, controller_now_ns)
        if last_status_ns is None or controller_now_ns - last_status_ns >= 1_000_000_000:
            _emit_record(
                dependencies,
                build_status_record(
                    resources.controller,
                    evidence,
                    controller_now_ns,
                    attempted_left,
                    attempted_right,
                    zmq_rejected=resources.zmq_rejected,
                    zmq_oversized=resources.zmq_oversized,
                    zmq_backlog_discarded=resources.zmq_backlog_discarded,
                ),
            )
            last_status_ns = controller_now_ns

        ticks += 1
        last_tick_start_ns = tick_start_ns
        # The safety controller timestamps decisions after ingress. Do not schedule
        # the next decision before its full 100ms period when ingress jitter changes.
        next_deadline_ns = max(next_deadline_ns + CONTROL_PERIOD_NS, controller_now_ns + CONTROL_PERIOD_NS)
        now_ns = dependencies.monotonic_ns()
        if next_deadline_ns < now_ns:
            skipped = (now_ns - next_deadline_ns) // CONTROL_PERIOD_NS + 1
            next_deadline_ns += skipped * CONTROL_PERIOD_NS
    if resources.controller.state is BridgeState.SHUTDOWN_OPENING:
        if resources.left_writer is None and resources.right_writer is None:
            return RuntimeResult(0, "monitor shutdown", ticks)
        result_now_ns = dependencies.monotonic_ns()
        return RuntimeResult(
            1,
            runtime_fault or "shutdown opening incomplete at tick limit",
            ticks,
            opening_unconfirmed=True,
            measured_feedback_confirmed=_measured_open_feedback_confirmed(
                resources.controller,
                result_now_ns,
            ),
        )
    return RuntimeResult(1 if runtime_fault else 0, runtime_fault or "tick limit", ticks)


def _initialize_channel(channel: Any, *args: Any) -> None:
    init = getattr(channel, "Init", None)
    if callable(init):
        init(*args)


def _close_resource(resource: Any) -> None:
    if resource is None:
        return
    for name in ("Close", "close"):
        close = getattr(resource, name, None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
            return


def run_bridge(args: argparse.Namespace, dependencies: RuntimeDependencies | None = None) -> RuntimeResult:
    """Construct, run, and clean up the monitor-first PC2 bridge."""

    dependencies = RuntimeDependencies() if dependencies is None else dependencies
    source_profile = getattr(args, "source_profile", "pico")
    if source_profile not in SOURCE_PROTOCOLS:
        raise ValueError("source_profile must be 'pico' (optical q6)")
    signal_source = None
    socket = None
    status_publisher = None
    dds = None
    subscribers: list[Any] = []
    writers: list[Any] = []
    evidence_sink = None
    result = RuntimeResult(1, "initialization did not complete", 0)
    tracker = PublicationOwnershipTracker()
    controller = None
    left_writer = right_writer = left_message = right_message = None
    active_dependencies = dependencies
    ownership_registered = False
    resources: BridgeRuntimeResources | None = None
    observed_signal_count = 0
    try:
        signal_source = dependencies.signal_source_factory()
        if dependencies.zmq_socket_factory is not None:
            socket = dependencies.zmq_socket_factory(args.host, args.port)
        else:
            socket = create_zmq_subscriber(
                args.host,
                args.port,
                context=dependencies.zmq_context,
                zmq_module=dependencies.zmq_module,
                source_profile=source_profile,
            )
        status_publisher = (
            dependencies.status_publisher_factory(args.status_port)
            if dependencies.status_publisher_factory is not None
            else InspireStatusPublisher(
                args.status_port, context=dependencies.zmq_context, zmq_module=dependencies.zmq_module
            )
        )
        dds = dependencies.dds_loader(args.network_interface)
        controller = dependencies.controller_factory(
            source_profile=source_profile,
        )
        inbox = InboundInbox()
        for side, topic in STATE_TOPICS.items():
            subscriber = dds.subscriber_factory(topic, dds.state_type)
            subscribers.append(subscriber)
            _initialize_channel(
                subscriber,
                create_dds_state_handler(side, inbox, monotonic_ns=dependencies.monotonic_ns),
                10,
            )

        if args.evidence_log and dependencies.evidence_sink is None:
            evidence_sink = JsonlEvidenceSink(args.evidence_log)
            active_dependencies = dependencies.replaced(evidence_sink=evidence_sink)

        if not args.publish:
            # A new built-in reader receives the existing DDS publication snapshot in
            # one potentially expensive batch. Consume that one-time startup work
            # before establishing the 10 Hz control-loop timing baseline.
            tracker.observe(dds.drain_publications(monotonic_ns=dependencies.monotonic_ns))

        if args.publish:

            def poll() -> tuple[PublicationLifecycleSample, ...]:
                nonlocal observed_signal_count
                new_signals = signal_source.drain()
                observed_signal_count += new_signals
                if new_signals:
                    raise _PreControlSignal("signal received before control loop", observed_signal_count)
                samples = dds.drain_publications(monotonic_ns=dependencies.monotonic_ns)
                drain_controller_events(inbox, controller)
                _require_controller_not_faulted(controller)
                return samples

            preflight_started_ns = dependencies.monotonic_ns()
            try:
                before = require_zero_writer_preflight(
                    tracker,
                    poll,
                    monotonic_ns=dependencies.monotonic_ns,
                    sleep=dependencies.sleep,
                    timeout_ns=dependencies.discovery_timeout_ns,
                )
            except Exception as error:
                _emit_record(
                    active_dependencies,
                    _discovery_phase_record(
                        "preflight",
                        tracker.evidence(),
                        preflight_started_ns,
                        dependencies.monotonic_ns(),
                        f"failed: {_bounded_text(error)}",
                    ),
                )
                raise
            _emit_record(
                active_dependencies,
                _discovery_phase_record(
                    "preflight",
                    before,
                    preflight_started_ns,
                    dependencies.monotonic_ns(),
                    "healthy",
                ),
            )
            drain_controller_events(inbox, controller)
            _require_controller_not_faulted(controller)
            left_state, right_state = controller.hand_states
            if left_state is None or right_state is None:
                raise DiscoveryFailure("fresh left/right state feedback missing before writer creation")
            armed_from_state = controller.state
            armed_at_ns = dependencies.monotonic_ns()
            controller.arm_publish_from_feedback(
                left_state,
                right_state,
                armed_at_ns,
            )
            _emit_record(
                active_dependencies,
                {
                    "event": "transition",
                    "timestamp_ns": armed_at_ns,
                    "from": armed_from_state.name,
                    "to": controller.state.name,
                    "fault": controller.fault_reason,
                },
            )
            new_signals = signal_source.drain()
            observed_signal_count += new_signals
            if new_signals:
                raise _PreControlSignal("signal received before writer creation", observed_signal_count)
            left_writer = dds.publisher_factory(COMMAND_TOPICS["left"], dds.command_type)
            writers.append(left_writer)
            _initialize_channel(left_writer)
            new_signals = signal_source.drain()
            observed_signal_count += new_signals
            if new_signals:
                raise _PreControlSignal("signal received during writer creation", observed_signal_count)
            right_writer = dds.publisher_factory(COMMAND_TOPICS["right"], dds.command_type)
            writers.append(right_writer)
            _initialize_channel(right_writer)
            registration_started_ns = dependencies.monotonic_ns()
            try:
                left_participant_key = dds.publisher_participant_key(left_writer)
                right_participant_key = dds.publisher_participant_key(right_writer)
                if left_participant_key != right_participant_key:
                    raise DiscoveryFailure("created publisher participant identities do not match")
                left_message = dds.message_factory()
                right_message = dds.message_factory()
                registered = require_bridge_writer_registration(
                    tracker,
                    before,
                    poll,
                    expected_participant_key=left_participant_key,
                    monotonic_ns=dependencies.monotonic_ns,
                    sleep=dependencies.sleep,
                    timeout_ns=dependencies.discovery_timeout_ns,
                )
            except Exception as error:
                _emit_record(
                    active_dependencies,
                    _discovery_phase_record(
                        "owned",
                        tracker.evidence(),
                        registration_started_ns,
                        dependencies.monotonic_ns(),
                        f"failed: {_bounded_text(error)}",
                    ),
                )
                raise
            ownership_registered = True
            _emit_record(
                active_dependencies,
                _discovery_phase_record(
                    "owned",
                    registered,
                    registration_started_ns,
                    dependencies.monotonic_ns(),
                    "healthy",
                ),
            )
            drain_controller_events(inbox, controller)
            _require_controller_not_faulted(controller)
            controller.require_publish_feedback_healthy(dependencies.monotonic_ns())

        resources = BridgeRuntimeResources(
            controller=controller,
            inbox=inbox,
            zmq_socket=socket,
            discovery=dds,
            ownership=tracker,
            signal_source=signal_source,
            status_publisher=status_publisher,
            left_writer=left_writer,
            right_writer=right_writer,
            left_message=left_message,
            right_message=right_message,
            subscribers=tuple(subscribers),
        )
        result = run_control_loop(resources, active_dependencies)
    except _PreControlSignal as error:
        command_resources_exist = bool(writers)
        result = RuntimeResult(
            1 if command_resources_exist else 0,
            _bounded_text(error),
            0,
            forced_exit=error.count >= 2,
            opening_unconfirmed=command_resources_exist,
            measured_feedback_confirmed=_measured_open_feedback_confirmed(
                controller,
                dependencies.monotonic_ns(),
            ),
        )
    except Exception as error:
        emergency_signal_count = observed_signal_count if resources is None else resources.signal_count

        def emergency_abort() -> bool:
            nonlocal emergency_signal_count
            emergency_signal_count += signal_source.drain()
            return emergency_signal_count >= 2

        emergency_attempted = bool(writers and controller is not None and ownership_registered)
        emergency_completed = False
        if writers and controller is not None and ownership_registered:
            emergency_completed = _emergency_open_existing_writers(
                left_writer,
                right_writer,
                left_message,
                right_message,
                controller.last_successful_left,
                controller.last_successful_right,
                monotonic_ns=dependencies.monotonic_ns,
                sleep=dependencies.sleep,
                abort=emergency_abort,
            )
        if resources is not None:
            resources.signal_count = max(resources.signal_count, emergency_signal_count)
        else:
            observed_signal_count = max(observed_signal_count, emergency_signal_count)
        result = RuntimeResult(
            1,
            _bounded_text(error) or type(error).__name__,
            0,
            forced_exit=emergency_signal_count >= 2,
            opening_confirmed=emergency_completed,
            opening_unconfirmed=emergency_attempted and not emergency_completed,
            measured_feedback_confirmed=_measured_open_feedback_confirmed(
                controller,
                dependencies.monotonic_ns(),
            ),
            emergency_attempted=emergency_attempted,
            emergency_completed=emergency_completed,
        )
    finally:
        for writer in writers:
            _close_resource(writer)
        forced_signal_exit = result.forced_exit
        if writers and dds is not None and not forced_signal_exit:
            try:
                final_started_ns = dependencies.monotonic_ns()

                def poll_final() -> tuple[PublicationLifecycleSample, ...]:
                    nonlocal observed_signal_count
                    new_signals = signal_source.drain()
                    if resources is not None:
                        resources.signal_count += new_signals
                        total_signals = resources.signal_count
                    else:
                        observed_signal_count += new_signals
                        total_signals = observed_signal_count
                    if total_signals >= 2:
                        raise _PreControlSignal(
                            "second signal during final writer closure",
                            total_signals,
                        )
                    return dds.drain_publications(monotonic_ns=dependencies.monotonic_ns)

                try:
                    final_closure = require_final_zero_writers(
                        tracker,
                        poll_final,
                        monotonic_ns=dependencies.monotonic_ns,
                        sleep=dependencies.sleep,
                        timeout_ns=dependencies.discovery_timeout_ns,
                    )
                except Exception as error:
                    _emit_record(
                        active_dependencies,
                        _discovery_phase_record(
                            "final_zero",
                            tracker.evidence(),
                            final_started_ns,
                            dependencies.monotonic_ns(),
                            f"failed: {_bounded_text(error)}",
                        ),
                    )
                    raise
                _emit_record(
                    active_dependencies,
                    _discovery_phase_record(
                        "final_zero",
                        final_closure.ownership,
                        final_closure.zero_since_ns,
                        final_closure.observation_end_ns,
                        "healthy",
                        zero_since_ns=final_closure.zero_since_ns,
                    ),
                )
            except _PreControlSignal:
                if result.exit_code == 0:
                    result = replace(result, exit_reason="second signal", forced_exit=True)
                else:
                    result = replace(result, forced_exit=True)
            except Exception as error:
                if result.exit_code == 0:
                    result = replace(result, exit_code=1, exit_reason=_bounded_text(error))
        elif writers and forced_signal_exit:
            try:
                skipped_at_ns = dependencies.monotonic_ns()
                _emit_record(
                    active_dependencies,
                    _discovery_phase_record(
                        "final_zero_skipped_forced_signal",
                        tracker.evidence(),
                        skipped_at_ns,
                        skipped_at_ns,
                        "skipped: forced second signal",
                    ),
                )
            except Exception:
                pass
        result = _emit_terminal_result(
            active_dependencies,
            result,
            dependencies.monotonic_ns(),
        )
        for subscriber in subscribers:
            _close_resource(subscriber)
        if dds is not None:
            _close_resource(getattr(dds, "publication_reader", None))
            _close_resource(getattr(dds, "participant", None))
        _close_resource(status_publisher)
        _close_resource(socket)
        _close_resource(signal_source)
        _close_resource(evidence_sink)
    return result


def main(argv: list[str] | None = None) -> int:
    return run_bridge(parse_args(argv)).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
