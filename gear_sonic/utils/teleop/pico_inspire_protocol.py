"""Exact optical Inspire q6 and physical-feedback wire schemas."""

import numpy as np

from gear_sonic.utils.teleop.pico_recording import parse_pv

HAND_SCHEMA = (
    ("pv", "u1", 28),
    ("message_seq", "<i8", 1),
    ("sample_generation", "<i8", 1),
    ("left_command", "<f4", 6),
    ("right_command", "<f4", 6),
    ("left_tracking_state", "<i4", 1),
    ("right_tracking_state", "<i4", 1),
    ("left_source_epoch", "<i8", 1),
    ("right_source_epoch", "<i8", 1),
    ("left_source_timestamp_ns", "<i8", 1),
    ("right_source_timestamp_ns", "<i8", 1),
)
STATUS_SCHEMA = (
    ("pc2_session_id", "u1", 16),
    ("status_seq", "<i8", 1),
    ("accepted_pv", "u1", 28),
    ("bridge_state", "<i4", 1),
    ("last_applied_message_seq", "<i8", 1),
    ("left_angle_act", "<i4", 6),
    ("right_angle_act", "<i4", 6),
    ("left_err", "u1", 6),
    ("right_err", "u1", 6),
    ("left_applied", "<f4", 6),
    ("right_applied", "<f4", 6),
    ("feedback_healthy", "?", 1),
    ("fault_code", "<i4", 1),
)


def _validate(fields, schema):
    if (
        not isinstance(fields, dict)
        or type(fields.get("version")) is not int
        or fields["version"] != 1
        or fields.get("endian") != "le"
    ):
        raise ValueError("Inspire protocol must be version 1 little-endian")
    if tuple(k for k in fields if k not in ("version", "endian")) != tuple(s[0] for s in schema):
        raise ValueError("Inspire fields/order mismatch")
    for name, dtype, size in schema:
        value = fields[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != (size,):
            raise ValueError(f"Invalid Inspire field {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite Inspire field {name}")
    return fields


def validate_inspire_hand(fields):
    _validate(fields, HAND_SCHEMA)
    parse_pv(fields["pv"])
    for name in (
        "message_seq",
        "sample_generation",
        "left_source_epoch",
        "right_source_epoch",
        "left_source_timestamp_ns",
        "right_source_timestamp_ns",
    ):
        if fields[name][0] < 0:
            raise ValueError(f"Negative Inspire {name}")
    for side in ("left", "right"):
        if not 0 <= fields[f"{side}_tracking_state"][0] <= 3:
            raise ValueError("Invalid optical state")
        if np.any((fields[f"{side}_command"] < 0) | (fields[f"{side}_command"] > 1)):
            raise ValueError("Inspire command outside normalized limits")
    return fields


def validate_inspire_status(fields):
    _validate(fields, STATUS_SCHEMA)
    if not fields["pc2_session_id"].any() or fields["status_seq"][0] < 0:
        raise ValueError("Invalid PC2 identity/sequence")
    if fields["accepted_pv"].any():
        parse_pv(fields["accepted_pv"])
    if not 0 <= fields["bridge_state"][0] <= 5 or fields["last_applied_message_seq"][0] < -1:
        raise ValueError("Invalid PC2 state/applied sequence")
    if fields["fault_code"][0] < 0:
        raise ValueError("Invalid PC2 fault code")
    for side in ("left", "right"):
        measured, applied = fields[f"{side}_angle_act"], fields[f"{side}_applied"]
        if np.any((measured < 0) | (measured > 1000)) or np.any((applied < 0) | (applied > 1)):
            raise ValueError("PC2 measured/applied range violation")
        if fields["feedback_healthy"][0] and fields[f"{side}_err"].any():
            raise ValueError("Healthy PC2 status contains motor errors")
    return fields
