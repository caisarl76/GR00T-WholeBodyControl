"""Data-only local RPC contract. Mirrored into the native executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from typing import get_type_hints

PHASES = {
    "IDLE",
    "MANIPULATING",
    "PAUSED",
    "RESETTING",
    "WALKING",
    "TURNING",
    "COMPLETED",
    "INTERRUPTED",
    "FAULT",
}
ERROR_CODES = {
    "INVALID_REQUEST",
    "PROFILE_MISMATCH",
    "BUSY",
    "NOT_READY",
    "NOT_OWNER",
    "EXPIRED_LEASE",
    "STALE_RUNTIME",
    "REQUEST_CONFLICT",
    "UNAVAILABLE",
    "UNSUPPORTED_SKILL",
    "FAULT",
}
PARAMS = {
    "get_status": {},
    "observe": {},
    "heartbeat": {},
    "release_control": {},
    "claim_control": {"registry_sha256": str},
    "start_manipulation": {"skill_id": str},
    "pause_manipulation": {"execution_id": str},
    "cancel": {"execution_id": str},
    "reset_standing": {"execution_id": str, "open_hands": bool},
    "reset_ready": {"execution_id": str},
    "walk_for": {"direction": str, "duration_s": float, "speed_mps": float},
    "turn_by": {"angle_rad": float, "rate_rps": float},
}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _load(payload: bytes) -> dict:
    def reject(value):
        raise ValueError(f"Non-finite JSON: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject, object_pairs_hook=unique)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value


def validate_params(method: str, params: dict) -> None:
    schema = PARAMS.get(method)
    if schema is None or not isinstance(params, dict) or params.keys() != schema.keys():
        raise ValueError("Unknown method or invalid parameters")
    for key, kind in schema.items():
        value = params[key]
        if kind is float:
            if not finite_number(value):
                raise ValueError(f"{key} must be finite")
        elif type(value) is not kind:
            raise ValueError(f"Invalid {key}")
        elif (
            kind is str
            and not value
            and not (
                method in {"reset_standing", "reset_ready"} and key == "execution_id"
            )
        ):
            raise ValueError(f"Empty {key}")
    if method == "walk_for":
        if (
            params["direction"] not in {"forward", "backward", "left", "right"}
            or not 0 < params["duration_s"] <= 5
            or not 0 < params["speed_mps"] <= 0.2
        ):
            raise ValueError("Walk exceeds limits")
    if method == "turn_by" and not (
        0 < abs(params["angle_rad"]) <= math.pi / 4
        and 0 < params["rate_rps"] <= math.pi / 18
    ):
        raise ValueError("Turn exceeds limits")


@dataclass(frozen=True)
class SkillCall:
    skill_id: str
    params: dict[str, object]


@dataclass(frozen=True)
class Request:
    version: int
    request_id: str
    runtime_id: str | None
    session_id: str | None
    lease_id: str | None
    method: str
    params: dict[str, object]


@dataclass(frozen=True)
class Status:
    runtime_id: str
    phase: str
    skill_id: str | None
    execution_id: str | None
    inference_epoch: int
    controller_running: bool
    policy_ready: bool
    mode_requested: str | None
    planner_reference_active: bool | None
    hold_confirmed: bool
    telemetry_age_s: float | None
    telemetry_index: int | None
    observation_age_s: float | None
    frame_id: str | None
    registry_sha256: str
    policy_host: str
    policy_port: int
    checkpoint_expected: str
    owner_session_id: str | None
    reason: str | None
    locomotion_enabled: bool = False
    ready_return_enabled: bool = False
    right_hand_open: bool | None = None


@dataclass(frozen=True)
class Lease:
    runtime_id: str
    session_id: str
    lease_id: str
    expires_at: float


@dataclass(frozen=True)
class Execution:
    runtime_id: str
    execution_id: str
    skill_id: str
    phase: str
    inference_epoch: int
    started_at: float
    reason: str | None


@dataclass(frozen=True)
class ObservationSnapshot:
    runtime_id: str
    execution_id: str | None
    inference_epoch: int
    camera_key: str
    source_timestamp: float
    frame_id: str
    received_at: float
    age_s: float
    jpeg_rgb_b64: str


@dataclass(frozen=True)
class Response:
    request_id: str
    runtime_id: str
    result: Status | Lease | Execution | ObservationSnapshot | None = None
    error: dict[str, str] | None = None


def decode_request(payload: bytes) -> Request:
    data = _load(payload)
    if data.keys() != {f.name for f in fields(Request)}:
        raise ValueError("Invalid request fields")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported version")
    for key in ("request_id", "runtime_id", "session_id", "lease_id"):
        value = data[key]
        if value is not None and (
            type(value) is not str or not value or len(value) > 256
        ):
            raise ValueError(f"Invalid {key}")
    if data["request_id"] is None or type(data["method"]) is not str:
        raise ValueError("Missing request identity")
    validate_params(data["method"], data["params"])
    if data["method"] != "get_status" and not data["runtime_id"]:
        raise ValueError("Missing runtime ID")
    if data["method"] not in {"get_status", "observe"}:
        if not data["session_id"] or (
            data["method"] != "claim_control" and not data["lease_id"]
        ):
            raise ValueError("Missing ownership context")
    return Request(**data)


def _typed(cls, data):
    if not isinstance(data, dict) or data.keys() != {f.name for f in fields(cls)}:
        raise ValueError(f"Invalid {cls.__name__} fields")
    for name, hint in get_type_hints(cls).items():
        value = data[name]
        kinds = getattr(hint, "__args__", (hint,))
        if value is None and type(None) in kinds:
            continue
        if float in kinds:
            if finite_number(value):
                continue
            raise ValueError(f"Invalid finite {name}")
        if type(value) not in kinds:
            raise ValueError(f"Invalid {name} type")
    if "phase" in data and data["phase"] not in PHASES:
        raise ValueError("Invalid phase")
    return cls(**data)


def decode_response(payload: bytes) -> Response:
    data = _load(payload)
    if (
        data.keys() != {"request_id", "runtime_id", "result", "error"}
        or type(data["request_id"]) is not str
        or type(data["runtime_id"]) is not str
    ):
        raise ValueError("Invalid response")
    result, error = data["result"], data["error"]
    if (result is None) == (error is None):
        raise ValueError("Expected one result or error")
    if error is not None:
        if (
            not isinstance(error, dict)
            or error.keys() != {"code", "message"}
            or error["code"] not in ERROR_CODES
            or type(error["message"]) is not str
        ):
            raise ValueError("Invalid error")
    else:
        if not isinstance(result, dict):
            raise ValueError("Invalid result")
        cls = next(
            (
                c
                for key, c in [
                    ("policy_ready", Status),
                    ("jpeg_rgb_b64", ObservationSnapshot),
                    ("lease_id", Lease),
                    ("started_at", Execution),
                ]
                if key in result
            ),
            None,
        )
        if cls is None:
            raise ValueError("Unknown result")
        result = _typed(cls, result)
        if result.runtime_id != data["runtime_id"]:
            raise ValueError("Result runtime mismatch")
    return Response(data["request_id"], data["runtime_id"], result, error)


def encode_request(request: Request) -> bytes:
    payload = json.dumps(asdict(request), allow_nan=False).encode()
    decode_request(payload)
    return payload


def encode_response(response: Response) -> bytes:
    payload = json.dumps(asdict(response), allow_nan=False).encode()
    decode_response(payload)
    return payload
