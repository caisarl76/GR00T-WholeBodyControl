"""Strict decoder for GEAR-SONIC's topic-prefixed binary ZMQ messages."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

_DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


def _decode_header(header_bytes: bytes) -> dict[str, Any]:
    encoded = header_bytes.split(b"\x00", maxsplit=1)[0]
    if not encoded:
        raise ValueError("ZMQ message header is empty")
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ZMQ message header is not valid UTF-8 JSON") from error
    if not isinstance(header, dict):
        raise ValueError("ZMQ message header must be a JSON object")
    if header.get("endian", "le") != "le":
        raise ValueError("only little-endian ZMQ payloads are supported")
    if not isinstance(header.get("fields", []), list):
        raise ValueError("ZMQ message fields must be a list")
    return header


def _field_layout(field: object) -> tuple[str, np.dtype, tuple[int, ...], int]:
    if not isinstance(field, dict):
        raise ValueError("each ZMQ field descriptor must be an object")
    name = field.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("each ZMQ field must have a non-empty string name")
    dtype_name = field.get("dtype")
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported ZMQ field dtype for {name!r}: {dtype_name!r}")
    raw_shape = field.get("shape")
    if not isinstance(raw_shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in raw_shape
    ):
        raise ValueError(f"invalid ZMQ field shape for {name!r}: {raw_shape!r}")
    shape = tuple(raw_shape)
    dtype = _DTYPES[dtype_name]
    n_bytes = math.prod(shape) * dtype.itemsize
    return name, dtype, shape, n_bytes


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict[str, Any]:
    """Decode one message while enforcing its complete self-described layout.

    Wire format: ``[topic][1280-byte JSON header][concatenated field payloads]``.
    The historical function name is retained because the exporter consumes the
    same function for ``pose``, ``planner``, and ``manager_state`` topics.
    """

    if not isinstance(packed_data, bytes):
        raise TypeError("packed_data must be bytes")
    if not isinstance(topic, str) or not topic:
        raise ValueError("topic must be a non-empty string")
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic {topic!r}")

    payload_start = len(topic_bytes) + HEADER_SIZE
    if len(packed_data) < payload_start:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {payload_start}")
    header = _decode_header(packed_data[len(topic_bytes) : payload_start])

    result: dict[str, Any] = {
        "version": header.get("v", 0),
        "endian": header.get("endian", "le"),
    }
    layouts = []
    expected_payload_size = 0
    for field in header.get("fields", []):
        layout = _field_layout(field)
        name = layout[0]
        if name in result:
            raise ValueError(f"duplicate ZMQ field name: {name!r}")
        result[name] = None
        layouts.append(layout)
        expected_payload_size += layout[3]

    actual_payload_size = len(packed_data) - payload_start
    if actual_payload_size != expected_payload_size:
        raise ValueError(
            "ZMQ payload length does not match header: "
            f"expected {expected_payload_size}, got {actual_payload_size}"
        )

    offset = payload_start
    for name, dtype, shape, n_bytes in layouts:
        result[name] = np.frombuffer(packed_data[offset : offset + n_bytes], dtype=dtype).reshape(shape).copy()
        offset += n_bytes
    return result
