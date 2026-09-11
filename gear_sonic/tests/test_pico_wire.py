"""Strict binary round trips and full-sized manager/feedback header budgets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.teleop.pico_hand_runtime import PicoPublisher
from gear_sonic.utils.teleop.pico_inspire_protocol import (
    HAND_SCHEMA,
    STATUS_SCHEMA,
    validate_inspire_hand,
    validate_inspire_status,
)
from gear_sonic.utils.teleop.pico_recording import make_pv
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_planner_message,
    pack_pose_message,
    unpack_pose_message,
)

PV = make_pv(b"m" * 16, 2, 1)


def wire(header, payload=b"", topic="pose"):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    assert len(encoded) <= HEADER_SIZE
    return topic.encode() + encoded.ljust(HEADER_SIZE, b"\0") + payload


def header(fields):
    return {"v": 1, "endian": "le", "count": 1, "fields": fields}


def header_size(raw, topic):
    return len(raw[len(topic) : len(topic) + HEADER_SIZE].split(b"\0", 1)[0])


def test_uint8_and_explicit_big_endian_input_are_native_little_endian_on_wire():
    values = {
        "pv": PV,
        "be_float": np.array([1.25, -2.5], dtype=">f4"),
        "be_int": np.array([2**40, -8], dtype=">i8"),
        "strided": np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
        "boolean": np.array([True, False]),
    }
    raw = pack_pose_message(values)
    result = unpack_pose_message(raw)
    assert result["pv"].dtype == np.uint8
    assert result["be_float"].dtype == np.dtype("<f4")
    for name, value in values.items():
        np.testing.assert_array_equal(result[name], value)
    assert raw[4 + HEADER_SIZE : 4 + HEADER_SIZE + 28] == PV.tobytes()
    assert raw[4 + HEADER_SIZE + 28 : 4 + HEADER_SIZE + 36] == np.array([1.25, -2.5], "<f4").tobytes()


@pytest.mark.parametrize(
    "bad_header,payload",
    [
        ([], b""),
        (header([None]), b""),
        (header([{"name": "q", "dtype": [], "shape": [1]}]), b""),
        (header([{"name": "q", "dtype": "uint8", "shape": [1]}]), b"\0"),
        (header([{"name": "q", "dtype": "i32", "shape": [-1]}]), b""),
        (header([{"name": "q", "dtype": "i32", "shape": [True]}]), b""),
        (header([{"name": "q", "dtype": "i32", "shape": [1.0]}]), b""),
        (header([{"name": "q", "dtype": "i32", "shape": [0, 2**100]}]), b""),
        (header([{"name": "q", "dtype": "i32", "shape": [1] * 33}]), b""),
        (header([{"name": "q", "dtype": "i32", "shape": [1]}]), b"\0"),
        (header([{"name": "q", "dtype": "bool", "shape": [1]}]), b"\2"),
        (header([{"name": "version", "dtype": "u8", "shape": [1]}]), b"\0"),
        (header([{"name": "q", "dtype": "u8", "shape": [1]}] * 2), b"\0\0"),
        ({**header([]), "endian": "be"}, b""),
        (header([]), b"extra"),
    ],
)
def test_decoder_rejects_invalid_shapes_types_duplicate_names_and_lengths(bad_header, payload):
    with pytest.raises(ValueError):
        unpack_pose_message(wire(bad_header, payload))


@pytest.mark.parametrize(
    "value",
    [
        np.array([1], np.uint16),
        np.array([1], np.int16),
        np.array([1.0], np.float16),
        np.array(["string"]),
        np.array([object()]),
    ],
)
def test_sender_rejects_unsupported_dtype_without_coercing(value):
    with pytest.raises(ValueError):
        pack_pose_message({"bad": value})


def schema_fields(schema):
    fields = {name: np.zeros(size, dtype) for name, dtype, size in schema}
    if schema == HAND_SCHEMA:
        fields["pv"][:] = PV
    else:
        fields["pc2_session_id"][:] = 1
        fields["accepted_pv"][:] = PV
        fields["last_applied_message_seq"][:] = -1
    return fields


@pytest.mark.parametrize(
    "schema,topic,validate",
    [
        (HAND_SCHEMA, "inspire_hand", validate_inspire_hand),
        (STATUS_SCHEMA, "inspire_hand_status", validate_inspire_status),
    ],
)
def test_exact_protocol_order_dtype_shape_and_header_budget(schema, topic, validate):
    values = schema_fields(schema)
    raw = pack_pose_message(values, topic, 1)
    assert header_size(raw, topic) <= 1200
    decoded = unpack_pose_message(raw, topic)
    validate(decoded)
    assert tuple(k for k in decoded if k not in ("version", "endian")) == tuple(row[0] for row in schema)
    assert len(raw) == len(topic) + HEADER_SIZE + sum(
        np.dtype(dtype).itemsize * count for _, dtype, count in schema
    )
    for bad in (
        {**decoded, "extra": np.zeros(1, np.uint8)},
        {**decoded, "version": True},
        {**decoded, "version": 2},
        {**decoded, "endian": "be"},
        {**decoded, schema[0][0]: decoded[schema[0][0]].astype(np.int32)},
        {**decoded, schema[0][0]: decoded[schema[0][0]][:-1]},
        dict(reversed(list(decoded.items()))),
    ):
        with pytest.raises(ValueError):
            validate(bad)


def test_protocol_range_and_health_contradictions():
    fields = {**schema_fields(HAND_SCHEMA), "version": 1, "endian": "le"}
    for name, value in (
        ("left_command", np.full(6, np.nan, np.float32)),
        ("right_command", np.full(6, 1.01, np.float32)),
        ("left_tracking_state", np.array([4], np.int32)),
        ("left_source_epoch", np.array([-1], np.int64)),
    ):
        with pytest.raises(ValueError):
            validate_inspire_hand({**fields, name: value})
    fields = {**schema_fields(STATUS_SCHEMA), "version": 1, "endian": "le"}
    fields["feedback_healthy"][0] = True
    fields["left_err"][3] = 1
    with pytest.raises(ValueError, match="errors"):
        validate_inspire_status(fields)


def test_production_pose_planner_and_diagnostic_headers_fit_1200_bytes():
    # Shapes and field names match PoseStreamer.send_pose_data, including its windowed body.
    data = {
        "smpl_pose": np.zeros((10, 24, 3)),
        "smpl_joints": np.zeros((10, 24, 3)),
        "body_quat_w": np.zeros((10, 4)),
        "joint_pos": np.zeros((10, 29)),
        "joint_vel": np.zeros((10, 29)),
        "vr_position": np.zeros(9),
        "vr_orientation": np.zeros(12),
        "frame_index": np.arange(10, dtype=np.int64),
    }
    for field in ("left_trigger", "right_trigger", "left_grip", "right_grip", "pico_dt", "pico_fps"):
        data[field] = np.zeros(1, np.float32)
    for field in ("timestamp_realtime", "timestamp_monotonic"):
        data[field] = np.zeros(1, np.float64)
    for side in ("left", "right"):
        data[f"{side}_hand_joints"] = np.zeros(7, np.float32)
    for field in ("toggle_data_collection", "toggle_data_abort"):
        data[field] = np.zeros(1, bool)
    data["heading_increment"] = np.zeros(1, np.float32)
    sent = []
    held = [np.full(7, 0.25, np.float32), np.full(7, 0.5, np.float32)]
    publisher = PicoPublisher(
        SimpleNamespace(send=sent.append), "optical", "dex3", SimpleNamespace(commands=lambda: held)
    )
    publisher.pv, publisher.generation = PV, 17
    publisher.send(pack_pose_message(data))
    assert header_size(sent[-1], "pose") <= 1200
    decoded = unpack_pose_message(sent[-1])
    np.testing.assert_array_equal(decoded["left_hand_joints"], held[0])
    assert decoded["sample_generation"][0] == 17
    assert "toggle_data_collection" not in decoded
    assert publisher.body_sent

    planner = build_planner_message(
        0,
        [0.0] * 3,
        [1.0, 0.0, 0.0],
        upper_body_position=[0.0] * 17,
        upper_body_velocity=[0.0] * 17,
        left_hand_position=[0.0] * 7,
        right_hand_position=[0.0] * 7,
        vr_3pt_position=[0.0] * 9,
        vr_3pt_orientation=[0.0] * 12,
        vr_3pt_compliance=[0.0] * 3,
    )
    publisher.send(planner)
    assert header_size(sent[-1], "planner") <= 1200
    decoded = unpack_pose_message(sent[-1], "planner")
    np.testing.assert_array_equal(decoded["right_hand_joints"], held[1])

    diagnostic = {
        "pv": PV,
        "sample_generation": np.zeros(1, np.int64),
        "hand_profile": np.zeros(1, np.int32),
        "body_sent": np.ones(1, bool),
        "hand_input": np.zeros(1, np.int32),
    }
    for side in ("left", "right"):
        for field, dtype, shape in (
            ("state", np.int32, 1),
            ("source_epoch", np.int64, 1),
            ("source_timestamp_ns", np.int64, 1),
            ("valid", bool, 1),
            ("command", np.float32, 7),
        ):
            diagnostic[f"{side}_{field}"] = np.zeros(shape, dtype)
    assert header_size(pack_pose_message(diagnostic, "hand_tracking", 1), "hand_tracking") <= 1200
