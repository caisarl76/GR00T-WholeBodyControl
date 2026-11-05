import msgpack
import msgpack_numpy as m
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema

INTR = {
    "fx": 600.0,
    "fy": 600.0,
    "ppx": 320.0,
    "ppy": 240.0,
    "width": 640,
    "height": 480,
    "depth_scale": 0.001,
}


def test_intrinsics_survive_serialization():
    msg = ImageMessageSchema(
        timestamps={"ego_view": 1.0},
        images={"ego_view": np.zeros((4, 4, 3), np.uint8)},
        intrinsics={"ego_view": INTR},
    )
    out = ImageMessageSchema.deserialize(msg.serialize())
    assert out.intrinsics["ego_view"]["fx"] == 600.0


def test_no_intrinsics_key_on_wire_without_depth():
    msg = ImageMessageSchema(timestamps={}, images={"ego_view": np.zeros((4, 4, 3), np.uint8)})
    assert "intrinsics" not in msg.serialize()


def test_uint16_depth_is_lossless_over_msgpack():
    depth = np.full((4, 4), 1500, np.uint16)
    schema = ImageMessageSchema(timestamps={}, images={"ego_view_depth": depth})
    wire = msgpack.packb(schema.serialize(), use_bin_type=True)
    out = ImageMessageSchema.deserialize(msgpack.unpackb(wire, object_hook=m.decode))
    assert out.images["ego_view_depth"].dtype == np.uint16
    assert (out.images["ego_view_depth"] == 1500).all()


def test_uint8_ir_round_trips_as_jpeg():
    ir = np.tile(np.arange(0, 256, 16, dtype=np.uint8), (8, 1))
    schema = ImageMessageSchema(timestamps={}, images={"ego_view_ir": ir})
    wire = msgpack.packb(schema.serialize(), use_bin_type=True)
    out = ImageMessageSchema.deserialize(msgpack.unpackb(wire, object_hook=m.decode))
    # Decoded with IMREAD_COLOR: grayscale comes back replicated across 3 channels.
    assert out.images["ego_view_ir"].shape == (8, 16, 3)
    assert np.abs(out.images["ego_view_ir"][..., 0].astype(int) - ir).max() <= 8
