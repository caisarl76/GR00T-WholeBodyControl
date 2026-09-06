import numpy as np

from gear_sonic.scripts.run_camera_viewer import _prepare_display_tiles


def test_prepare_display_tiles_skips_d435_raw_depth_payload():
    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth_payload = {
        "data": b"\x00" * (4 * 6 * 2),
        "shape": [4, 6],
        "dtype": "uint16",
    }

    tiles = _prepare_display_tiles(
        {
            "ego_view": color,
            "ego_view_depth": depth_payload,
        },
        camera_names=["ego_view", "ego_view_depth"],
        max_display_width=640,
        is_recording=False,
    )

    assert len(tiles) == 1
    assert tiles[0].shape == color.shape
