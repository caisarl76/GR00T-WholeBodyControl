import importlib
import importlib.util

import numpy as np


def cache():
    assert importlib.util.find_spec("gear_sonic.utils.inference.observation_snapshot"), "Snapshot cache missing"
    cls = importlib.import_module("gear_sonic.utils.inference.observation_snapshot").ObservationSnapshotCache
    return cls(lambda: None, "ego_view")


def frame(timestamp):
    return {"timestamps": {"ego_view": timestamp}, "images": {"ego_view": np.zeros((12, 16, 3), np.uint8)}}


def test_republished_camera_timestamp_ages():
    c = cache()
    c.ingest(frame(10.0), 1.0)
    first = c.latest(1.1)
    c.ingest(frame(10.0), 1.4)
    assert c.latest(1.4)["frame_id"] == first["frame_id"]
    assert c.latest(1.4)["received_at"] == 1.0
    assert c.latest(1.6) is None


def test_missing_ego_view_is_unavailable():
    c = cache()
    c.ingest(frame(10.0), 1.0)
    c.ingest({"timestamps": {}, "images": {}}, 1.1)
    assert c.latest(1.1) is None


def test_regressed_timestamp_clears_old_stream():
    c = cache()
    c.ingest(frame(10.0), 1.0)
    old = c.latest(1.0)["frame_id"]
    c.ingest(frame(1.0), 1.1)
    assert c.latest(1.1) is None
    c.ingest(frame(2.0), 1.2)
    assert c.latest(1.2)["frame_id"] != old


def test_camera_owned_by_one_worker_thread():
    import threading
    import time

    from gear_sonic.utils.inference.observation_snapshot import ObservationSnapshotCache

    calls = []

    class Camera:
        def __init__(self):
            calls.append(("create", threading.get_ident()))

        def read(self):
            calls.append(("read", threading.get_ident()))
            return frame(time.monotonic())

        def close(self):
            calls.append(("close", threading.get_ident()))

    c = ObservationSnapshotCache(Camera, "ego_view")
    c.start()
    deadline = time.monotonic() + 1
    while c.latest(time.monotonic()) is None and time.monotonic() < deadline:
        time.sleep(0.01)
    c.close()
    assert {name for name, _ in calls} == {"create", "read", "close"}
    assert len({tid for _, tid in calls}) == 1
    assert calls[0][1] != threading.get_ident()
