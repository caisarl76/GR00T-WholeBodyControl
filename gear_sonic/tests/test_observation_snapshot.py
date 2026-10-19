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


def test_worker_uses_continuous_camera_capture_time():
    c = cache()
    c.ingest(frame(100.0), 0.0)
    c.ingest(frame(101.0), 0.1)  # Arrives while inference is busy.
    message, received = c.latest_camera(0.2)
    assert message["timestamps"]["ego_view"] == 101.0 and received == 0.1
    assert c.latest_camera(0.75) is None  # Cannot rebase frozen frame.


def test_raw_frame_and_timestamp_stay_paired_across_camera_updates():
    c = cache()
    first = frame(100.0)
    first["images"]["ego_view"][:] = 11
    c.ingest(first, 1.0)
    old_message, old_received = c.latest_camera(1.01)
    second = frame(101.0)
    second["images"]["ego_view"][:] = 22
    c.ingest(second, 1.02)
    message, received = c.latest_camera(1.025)
    assert message["timestamps"]["ego_view"] == 101.0 and received == 1.02
    assert np.all(message["images"]["ego_view"] == 22)
    assert old_message["timestamps"]["ego_view"] == 100.0 and old_received == 1.0
    assert np.all(old_message["images"]["ego_view"] == 11)
    message["images"]["ego_view"][:] = 99
    assert np.all(c.latest_camera(1.03)[0]["images"]["ego_view"] == 22)


def test_cache_update_after_caller_clock_sample_remains_fresh():
    c = cache()
    sampled_at = 1.0
    c.ingest(frame(101.0), 1.007)  # Concurrent update after caller sampled now.
    assert c.latest(sampled_at)["age_s"] == 0.0
    assert c.latest_camera(sampled_at)[1] == 1.007
    assert c.latest(1.508) is None
