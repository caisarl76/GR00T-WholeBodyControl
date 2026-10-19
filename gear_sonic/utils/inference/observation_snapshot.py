"""Camera freshness and background JPEG snapshots for the local harness."""

from __future__ import annotations

import base64
import io
import math
import threading
import time
import uuid

import numpy as np
from PIL import Image


class SensorFreshness:
    def __init__(self, max_age_s=0.5):
        self.max_age_s = max_age_s
        self.timestamp = None
        self.received_at = None
        self.generation = uuid.uuid4().hex

    def capture(self, camera_message, now):
        try:
            stamp = float(camera_message["timestamps"]["ego_view"])
            if not math.isfinite(stamp) or camera_message["images"].get("ego_view") is None:
                raise ValueError("Invalid camera frame")
        except (TypeError, KeyError, ValueError):
            self.received_at = None
            return None
        if self.timestamp is not None and stamp < self.timestamp:
            self.generation = uuid.uuid4().hex
            self.timestamp, self.received_at = stamp, None
            return None
        if self.timestamp != stamp:
            self.timestamp, self.received_at = stamp, now
        if self.received_at is None or not 0 <= now - self.received_at <= self.max_age_s:
            return None
        return self.received_at


class ObservationSnapshotCache:
    def __init__(self, camera_factory, camera_key, max_age_s=0.5):
        if camera_key != "ego_view":
            raise ValueError("Expected ego_view")
        self.camera_factory, self.camera_key, self.max_age_s = camera_factory, camera_key, max_age_s
        self.freshness = SensorFreshness(max_age_s)
        self._snapshot = None
        self._camera = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def ingest(self, message, now):
        captured = self.freshness.capture(message, now)
        if captured is None:
            with self._lock:
                self._snapshot = None
                self._camera = None
            return
        frame_id = f"{self.freshness.generation}:{self.freshness.timestamp}"
        with self._lock:
            if self._snapshot and self._snapshot["frame_id"] == frame_id:
                return
        try:
            array = np.asarray(message["images"][self.camera_key])
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError("Invalid RGB frame")
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="JPEG", quality=85)
            snapshot = dict(
                camera_key=self.camera_key,
                source_timestamp=self.freshness.timestamp,
                frame_id=frame_id,
                received_at=captured,
                age_s=0.0,
                jpeg_rgb_b64=base64.b64encode(stream.getvalue()).decode("ascii"),
            )
        except (TypeError, ValueError, OSError):
            snapshot = None
        with self._lock:
            self._snapshot = snapshot
            self._camera = (
                None
                if snapshot is None
                else (
                    {
                        "timestamps": {self.camera_key: snapshot["source_timestamp"]},
                        "images": {self.camera_key: array.copy()},
                    },
                    captured,
                )
            )

    def latest(self, now):
        with self._lock:
            snapshot = self._snapshot
        # A concurrent local capture can follow the caller's sampled clock.
        if snapshot is None or max(0.0, now - snapshot["received_at"]) > self.max_age_s:
            return None
        return {**snapshot, "age_s": max(0.0, now - snapshot["received_at"])}

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Snapshot worker already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest_camera(self, now):
        with self._lock:
            camera = self._camera
        if camera is None or max(0.0, now - camera[1]) > self.max_age_s:
            return None
        message, received_at = camera
        return {
            "timestamps": dict(message["timestamps"]),
            "images": {self.camera_key: message["images"][self.camera_key].copy()},
        }, received_at

    def _run(self):
        camera = None
        try:
            camera = self.camera_factory()  # Socket is created/read/closed on this thread.
            while not self._stop.is_set():
                try:
                    self.ingest(camera.read(), time.monotonic())
                except Exception:
                    with self._lock:
                        self._snapshot = None
                        self._camera = None
                self._stop.wait(0.02)
        finally:
            if camera is not None:
                close = getattr(camera, "close", None) or getattr(camera, "close_client", None)
                if close:
                    close()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(1.0)
