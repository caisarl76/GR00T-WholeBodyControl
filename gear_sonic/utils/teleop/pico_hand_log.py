"""Bounded NPZ capture of already-read hand samples; never reads the XR SDK.

Required per-tick arrays are documented by INPUT_SCHEMA. Missing binding
snapshots have snapshot_present=False; NaN wrists/measurements mean unavailable.
Writes use one worker with at most two queued shards. A full/failing writer
raises an error for the caller to report and disable logging, not stop motion.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

INPUT_SCHEMA = {
    "poses": (np.float64, (2, 26, 7)),
    "flags": (np.uint64, (2, 26)),
    "radius": (np.float64, (2, 26)),
    "scale": (np.float64, (2,)),
    "active": (np.int32, (2,)),
    "source_timestamp_ns": (np.int64, (2,)),
    "binding_generation": (np.uint64, (2,)),
    "body_wrist": (np.float64, (2, 3)),
    "tick_ns": (np.int64, ()),
}


def validate_capture(arrays, profile):
    """Validate storage schema; invalid sensor numeric values remain replayable."""
    if profile not in ("dex3", "inspire_ftp"):
        raise ValueError("unknown hand profile")
    tick = arrays.get("tick_ns")
    if not isinstance(tick, np.ndarray) or tick.ndim != 1 or len(tick) == 0:
        raise ValueError("tick_ns must contain at least one tick")
    count = len(tick)
    schema = {**INPUT_SCHEMA, "measured": (np.float64, (2, 7 if profile == "dex3" else 6))}
    for key, (dtype, suffix) in schema.items():
        value = arrays.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != (count, *suffix):
            raise ValueError(f"invalid capture field {key}: expected {np.dtype(dtype)} {(count, *suffix)}")
    if np.any(tick < 0) or any(int(b) <= int(a) for a, b in zip(tick, tick[1:])):
        raise ValueError("capture ticks must be strictly increasing nonnegative monotonic ns")
    for name, dtype, suffix in (
        ("snapshot_present", np.bool_, (2,)),
        ("enabled", np.bool_, ()),
        ("body_sent", np.bool_, ()),
        ("hand_sent", np.bool_, (2,)),
        ("pv", np.uint8, (28,)),
        ("sample_generation", np.int64, ()),
        ("timestamp_source", np.int8, (2,)),
    ):
        if name in arrays:
            value = arrays[name]
            if value.dtype != np.dtype(dtype) or value.shape != (count, *suffix):
                raise ValueError(f"invalid optional capture field {name}")
    if "timestamp_source" in arrays and not np.isin(arrays["timestamp_source"], (0, 1)).all():
        raise ValueError("invalid hand timestamp source")
    if "pose_label" in arrays and (
        arrays["pose_label"].shape != (count,) or arrays["pose_label"].dtype.kind != "U"
    ):
        raise ValueError("pose_label must be a unicode string per tick")
    return count


def snapshot_at(arrays, index, side):
    if "snapshot_present" in arrays and not arrays["snapshot_present"][index, side]:
        return None
    return {
        "pose": arrays["poses"][index, side],
        "location_flags": arrays["flags"][index, side],
        "radius": arrays["radius"][index, side],
        "scale": arrays["scale"][index, side],
        "is_active": arrays["active"][index, side],
        "source_timestamp_ns": arrays["source_timestamp_ns"][index, side],
        "binding_generation": arrays["binding_generation"][index, side],
        "timestamp_source": arrays["timestamp_source"][index, side] if "timestamp_source" in arrays else 0,
    }


class HandCaptureLog:
    def __init__(self, output_dir, profile, shard_size=250, *, source_kind="runtime_capture"):
        if profile not in ("dex3", "inspire_ftp") or source_kind not in ("runtime_capture", "synthetic"):
            raise ValueError("invalid capture profile/source kind")
        if not isinstance(shard_size, int) or not 1 <= shard_size <= 250:
            raise ValueError("shard_size must be 1..250")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if list(self.output_dir.glob("capture_*.npz*")) or (self.output_dir / "transitions.jsonl").exists():
            raise FileExistsError("use a new capture directory to preserve previous artifacts")
        self.profile, self.shard_size, self.source_kind = profile, shard_size, source_kind
        self._rows, self._events, self._pending = [], [], []
        self._last_tick = None
        self._last_states = [None, None]
        self._index = 0
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hand-capture")
        self._closed = False

    def record(
        self,
        tick_ns,
        snapshots,
        body_wrists,
        measured,
        outputs=None,
        enabled=True,
        *,
        pose_label="",
        body_sent=None,
        hand_sent=None,
        pv=None,
        sample_generation=None,
    ):
        if self._closed:
            raise RuntimeError("capture is closed")
        if type(tick_ns) is not int or tick_ns < 0 or (self._last_tick is not None and tick_ns <= self._last_tick):
            raise ValueError("record ticks must be strictly increasing")
        if len(snapshots) != 2 or len(body_wrists) != 2 or len(measured) != 2:
            raise ValueError("capture requires exactly two sides")
        self._check_writer()
        size = 7 if self.profile == "dex3" else 6
        row = {key: np.zeros(shape, dtype=dtype) for key, (dtype, shape) in INPUT_SCHEMA.items()}
        row["poses"].fill(np.nan)
        row["radius"].fill(np.nan)
        row["body_wrist"].fill(np.nan)
        row["measured"] = np.full((2, size), np.nan, np.float64)
        row["snapshot_present"] = np.zeros(2, np.bool_)
        row["timestamp_source"] = np.zeros(2, np.int8)
        row["tick_ns"] = np.asarray(tick_ns, np.int64)
        row["enabled"] = np.asarray(enabled, np.bool_)
        row["pose_label"] = np.asarray(pose_label, dtype="U64")
        mapping = {
            "poses": "pose",
            "flags": "location_flags",
            "radius": "radius",
            "scale": "scale",
            "active": "is_active",
            "source_timestamp_ns": "source_timestamp_ns",
            "binding_generation": "binding_generation",
        }
        for side in range(2):
            snapshot = snapshots[side]
            if snapshot is not None:
                clock_source = snapshot.get("timestamp_source", 0)
                if (
                    isinstance(clock_source, (bool, np.bool_))
                    or not isinstance(clock_source, (int, np.integer))
                    or clock_source not in (0, 1)
                ):
                    raise ValueError("invalid hand timestamp source")
                row["timestamp_source"][side] = clock_source
                for target, source in mapping.items():
                    value = np.asarray(snapshot[source])
                    if value.shape != row[target][side].shape:
                        raise ValueError(f"invalid stored snapshot shape {source}")
                    if source in ("pose", "location_flags", "radius") and value.dtype != row[target].dtype:
                        raise ValueError(f"invalid stored snapshot dtype {source}")
                    row[target][side] = value
                row["snapshot_present"][side] = True
            if body_wrists[side] is not None:
                row["body_wrist"][side] = body_wrists[side]
            if measured[side] is not None:
                row["measured"][side] = measured[side]
        if body_sent is not None:
            row["body_sent"] = np.asarray(body_sent, np.bool_)
        if hand_sent is not None:
            row["hand_sent"] = np.asarray(hand_sent, np.bool_)
        if pv is not None:
            row["pv"] = np.asarray(pv, dtype=np.uint8).copy()
        if sample_generation is not None:
            row["sample_generation"] = np.asarray(sample_generation, dtype=np.int64)
        if outputs is not None:
            if len(outputs) != 2:
                raise ValueError("capture outputs must contain both sides")
            row["captured_command"] = np.full((2, size), np.nan, np.float32)
            row["captured_target"] = np.full((2, size), np.nan, np.float32)
            row["captured_state"] = np.empty(2, np.int32)
            row["captured_reason"] = np.empty(2, np.int32)
            for side, output in enumerate(outputs):
                if output is None:
                    row["captured_state"][side] = row["captured_reason"][side] = -1
                    continue
                row["captured_state"][side], row["captured_reason"][side] = int(output.state), int(output.reason)
                for name in ("command", "target"):
                    value = getattr(output, name)
                    if value is not None:
                        row[f"captured_{name}"][side] = value
                state = (int(output.state), int(output.reason))
                if state != self._last_states[side]:
                    self._events.append(
                        {
                            "tick_ns": tick_ns,
                            "side": ("left", "right")[side],
                            "state": output.state.name,
                            "reason": output.reason.name,
                        }
                    )
                    self._last_states[side] = state
        if self._rows and row.keys() != self._rows[0].keys():
            raise ValueError("optional capture evidence must be present consistently within each shard")
        self._rows.append(row)
        self._last_tick = tick_ns
        if len(self._rows) >= self.shard_size:
            self.flush()

    def _check_writer(self):
        remaining = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        self._pending = remaining

    def flush(self):
        self._check_writer()
        if not self._rows:
            return
        if len(self._pending) >= 2:
            raise RuntimeError("capture writer queue full; disable logging and preserve motion")
        arrays = {key: np.stack([row[key] for row in self._rows]) for key in self._rows[0]}
        validate_capture(arrays, self.profile)
        arrays.update(
            schema_version=np.array([1], np.int32),
            profile=np.array([self.profile]),
            source_kind=np.array([self.source_kind]),
        )
        self._pending.append(self._worker.submit(self._write, self._index, arrays, self._events))
        self._index += 1
        self._rows, self._events = [], []

    def _write(self, index, arrays, events):
        path = self.output_dir / f"capture_{index:06d}.npz"
        with path.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez(stream, **arrays)
        path.with_suffix(".npz.tmp").replace(path)
        with (self.output_dir / "transitions.jsonl").open("a") as stream:
            for event in events:
                stream.write(json.dumps(event) + "\n")

    def close(self):
        if self._closed:
            return
        try:
            for future in self._pending:
                future.result()
            self.flush()
            for future in self._pending:
                future.result()
        finally:
            self._closed = True
            self._worker.shutdown(wait=True)
