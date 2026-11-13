"""Synchronized gate recorder, sync checker and source-time aligner.

Every analysis aligns on *source* time (the publisher's CLOCK_MONOTONIC stamp).
Receipt time is only used to measure skew. All times in ``check_sync`` /
``align`` windows are seconds on that monotonic clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Iterable, Mapping

import msgpack
import numpy as np
import zmq

# ---- Sync contract -------------------------------------------------------
SKEW_MIN_MS = -1.0
SKEW_P99_MAX_MS = 20.0
SKEW_MAX_MS = 50.0
GAP_MIN_S = 0.1  # gap limit = max(GAP_PERIODS * period, GAP_MIN_S)
GAP_PERIODS = 3.0
ABSENT_END_SLACK_S = 1.0  # restart allowance after an absence interval
BASELINE_S = 1.0
RTF_MIN = 0.9
INDEPENDENT = ("sim_gt", "g1_debug", "markers")  # never allowed an absence


@dataclass(frozen=True)
class StreamReq:
    period: float
    absent: tuple[tuple[str, str], ...] = ()


@dataclass
class SyncReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    skew_p99_ms: dict = field(default_factory=dict)
    max_gap_ms: dict = field(default_factory=dict)
    rtf: float | None = None


# ---- Recorder ------------------------------------------------------------
def _decode(name: str, topic: str, raw: bytes) -> dict:
    if name == "sim_gt":
        return msgpack.unpackb(raw[len(topic) :], raw=False)
    if name == "g1_debug":
        from gear_sonic.utils.data_collection.zmq_state_subscriber import _unpack_msgpack_zmq

        return _unpack_msgpack_zmq(raw, topic)
    from gear_sonic.scripts.run_data_exporter import unpack_pose_message

    return unpack_pose_message(raw, topic)


def _scalar(x) -> float:
    return float(np.ravel(x)[0])


def _src_ns(msg: dict) -> int:
    """Source time of a decoded message in ns, or -1 if the stream has none."""
    if "t_mono_ns" in msg:
        return int(_scalar(msg["t_mono_ns"]))
    if "timestamp_monotonic" in msg:  # pose: seconds
        return int(round(_scalar(msg["timestamp_monotonic"]) * 1e9))
    return -1


class GateRecorder:
    """Records ZMQ streams (and local pushes) with receipt + source timestamps."""

    def __init__(self, streams: dict[str, tuple[str, int, str]]):
        self._streams = streams
        self._ctx = zmq.Context.instance()
        self._socks: dict[str, zmq.Socket] = {}
        self._d: dict[str, dict[str, list]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Exception | None = None
        for name, (host, port, topic) in streams.items():
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.RCVHWM, 0)  # keep every message
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt_string(zmq.SUBSCRIBE, topic)
            sock.connect(f"tcp://{host}:{port}")
            self._socks[name] = sock
            self._d[name] = {"recv": [], "src": [], "raw": [], "t_sim": []}
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        if self._socks:
            self._thread.start()

    def _recv_loop(self) -> None:
        # Receipt time is stamped here, on arrival, independent of the caller's loop.
        poller = zmq.Poller()
        for sock in self._socks.values():
            poller.register(sock, zmq.POLLIN)
        by_sock = {s: n for n, s in self._socks.items()}
        try:
            while not self._stop.is_set():
                for sock, _ in poller.poll(1):
                    name = by_sock[sock]
                    while True:
                        try:
                            raw = sock.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        recv = time.monotonic_ns()
                        msg = _decode(name, self._streams[name][2], raw)
                        with self._lock:
                            self._append(name, msg, _src_ns(msg), recv)
        except Exception as e:  # never die silently: save()/close() re-raise
            self._error = e

    def _raise_if_died(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"receive thread died: {self._error!r}") from self._error

    def poll_all(self) -> None:
        """Kept for API compatibility: receiving runs in a background thread,
        so this is a no-op. Receipt stamps never depend on how often it is called."""

    def push(self, name: str, msg: dict, src_ns: int) -> None:
        recv = time.monotonic_ns()
        with self._lock:
            self._append(name, msg, int(src_ns), recv)

    def last(self, name: str) -> dict | None:
        """Most recent decoded message of a stream, or None if nothing arrived yet."""
        with self._lock:
            raw = self._d.get(name, {}).get("raw")
            return raw[-1] if raw else None

    def mark(self, name: str, t_mono_ns: int | None = None) -> None:
        self.push("markers", {"name": name}, time.monotonic_ns() if t_mono_ns is None else t_mono_ns)

    def _append(self, name: str, msg: dict, src: int, recv: int) -> None:
        d = self._d.setdefault(name, {"recv": [], "src": [], "raw": [], "t_sim": []})
        t_sim = float(msg["t_sim"]) if name == "sim_gt" else None  # validate before appending
        d["recv"].append(recv)
        d["src"].append(src)
        d["raw"].append(msg)
        if t_sim is not None:
            d["t_sim"].append(t_sim)

    def save(self, path: str) -> None:
        self._raise_if_died()
        out = {}
        with self._lock:
            snap = {n: {k: list(v) for k, v in d.items()} for n, d in self._d.items()}
        for name, d in snap.items():
            out[f"{name}_recv_ns"] = np.array(d["recv"], dtype=np.int64)
            out[f"{name}_src_ns"] = np.array(d["src"], dtype=np.int64)
            raw = np.empty(len(d["raw"]), dtype=object)
            raw[:] = d["raw"]
            out[f"{name}_raw"] = raw
            if name == "sim_gt":
                out["sim_gt_t_sim"] = np.array(d["t_sim"], dtype=np.float64)
        np.savez(path, **out)

    def close(self) -> None:
        self._stop.set()
        if self._socks:
            self._thread.join()
        for s in self._socks.values():
            s.close()
        self._raise_if_died()


# ---- Sync check ----------------------------------------------------------
def _load(npz):
    return np.load(npz, allow_pickle=True) if isinstance(npz, (str, bytes)) or hasattr(npz, "__fspath__") else npz


def _stream_arrays(d, name):
    if f"{name}_src_ns" not in d:
        return None
    return np.asarray(d[f"{name}_src_ns"]), np.asarray(d[f"{name}_recv_ns"])


def _marker_times(d) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    a = _stream_arrays(d, "markers")
    if a is None:
        return out
    src, recv = a
    for s, r, raw in zip(src, recv, d["markers_raw"]):
        out.setdefault(raw["name"], []).append((s if s >= 0 else r) / 1e9)
    return out


def check_sync(npz, required: dict[str, StreamReq], window: tuple[float, float]) -> SyncReport:
    d = _load(npz)
    rep = SyncReport(ok=True)
    why = rep.reasons
    t0, t1 = window
    lo = t0 - BASELINE_S
    markers = _marker_times(d)

    for name, req in required.items():
        g = max(GAP_PERIODS * req.period, GAP_MIN_S)
        if req.absent and name in INDEPENDENT:
            why.append(f"{name}: independent evidence stream cannot declare absences")
            continue
        arr = _stream_arrays(d, name)
        if arr is None or len(arr[0]) == 0:
            why.append(f"{name}: missing stream")
            continue
        src_ns, recv_ns = arr
        if np.any(src_ns < 0):
            why.append(f"{name}: no source timestamps")
            continue
        src = src_ns / 1e9
        in_win = (src >= lo) & (src <= t1)

        # skew
        skew = (recv_ns[in_win] - src_ns[in_win]) / 1e6
        if len(skew):
            rep.skew_p99_ms[name] = float(np.percentile(skew, 99))
            if skew.min() < SKEW_MIN_MS:
                why.append(f"{name}: negative skew {skew.min():.2f} ms (clock mismatch)")
            if rep.skew_p99_ms[name] > SKEW_P99_MAX_MS:
                why.append(f"{name}: skew p99 {rep.skew_p99_ms[name]:.1f} ms > {SKEW_P99_MAX_MS}")
            if skew.max() > SKEW_MAX_MS:
                why.append(f"{name}: skew max {skew.max():.1f} ms > {SKEW_MAX_MS}")

        # declared absences -> allowed intervals + observed-absence check
        allowed = []
        for start, end in req.absent:
            ts = markers.get(start)
            te = [t for t in markers.get(end, []) if ts and t >= ts[0]]
            if not ts or not te:
                why.append(f"{name}: marker '{start}'/'{end}' not found")
                continue
            ts, te = ts[0], te[0]
            if np.any((src > ts + g) & (src < te)):
                why.append(f"{name}: fault not applied (messages during {start}..{end})")
            allowed.append((ts - g, te + ABSENT_END_SLACK_S))

        # gaps incl. coverage of [lo, t1] via sentinels
        pts = np.concatenate([[lo], np.sort(src[in_win]), [t1]])
        gaps = np.diff(pts)
        rep.max_gap_ms[name] = float(gaps.max() * 1e3)
        for i in np.nonzero(gaps > g)[0]:
            a, b = pts[i], pts[i + 1]
            if not any(a >= x and b <= y for x, y in allowed):
                why.append(f"{name}: gap {gaps[i] * 1e3:.0f} ms at t={a:.3f} (limit {g * 1e3:.0f} ms)")

    # real-time factor
    if "sim_gt" in required and "sim_gt_src_ns" in d and "sim_gt_t_sim" not in d:
        why.append("sim_gt: missing t_sim (cannot compute rtf)")
    if "sim_gt_t_sim" in d and "sim_gt_src_ns" in d:
        src = np.asarray(d["sim_gt_src_ns"]) / 1e9
        ts = np.asarray(d["sim_gt_t_sim"])
        m = (src >= t0) & (src <= t1)
        if m.sum() >= 2 and src[m][-1] > src[m][0]:
            rep.rtf = float((ts[m][-1] - ts[m][0]) / (src[m][-1] - src[m][0]))
            if rep.rtf < RTF_MIN:
                why.append(f"rtf {rep.rtf:.2f} < {RTF_MIN}")
    if "sim_gt" in required and rep.rtf is None and not any("t_sim" in x for x in why):
        why.append("sim_gt: cannot compute rtf (fewer than 2 samples in window)")

    rep.ok = not why
    return rep


# ---- Alignment -----------------------------------------------------------
def align(
    npz,
    signals: dict[str, Callable[[dict], np.ndarray]],
    window: tuple[float, float],
    hz: float = 100.0,
    angles: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Resample signals onto one source-time grid (returned under key ``"t"``).

    A signal key ``"<stream>"`` or ``"<stream>.<label>"`` reads from that stream.
    Keys listed in ``angles`` are unwrapped before linear interpolation (the
    result stays unwrapped). Grid points more than one stream period outside
    the stream's data are NaN.
    """
    d = _load(npz)
    t = np.arange(window[0], window[1] + 0.5 / hz, 1.0 / hz)
    out = {"t": t}
    angles = set(angles)
    for key, fn in signals.items():
        stream = key.split(".", 1)[0]
        src = np.asarray(d[f"{stream}_src_ns"]) / 1e9
        if len(src) == 0 or np.any(src < 0):
            raise ValueError(f"{stream}: no source-time samples to align")
        order = np.argsort(src, kind="stable")
        src = src[order]
        raw = d[f"{stream}_raw"]
        v = np.array([np.asarray(fn(raw[i]), dtype=float) for i in order])
        if key in angles:
            v = np.unwrap(v, axis=0)
        period = float(np.median(np.diff(src))) if len(src) > 1 else 0.0
        v2 = v.reshape(len(src), -1)
        res = np.stack([np.interp(t, src, v2[:, j]) for j in range(v2.shape[1])], axis=1)
        idx = np.clip(np.searchsorted(src, t), 1, max(len(src) - 1, 1)) if len(src) > 1 else None
        if idx is None:
            near = np.abs(t - src[0])
        else:
            near = np.minimum(np.abs(t - src[idx - 1]), np.abs(src[idx] - t))
        res[near > period + 1e-9] = np.nan
        out[key] = res.reshape((len(t),) + v.shape[1:])
    return out
