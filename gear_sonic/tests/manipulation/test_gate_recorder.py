"""Tests for the gate recorder: capture, sync contract and source-time alignment."""

import socket
import time

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.scripts.gates.recorder import GateRecorder, StreamReq, align, check_sync

S = 1_000_000_000
T0 = 1000 * S  # arbitrary monotonic origin
WIN = (1002.0, 1006.0)  # analysis window, seconds


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pub(port):
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(f"tcp://127.0.0.1:{port}")
    return sock


def test_recorder_timestamps_and_decodes(tmp_path):
    p1, p2 = _free_port(), _free_port()
    pub1, pub2 = _pub(p1), _pub(p2)
    rec = GateRecorder(
        {"sim_gt": ("127.0.0.1", p1, "sim_gt"), "g1_debug": ("127.0.0.1", p2, "g1_debug")}
    )
    time.sleep(0.5)  # slow joiner
    sent = []
    for i in range(3):
        src = time.monotonic_ns()
        sent.append(src)
        msg = {"t_mono_ns": src, "t_sim": 0.01 * i, "pelvis_pos": [i, 2.0, 3.0]}
        pub1.send(b"sim_gt" + msgpack.packb(msg))
    sent2 = []
    for _ in range(2):
        src = time.monotonic_ns()
        sent2.append(src)
        pub2.send(b"g1_debug" + msgpack.packb({"t_mono_ns": src, "body_q": [0.0]}))
    deadline = time.time() + 3
    while time.time() < deadline:
        rec.poll_all()
        if len(rec._d["sim_gt"]["src"]) == 3 and len(rec._d["g1_debug"]["src"]) == 2:
            break
        time.sleep(0.01)
    path = str(tmp_path / "rec.npz")
    rec.save(path)
    d = np.load(path, allow_pickle=True)
    assert len(d["sim_gt_recv_ns"]) == 3 and len(d["g1_debug_recv_ns"]) == 2
    assert list(d["sim_gt_src_ns"]) == sent and list(d["g1_debug_src_ns"]) == sent2
    assert np.all(d["sim_gt_recv_ns"] >= d["sim_gt_src_ns"])
    assert np.all(d["g1_debug_recv_ns"] >= d["g1_debug_src_ns"])
    assert d["sim_gt_raw"][2]["pelvis_pos"] == [2, 2.0, 3.0]
    assert list(d["sim_gt_t_sim"]) == [0.0, 0.01, 0.02]


def test_push_and_mark_saved(tmp_path):
    rec = GateRecorder({})
    rec.push("operator_ref", {"x": 1}, 5)
    rec.mark("kill", 7)
    path = str(tmp_path / "r.npz")
    rec.save(path)
    d = np.load(path, allow_pickle=True)
    assert list(d["operator_ref_src_ns"]) == [5] and d["operator_ref_raw"][0] == {"x": 1}
    assert list(d["markers_src_ns"]) == [7] and d["markers_raw"][0] == {"name": "kill"}
    assert d["markers_recv_ns"][0] >= 0


def _stream(t_start, t_end, dt, skew_ms=2.0, vals=None):
    """Source times t_start..t_end (s) with fixed skew; returns dict pieces."""
    src = np.arange(round(t_start * 1000), round(t_end * 1000) + 1, round(dt * 1000)) * (S // 1000)
    return src.astype(np.int64), (src + int(skew_ms * 1e6)).astype(np.int64)


def _add(d, name, src, recv, raw=None):
    d[f"{name}_src_ns"] = src
    d[f"{name}_recv_ns"] = recv
    d[f"{name}_raw"] = np.array(raw if raw is not None else [{}] * len(src), dtype=object)


def _good(rtf=1.0):
    d = {}
    src, recv = _stream(1001.0, 1006.5, 0.02)
    _add(d, "g1_debug", src, recv)
    src, recv = _stream(1001.0, 1006.5, 0.005)
    _add(d, "sim_gt", src, recv)
    d["sim_gt_t_sim"] = (src - src[0]) / S * rtf
    return d


REQ = {"sim_gt": StreamReq(0.005), "g1_debug": StreamReq(0.02)}


def test_sync_ok_baseline():
    r = check_sync(_good(), REQ, WIN)
    assert r.ok, r.reasons
    assert r.rtf == pytest.approx(1.0)
    assert r.skew_p99_ms["g1_debug"] == pytest.approx(2.0)


def test_sync_rejects_missing_stream():
    d = _good()
    r = check_sync(d, {**REQ, "pose": StreamReq(0.02)}, WIN)
    assert not r.ok and any("pose" in x and "missing" in x for x in r.reasons)


def test_sync_rejects_gap():
    d = _good()
    src, recv = _stream(1001.0, 1006.5, 0.02)
    keep = ~((src > int(1003.0 * S)) & (src < int(1003.15 * S)))
    _add(d, "g1_debug", src[keep], recv[keep])
    r = check_sync(d, REQ, WIN)
    assert not r.ok and any("gap" in x for x in r.reasons)
    assert r.max_gap_ms["g1_debug"] > 100


def test_sync_rejects_negative_skew():
    d = _good()
    src, recv = _stream(1001.0, 1006.5, 0.02, skew_ms=-3.0)
    _add(d, "g1_debug", src, recv)
    r = check_sync(d, REQ, WIN)
    assert not r.ok and any("negative skew" in x for x in r.reasons)


def test_sync_rejects_high_skew():
    d = _good()
    src, recv = _stream(1001.0, 1006.5, 0.02, skew_ms=60.0)
    _add(d, "g1_debug", src, recv)
    r = check_sync(d, REQ, WIN)
    assert not r.ok and any("skew" in x for x in r.reasons)


def test_sync_rejects_partial_coverage():
    d = _good()
    src, recv = _stream(1001.5, 1006.5, 0.02)  # window baseline starts at 1001.0
    _add(d, "g1_debug", src, recv)
    r = check_sync(d, REQ, WIN)
    assert not r.ok and any(x.startswith("g1_debug: gap") for x in r.reasons)


def test_sync_rejects_slow_sim():
    r = check_sync(_good(rtf=0.8), REQ, WIN)
    assert not r.ok and any("rtf" in x for x in r.reasons)
    assert r.rtf == pytest.approx(0.8)


def _with_fault(gap_end=None, extra_gap=None, kill=1003.0, restart=1004.0, keep_streaming=False):
    """manip_debug (10 ms period) silent between kill and restart (or not)."""
    d = _good()
    src, recv = _stream(1001.0, 1006.5, 0.01)
    if not keep_streaming:
        keep = ~((src > int(kill * S)) & (src < int((restart + 0.05) * S)))
        src, recv = src[keep], recv[keep]
    if extra_gap:
        keep = ~((src > int(extra_gap[0] * S)) & (src < int(extra_gap[1] * S)))
        src, recv = src[keep], recv[keep]
    _add(d, "manip_debug", src, recv)
    msrc = np.array([int(kill * S), int(restart * S)], dtype=np.int64)
    _add(d, "markers", msrc, msrc + 1_000_000, [{"name": "kill"}, {"name": "restart"}])
    return d


MREQ = {
    **REQ,
    "manip_debug": StreamReq(0.01, absent=(("kill", "restart"),)),
}


def test_declared_absence_accepted():
    r = check_sync(_with_fault(), MREQ, WIN)
    assert r.ok, r.reasons


def test_gap_outside_declared_absence_rejected():
    # restart at 1003.0; a 300 ms hole 2 s later
    d = _with_fault(kill=1002.5, restart=1003.0, extra_gap=(1005.0, 1005.3))
    r = check_sync(d, MREQ, WIN)
    assert not r.ok and any("gap" in x for x in r.reasons)


def test_declared_absence_not_observed_rejected():
    r = check_sync(_with_fault(keep_streaming=True), MREQ, WIN)
    assert not r.ok and any("fault not applied" in x for x in r.reasons)


def test_absence_on_independent_stream_rejected():
    req = {**REQ, "g1_debug": StreamReq(0.02, absent=(("kill", "restart"),))}
    d = _with_fault()
    r = check_sync(d, req, WIN)
    assert not r.ok and any("independent" in x for x in r.reasons)


def test_check_sync_accepts_path(tmp_path):
    p = str(tmp_path / "x.npz")
    np.savez(p, **_good())
    assert check_sync(p, REQ, WIN).ok


def _sig_stream(offset_ns):
    src, recv = _stream(1001.0, 1006.5, 0.02)
    raw = [{"v": float(s) / S * 2.0} for s in src]
    d = {}
    _add(d, "a", src, recv + offset_ns, raw)
    return d


def test_align_uses_source_time():
    d = _sig_stream(0)
    d2 = _sig_stream(15_000_000)
    f = {"a": lambda m: np.array([m["v"]])}
    o1 = align(d, f, (1002.0, 1004.0), hz=100.0)
    o2 = align(d2, f, (1002.0, 1004.0), hz=100.0)
    assert np.array_equal(o1["a"], o2["a"])
    assert o1["a"][0, 0] == pytest.approx(2.0 * 1002.0)  # v = 2*t
    assert o1["a"][50, 0] == pytest.approx(2.0 * 1002.5)


def test_align_no_extrapolation():
    d = _sig_stream(0)
    out = align(d, {"a": lambda m: np.array([m["v"]])}, (1006.0, 1008.0), hz=100.0)
    a = out["a"][:, 0]
    assert np.isfinite(a[0])
    assert np.all(np.isnan(a[int(round((0.5 + 0.03) * 100)) :]))  # last sample 1006.5, period 20 ms


def test_align_angle_unwrap():
    src, recv = _stream(1001.0, 1003.0, 0.1)
    ang = np.angle(np.exp(1j * (src / S * 3.0)))  # wrapped ramp
    d = {}
    _add(d, "a", src, recv, [{"x": float(x)} for x in ang])
    out = align(d, {"a": lambda m: m["x"]}, (1001.5, 1002.5), hz=10.0, angles=("a",))
    assert np.all(np.diff(out["a"]) > 0)


def test_align_nan_in_interior_hole():
    src, recv = _stream(1001.0, 1006.5, 0.02)
    keep = ~((src > int(1003.0 * S)) & (src < int(1004.0 * S)))  # 1 s hole
    d = {}
    _add(d, "a", src[keep], recv[keep], [{"v": float(s) / S} for s in src[keep]])
    out = align(d, {"a": lambda m: np.array([m["v"]])}, (1002.0, 1005.0), hz=100.0)
    a, t = out["a"][:, 0], out["t"]
    assert np.all(np.isnan(a[(t > 1003.05) & (t < 1003.95)]))
    assert np.all(np.isfinite(a[(t < 1002.99) | (t > 1004.01)]))


def test_sync_rejects_missing_t_sim():
    d = _good()
    del d["sim_gt_t_sim"]
    r = check_sync(d, REQ, WIN)
    assert not r.ok and any("t_sim" in x for x in r.reasons)


def test_recv_stamped_on_arrival_not_poll(tmp_path):
    port = _free_port()
    pub = _pub(port)
    rec = GateRecorder({"sim_gt": ("127.0.0.1", port, "sim_gt")})
    time.sleep(0.5)
    for i in range(5):
        src = time.monotonic_ns()
        pub.send(b"sim_gt" + msgpack.packb({"t_mono_ns": src, "t_sim": 0.0}))
        time.sleep(0.01)
    time.sleep(0.2)  # caller busy, not polling
    rec.poll_all()
    path = str(tmp_path / "r.npz")
    rec.save(path)
    rec.close()
    d = np.load(path, allow_pickle=True)
    skew_ms = (d["sim_gt_recv_ns"] - d["sim_gt_src_ns"]) / 1e6
    assert len(skew_ms) == 5 and skew_ms.max() < 10


def test_recv_thread_error_surfaces_on_save(tmp_path):
    port = _free_port()
    pub = _pub(port)
    rec = GateRecorder({"sim_gt": ("127.0.0.1", port, "sim_gt")})
    time.sleep(0.5)
    pub.send(b"sim_gt" + msgpack.packb({"t_mono_ns": 1}))  # no t_sim
    time.sleep(0.3)
    with pytest.raises(RuntimeError, match="receive thread died"):
        rec.save(str(tmp_path / "r.npz"))
    assert rec._d["sim_gt"]["src"] == rec._d["sim_gt"]["t_sim"] == []  # all-or-nothing
    with pytest.raises(RuntimeError, match="receive thread died"):
        rec.close()


def test_sync_rejects_uncomputable_rtf():
    d = _good()
    r = check_sync(d, REQ, (1006.4999, 1006.5))  # <2 sim_gt samples in window
    assert not r.ok and any("cannot compute rtf" in x for x in r.reasons)


def test_last_returns_latest_or_none():
    rec = GateRecorder({})
    assert rec.last("sim_gt") is None
    rec.push("sim_gt", {"t_sim": 0.0, "k": 1}, 1)
    rec.push("sim_gt", {"t_sim": 0.1, "k": 2}, 2)
    assert rec.last("sim_gt")["k"] == 2
    rec.close()
