"""G1b analysis tests on synthetic recordings in the Task 1 recorder save format.

Marker protocol (shared with g1b_walk_handoff_probe.py): ``settle:0`` 3 s, then per DIR in fwd/back/turn and
run K the walk trial ``prehold`` 3 s, ``ramp`` 2 s, ``hold`` 2 s, ``trial`` spanning ``sw1`` 0.5 s, ``walk``
10 s, ``stop`` 0.5 s, ``back`` 3 s; then per run K the isolated block ``switch:hold:K``,
``switch:to_sentinel:K``, ``switch:to_height:K`` (3 s each). ``build`` z hooks get (segment, dir, run) and
return a float or a function of the time u (s, array) inside the segment.
"""

import numpy as np

from gear_sonic.scripts.gates.g1b_analyze import _stationary, analyze_handoff

NS = 1_000_000_000
T0 = 1000.0
DIRS = ("fwd", "back", "turn")
Z = 0.787
WALK = (("prehold", 3.0), ("ramp", 2.0), ("hold", 2.0), ("sw1", 0.5), ("walk", 10.0), ("stop", 0.5), ("back", 3.0))
ISO = (("hold", "switch:hold"), ("to_sentinel", "switch:to_sentinel"), ("to_height", "switch:to_height"))


def _objs(items):
    a = np.empty(len(items), dtype=object)
    a[:] = items
    return a


def build(*, z=lambda seg, d, k: Z, moves=lambda d, k: True, drop_markers=(), extra_markers=()):
    """Returns (npz dict, {(seg, dir, run): start time}); isolated segments use dir ``iso``."""
    t, starts = T0 + 3.0, {}
    markers = [("settle:0:start", T0), ("settle:0:end", T0 + 3.0)]
    segs = [(3.0, ("settle", "", 0), None)]

    def add(name, key, dur, mot=None):
        nonlocal t
        starts[key] = t
        markers.extend([(f"{name}:start", t), (f"{name}:end", t + dur)])
        segs.append((dur, key, mot))
        t += dur

    for k in range(5):
        for d in DIRS:
            for seg, dur in WALK:
                if seg == "sw1":
                    markers.append((f"trial:{d}:{k}:start", t))
                add(f"{seg}:{d}:{k}", (seg, d, k), dur, d if seg == "walk" and moves(d, k) else None)
                if seg == "back":
                    markers.append((f"trial:{d}:{k}:end", t))
    for k in range(5):
        for seg, name in ISO:
            add(f"{name}:{k}", (seg, "iso", k), 3.0)
    markers = [m for m in markers if not any(m[0].startswith(p) for p in drop_markers)] + list(extra_markers)
    parts = [(np.arange(T0 - 2, T0, 0.005), np.full(400, Z), np.zeros(400), np.zeros(400))]
    x = yaw = 0.0
    tc = T0
    for dur, key, mot in segs:
        n = int(round(dur / 0.005))
        u = 0.005 * np.arange(n)
        zz = Z if key[0] == "settle" else z(*key)
        zz = zz(u) if callable(zz) else np.full(n, zz)
        v = {"fwd": 0.3, "back": -0.3}.get(mot, 0.0)
        w = 0.5 if mot == "turn" else 0.0
        parts.append((tc + u, zz, x + v * u, yaw + w * u))
        x, yaw, tc = x + v * dur, yaw + w * dur, tc + dur
    parts.append((tc + 0.005 * np.arange(400), np.full(400, Z), np.full(400, x), np.full(400, yaw)))
    tt, zz, xx, yy = (np.concatenate(c) for c in zip(*parts))
    src = np.round(tt * NS).astype(np.int64)
    g_src = np.round(np.arange(tt[0], tt[-1], 0.02) * NS).astype(np.int64)
    m_src = np.array([int(round(tm * NS)) for _, tm in markers], dtype=np.int64)
    d = {
        "sim_gt_src_ns": src,
        "sim_gt_recv_ns": src + 200_000,
        "sim_gt_raw": _objs([
            {"pelvis_pos": [float(a), 0.0, float(c)], "pelvis_quat_wxyz": [np.cos(q / 2), 0.0, 0.0, np.sin(q / 2)]}
            for a, c, q in zip(xx, zz, yy)
        ]),
        "sim_gt_t_sim": tt - T0,
        "g1_debug_src_ns": g_src,
        "g1_debug_recv_ns": g_src + 200_000,
        "g1_debug_raw": _objs([{} for _ in g_src]),
        "markers_src_ns": m_src,
        "markers_recv_ns": m_src,
        "markers_raw": _objs([{"name": n} for n, _ in markers]),
    }
    return d, starts


def _drop(d, stream, t0, t1):
    src = d[f"{stream}_src_ns"]
    keep = ~((src > int(t0 * NS)) & (src < int(t1 * NS)))
    for k in [k for k in d if k.startswith(stream + "_")]:
        d[k] = d[k][keep]


def _diag(r, kind, dr, k):
    return next(s["dz_m"] for s in r["walk_switch_steps_diag"] if (s["kind"], s["dir"], s["run"]) == (kind, dr, k))


def _iso(r, kind, k):
    return next(s for s in r["isolated_switches"] if (s["kind"], s["run"]) == (kind, k))


def _at(seg, dr, k, f):
    """z hook: base Z everywhere, ``f`` (u -> z) in one segment."""
    return lambda s, d, r: f if (s, d, r) == (seg, dr, k) else Z


def test_handoff_pass():
    d, _ = build()
    r = analyze_handoff(d)
    assert r["gate_pass"] is True and r["reasons"] == []
    assert r["dir_passes"] == {"fwd": 5, "back": 5, "turn": 5}
    assert len(r["isolated_switches"]) == 10 and len(r["walk_switch_steps_diag"]) == 30
    assert r["max_dev"] < 1e-6 and r["max_dev_vs_settle0"] < 1e-6 and r["falls"] == [] and r["sync"]["ok"]
    assert r["trials"]["fwd"][0]["disp_m"] > 2.5 and r["trials"]["turn"][0]["dyaw_rad"] > 4.0


def test_back_first_half_second_offset_is_diag_only():
    d, _ = build(z=_at("back", "fwd", 0, lambda u: Z + 0.012 * (u < 0.5)))
    r = analyze_handoff(d)
    assert abs(_diag(r, "to_height", "fwd", 0) - 0.012) < 1e-6
    assert r["max_walk_switch_step_diag_m"] > 0.0119 and r["gate_pass"] is True  # diagnostics are not gated


def test_negative_diag_step_reported():
    d, _ = build(z=_at("sw1", "back", 1, Z - 0.012))
    assert abs(_diag(analyze_handoff(d), "to_sentinel", "back", 1) + 0.012) < 1e-6


def test_missing_stop_or_back_marker_fails():
    for seg in ("stop", "back"):
        d, _ = build(drop_markers=(f"{seg}:turn:4:",))
        r = analyze_handoff(d)
        assert r["gate_pass"] is False and r["trials"]["turn"][4]["pass"] is False
        assert [s["dz_m"] for s in r["walk_switch_steps_diag"] if (s["kind"], s["dir"], s["run"]) == ("to_height", "turn", 4)] == [None]


def test_missing_isolated_marker_fails():
    d, _ = build(drop_markers=("switch:to_height:2:",))
    r = analyze_handoff(d)
    assert r["gate_pass"] is False and _iso(r, "to_height", 2)["steady_dz_m"] is None


def test_hold_first_half_offset_is_ignored():
    d, _ = build(z=_at("hold", "fwd", 0, lambda u: Z + 0.02 * (u < 1.0)))
    r = analyze_handoff(d)
    assert abs(_diag(r, "to_sentinel", "fwd", 0)) < 1e-6 and r["trials"]["fwd"][0]["max_dev"] < 1e-6
    assert r["gate_pass"] is True


def test_trial_dev_over_3cm_fails():
    d, _ = build(z=_at("walk", "back", 1, Z - 0.04))
    r = analyze_handoff(d)
    assert r["trials"]["back"][1]["pass"] is False and r["trials"]["back"][1]["max_dev"] > 0.03
    assert r["dir_passes"]["back"] == 4 and r["gate_pass"] is False


def test_trial_reference_is_settled_hold_and_settle0_reported():
    hold_z = lambda s, d, k: Z + 0.02 if s in ("hold", "sw1", "walk", "stop", "back") else Z
    r = analyze_handoff(build(z=hold_z)[0])
    assert r["max_dev"] < 1e-6 and abs(r["max_dev_vs_settle0"] - 0.02) < 1e-6


def test_no_motion_evidence_fails():
    d, _ = build(moves=lambda dr, k: not (dr == "turn" and k == 2))
    r = analyze_handoff(d)
    assert "no motion evidence" in r["trials"]["turn"][2]["fail"] and r["gate_pass"] is False


def test_sync_gap_in_switch_window_fails():
    d, st = build()
    t = st[("to_sentinel", "iso", 0)]
    _drop(d, "g1_debug", t + 1.0, t + 2.9)
    r = analyze_handoff(d)
    assert r["gate_pass"] is False and not r["sync"]["ok"] and any("sync" in x for x in r["reasons"])
    assert _iso(r, "to_sentinel", 0)["steady_dz_m"] is None


def test_fall_marker_alone_fails():
    d, _ = build(extra_markers=[("fall:fwd3:start", T0 + 400.0)])
    r = analyze_handoff(d)
    assert r["falls"] == ["fwd3"] and r["gate_pass"] is False


def test_fall_z_fails():
    d, _ = build(z=_at("walk", "fwd", 3, 0.2))
    r = analyze_handoff(d)
    assert "fall" in r["trials"]["fwd"][3]["fail"] and r["gate_pass"] is False


def test_isolated_step_1p2cm_fails_and_0p5cm_passes():
    for kind, seg in (("to_sentinel", "to_sentinel"), ("to_height", "to_height")):
        r = analyze_handoff(build(z=_at(seg, "iso", 1, Z + 0.012))[0])
        assert r["gate_pass"] is False and _iso(r, kind, 1)["pass"] is False
        assert abs(_iso(r, kind, 1)["steady_dz_m"] - 0.012) < 1e-6
        r = analyze_handoff(build(z=_at(seg, "iso", 1, Z - 0.005))[0])
        assert _iso(r, kind, 1)["pass"] is True and abs(_iso(r, kind, 1)["steady_dz_m"] + 0.005) < 1e-6
    assert r["gate_pass"] is True  # 0.5 cm steps (and the matching return steps) are under 1 cm


def test_isolated_oscillation_not_stationary():
    osc = lambda u: Z + 0.02 * np.sin(2 * np.pi * u / 0.6) * (u > 1.9)
    r = analyze_handoff(build(z=_at("to_sentinel", "iso", 2, osc))[0])
    s = _iso(r, "to_sentinel", 2)
    assert s["pass"] is False and any("not stationary" in f for f in s["fail"]) and r["gate_pass"] is False


def test_isolated_transient_peak_fails_though_steady_returns():
    d, _ = build(z=_at("to_sentinel", "iso", 3, lambda u: Z + 0.015 * (u < 1.0)))
    s = _iso(analyze_handoff(d), "to_sentinel", 3)
    assert abs(s["steady_dz_m"]) < 1e-6 and abs(s["peak_m"] - 0.015) < 1e-6
    assert s["pass"] is False and any("transient peak" in f for f in s["fail"])


def test_missing_sim_gt_is_missing_evidence_not_keyerror():
    d, _ = build()
    for k in [k for k in d if k.startswith("sim_gt")]:
        del d[k]
    r = analyze_handoff(d)
    assert r["gate_pass"] is False and r["reasons"] and r["dir_passes"] == {"fwd": 0, "back": 0, "turn": 0}


def _seg(z=lambda t: 0 * t, x=lambda t: 0 * t, yaw=lambda t: 0 * t):
    t = np.arange(0, 1.0, 0.01)
    return t, Z + z(t), np.stack([x(t), 0 * t], axis=1), yaw(t)


def test_stationary_per_axis():
    assert _stationary(_seg()) is None
    assert "z slope" in _stationary(_seg(z=lambda t: 0.004 * t))
    assert _stationary(_seg(x=lambda t: 0.03 * t)) == "xy speed"  # xy only
    assert _stationary(_seg(yaw=lambda t: 0.1 * t)) == "yaw rate"  # yaw only
    assert _stationary(_seg(x=lambda t: 0.01 * t, yaw=lambda t: 0.02 * t)) is None  # under both limits


def test_missing_settle0_fails():
    r = analyze_handoff(build(drop_markers=("settle:0:",))[0])
    assert r["gate_pass"] is False and any("settle:0" in x for x in r["reasons"])


def test_unsettled_reference_fails():
    d, _ = build(z=_at("hold", "back", 2, lambda u: Z + 0.004 * u))  # last 1 s still drifting 4 mm/s
    r = analyze_handoff(d)
    t = r["trials"]["back"][2]
    assert t["pass"] is False and any("pre-walk hold not settled" in f for f in t["fail"]) and r["gate_pass"] is False


def test_settle_s_measured_and_reported():
    d, _ = build(z=_at("hold", "fwd", 1, lambda u: Z + 0.02 * np.exp(-u / 0.3)))
    r = analyze_handoff(d)
    s = r["trials"]["fwd"][1]["settle_s"]
    assert 0.9 < s < 1.3 and r["max_settle_s"]["fwd"] == s  # slope < 2 mm/s at ~1.05 s
    assert r["trials"]["fwd"][0]["settle_s"] - 0.25 < 1e-3 and r["unsettled_holds"] == 0


def test_never_settled_hold_is_none():
    d, _ = build(z=_at("hold", "turn", 0, lambda u: Z + 0.01 * u))
    r = analyze_handoff(d)
    assert r["trials"]["turn"][0]["settle_s"] is None and r["unsettled_holds"] == 1


def test_sync_failure_reported_once():
    d, st = build()
    t = st[("hold", "fwd", 0)]
    _drop(d, "g1_debug", t + 0.2, t + 1.8)
    r = analyze_handoff(d)
    assert len([x for x in r["sync"]["failures"] if x.startswith("hold:fwd:0:")]) == 1
