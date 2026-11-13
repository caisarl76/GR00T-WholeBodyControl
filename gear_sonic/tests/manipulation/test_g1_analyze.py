"""G1 analysis tests on synthetic recordings in the Task 1 recorder save format.

Marker protocol (shared with g1_height_probe.py), each a ``<name>:start`` / ``<name>:end`` pair:
``settle:K`` (K=0..2, -1 sentinel), ``step:IDLE:H`` / ``step:IDLE_SQUAT:H`` (3 s holds),
``stat:H`` (3 s stationary hold per walk height; H=``-1`` for the sentinel),
``trial:H:DIR:K`` (10 s motion + 3 s stationary).
"""

import numpy as np

from gear_sonic.scripts.gates.g1_analyze import analyze_height

NS = 1_000_000_000
T0 = 1000.0
GRID = [round(0.8 - 0.05 * i, 2) for i in range(9)]
DIRS = ("fwd", "back", "turn")
STEPS = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def squat_z(h):
    return 0.74 + 0.8 * (h - 0.75)  # command 0.75 -> 0.74 m


def build(
    *,
    idle_z=lambda h: 0.74,  # IDLE ignores height by default
    squat=squat_z,
    settle_z=(0.74, 0.74, 0.74),
    walk_heights=None,  # heights with walk trials; default: sentinel + grid + 0.73
    trial_ok=lambda h, d, k: True,
    skip=lambda h, d, k: False,
    moves=lambda h, d, k: True,  # False -> robot stays still during the trial
    steps=STEPS,
):
    segs = []  # (marker, duration, z, motion kind or None)

    def hold(name, dur, z):
        segs.append((name, dur, z, None))

    for k, z in enumerate(settle_z):
        hold(f"settle:{k}", 3.0, z)
    for h in steps:
        hold(f"step:IDLE:{h:.2f}", 3.0, idle_z(h))
        hold(f"step:IDLE_SQUAT:{h:.2f}", 3.0, squat(h))
    honors = abs(idle_z(0.8) - idle_z(0.4)) > 0.03
    zmap = idle_z if honors else squat
    heights = [-1.0] + [0.73] + GRID if walk_heights is None else walk_heights
    for h in heights:
        hk = "-1" if h == -1.0 else f"{h:.2f}"
        zs = settle_z[0] if h == -1.0 else zmap(h)
        hold(f"stat:{hk}", 3.0, zs)
        for k in range(5):
            for d in DIRS:
                if skip(h, d, k):
                    continue
                z = zs if trial_ok(h, d, k) else zs + 0.06
                segs.append((f"trial:{hk}:{d}:{k}", 13.0, z, d if moves(h, d, k) else None))
    return _assemble(segs)


def _assemble(segs):
    t = T0
    markers = []
    parts_t, parts_z = [np.arange(T0 - 2.0, T0, 0.005)], [np.full(400, 0.74)]
    parts_x, parts_yaw = [np.zeros(400)], [np.zeros(400)]
    x = yaw = 0.0
    for name, dur, z, mot in segs:
        markers += [(f"{name}:start", t), (f"{name}:end", t + dur)]
        n = int(round(dur / 0.005))
        u = np.minimum(0.005 * np.arange(n), 10.0)  # motion lasts the first 10 s
        v = {"fwd": 0.3, "back": -0.3}.get(mot, 0.0)
        w = 0.5 if mot == "turn" else 0.0
        parts_t.append(t + 0.005 * np.arange(n))
        parts_z.append(np.full(n, z))
        parts_x.append(x + v * u)
        parts_yaw.append(yaw + w * u)
        x, yaw = x + v * 10.0, yaw + w * 10.0
        t += n * 0.005
    parts_t.append(t + 0.005 * np.arange(400))  # trailing coverage
    parts_z.append(np.full(400, 0.74))
    parts_x.append(np.full(400, x))
    parts_yaw.append(np.full(400, yaw))
    tt, zz = np.concatenate(parts_t), np.concatenate(parts_z)
    xx, yy = np.concatenate(parts_x), np.concatenate(parts_yaw)
    src = np.round(tt * NS).astype(np.int64)
    g_src = np.round(np.arange(tt[0], tt[-1], 0.02) * NS).astype(np.int64)
    m_src = np.array([int(round(tm * NS)) for _, tm in markers], dtype=np.int64)
    return {
        "sim_gt_src_ns": src,
        "sim_gt_recv_ns": src + 200_000,
        "sim_gt_raw": _objs(
            [
                {"pelvis_pos": [float(a), 0.0, float(z)], "pelvis_quat_wxyz": [np.cos(q / 2), 0.0, 0.0, np.sin(q / 2)]}
                for a, z, q in zip(xx, zz, yy)
            ]
        ),
        "sim_gt_t_sim": tt - T0,
        "g1_debug_src_ns": g_src,
        "g1_debug_recv_ns": g_src + 200_000,
        "g1_debug_raw": _objs([{} for _ in g_src]),
        "markers_src_ns": m_src,
        "markers_recv_ns": m_src,
        "markers_raw": _objs([{"name": n} for n, _ in markers]),
    }


def _objs(items):
    a = np.empty(len(items), dtype=object)
    a[:] = items
    return a


# ---- tests ---------------------------------------------------------------
def test_idle_ignores_height():
    r = analyze_height(build(walk_heights=[]))
    assert r["idle_honors_height"] is False


def test_idle_honors_height():
    r = analyze_height(build(idle_z=squat_z, walk_heights=[]))
    assert r["idle_honors_height"] is True
    assert any(abs(h - 0.8) < 1e-9 and abs(z - squat_z(0.8)) < 1e-3 for h, z in r["cmd_to_z"])


def test_h_default_interpolated():
    sq = lambda h: {0.8: 0.78, 0.6: 0.62}.get(round(h, 2), 0.5)
    r = analyze_height(build(squat=sq, steps=[0.8, 0.6], walk_heights=[]))
    assert abs(r["h_default"] - 0.75) < 1e-3
    assert len(r["cmd_to_z"]) == 2


def test_h_default_none_when_not_bracketed():
    r = analyze_height(build(settle_z=(0.9, 0.9, 0.9), walk_heights=[]))
    assert r["h_default"] is None and r["gate_pass"] is False


def test_h_default_none_when_standing_z_drifts():
    r = analyze_height(build(settle_z=(0.74, 0.74, 0.76), walk_heights=[]))
    assert r["h_default"] is None


def _fails(bad):
    """trial_ok that fails every trial at heights in ``bad``."""
    return lambda h, d, k: round(h, 2) not in bad


def test_walk_band_margin():
    r = analyze_height(build(trial_ok=_fails({0.45, 0.40})))
    assert r["h_walk_min"] == 0.53
    assert r["gate_pass"] is True
    assert all(len(v) == 5 for v in r["walk_table"][0.8].values())


def test_gap_in_band_truncates():
    r = analyze_height(build(trial_ok=_fails({0.65})))
    assert r["h_walk_min"] == 0.73


def test_one_direction_fail_fails_height():
    r = analyze_height(build(trial_ok=lambda h, d, k: not (round(h, 2) == 0.55 and d == "turn" and k == 2)))
    assert r["walk_table"][0.55]["turn"].count(False) == 1
    assert r["h_walk_min"] == 0.63  # 0.60 is the lowest passing height before the stop at 0.55


def test_only_top_passes_band_unknown():
    r = analyze_height(build(trial_ok=_fails({0.75})))
    assert r["h_walk_min"] is None
    assert r["band_assumption"] is None


def test_band_assumption_reported():
    r = analyze_height(build())
    assert r["band_assumption"] == "interpolated between grid points, step 0.05"
    assert r["h_walk_min"] == 0.43


def test_top_fails_band_none():
    r = analyze_height(build(trial_ok=_fails({0.8})))
    assert r["h_walk_min"] is None
    assert r["gate_pass"] is True  # depends only on sentinel and h_default


def test_untested_below_grid_not_in_band():
    r = analyze_height(build())
    assert r["h_walk_min"] == 0.43  # lowest tested 0.40 + 0.03; nothing below the grid
    assert min(h for h in r["walk_table"] if h > 0) == 0.40


def test_gate_fails_when_default_height_walk_fails():
    # settle z 0.724 -> h_default 0.73 (off-grid), whose trials fail
    r = analyze_height(build(settle_z=(0.724,) * 3, trial_ok=_fails({0.73})))
    assert r["h_default"] is not None and abs(r["h_default"] - 0.73) < 1e-3
    assert r["gate_pass"] is False
    assert any("h_default" in x for x in r["reasons"])


def test_gate_fails_when_h_default_none():
    r = analyze_height(build(settle_z=(0.9, 0.9, 0.9)))
    assert r["gate_pass"] is False


def test_sentinel_fail_fails_gate():
    r = analyze_height(build(trial_ok=lambda h, d, k: h != -1.0))
    assert r["gate_pass"] is False


def test_missing_trials_count_as_failed():
    r = analyze_height(build(skip=lambda h, d, k: round(h, 2) <= 0.6 and h > 0))
    assert r["walk_table"][0.6]["fwd"] == [False] * 5
    assert r["h_walk_min"] == 0.68  # 0.65 lowest passing


def _drop(d, stream, t0, t1):
    src = d[f"{stream}_src_ns"]
    keep = ~((src > int(t0 * NS)) & (src < int(t1 * NS)))
    for k in [k for k in d if k.startswith(stream + "_")]:
        d[k] = d[k][keep]


def test_sync_failure_in_trial_is_missing_evidence():
    d = build(walk_heights=[-1.0])
    t_trial = T0 + 9 + 42 + 3  # sentinel trial fwd:0 start
    _drop(d, "g1_debug", t_trial + 4, t_trial + 6)
    r = analyze_height(d)
    assert r["walk_table"][-1.0]["fwd"][0] is False
    assert any("sync" in x for x in r["reasons"]) and r["gate_pass"] is False


def test_small_sim_hole_does_not_fail_trial():
    d = build(walk_heights=[-1.0])
    t_trial = T0 + 9 + 42 + 3
    _drop(d, "sim_gt", t_trial + 5.0, t_trial + 5.012)  # 12 ms jitter hole, under the 100 ms limit
    assert analyze_height(d)["walk_table"][-1.0]["fwd"][0] is True


def test_sync_failure_on_idle_step_is_not_a_pass():
    d = build(idle_z=squat_z)  # IDLE honors height
    t = T0 + 9 + 4 * 6  # step:IDLE:0.40 start
    _drop(d, "g1_debug", t + 1.0, t + 1.5)
    r = analyze_height(d)
    assert r["gate_pass"] is False and r["reasons"]


def test_stationary_trial_has_no_motion_evidence():
    r = analyze_height(build(walk_heights=[-1.0], moves=lambda h, d, k: not (d == "fwd" and k == 1)))
    assert r["walk_table"][-1.0]["fwd"] == [True, False, True, True, True]
    assert r["trials"]["-1"]["fwd"][1]["fail"] == "no motion evidence"
    assert not any(not x for x in r["walk_table"][-1.0]["turn"])
    r2 = analyze_height(build(walk_heights=[-1.0], moves=lambda h, d, k: d != "turn"))
    assert r2["walk_table"][-1.0]["turn"] == [False] * 5 and r2["gate_pass"] is False


def test_traceability_fields():
    r = analyze_height(build(walk_heights=[-1.0]))
    assert len(r["settle_z"]) == 3 and r["idle_cmd_to_z"] and "0:IDLE" in r["switch_z"]
    t = r["trials"]["-1"]["fwd"][0]
    assert t["disp_m"] > 2.5 and t["max_dev"] < 0.01 and t["min_z"] > 0.7
    assert abs(r["trials"]["-1"]["turn"][0]["dyaw_rad"] - 5.0) < 0.1
