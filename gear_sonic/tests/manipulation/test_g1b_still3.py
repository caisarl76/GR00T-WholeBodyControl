"""Tests for the Step 9 probe plan and analysis (synthetic data, no sim)."""

import hashlib
import json

import numpy as np
import pytest

from gear_sonic.scripts.gates import g1b_still3 as S
from gear_sonic.scripts.gates import g1b_still3_probe as P

NS = 1_000_000_000
T0 = 1000.0
CHANGE = ("prehold", "ramp", "hold", "sw1", "back", "switch")  # Run.__init__ static-command-change prefixes


# ---- fake probe: runs the real block executors on a virtual clock --------
class FakeRec:
    def __init__(self, p):
        self.p, self.marks = p, []

    def mark(self, name, t_ns=None):
        self.marks.append((name, self.p.t if t_ns is None else t_ns))


class FakeProbe:
    def __init__(self):
        self.t, self.fall_key, self.profile, self.rec = int(T0 * NS), None, None, FakeRec(self)
        self.runs, self.gap_ns = [], 0

    def now_ns(self):
        return self.t

    def run(self, secs, mode, height, *, name=None, **kw):
        if name:
            self.rec.mark(f"{name}:start")
        self.runs.append((name, secs, mode))
        self.t += int(round(secs * NS))
        if name:
            self.rec.mark(f"{name}:end")
            self.t += self.gap_ns  # the next start is stamped separately (a gap like Probe.run's)


MAGS = {"nudge": 100, "lateral": 60, "down": 200, "yaw": 20}


def _plan(kind):
    p = FakeProbe()
    p.run(3.0, 0, -1.0, name="settle:0")
    (P.run_validate(p, p.rec, 0.7984, MAGS) if kind == "validate" else P.run_calibrate(p, p.rec, 0.7984, P.CAL_TYPES))
    return p


def _starts(marks, pref):
    return sorted(t / NS for n, t in marks if n.endswith(":start") and n.split(":")[0] in pref)


def test_disturbance_not_a_command_change():
    for kind in ("calibrate", "validate"):
        marks = _plan(kind).rec.marks
        names = {n for n, _ in marks}
        change = _starts(marks, CHANGE)
        dists = [(n[:-6], t / NS) for n, t in marks if n.startswith("dist:") and n.endswith(":start")]
        assert dists
        for n, on in dists:
            end = next(t for m, t in marks if m == n + ":end") / NS
            assert not any(on <= c <= end for c in change), n  # no command change from onset to window end
            assert end - on == pytest.approx(P.WINDOW_S["nudge" if n.startswith("dist:nudge") else "creep"])
        # the names added here never match what Run.__init__ / Rec.__init__ treat specially
        for n in names:
            if n.startswith(("dist:", "req:")):
                assert n.split(":")[0] not in CHANGE + ("walk", "stop", "settle") and not n.startswith("walk:")
        assert len(names) == len(marks)  # every marker name is unique


def test_validate_plan_matches_prd():
    marks = _plan("validate").rec.marks
    req = [n for n, _ in marks if n.startswith("req:")]
    assert len(req) == 45
    for k in range(3):  # delta order per rep
        assert [float(n.split(":")[-1]) for n in req if n.split(":")[2] == str(k) and n.split(":")[1] == "fwd"] == list(P.DELTAS[k])
    assert [n for n, _ in marks if n.startswith("walk:")].__len__() == 2 * 54
    assert len([n for n, _ in marks if n.startswith("dist:") and n.endswith(":start")]) == 21
    t = dict((n, v) for n, v in marks)
    # request delta after the last static-command change, and the next hand-off starts at the request instant
    for i, d in enumerate(P.DELTAS[0]):
        back, r = t[f"back:A:fwd:0:{i}:start"], t[f"req:fwd:0:{i}:{d:.1f}"]
        assert abs((r - back) / NS - d) < 1e-6
        assert t[f"sw1:A:fwd:0:{i + 1}:start"] == r
    # onset 6.0 s after the block's last static-command change (rep 0 hold, rep 1 the -1 switch)
    assert abs((t["dist:nudge+x:0:start"] - t["hold:B:nudge+x:0:start"]) / NS - 6.0) < 1e-6
    assert abs((t["dist:nudge+x:1:start"] - t["switch:to_sentinel:B:nudge+x:1:start"]) / NS - 6.0) < 1e-6
    assert abs((t["hold:pre:B:nudge+x:1:end"] - t["switch:to_sentinel:B:nudge+x:1:start"])) < 1e-3 * NS


def test_calibrate_plan_matches_prd():
    marks = _plan("calibrate").rec.marks
    d = [n[5:-6] for n, _ in marks if n.startswith("dist:") and n.endswith(":start")]
    assert len(d) == 2 * (2 * 5 + 4 + 4 + 4)
    assert d[:4] == ["nudge+x@25:0", "nudge+x@25:1", "nudge+x@50:0", "nudge+x@50:1"]  # ascending, each value x2


def test_profiles():
    assert [P.profile_scale("nudge", t) for t in (-0.1, 0.0, 0.09, 0.1, 1.0)] == [0, 1, 1, 0, 0]
    got = [P.profile_scale("creep", t) for t in (0, 2, 4, 6.9, 7, 9, 11, 12)]
    assert got == pytest.approx([0, 0.5, 1, 1, 1, 0.5, 0, 0])


# ---- effectiveness ---------------------------------------------------------
class _W:
    def __init__(self, z, x=None, yaw=None, t1=30.0):
        self.t = np.arange(0, t1, 0.01)
        self.z, self.xy = z(self.t), np.stack([x(self.t) if x else np.zeros_like(self.t), np.zeros_like(self.t)], 1)
        self.yaw = np.zeros_like(self.t) if yaw is None else yaw(self.t)


def test_effective_uses_stationarity_not_z_band():
    seg, win = (0.0, 25.0), (3.0, 11.0)
    band = _W(z=lambda t: 0.8 + 0.004 * (1 - t / 20))  # slow 4 mm drift: violates only the 3 mm band (slope 0.2 mm/s)
    r = S.effectiveness(band, win, seg, "nudge")
    assert not r["effective"] and r["effective_s"] == 0.0 and r["z_band_only_s"] > 0.5
    fast = _W(z=lambda t: np.full_like(t, 0.8), x=lambda t: 0.05 * np.clip(t - 5, 0, 3))  # 5 cm/s for 3 s
    r = S.effectiveness(fast, win, seg, "nudge")
    assert r["effective"] and r["effective_s"] >= 0.5
    r = S.effectiveness(fast, win, seg, "creep")
    assert r["effective"] == (r["effective_s"] >= 1.0)
    holes = _W(z=lambda t: np.full_like(t, 0.8), x=lambda t: 0.05 * np.clip(t - 5, 0, 3))
    holes.z[400:700] = np.nan  # unknown instants do not count
    holes.xy[400:700] = np.nan
    assert S.effectiveness(holes, win, seg, "nudge")["unknown_s"] > 0


def _blk(eff):
    return {"effective": eff, "effective_s": 1.0 if eff else 0.0}


def test_calibration_picks_smallest_2of2():
    blocks = {("nudge+x", "25", 0): _blk(1), ("nudge+x", "25", 1): _blk(0),  # 1/2: not chosen
              ("nudge+x", "50", 0): _blk(1), ("nudge+x", "50", 1): _blk(1),
              ("nudge+y", "25", 0): _blk(1), ("nudge+y", "25", 1): _blk(1),
              ("lateral", "15", 0): _blk(0), ("lateral", "15", 1): _blk(0),
              ("lateral", "30", 0): _blk(1), ("lateral", "30", 1): _blk(1),
              ("down", "50", 0): _blk(0), ("down", "50", 1): _blk(0),
              ("down", "100", 0): _blk(1), ("down", "200", 0): _blk(1), ("down", "200", 1): _blk(1)}
    out = S.choose(blocks, {"down@100:1"})  # fall at 100 (rep 1): 100 and the rest of the ladder are unusable
    assert out["chosen"]["nudge"] == 50 and out["chosen"]["lateral"] == 30  # nudge = larger of +x (50) and +y (25)
    assert out["chosen_nudge_by_direction"] == {"nudge+x": 50, "nudge+y": 25}
    assert out["per_type"]["down"]["chosen"] is None and "down" in out["stop_reason"]
    assert out["per_type"]["down"]["values"][2]["unusable_after_fall"]
    assert out["needs_d0b_types"] == ["yaw"] and not out["complete"]


# ---- verdict, manifest ------------------------------------------------------
def test_verdict_order():
    ok = {t: 3 for t in S.TYPES}
    poor = dict(ok, yaw=1)
    assert S.verdict(1, poor, 0, ["x"]) == "FAILED"  # a false acceptance beats ineffective / missing / holds
    assert S.verdict(0, poor, 0, []) == "INCONCLUSIVE"  # ineffective beats usefulness
    assert S.verdict(0, ok, 30, ["missing"]) == "INCONCLUSIVE"
    assert S.verdict(0, ok, 25, []) == "FAILED" and S.verdict(0, ok, 26, []) == "PASS"
    assert S.verdict(0, dict(ok, yaw=2), 30, []) == "PASS"  # 2/3 is enough


def test_manifest_covers_imported_helpers():
    files = S.build_manifest(extra=[e for e in S.EXTRA if (S.ROOT / e).exists()])["files"]
    for f in ("g1b_still2.py", "g1b_still_thresholds.py", "g1b_analyze.py", "g1_analyze.py", "recorder.py", "g1b_still3.py"):
        assert f"gear_sonic/scripts/gates/{f}" in files, f
    for f in ("gear_sonic/utils/data_collection/zmq_state_subscriber.py", "outputs/gates/run_g1b_still3.sh",
              "gear_sonic/scripts/gates/g1b_still3_probe.py", "gear_sonic/utils/mujoco_sim/base_sim.py"):
        assert f in files, f
    assert files["outputs/gates/g1b_still2_frozen.json"] == S.STEP8_FROZEN_SHA


def test_validate_refuses_changed_helper(tmp_path, monkeypatch):
    helper, script, man, led = (tmp_path / n for n in ("helper.py", "script.py", "m.json", "l.md"))
    helper.write_text("v1")
    script.write_text("s")
    monkeypatch.setattr(S, "repo_modules", lambda root: {helper, script})
    man.write_text(json.dumps(S.build_manifest(tmp_path, [])))
    led.write_text(f"Step 9 FREEZE: manifest sha256={hashlib.sha256(man.read_bytes()).hexdigest()}\n")
    assert S.verify_freeze(man, led, tmp_path, []) is None
    helper.write_text("v2")
    assert "helper.py: changed" in S.verify_freeze(man, led, tmp_path, [])
    helper.write_text("v1")
    monkeypatch.setattr(S, "repo_modules", lambda root: {helper, script, tmp_path / "new.py"})
    (tmp_path / "new.py").write_text("n")
    assert "new.py: module not in the manifest" in S.verify_freeze(man, led, tmp_path, [])
    monkeypatch.setattr(S, "repo_modules", lambda root: {helper, script})
    man.write_text(man.read_text() + " ")
    assert "manifest sha256 differs" in S.verify_freeze(man, led, tmp_path, [])
    led.write_text("nothing\n")
    assert "ledger line" in S.verify_freeze(man, led, tmp_path, [])


# ---- end to end on a synthetic recording of the real plan -----------------
def _synth(marks, *, dist_motion=True, tracks=True, wrench=True, z_step=False):
    """sim_gt + g1_debug for the plan's markers: robot walks in walk windows (from the 0.5 s hand-off before), is pushed (0.1 m/s for 3 s) in
    disturbance windows if ``dist_motion``; the leg joints follow the pelvis x if ``tracks`` (else the feedback is blind)."""
    t_end = max(t for _, t in marks) / NS + 3
    t = np.arange(T0 - 2, t_end, 0.01)
    v = np.zeros_like(t)
    for n, ts in marks:
        if n.endswith(":start"):
            w0, w1 = ts / NS, next(e for m, e in marks if m == n[:-6] + ":end") / NS
            if n.startswith("walk:"):
                v[(t >= w0 - 0.5) & (t < w1)] = 0.3  # the robot starts moving at the hand-off
            if n.startswith("dist:") and dist_motion:
                v[(t >= w0) & (t < w0 + 3)] = 0.1
    x = np.cumsum(v) * 0.01
    z = np.full_like(t, 0.7984)
    wr = np.zeros((len(t), 6))
    for n, ts in marks:
        if n.startswith("dist:") and n.endswith(":start"):
            w0 = ts / NS
            typ = n[5:-6].rsplit(":", 1)[0].split("@")[0]
            kind, vec = P.TYPES[typ]
            mag = MAGS["nudge" if typ.startswith("nudge") else typ]
            m = np.nonzero((t >= w0) & (t < w0 + P.WINDOW_S[kind]))[0]
            wr[m] = np.array([mag * P.profile_scale(kind, t[j] - w0) for j in m])[:, None] * np.array(vec)
            if z_step == "bump":  # slow hump (1.5 mm/s, back to the shifted z before the segment end) peaking 4 mm above the pre-onset z
                tau = t - w0
                z += np.clip(0.009 * (1 - np.abs(tau - 7) / 6), 0, None) * (tau >= 1) * (tau < 13)
            if z_step:  # the disturbance leaves z 5 mm lower (ramped over its 3 s of motion) until the end of the segment
                end = next(e for m2, e in marks if m2.endswith(":end") and (m2.startswith(("hold:B:", "switch:to_sentinel:B:")))
                           and next(s_ for n2, s_ in marks if n2 == m2[:-4] + ":start") <= ts < e) / NS
                z[(t >= w0) & (t < end)] -= 0.005 * np.clip((t[(t >= w0) & (t < end)] - w0) / 3.0, 0, 1)  # while the robot is moving
    src = np.round(t * NS).astype(np.int64)
    g = np.arange(0, len(t), 2)  # 50 Hz feedback
    q = np.repeat((10 * x[g] if tracks else np.zeros(len(g)))[:, None], 12, axis=1)
    m_src = np.array([ts for _, ts in marks], dtype=np.int64)

    def objs(items):
        a = np.empty(len(items), dtype=object)
        a[:] = items
        return a

    return {
        "sim_gt_src_ns": src, "sim_gt_recv_ns": src + 200_000, "sim_gt_t_sim": t - T0,
        "sim_gt_raw": objs([dict({"pelvis_pos": [float(a), 0.0, float(c)], "pelvis_quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                            **({"pelvis_wrench": list(map(float, w))} if wrench else {})) for a, c, w in zip(x, z, wr)]),
        "g1_debug_src_ns": src[g], "g1_debug_recv_ns": src[g] + 200_000,
        "g1_debug_raw": objs([{"body_q": list(qq) + [0.0] * 17, "body_dq": [0.0] * 29, "base_ang_vel": [0.0, 0.0, 0.0]} for qq in q]),
        "markers_src_ns": m_src, "markers_recv_ns": m_src, "markers_raw": objs([{"name": n} for n, _ in marks]),
    }


def _mini():
    """Part A rep 0 (fwd) plus two nudge blocks: enough structure for the false-acceptance paths, ~2 min of sim time."""
    p = FakeProbe()
    p.run(3.0, 0, -1.0, name="settle:0")
    P.walk_stop_walk(p, p.rec, 0.7984, "fwd", 0)
    for rep in (0, 1):
        P.dist_block(p, p.rec, 0.7984, "nudge+x", 100, rep, "validate")
    return p.rec.marks


D_MAX, G_MAX = 0.0135184, 0.0037496


def test_validate_end_to_end_pass():
    marks = _plan("validate").rec.marks
    r = S.validate_run(_synth(marks), D_MAX, G_MAX, MAGS)
    assert r["missing"] == [] and len(r["request_table"]) == 45 and r["settled_holds"]["n"] == 30
    assert r["false_acceptances"] == 0 and r["effective_by_type"] == {t: 3 for t in S.TYPES}, r["false_by_class"]
    assert r["verdict"] == "PASS" and r["settled_holds"]["accepted"] == 30
    rows = {(x["dir"], x["rep"], x["i"]): x for x in r["request_table"]}
    assert rows[("fwd", 0, 2)]["delta_s"] == 4.5 and rows[("fwd", 0, 2)]["deployed_accept"]
    assert not rows[("fwd", 0, 0)]["deployed_accept"]  # delta 1.0 s: diagnostic only


def test_validate_blind_feedback_fails_and_missing_evidence_is_inconclusive():
    marks = _mini()
    r = S.validate_run(_synth(marks, tracks=False), D_MAX, G_MAX, MAGS)  # check accepts while moving: FAILED, even with data missing
    assert r["verdict"] == "FAILED" and r["false_by_class"]["walking"] > 0 and r["false_by_class"]["nudge"] > 0
    r = S.validate_run(_synth(marks), D_MAX, G_MAX, MAGS)  # no false acceptance, but most of the plan was not recorded
    assert r["false_acceptances"] == 0 and r["verdict"] == "INCONCLUSIVE" and r["missing"]
    r = S.validate_run(_synth(marks, dist_motion=False), D_MAX, G_MAX, MAGS)  # nothing moves in the windows: not effective
    assert r["effective_by_type"]["nudge+x"] == 0 and r["verdict"] == "INCONCLUSIVE"


def test_calibrate_reads_sim_gt_only(monkeypatch):
    import subprocess
    import sys

    code = "import sys, gear_sonic.scripts.gates.g1b_still3; sys.exit('g1b_still2' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0  # the feature code is not even imported
    monkeypatch.setattr(S, "_s2", lambda: (_ for _ in ()).throw(AssertionError("feature code used")))
    p = FakeProbe()
    p.run(3.0, 0, -1.0, name="settle:0")
    P.run_calibrate(p, p.rec, 0.7984, ("nudge+x",))
    d = _synth(p.rec.marks)
    d["g1_debug_raw"] = d["g1_debug_raw"][:0]  # no feedback at all
    blocks, falls = S.analyze_d0(d)
    assert len(blocks) == 10 and all(b["effective"] for b in blocks.values())
    out = S.choose(blocks, falls)
    assert out["per_type"]["nudge+x"]["chosen"] == 25 and out["needs_d0b_types"] == ["nudge+y", "lateral", "down", "yaw"]


def test_tick_sends_one_wrench_per_planner_message():
    sent = []
    sock = type("S", (), {"send": lambda self, d, *a: sent.append(("planner", d))})()
    t = P._Tick(sock, lambda: sent.append(("wrench", None)))
    t.send(b"abc")
    t.send(b"def")
    assert sent == [("planner", b"abc"), ("wrench", None), ("planner", b"def"), ("wrench", None)]


def test_marker_gap_instant_is_unknown_not_a_false_acceptance():
    p = FakeProbe()
    p.gap_ns = 15_000_000  # 15 ms between one segment's end and the next start: at least one 10 ms grid instant
    p.run(3.0, 0, -1.0, name="settle:0")
    P.walk_stop_walk(p, p.rec, 0.7984, "fwd", 0)
    for rep in (0, 1):
        P.dist_block(p, p.rec, 0.7984, "nudge+x", MAGS["nudge"], rep, "validate")
    d = _synth(p.rec.marks)
    rec = S._s2().Rec(d)
    a = rec.accept(D_MAX, G_MAX)
    cov = np.zeros(len(rec.run.t), bool)
    for w in rec.run.wins.values():
        cov |= (rec.run.t >= w[0]) & (rec.run.t < w[1])
    g = ~cov & a & rec.static & (rec.since >= 4.5)
    assert g.any() and not rec.run.gt[g].any() and not rec.run.unk[g].any()  # unfixed Run: a false acceptance
    r = S.validate_run(d, D_MAX, G_MAX, MAGS)
    assert r["marker_gap_instants"] > 0 and r["false_acceptances"] == 0
    assert all(x["ground_truth_still"] is not None for x in r["request_table"])  # request rows skip the gap


def test_pre_onset_uses_z_final_before_onset():
    r = S.validate_run(_synth(_mini(), z_step=True), D_MAX, G_MAX, MAGS)
    # a 5 mm final shift: settled pre-onset instants stay ground-truth still, and so do the first instants after onset
    # (within 3 mm of the pre-onset z_final), so the blind-to-z check has no false acceptance anywhere
    assert r["false_acceptances"] == 0, r["false_acceptance_runs"]


def test_band_violated_only_if_far_from_both_references():
    r = S.validate_run(_synth(_mini(), z_step="bump"), D_MAX, G_MAX, MAGS)
    f = r["false_acceptance_runs"]  # z 3 mm+ from both the pre-onset and the segment-end z_final, slow: band only
    assert f and all(x["ground_truth_cause"] == "band" for x in f)
    assert all(x["since_s"] > 6.0 + 4 for x in f)  # near the hump peak, not at the onset


def test_wrench_must_match_chosen_magnitude():
    marks = _mini()
    assert not [m for m in S.validate_run(_synth(marks), D_MAX, G_MAX, MAGS)["missing"] if "wrench" in m]
    bad = S.validate_run(_synth(marks), D_MAX, G_MAX, dict(MAGS, nudge=50))  # run recorded at 100 N, chosen 50
    assert any("peak wrench" in m for m in bad["missing"]) and bad["verdict"] == "INCONCLUSIVE"
    none = S.validate_run(_synth(marks, wrench=False), D_MAX, G_MAX, MAGS)
    assert any("no pelvis_wrench" in m for m in none["missing"])


def test_false_acceptance_runs_listed():
    r = S.validate_run(_synth(_mini(), tracks=False), D_MAX, G_MAX, MAGS)
    runs = r["false_acceptance_runs"]
    assert 0 < len(runs) <= 50 and r["false_acceptance_runs_total"] >= len(runs)
    f = runs[0]
    assert {"t_start", "t_end", "windows", "since_s", "D_over_Dmax", "G_over_Gmax", "ground_truth_cause"} <= f.keys()
    assert f["ground_truth_cause"] in ("walking", "stationarity", "band", "no segment") and f["windows"]
    assert set(r["post_stop_time_to_first_acceptance_s"]) == {"raw", "deployed"} and "sync_summary" in r
