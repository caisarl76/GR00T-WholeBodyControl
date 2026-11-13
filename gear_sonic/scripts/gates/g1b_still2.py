"""G1b Step 8: displacement-based still-standing check, pre-registered in PRD section 9 G1b Step 8.

Subcommands (JSON on stdout):
  develop A.npz B.npz           pooled development; writes outputs/gates/g1b_still2_develop.json
  freeze                        writes outputs/gates/g1b_still2_frozen.json, prints sha256 of script and JSON;
                                the operator then records them in the ledger line ``Step 8 FREEZE:``
  validate C.npz                refuses unless script and frozen JSON hashes match the ledger line; then validates once

Feature (trailing 1 s window, only samples with recv <= t; newest age < 0.1 s, no gap > 2 periods -- the
``g1b_still_thresholds.window_stats`` rules, imported):
  D = max over the 12 leg joints of |body_q(newest) - body_q(oldest sample used by the window)| [rad]
  G = || mean of base_ang_vel over the window samples || [rad/s]
  still iff D < D_max and G < G_max. Ground truth, NaN handling and class labels: g1b_still_thresholds.
Conservative readings: "oldest in window" is the last sample at or before t-1.0 s (it also anchors the
coverage check, so the baseline spans >= 1.0 s: larger D, fewer acceptances); the "static-command change" set
is the Step 7 set (starts of prehold, ramp, hold, sw1, back, switch:*) plus settle:0 and walk-stop starts, so
scope starts are never earlier than the stricter reading; ramp windows are never static instants; walking
instants always count as false-acceptance candidates when validating (class d, in the total).
Development: static instants >= 0.5 s after a change, A and B pooled; boundary = zero-false-acceptance grid
pair maximizing accepted gt-still instants (ties -> smaller); final = 0.8 x boundary; gate = zero in-sample
false acceptances and >= 26/30 pre-walk holds accepted before hold end.
Validation (run C, once): instants >= 4.5 s since the last change; PASS = zero false acceptances (overall and by
class) and >= 13/15 pre-walk and >= 9/10 isolated holds accepted within [4.5 s, 6.0 s] after their change.
Otherwise FAILED; INCONCLUSIVE only for missing data. No re-tuning after C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from gear_sonic.scripts.gates.g1b_still_thresholds import N_LEG, WINDOW_S, Run, window_stats
from gear_sonic.scripts.gates.recorder import _load

GRID = np.round(np.geomspace(1e-4, 1.0, 80), 6)  # candidate D_max (rad) and G_max (rad/s), fixed before data
MARGIN = 0.8
DEV_SCOPE_S, VAL_SCOPE_S, VAL_WIN_S = 0.5, 4.5, 6.0
DEV_HOLDS_MIN = 26  # of 30 pre-walk holds in A+B
VAL_HOLDS_MIN, VAL_ISO_MIN = 13, 9
HERE = Path(__file__).resolve()
OUT = Path("outputs/gates")
DEVELOP_JSON, FROZEN_JSON = OUT / "g1b_still2_develop.json", OUT / "g1b_still2_frozen.json"
LEDGER = Path("/home/jihun/orca/workspaces/GR00T-WholeBodyControl/teleop/.superpowers/sdd/2026-11-09-pico-manipulation-mode/progress.md")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_features(samples, t):
    """(D, G) at each instant of ``t`` (nan where the structural window rules fail); samples carry q (N,12), w (N,3)."""
    r = samples["recv_s"]
    hi = np.searchsorted(r, t, side="right") - 1
    lo = np.searchsorted(r, t - WINDOW_S, side="right") - 1
    D, G = np.full(len(t), np.nan), np.full(len(t), np.nan)
    cs = np.vstack([np.zeros(3), np.cumsum(samples["w"], axis=0)])
    for j in range(len(t)):
        if lo[j] < 0 or hi[j] <= lo[j] or window_stats(samples, t[j])[0] is not None:
            continue
        D[j] = np.abs(samples["q"][hi[j]] - samples["q"][lo[j]]).max()
        G[j] = np.linalg.norm((cs[hi[j] + 1] - cs[lo[j] + 1]) / (hi[j] - lo[j]))
    return D, G


class Rec:
    """A Run plus D/G features and scope labels."""

    def __init__(self, npz):
        self.run = run = Run(npz)
        d = _load(npz)
        raw = d["g1_debug_raw"]
        run.samples["q"] = np.stack([np.asarray(m["body_q"], float)[:N_LEG] for m in raw])
        run.samples["w"] = np.stack([np.asarray(m["base_ang_vel"], float) for m in raw])
        self.D, self.G = window_features(run.samples, run.t)
        ok = ~np.isnan(self.D)
        walk = np.array([any(a <= t < b for a, b in run.walk) for t in run.t])
        ramp = np.array([any(w[0] <= t < w[1] for n, w in run.wins.items() if n.startswith("ramp:")) for t in run.t])
        stops = [w[0] for n, w in run.wins.items() if n.startswith("stop:")]
        settle = [w[0] for n, w in run.wins.items() if n.startswith("settle:")]
        ch = np.sort(np.array(list(run.change) + stops + settle))
        k = np.searchsorted(ch, run.t, side="right") - 1
        self.since = np.where(k >= 0, run.t - ch[np.maximum(k, 0)], -1.0)
        self.walk, self.static = walk, ~walk & ~ramp
        self.ok = ok

    def accept(self, d_max, g_max):
        return self.ok & (self.D < d_max) & (self.G < g_max)


def in_scope(rec, scope_s, include_walk=False):
    m = rec.static & (rec.since >= scope_s)
    return m | rec.walk if include_walk else m


def pooled_boundary(recs, scope_s=DEV_SCOPE_S):
    """Zero-false-acceptance pair maximizing accepted gt-still instants on the pooled static scope."""
    D = np.concatenate([r.D[in_scope(r, scope_s)] for r in recs])
    G = np.concatenate([r.G[in_scope(r, scope_s)] for r in recs])
    gt = np.concatenate([r.run.gt[in_scope(r, scope_s)] for r in recs])
    unk = np.concatenate([r.run.unk[in_scope(r, scope_s)] for r in recs])
    ok = ~np.isnan(D)
    best = None
    for d in GRID:
        for g in GRID:
            a = ok & (D < d) & (G < g)
            if (a & ~gt & ~unk).any():
                continue
            key = (-int((a & gt).sum()), d, g)
            if best is None or key < best:
                best = key
    return None if best is None else {"D_boundary": float(best[1]), "G_boundary": float(best[2]), "accepted_gt_still": -best[0]}


def hold_counts(rec, a, names, lo_s, hi_s):
    """(n, accepted) of holds with an accepted instant in [start+lo_s, start+hi_s) and first-acceptance delays."""
    wins = [w for n, w in sorted(rec.run.wins.items()) if n.startswith(names)]
    acc, delays = 0, []
    for w0, w1 in wins:
        m = (rec.run.t >= w0 + lo_s) & (rec.run.t < min(w1, w0 + hi_s)) & a
        if m.any():
            acc += 1
            delays.append(float(rec.run.t[np.nonzero(m)[0][0]] - w0))
    return {"n": len(wins), "accepted": acc, "first_accept_after_change_s": delays}


def evaluate(rec, d_max, g_max, scope_s, win_hi_s, include_walk):
    run, a = rec.run, rec.accept(d_max, g_max)
    sc = in_scope(rec, scope_s, include_walk)
    fa = a & ~run.gt & ~run.unk & sc
    by = {c: int(sum(1 for j in np.nonzero(fa)[0] if run.cls[j] == c)) for c in "abcd"}
    gts = sc & run.gt
    return {
        "scope_s": scope_s, "scope_instants": int(sc.sum()), "unknown_gt_instants": int((run.unk & sc).sum()),
        "gt_still_instants": int(gts.sum()), "accepted_gt_still_fraction": float((a & gts).sum() / max(1, gts.sum())),
        "false_acceptances": int(fa.sum()), "false_by_class": by,
        "pre_walk_holds": hold_counts(rec, a & sc, ("hold:",), scope_s, win_hi_s),
        "isolated_holds": hold_counts(rec, a & sc, ("switch:to_sentinel:", "switch:to_height:"), scope_s, win_hi_s),
    }


def verdict(res, n_holds=15, n_iso=10):
    """PASS / FAILED / INCONCLUSIVE (missing data only) for a validation result."""
    h, i = res["pre_walk_holds"], res["isolated_holds"]
    if h["n"] != n_holds or i["n"] != n_iso or res["gt_still_instants"] == 0:
        return "INCONCLUSIVE"
    if res["false_acceptances"] or h["accepted"] < VAL_HOLDS_MIN or i["accepted"] < VAL_ISO_MIN:
        return "FAILED"
    return "PASS"


def cmd_develop(a):
    recs = [Rec(a.A), Rec(a.B)]
    bd = pooled_boundary(recs)
    if bd is None:
        out = {"verdict": "FAILED", "reason": "no zero-false-acceptance pair in development"}
    else:
        d, g = MARGIN * bd["D_boundary"], MARGIN * bd["G_boundary"]
        ev = [evaluate(r, d, g, DEV_SCOPE_S, 1e9, False) for r in recs]
        fa = sum(e["false_acceptances"] for e in ev)
        holds = sum(e["pre_walk_holds"]["accepted"] for e in ev)
        n = sum(e["pre_walk_holds"]["n"] for e in ev)
        fa_all = sum(evaluate(r, d, g, DEV_SCOPE_S, 1e9, True)["false_acceptances"] for r in recs)
        ok = fa == 0 and n == 30 and holds >= DEV_HOLDS_MIN
        out = {"boundary": bd, "D_max": d, "G_max": g, "in_sample_false_acceptances": fa,
               "in_sample_false_acceptances_incl_walking": fa_all, "pre_walk_holds_accepted": f"{holds}/{n}",
               "isolated_holds_info": [e["isolated_holds"] for e in ev], "runs": {"A": ev[0], "B": ev[1]},
               "development_gate": "PASS" if ok else "FAILED", "verdict": None if ok else "FAILED (development gate)"}
    OUT.mkdir(parents=True, exist_ok=True)
    DEVELOP_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out.get("development_gate") == "PASS" else 1


def cmd_freeze(a):
    dev = json.loads(DEVELOP_JSON.read_text())
    if dev.get("development_gate") != "PASS":
        print("development gate not passed: nothing to freeze", file=sys.stderr)
        return 1
    FROZEN_JSON.write_text(json.dumps({"D_max": dev["D_max"], "G_max": dev["G_max"], "script_sha256": sha256(HERE)}, indent=2))
    print(f"script_sha256={sha256(HERE)} frozen_json_sha256={sha256(FROZEN_JSON)}")
    return 0


def check_hashes(ledger=LEDGER, frozen=FROZEN_JSON, script=HERE):
    """None if script and frozen JSON match the ledger ``Step 8 FREEZE:`` line, else a reason."""
    lines = [l for l in Path(ledger).read_text().splitlines() if l.startswith("Step 8 FREEZE:")]
    if not lines:
        return "no 'Step 8 FREEZE:' ledger line"
    line = lines[-1]
    for label, path in (("script", script), ("frozen json", frozen)):
        if sha256(path) not in line:
            return f"{label} sha256 changed since the freeze"
    if json.loads(Path(frozen).read_text())["script_sha256"] != sha256(script):
        return "script sha256 in the frozen json differs"
    return None


def cmd_validate(a):
    why = check_hashes(a.ledger)
    if why:
        print(json.dumps({"verdict": "REFUSED", "reason": why}, indent=2))
        return 2
    fz = json.loads(FROZEN_JSON.read_text())
    try:
        rec = Rec(a.C)
    except ValueError as e:
        print(json.dumps({"verdict": "INCONCLUSIVE", "reason": str(e)}, indent=2))
        return 1
    d, g = fz["D_max"], fz["G_max"]
    main = evaluate(rec, d, g, VAL_SCOPE_S, VAL_WIN_S, True)
    out = {"run": a.C, "D_max": d, "G_max": g, "validation_4p5s": main, "verdict": verdict(main),
           "diagnostic_0p5s_scope": evaluate(rec, d, g, DEV_SCOPE_S, 1e9, True),
           "nominal_period_s": rec.run.samples["period_s"]}
    print(json.dumps(out, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("develop"); p.add_argument("A"); p.add_argument("B"); p.set_defaults(f=cmd_develop)
    p = sub.add_parser("freeze"); p.set_defaults(f=cmd_freeze)
    p = sub.add_parser("validate"); p.add_argument("C"); p.add_argument("--ledger", default=str(LEDGER)); p.set_defaults(f=cmd_validate)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
