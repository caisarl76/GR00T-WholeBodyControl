"""G1b Step 9 analysis: decision-time stress test of the Step 8 check, pre-registered in PRD section 9 "Step 9".

Subcommands (JSON on stdout):
  calibrate D0.npz [D0b.npz]  run D0 (sim_gt only; the D/G feature is never computed): effectiveness of every
                              disturbance, chosen magnitude per type; writes outputs/gates/g1b_still3_calibration.json
  freeze                      writes outputs/gates/g1b_still3_manifest.json (sha256 of this script, every repo-local
                              module in sys.modules after importing it, the probe, the sim files, the Step 8 frozen
                              JSON and the calibration JSON); prints the manifest sha256 for the ledger line
                              ``Step 9 FREEZE:``
  validate D.npz              refuses (exit 2, REFUSED) unless every manifest entry and the manifest sha256 in the
                              ledger match; then evaluates run D once with the frozen Step 8 thresholds

Effectiveness of a disturbance: instants (100 Hz grid) inside its window where the G1b ``_stationary`` criterion is
violated on the trailing 1 s sim_gt window (``g1b_still_thresholds.gt_still`` with the 3 mm z band switched off), x 0.01 s:
>= 0.5 s (nudge) / >= 1.0 s (creep). Unknown instants (sim_gt holes) do not count. Non-still instants caused by the
z band alone are reported separately. Calibration: smallest ladder value effective in 2/2 reps without a fall; a fall
makes that value and the rest of its ladder unusable.
Validation classes of false acceptances (precedence walking, post-stop <= 10 s after a walk end, nudge window, creep
window, other static); walking counts every walking instant, as in Step 8 (also reported at >= 4.5 s since the change).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from gear_sonic.scripts.gates.g1_analyze import STREAMS, _Windows
from gear_sonic.scripts.gates.g1b_still3_probe import CAL_TYPES, TYPES, ladder
from gear_sonic.scripts.gates.g1b_still_thresholds import FINAL_S, HZ, STATIC, gt_still
from gear_sonic.scripts.gates.recorder import _load, check_sync, check_sync

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/gates"
CAL_JSON, MANIFEST = OUT / "g1b_still3_calibration.json", OUT / "g1b_still3_manifest.json"
STEP8_FROZEN = OUT / "g1b_still2_frozen.json"
STEP8_FROZEN_SHA = "678b6e2fa3f84a1cce5784eb733dd36ff82d34dcb95d2697b7fcc94b41a6f685"
FREEZE_KEY = "Step 9 FREEZE:"
EXTRA = [  # hashed besides the imported repo-local modules (the sim files are not importable in the analysis env)
    "gear_sonic/scripts/gates/g1b_still3_probe.py", "gear_sonic/utils/mujoco_sim/base_sim.py",
    "gear_sonic/utils/mujoco_sim/configs.py", "gear_sonic/scripts/run_sim_loop.py",
    "gear_sonic/utils/data_collection/zmq_state_subscriber.py", "outputs/gates/run_g1b_still3.sh",
    str(STEP8_FROZEN.relative_to(ROOT)), str(CAL_JSON.relative_to(ROOT)),
]
EFFECTIVE_S = {"nudge": 0.5, "creep": 1.0}
DT = 1.0 / HZ
SCOPE_S, SETTLED_HI_S, ONSET_S, POST_STOP_S = 4.5, 6.0, 0.25, 10.0
N_REQ, N_HOLDS, HOLDS_MIN, EFF_MIN = 45, 30, 26, 2
CLASSES = ("post_stop", "nudge", "creep", "other", "walking")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _s2():
    return importlib.import_module("gear_sonic.scripts.gates.g1b_still2")  # D/G feature; never loaded by calibrate


# ---- effectiveness (sim_gt only) ----------------------------------------
def seg_final(W, w):
    """z_final of a static segment as in Run: median z over its last 0.5 s (None without data)."""
    m = (W.t >= w[1] - FINAL_S) & (W.t < w[1]) & ~np.isnan(W.z)
    return float(np.median(W.z[m])) if m.any() else None


def stationarity(W, i, final):
    """(stationary, ground truth) at grid index ``i``; each True / False / None (unknown).

    ``gt_still`` with ``seg_final`` set to the newest valid z of the window has a zero z-drift term, so it reduces to
    the unchanged G1b ``_stationary`` criterion (and its NaN / hole rules); with the real ``final`` it is Step 8 truth.
    """
    z = W.z[max(i - HZ, 0) : i + 1]
    z = z[~np.isnan(z)]
    st = gt_still(W.t, W.z, W.xy, W.yaw, float(z[-1]) if len(z) else 0.0, i)
    return st, gt_still(W.t, W.z, W.xy, W.yaw, final, i)


def effectiveness(W, win, seg, kind):
    """Motion confirmation of one disturbance window ``win`` (onset, end) inside static segment ``seg``."""
    final = None if seg is None else seg_final(W, seg)
    i0, i1 = np.searchsorted(W.t, [win[0], win[1]])
    viol = band = unk = 0
    for i in range(i0, i1):
        st, gt = stationarity(W, i, final) if final is not None else (None, None)
        unk += st is None
        viol += st is False
        band += st is True and gt is False
    need = EFFECTIVE_S[kind]
    return {"effective_s": round(viol * DT, 4), "z_band_only_s": round(band * DT, 4), "unknown_s": round(unk * DT, 4),
            "effective": bool(final is not None and viol * DT >= need - 1e-9), "needed_s": need}


def split_tag(tag):
    typ, rep = tag.rsplit(":", 1)
    return typ, int(rep)


def containing(W, prefixes, t):
    """The static window starting with one of ``prefixes`` that contains instant ``t``."""
    for n in sorted(k[: -len(":start")] for k in W.markers if k.endswith(":start") and k.startswith(prefixes)):
        w = W.win(n)
        if w and w[0] <= t < w[1]:
            return w
    return None


def analyze_d0(npz):
    """({(type, mag, rep): result}, {fall tags}) of one calibration recording."""
    W = _Windows(_load(npz))
    if W.z is None:
        raise ValueError("missing sim_gt")
    blocks, falls = {}, set()
    for k in W.markers:
        if k.startswith("fall:") and k.endswith(":start"):
            falls.add(k[len("fall:") : -len(":start")])
        if not (k.startswith("dist:") and k.endswith(":start")):
            continue
        tag = k[len("dist:") : -len(":start")]
        win = W.win(f"dist:{tag}")
        typ_mag, rep = split_tag(tag)
        typ, mag = typ_mag.split("@")
        if win is None:
            continue
        seg = containing(W, ("hold:C:",), win[0])
        blocks[(typ, mag, rep)] = effectiveness(W, win, seg, TYPES[typ][0])
    return blocks, falls


def choose(blocks, falls):
    """Per type and value rows, the chosen magnitude per type, per the PRD rule."""
    per_type, chosen = {}, {}
    for typ in CAL_TYPES:
        rows, fell, pick = [], False, None
        has_data = any(k[0] == typ for k in blocks) or any(f.startswith(typ + "@") for f in falls)
        for mag in ladder(typ):
            m = f"{mag:g}"
            reps = [blocks.get((typ, m, r)) for r in (0, 1)]
            fall = any(f"{typ}@{m}:{r}" in falls for r in (0, 1))
            usable = not fell and not fall
            eff = usable and all(r is not None and r["effective"] for r in reps)
            rows.append({"magnitude": mag, "fall": fall, "unusable_after_fall": fell, "effective_2of2": eff,
                         "reps": [dict(r, rep=i) if r else {"rep": i, "missing": True} for i, r in enumerate(reps)]})
            fell = fell or fall
            if eff and pick is None:
                pick = mag
        per_type[typ] = {"values": rows, "chosen": pick, "status": ("chosen" if pick is not None else
                         "no_effective_value" if has_data else "no_data")}
        chosen[typ] = pick
    stop = [f"{t}: no ladder value is effective in 2/2 without a fall (ladder not extended without approval)"
            for t, v in per_type.items() if v["status"] == "no_effective_value"]
    needs = [t for t, v in per_type.items() if v["status"] == "no_data"]
    final = {"lateral": chosen["lateral"], "down": chosen["down"], "yaw": chosen["yaw"],
             "nudge": None if None in (chosen["nudge+x"], chosen["nudge+y"]) else max(chosen["nudge+x"], chosen["nudge+y"])}
    return {"per_type": per_type, "chosen": final, "chosen_nudge_by_direction": {k: chosen[k] for k in ("nudge+x", "nudge+y")},
            "stop_reason": "; ".join(stop) or None, "needs_d0b_types": needs,
            "complete": not stop and not needs and None not in final.values()}


def cmd_calibrate(a):
    blocks, falls = {}, set()
    try:
        for f in a.npz:
            b, fl = analyze_d0(f)
            for k, v in b.items():
                blocks.setdefault(k, v)
            falls |= fl
    except ValueError as e:
        print(json.dumps({"verdict": "INCONCLUSIVE", "reason": str(e)}, indent=2))
        return 1
    out = {"files": a.npz, **choose(blocks, falls), "falls": sorted(falls)}
    CAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAL_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["complete"] else (1 if out["stop_reason"] else 3)


# ---- freeze / manifest ----------------------------------------------------
def repo_modules(root=ROOT):
    """Files of every repo-local module currently in sys.modules."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py"):
            p = Path(f).resolve()
            if root in p.parents and "site-packages" not in p.parts:
                out.add(p)
    return out


def file_hashes(paths, root=ROOT):
    return {str(Path(p).resolve().relative_to(root)): sha256(p) for p in sorted({Path(q).resolve() for q in paths})}


def build_manifest(root=ROOT, extra=EXTRA):
    _s2()  # the validate path's imports must be in sys.modules before the scan
    paths = repo_modules(root) | {root / e for e in extra}
    return {"files": file_hashes(paths, root)}


def verify_freeze(manifest=MANIFEST, ledger=None, root=ROOT, extra=EXTRA):
    """None if the manifest, every entry and the Step 8 frozen JSON are as frozen, else a reason."""
    ledger = _s2().LEDGER if ledger is None else ledger
    if not Path(manifest).exists():
        return "no manifest"
    lines = [l for l in Path(ledger).read_text().splitlines() if l.startswith(FREEZE_KEY)]
    if not lines:
        return f"no '{FREEZE_KEY}' ledger line"
    if sha256(manifest) not in lines[-1]:
        return "manifest sha256 differs from the last ledger line"
    frozen = json.loads(Path(manifest).read_text())["files"]
    now = build_manifest(root, extra)["files"]
    bad = [f"{k}: {'missing' if k not in now else 'changed'}" for k in frozen if now.get(k) != frozen[k]]
    bad += [f"{k}: module not in the manifest" for k in now if k not in frozen]
    if bad:
        return "; ".join(bad)
    if root == ROOT and frozen.get(str(STEP8_FROZEN.relative_to(ROOT))) != STEP8_FROZEN_SHA:
        return "Step 8 frozen json is not the frozen one"
    return None


def cmd_freeze(a):
    if sha256(STEP8_FROZEN) != STEP8_FROZEN_SHA:
        print(f"refused: {STEP8_FROZEN} is not the Step 8 frozen JSON ({STEP8_FROZEN_SHA})", file=sys.stderr)
        return 1
    if not CAL_JSON.exists() or not json.loads(CAL_JSON.read_text())["complete"]:
        print("refused: no complete calibration JSON", file=sys.stderr)
        return 1
    MANIFEST.write_text(json.dumps(build_manifest(), indent=2))
    print(f"manifest_sha256={sha256(MANIFEST)} entries={len(json.loads(MANIFEST.read_text())['files'])}")
    return 0


# ---- validation (run D) ---------------------------------------------------
def verdict(false_acceptances, effective, holds_accepted, missing):
    """FAILED (any false acceptance) > INCONCLUSIVE (< 2/3 effective in a type, or missing data) > FAILED (< 26/30 holds) > PASS."""
    if false_acceptances:
        return "FAILED"
    if missing or any(effective.get(t, 0) < EFF_MIN for t in TYPES):
        return "INCONCLUSIVE"
    return "FAILED" if holds_accepted < HOLDS_MIN else "PASS"


def _first(mask, t, lo):
    """Seconds from ``lo`` to the first True of ``mask`` (None if never)."""
    k = np.nonzero(mask)[0]
    return float(t[k[0]] - lo) if len(k) else None


def wrench_problems(d, dists, mags):
    """Missing-data reasons from the sim_gt ``pelvis_wrench``: peak per window on the right axis/sign within 1 %,
    off-axis ~0 inside windows and ~0 outside all windows."""
    src = np.asarray(d["sim_gt_src_ns"]) / 1e9
    raw = d["sim_gt_raw"]
    if any("pelvis_wrench" not in m for m in raw):
        return ["sim_gt has no pelvis_wrench (sim started without --disturbance-port?)"]
    w = np.array([m["pelvis_wrench"] for m in raw], float).reshape(len(raw), 6)
    out, inside = [], np.zeros(len(src), bool)
    for n, win in sorted(dists.items()):
        typ, _ = split_tag(n[len("dist:"):])
        vec = np.array(TYPES[typ][1], float)
        mag = float(mags["nudge" if typ.startswith("nudge") else typ])
        m = (src >= win[0]) & (src < win[1])
        inside |= m
        if not m.any():
            out.append(f"{n}: no sim_gt in the window")
            continue
        peak = float((w[m] @ vec).max())
        off = float(np.abs(w[m] - np.outer(w[m] @ vec, vec)).max())
        if abs(peak - mag) > 0.01 * mag or off > 0.01 * mag:
            out.append(f"{n}: peak wrench {peak:g} (off-axis {off:g}) != chosen {mag:g}")
    stray = float(np.abs(w[~inside]).max(initial=0.0))
    if stray > 1e-6:
        out.append(f"wrench {stray:g} outside the disturbance windows")
    return out


def validate_run(npz, d_max, g_max, mags):
    s2 = _s2()
    rec = s2.Rec(npz)
    run, t = rec.run, rec.run.t
    d = _load(npz)
    W = _Windows(d)
    a = rec.accept(d_max, g_max)
    dists = {n: w for n, w in run.wins.items() if n.startswith("dist:")}
    statics = sorted((w, n) for n, w in run.wins.items() if n.split(":")[0] in STATIC)

    def static_seg(x):
        return next((w for w, _ in statics if w[0] <= x < w[1]), None)

    # amendment 2: before an onset the 3 mm band is taken around the z held just before it; after, around either reference
    pre_final = {}
    for n, w in dists.items():
        seg = static_seg(w[0])
        m = (W.t >= w[0] - FINAL_S) & (W.t < w[0]) & ~np.isnan(W.z)
        if seg is not None and m.any():
            pre_final[n] = (seg, w[0], float(np.median(W.z[m])))
            for j in np.nonzero((t >= seg[0]) & (t < w[0]))[0]:
                g = gt_still(W.t, W.z, W.xy, W.yaw, pre_final[n][2], int(np.searchsorted(W.t, t[j])))
                run.unk[j], run.gt[j] = g is None, bool(g)
            # from onset to the segment end the band is violated only if z is >= 3 mm from BOTH references
            for j in np.nonzero((t >= w[0]) & (t < seg[1]))[0]:
                g = gt_still(W.t, W.z, W.xy, W.yaw, pre_final[n][2], int(np.searchsorted(W.t, t[j])))
                run.gt[j] |= bool(g)
    # amendment 1: instants in no (non-dist) marker window are unknown (microsecond gaps between segments)
    covered = np.zeros(len(t), bool)
    for n, w in run.wins.items():
        if not n.startswith(("dist:", "req:")):
            covered |= (t >= w[0]) & (t < w[1])
    gap = ~covered
    run.unk |= gap
    run.gt &= ~gap

    sc = s2.in_scope(rec, SCOPE_S)
    dep = a & sc  # deployed decisions on static instants; walking instants all count (Step 8)
    cls = np.full(len(t), "other", dtype=object)
    for _, b in run.walk:
        cls[(t >= b) & (t < b + POST_STOP_S)] = "post_stop"
    onset = np.zeros(len(t), bool)
    for n, w in dists.items():
        cls[(t >= w[0]) & (t < w[1])] = "nudge" if n[len("dist:"):].startswith("nudge") else "creep"
        onset |= (t >= w[0]) & (t < w[0] + ONSET_S)
    cls[rec.walk] = "walking"
    bad = ~run.gt & ~run.unk
    fa = (dep & bad) | (a & rec.walk & bad)

    rows, req_fa = [], 0
    for k in sorted((k for k in W.markers if k.startswith("req:")), key=lambda k: W.markers[k][0]):
        _, dr, rep, i, delta = k.split(":")
        j = int(np.searchsorted(t, W.markers[k][0], side="left")) - 1  # last instant before the request
        while j >= 0 and gap[j]:  # not one in the [back:end, req) gap
            j -= 1
        raw = bool(a[j]) if j >= 0 else False
        deployed = raw and float(delta) >= SCOPE_S
        gt = None if j < 0 or run.unk[j] else bool(run.gt[j])
        req_fa += bool(deployed and gt is False and not fa[j])
        rows.append({"dir": dr, "rep": int(rep), "i": int(i), "delta_s": float(delta), "ground_truth_still": gt,
                     "check_accepts": raw, "deployed_accept": deployed,
                     "since_s": float(rec.since[j]) if j >= 0 else None,
                     "D_over_Dmax": float(rec.D[j] / d_max) if j >= 0 else None,
                     "G_over_Gmax": float(rec.G[j] / g_max) if j >= 0 else None})
    fa_total = int(fa.sum()) + req_fa
    by = {c: int((fa & (cls == c)).sum()) for c in CLASSES}

    # false-acceptance runs (contiguous instants merged), capped
    idx = np.nonzero(fa)[0]
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []
    fa_runs = []
    for r in runs[:50]:
        j = int(r[0])
        if rec.walk[j]:
            cause = "walking"
        elif (seg := static_seg(t[j])) is None:
            cause = "no segment"
        else:
            pf = next((v[2] for v in pre_final.values() if v[0] == seg and t[j] < v[1]), None)
            st, gt = stationarity(W, int(np.searchsorted(W.t, t[j])), seg_final(W, seg) if pf is None else pf)
            cause = "stationarity" if st is False else "band" if gt is False else "unknown"
        fa_runs.append({"t_start": float(t[r[0]]), "t_end": float(t[r[-1]]), "n_instants": len(r),
                        "windows": sorted(n for n, w in run.wins.items() if w[0] <= t[j] < w[1]),
                        "since_s": float(rec.since[j]), "D_over_Dmax": float(rec.D[j] / d_max), "G_over_Gmax": float(rec.G[j] / g_max),
                        "class": str(cls[j]), "ground_truth_cause": cause})

    gts = (sc | rec.walk) & run.gt
    holds = s2.hold_counts(rec, dep, ("hold:A:", "hold:B:", "switch:to_sentinel:B:"), SCOPE_S, SETTLED_HI_S)
    post = {"raw": [], "deployed": []}
    for n, w in sorted(run.wins.items()):
        if n.startswith("back:A:") and n.endswith(":post"):
            m = (t >= w[0]) & (t < w[1]) & rec.static
            post["raw"].append(_first(m & a, t, w[0]))
            post["deployed"].append(_first(m & dep, t, w[0]))

    eff, per_dist, missing = {k: 0 for k in TYPES}, [], []
    n_by = {k: 0 for k in TYPES}
    for n, w in sorted(dists.items()):
        typ, rep = split_tag(n[len("dist:"):])
        seg = containing(W, ("hold:B:", "switch:to_sentinel:B:"), w[0])
        r = effectiveness(W, w, seg, TYPES[typ][0])
        seg_end = seg[1] if seg else w[1]
        later = (t >= w[0]) & (t < seg_end) & rec.static
        nonstill = np.nonzero(later & bad)[0]
        delay = {"raw": None, "deployed": None}
        if len(nonstill) and nonstill[-1] + 1 < len(t):
            t_still = t[nonstill[-1] + 1]
            m = (t >= t_still) & (t < seg_end)
            delay = {"raw": _first(m & a, t, t_still), "deployed": _first(m & dep, t, t_still)}
        n_by[typ] += 1
        eff[typ] += r["effective"]
        per_dist.append({"window": n, "type": typ, "rep": rep, **r, "still_to_first_acceptance_s": delay})
    for typ in TYPES:
        if n_by[typ] != 3:
            missing.append(f"{typ}: {n_by[typ]} disturbance windows, expected 3")
    if len(rows) != N_REQ:
        missing.append(f"{len(rows)} walk requests, expected {N_REQ}")
    if holds["n"] != N_HOLDS:
        missing.append(f"{holds['n']} settled holds, expected {N_HOLDS}")
    missing += wrench_problems(d, dists, mags)
    falls = sorted(k for k in W.markers if k.startswith("fall:"))
    if falls:
        missing.append(f"fall recorded: {falls}")
    try:
        t0, t1 = min(w[0] for w in run.wins.values()), max(w[1] for w in run.wins.values())
        rep_ = check_sync(d, STREAMS, (t0, t1))
        sync = {"ok": rep_.ok, "reasons": rep_.reasons[:10], "skew_p99_ms": rep_.skew_p99_ms, "max_gap_ms": rep_.max_gap_ms, "rtf": rep_.rtf}
    except Exception as e:  # information only
        sync = {"error": str(e)}
    v = verdict(fa_total, eff, holds["accepted"], missing)
    return {"verdict": v, "false_acceptances": fa_total, "false_by_class": by, "false_onset_first_0p25s": int((fa & onset).sum()),
            "false_extra_from_requests": req_fa, "false_acceptance_runs": fa_runs, "false_acceptance_runs_total": len(runs),
            "walking_at_or_after_4p5s": int((a & rec.walk & (rec.since >= SCOPE_S)).sum()),
            "unknown_gt_instants_excluded": int((run.unk & (sc | rec.walk)).sum()), "marker_gap_instants": int(gap.sum()),
            "request_table": rows, "settled_holds": holds, "post_stop_time_to_first_acceptance_s": post,
            "accepted_gt_still_fraction": float((a & gts).sum() / max(1, gts.sum())), "gt_still_instants": int(gts.sum()),
            "effective_by_type": eff, "disturbances": per_dist, "missing": missing, "sync_summary": sync}


def cmd_validate(a):
    why = verify_freeze(ledger=a.ledger)
    if why:
        print(json.dumps({"verdict": "REFUSED", "reason": why}, indent=2))
        return 2
    fz = json.loads(STEP8_FROZEN.read_text())
    try:
        mags = json.loads(CAL_JSON.read_text())["chosen"]
        out = {"run": a.D, "D_max": fz["D_max"], "G_max": fz["G_max"], "magnitudes": mags,
               **validate_run(a.D, fz["D_max"], fz["G_max"], mags)}
    except ValueError as e:
        out = {"verdict": "INCONCLUSIVE", "reason": str(e)}
    print(json.dumps(out, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate"); p.add_argument("npz", nargs="+"); p.set_defaults(f=cmd_calibrate)
    p = sub.add_parser("freeze"); p.set_defaults(f=cmd_freeze)
    p = sub.add_parser("validate"); p.add_argument("D"); p.add_argument("--ledger"); p.set_defaults(f=cmd_validate)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
