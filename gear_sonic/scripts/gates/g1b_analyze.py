"""G1b analysis: restricted walking-height hand-off (PRD section 9 G1b).

Input is a recorder npz written by g1b_walk_handoff_probe.py. Markers (``:start`` / ``:end`` pairs):
``settle:0``; per direction DIR in fwd/back/turn and run K in 0..4 the walk trial:
``prehold:DIR:K`` (IDLE_SQUAT 0.60), ``ramp:DIR:K`` (not evaluated), ``hold:DIR:K`` (settled hold at
H_default), ``trial:DIR:K`` wrapping ``sw1`` 0.5 s, ``walk`` 10 s, ``stop`` 0.5 s, ``back`` 3 s;
then the isolated switch block per run K: ``switch:hold:K`` (3 s IDLE_SQUAT H_default),
``switch:to_sentinel:K`` (3 s IDLE -1), ``switch:to_height:K`` (3 s IDLE_SQUAT H_default).

Gate:
- walk trial: no fall, pelvis z within +-3 cm of the settled pre-walk z (median of the last 1.0 s of
  ``hold``) over sw1..back, motion evidenced (xy >= 1.0 m fwd/back, |dyaw| >= 2.0 rad turn).
  Deviation vs ``settle:0`` (median of its last 1.0 s) is reported too.
- isolated switch: steady step (median of the last 1.0 s after the switch minus the last 1.0 s before)
  < 1 cm and transient peak (max |z - before median| over the 3 s after) < 1 cm; both 1 s windows
  stationary (z slope < 2 mm/s, xy speed < 2 cm/s, |yaw rate| < 0.05 rad/s).
- In-walk 0.5 s switch steps (``walk_switch_steps_diag``) are diagnostics only, not gated.
Missing evidence of any kind (marker, sync, coverage, motion) fails; nothing passes without it.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from gear_sonic.scripts.gates.g1_analyze import (
    DIRS, EDGE_S, FALL_Z, MIN_DISP_M, MIN_DYAW_RAD, RUNS, STAT_TOL_M, _Windows,
)
from gear_sonic.scripts.gates.recorder import _load

SWITCH_TOL_M = 0.01
STAT_Z_M = 0.003  # |z - final| band of settle_time
PEAK_TOL_M = 0.01
Z_SLOPE_MAX = 0.002  # m/s
XY_SPEED_MAX = 0.02  # m/s
YAW_RATE_MAX = 0.05  # rad/s
HALF_S = 0.5
HZ = 100  # recorder align rate
MAX_REASONS = 10
SW = ("to_sentinel", "to_height")


def _step(before, after):
    """Median of ``after`` minus median of ``before`` (z arrays), None if either is missing."""
    if before is None or after is None or len(before) == 0 or len(after) == 0:
        return None
    return float(np.median(after) - np.median(before))


class _Seg(_Windows):
    """_Windows plus time-sliced pelvis (z, xy, yaw) with one sync check per window."""

    def zwin(self, name):
        """Cached: each window is sync-checked (and a failure reported) once."""
        if not hasattr(self, "_zw"):
            self._zw = {}
        if name not in self._zw:
            self._zw[name] = super().zwin(name)
        return self._zw[name]

    def seg(self, name, last=None):
        """(t, z, xy, yaw) of the window, or its last ``last`` seconds; None if missing/failed sync."""
        if not hasattr(self, "_ok"):
            self._ok = {}
        if name not in self._ok:
            self._ok[name] = self.zwin(name) is not None
        w = self.win(name)
        if not self._ok[name] or w is None:
            return None
        lo = w[0] if last is None else max(w[0], w[1] - last)
        m = (self.t >= lo) & (self.t < w[1] - EDGE_S) & ~np.isnan(self.z) & ~np.isnan(self.xy).any(axis=1) & ~np.isnan(self.yaw)
        return (self.t[m], self.z[m], self.xy[m], self.yaw[m]) if m.sum() >= 10 else None


def _stationary(s):
    """None if stationary, else a description. Slopes over the whole window and both halves."""
    t, z, xy, yaw = s
    h = t[0] + (t[-1] - t[0]) / 2
    for sel in (slice(None), t < h, t >= h):
        tt = t[sel]
        if len(tt) < 5:
            return "too few samples"
        sl = lambda y: np.polyfit(tt, y[sel], 1)[0]
        if abs(sl(z)) >= Z_SLOPE_MAX:
            return f"z slope {sl(z) * 1000:.1f} mm/s"
        if np.hypot(sl(xy[:, 0]), sl(xy[:, 1])) >= XY_SPEED_MAX:
            return "xy speed"
        if abs(sl(yaw)) >= YAW_RATE_MAX:
            return "yaw rate"
    return None


def settle_time(s):
    """Seconds from window start to the earliest sample from which, for the rest of the window, the trailing
    0.25 s z slope is < 2 mm/s, xy speed < 2 cm/s and |z - final| < 3 mm (final = median of the last 0.5 s);
    None if the window never settles."""
    t, z, xy, _ = s
    k = int(0.25 * HZ)
    if len(t) <= k:
        return None
    final = float(np.median(z[t >= t[-1] - HALF_S]))
    dt = t[k:] - t[:-k]
    ok = (np.abs(z[k:] - z[:-k]) / dt < Z_SLOPE_MAX) & (np.hypot(*(xy[k:] - xy[:-k]).T) / dt < XY_SPEED_MAX) \
        & (np.abs(z[k:] - final) < STAT_Z_M)
    bad = np.nonzero(~ok)[0]
    if len(bad) and bad[-1] == len(ok) - 1:
        return None
    return float(t[k + (bad[-1] + 1 if len(bad) else 0)] - t[0])


def _isolated(W, k):
    """Gated switch metrics of isolated block run ``k``: two records (to_sentinel, to_height)."""
    names = [f"switch:hold:{k}", f"switch:to_sentinel:{k}", f"switch:to_height:{k}"]
    recs = []
    for i, kind in enumerate(SW):
        before, after = W.seg(names[i], last=1.0), W.seg(names[i + 1], last=1.0)
        whole = W.seg(names[i + 1])
        r = {"kind": kind, "run": k, "steady_dz_m": None, "peak_m": None, "fail": []}
        if before is None or after is None or whole is None:
            r["fail"].append("missing evidence (switch window absent or failed sync)")
        else:
            ref = float(np.median(before[1]))
            r["steady_dz_m"] = float(np.median(after[1]) - ref)
            r["peak_m"] = float(np.abs(whole[1] - ref).max())
            if any(_stationary(s) for s in (before, after)):
                r["fail"].append("switch window not stationary: " + "; ".join(
                    f"{lab} {why}" for lab, s in (("before", before), ("after", after)) if (why := _stationary(s))))
            if abs(r["steady_dz_m"]) >= SWITCH_TOL_M:
                r["fail"].append(f"steady step {r['steady_dz_m'] * 100:.2f} cm >= 1 cm")
            if r["peak_m"] >= PEAK_TOL_M:
                r["fail"].append(f"transient peak {r['peak_m'] * 100:.2f} cm >= 1 cm")
        r["pass"] = not r["fail"]
        recs.append(r)
    return recs


def control_no_switch(npz) -> dict:
    """Same 0.5 s statistics on G1's -1 walk trials (no switch happens there): mm, per direction."""
    W = _Seg(_load(npz))
    out = {}
    for dr in DIRS:
        pre, post = [], []
        for k in range(RUNS):
            stat, tr = W.win("stat:-1"), W.win(f"trial:-1:{dr}:{k}")
            z = W.zwin(f"trial:-1:{dr}:{k}")
            if tr is None or z is None or stat is None or W.zwin("stat:-1") is None:
                continue
            at = lambda a, b: np.nanmedian(W.z[(W.t >= tr[0] + a) & (W.t < tr[0] + b)])
            pre.append(at(10.5, 11.0) - at(10.0, 10.5))  # walk end -> stop settling, no switch
            post.append(at(0.0, 0.5) - np.nanmedian(W.z[(W.t >= stat[1] - 0.5) & (W.t < stat[1])]))  # walk start
        out[dr] = {"after_walk_dz_mm": [round(x * 1000, 2) for x in pre],
                   "walk_start_dz_mm": [round(x * 1000, 2) for x in post]}
    return out


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def analyze_handoff(npz) -> dict:
    d = _load(npz)
    W = _Seg(d)
    n = int(HALF_S * HZ)
    trials: dict[str, list[dict]] = {}
    diag: list[dict] = []
    reasons: list[str] = []
    s0 = W.seg("settle:0", last=1.0)
    ref0 = None if s0 is None else float(np.median(s0[1]))
    if ref0 is None:
        reasons.append("settle:0 missing or failed sync")
    for dr in DIRS:
        rows = []
        for k in range(RUNS):
            tag = f"{dr}:{k}"
            hs, hfull = W.seg(f"hold:{tag}", last=1.0), W.seg(f"hold:{tag}")
            hold, sw1, stop, back = (W.zwin(f"{p}:{tag}") for p in ("hold", "sw1", "stop", "back"))
            trial, mv = W.zwin(f"trial:{tag}"), W.motion(f"walk:{tag}")
            ref = None if hs is None else float(np.median(hs[1]))
            for kind, dz in (("to_sentinel", _step(None if hold is None else hold[-n:], sw1)),
                             ("to_height", _step(stop, None if back is None else back[:n]))):
                diag.append({"kind": kind, "dir": dr, "run": k, "dz_m": dz})
            rec = {"pass": False, "settle_s": None if hfull is None else settle_time(hfull), "max_dev": None, "max_dev_vs_settle0": None, "min_z": None,
                   "disp_m": None, "dyaw_rad": None, "fail": []}
            if trial is None or ref is None:
                rec["fail"].append("missing evidence (trial/hold window absent or failed sync)")
            else:
                rec["min_z"], rec["max_dev"] = float(trial.min()), float(np.abs(trial - ref).max())
                if ref0 is not None:
                    rec["max_dev_vs_settle0"] = float(np.abs(trial - ref0).max())
                if rec["min_z"] <= FALL_Z:
                    rec["fail"].append("fall")
                if rec["max_dev"] > STAT_TOL_M:
                    rec["fail"].append("z deviation > 3 cm")
            if hs is not None and (why := _stationary(hs)):
                rec["fail"].append(f"pre-walk hold not settled: {why}")
            if sw1 is None or stop is None or back is None:
                rec["fail"].append("missing evidence (sw1/stop/back window absent or failed sync)")
            if mv is not None:
                rec["disp_m"], rec["dyaw_rad"] = mv
            if mv is None or (abs(mv[1]) < MIN_DYAW_RAD if dr == "turn" else mv[0] < MIN_DISP_M):
                rec["fail"].append("no motion evidence")
            rec["pass"] = not rec["fail"]
            rows.append(rec)
            if not rec["pass"]:
                reasons.append(f"{tag}: {'; '.join(rec['fail'])}")
        trials[dr] = rows
    iso = [r for k in range(RUNS) for r in _isolated(W, k)]
    reasons += [f"switch {r['kind']}:{r['run']}: {'; '.join(r['fail'])}" for r in iso if not r["pass"]]
    falls = sorted({m.split(":", 1)[1].rsplit(":", 1)[0] for m in W.markers if m.startswith("fall:")})
    reasons += [f"fall recorded: {f}" for f in falls]
    reasons += W.sync_failures[:MAX_REASONS]
    if len(W.sync_failures) > MAX_REASONS:
        reasons.append(f"... {len(W.sync_failures) - MAX_REASONS} more sync failures")
    devs = [r["max_dev"] for rows in trials.values() for r in rows if r["max_dev"] is not None]
    devs0 = [r["max_dev_vs_settle0"] for rows in trials.values() for r in rows if r["max_dev_vs_settle0"] is not None]
    steady = {k: [abs(r["steady_dz_m"]) for r in iso if r["kind"] == k and r["steady_dz_m"] is not None] for k in SW}
    peaks = {k: [r["peak_m"] for r in iso if r["kind"] == k and r["peak_m"] is not None] for k in SW}
    diag_abs = [abs(s["dz_m"]) for s in diag if s["dz_m"] is not None]
    return {
        "trials": trials,
        "dir_passes": {dr: sum(r["pass"] for r in rows) for dr, rows in trials.items()},
        "max_settle_s": {dr: _max([r["settle_s"] for r in rows]) for dr, rows in trials.items()},
        "unsettled_holds": sum(r["settle_s"] is None for rows in trials.values() for r in rows),
        "isolated_switches": iso,
        "max_isolated_steady_m": {k: max(v) if v else None for k, v in steady.items()},
        "max_isolated_peak_m": {k: max(v) if v else None for k, v in peaks.items()},
        "walk_switch_steps_diag": diag,
        "max_walk_switch_step_diag_m": max(diag_abs) if diag_abs else None,
        "max_dev": max(devs) if devs else None,
        "max_dev_vs_settle0": max(devs0) if devs0 else None,
        "falls": falls,
        "sync": {"ok": not W.sync_failures, "failures": W.sync_failures},
        "gate_pass": bool(not reasons and all(r["pass"] for rows in trials.values() for r in rows)
                          and len(iso) == 2 * RUNS and all(r["pass"] for r in iso)),
        "reasons": reasons,
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print("usage: g1b_analyze.py g1b.npz [g1_height.npz (no-switch control)]")
        return 2
    res = analyze_handoff(argv[0])
    if len(argv) == 2:
        res["control_no_switch_g1"] = control_no_switch(argv[1])
    print(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"gate_pass={res['gate_pass']} passes={res['dir_passes']} max_dev={res['max_dev']} "
          f"max_dev_vs_settle0={res['max_dev_vs_settle0']} iso_steady={res['max_isolated_steady_m']} "
          f"iso_peak={res['max_isolated_peak_m']} falls={res['falls']} sync_ok={res['sync']['ok']}", file=sys.stderr)
    return 0 if res["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
