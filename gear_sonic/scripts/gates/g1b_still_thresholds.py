"""G1b Step 7: still-standing thresholds (``still_dq_max``, ``still_gyro_max``) from recorded G1b runs, offline.

CLI: ``g1b_still_thresholds.py --select g1b_runA.npz --validate g1b_runB.npz`` (JSON on stdout).
Selection sees run A only; run B validates the selected pair unchanged (never re-tuned).

Ground truth (fixed before any feedback was examined)
  gt_still(t) = the G1b ``_stationary`` criterion on the trailing [t-1 s, t] window of ``sim_gt`` (aligned
  100 Hz; whole window and both halves: z slope < 2 mm/s, xy speed < 2 cm/s, |yaw rate| < 0.05 rad/s)
  AND |z(t) - z_final| < 3 mm, z_final = median z over the last 0.5 s of the static segment containing t.
  Outside static segments (``ramp``, ``walk``) gt_still = False. Static segments: settle:0, prehold, hold,
  sw1, stop, back, switch:hold, switch:to_sentinel, switch:to_height (g1b_analyze.py marker protocol).

Feedback check (``feedback_still``, reusable by Task 13's StillnessCheck)
  Uses only ``g1_debug`` samples with recv time <= t. Still iff the samples cover [t-1.0, t], the newest sample
  is < 0.1 s old, no inter-sample gap exceeds 2 nominal periods (nominal = median recv diff of the run), and
  over the window max |leg body_dq| < dq_max and max ||base_ang_vel|| < gyro_max. Leg dq = ``body_dq[0:12]``:
  the wire order is MuJoCo order (zmq_output_handler.hpp:25-26, 333-342) with legs first, left 6 then right 6
  (g1_supplemental_info.py:46-60).

Selection: among (dq_max, gyro_max) on a fixed log grid with zero false acceptances on A (check still, gt
not), maximize accepted gt-still instants; ties -> smaller values.
Usefulness (per run): >= 13/15 pre-walk ``hold`` windows and >= 9/10 isolated post-switch holds
(``switch:to_sentinel``, ``switch:to_height``) reach acceptance before the hold ends.
Instants whose trailing sim_gt window has a hole > 0.1 s (or too little data) are "unknown" and excluded from
all counts (reported per run); smaller NaN holes are skipped as in g1_analyze.py.
Verdict: PASS = zero false acceptances in A and B and usefulness in both; "FAILED: peak check not adopted" =
any false acceptance, usefulness not met, or no zero-false pair on A; INCONCLUSIVE = data missing only.
False-acceptance classes (precedence d, b, a, c): (d) in a walk window; (b) within 5 s after a walk end;
(a) within 5 s after the latest static command change (starts of prehold, ramp, hold, sw1, back, switch:*);
(c) other.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from gear_sonic.scripts.gates.g1b_analyze import _stationary
from gear_sonic.scripts.gates.g1_analyze import _Windows
from gear_sonic.scripts.gates.recorder import _load

WINDOW_S = 1.0
MAX_AGE_S = 0.1
GAP_PERIODS = 2.0
HZ = 100
DRIFT_M = 0.003
GAP_MIN_S = 0.1  # sim_gt sync contract: hole limit (recorder.GAP_MIN_S)
FINAL_S = 0.5
CLASS_S = 5.0
N_LEG = 12
GRID = np.round(np.geomspace(0.005, 2.0, 60), 5)  # candidate thresholds (rad/s), same for dq and gyro
STATIC = ("settle", "prehold", "hold", "sw1", "stop", "back", "switch")
USEFUL_HOLDS, USEFUL_ISO = 13, 9


def make_samples(recv_s, leg_dq, ang_vel, period_s=None):
    """Feedback samples (sorted by recv time) with per-sample reductions; period measured if not given."""
    recv_s = np.asarray(recv_s, float)
    return {
        "recv_s": recv_s,
        "dq_peak": np.abs(np.asarray(leg_dq, float)).max(axis=1),
        "gyro": np.linalg.norm(np.asarray(ang_vel, float), axis=1),
        "period_s": float(np.median(np.diff(recv_s))) if period_s is None else period_s,
    }


def window_stats(samples, t):
    """(reason | None, dq_peak, gyro_peak): structural check on samples with recv <= t, then window peaks."""
    r = samples["recv_s"]
    hi = np.searchsorted(r, t, side="right") - 1  # newest sample with recv <= t
    if hi < 0:
        return "no samples", np.inf, np.inf
    lo = np.searchsorted(r, t - WINDOW_S, side="right") - 1  # last sample at or before the window start
    if lo < 0:
        return "history does not cover the window", np.inf, np.inf
    if t - r[hi] >= MAX_AGE_S:
        return f"stale ({(t - r[hi]) * 1000:.0f} ms)", np.inf, np.inf
    if np.diff(r[lo : hi + 1]).max(initial=0.0) > GAP_PERIODS * samples["period_s"]:
        return "gap in feedback", np.inf, np.inf
    if hi == lo:
        return "no samples in window", np.inf, np.inf
    return None, float(samples["dq_peak"][lo + 1 : hi + 1].max()), float(samples["gyro"][lo + 1 : hi + 1].max())


def feedback_still(samples, t, dq_max, gyro_max):
    """Still-standing check at instant ``t`` from the feedback received by then: (bool, reason)."""
    why, dq, gy = window_stats(samples, t)
    if why:
        return False, why
    if dq >= dq_max:
        return False, f"leg dq {dq:.4f} >= {dq_max}"
    if gy >= gyro_max:
        return False, f"gyro {gy:.4f} >= {gyro_max}"
    return True, ""


def gt_still(t_grid, z, xy, yaw, seg_final, i):
    """Ground truth at grid index ``i``: True, False, or None (unknown: missing sim_gt beyond the sync contract).

    ``seg_final`` = z_final of the containing static segment, or None outside static segments (-> False).
    NaN grid points (sim_gt holes) are skipped as in g1_analyze.py (zwin); the window is unknown if the
    remaining samples leave a hole > 0.1 s (recorder GAP_MIN_S) anywhere in [t-1 s, t], or if it has too few.
    """
    if seg_final is None:
        return False
    if i < HZ:
        return None
    sl = slice(i - HZ, i + 1)
    ok = ~(np.isnan(z[sl]) | np.isnan(xy[sl]).any(axis=1) | np.isnan(yaw[sl]))
    v = np.nonzero(ok)[0]
    lim = round(GAP_MIN_S * HZ)
    if len(v) < HZ // 2 or v[0] > lim or HZ - v[-1] > lim or np.diff(v).max(initial=0) > lim:
        return None
    if abs(z[sl][v[-1]] - seg_final) >= DRIFT_M:  # newest valid sample (<= 0.1 s old) stands in for z(t)
        return False
    return _stationary((t_grid[sl][v], z[sl][v], xy[sl][v], yaw[sl][v])) is None


class Run:
    """One recording: instants, ground truth, class labels, holds and per-instant feedback stats."""

    def __init__(self, npz):
        d = _load(npz)
        W = _Windows(d)
        if W.z is None or "g1_debug_raw" not in d or len(d["g1_debug_recv_ns"]) == 0:
            raise ValueError("missing sim_gt or g1_debug")
        raw = d["g1_debug_raw"]
        self.samples = make_samples(
            np.asarray(d["g1_debug_recv_ns"]) / 1e9,
            [np.asarray(m["body_dq"], float)[:N_LEG] for m in raw],
            [np.asarray(m["base_ang_vel"], float) for m in raw],
        )
        names = sorted({k.rsplit(":", 1)[0] for k in W.markers if k.endswith(":start")})
        wins = {n: W.win(n) for n in names}
        wins = {n: w for n, w in wins.items() if w}
        walk = sorted(w for n, w in wins.items() if n.startswith("walk:"))
        change = sorted(w[0] for n, w in wins.items() if n.split(":")[0] in ("prehold", "ramp", "hold", "sw1", "back", "switch"))
        static = sorted((w, n) for n, w in wins.items() if n.split(":")[0] in STATIC)
        self.wins, self.walk, self.change = wins, walk, change  # exposed for g1b_still2.py
        t0, t1 = min(w[0] for w in wins.values()), max(w[1] for w in wins.values())
        self.t = W.t[(W.t >= t0) & (W.t < t1)]
        idx0 = int(np.searchsorted(W.t, self.t[0]))
        self.gt = np.zeros(len(self.t), bool)
        self.unk = np.zeros(len(self.t), bool)  # ground truth unknown (missing sim_gt): excluded from all counts
        self.cls = []
        for j, t in enumerate(self.t):
            seg = next((w for w, _ in static if w[0] <= t < w[1]), None)
            final = None
            if seg is not None:
                m = (W.t >= seg[1] - FINAL_S) & (W.t < seg[1]) & ~np.isnan(W.z)
                final = float(np.median(W.z[m])) if m.any() else None
            g = gt_still(W.t, W.z, W.xy, W.yaw, final, idx0 + j)
            self.unk[j], self.gt[j] = g is None, bool(g)
            if any(a <= t < b for a, b in walk):
                self.cls.append("d")
            elif any(b <= t < b + CLASS_S for _, b in walk):
                self.cls.append("b")
            elif any(c <= t < c + CLASS_S for c in change if c <= t):
                self.cls.append("a")
            else:
                self.cls.append("c")
        st = [window_stats(self.samples, t) for t in self.t]
        self.ok = np.array([s[0] is None for s in st])
        self.dq = np.array([s[1] for s in st])
        self.gy = np.array([s[2] for s in st])
        self.holds = {  # name -> window, for usefulness
            "hold": [w for n, w in sorted(wins.items()) if n.startswith("hold:")],
            "iso": [w for n, w in sorted(wins.items()) if n.startswith(("switch:to_sentinel:", "switch:to_height:"))],
        }

    def accept(self, dq_max, gyro_max):
        return self.ok & (self.dq < dq_max) & (self.gy < gyro_max)


def select_thresholds(run):
    """Pick (dq_max, gyro_max) from ONE recording: zero false acceptances, max accepted gt-still, ties smaller."""
    best = None
    for dq in GRID:
        for gy in GRID:
            a = run.accept(dq, gy)
            if (a & ~run.gt & ~run.unk).any():
                continue
            key = (-int((a & run.gt).sum()), dq, gy)
            if best is None or key < best:
                best = key
    return None if best is None else {"still_dq_max": float(best[1]), "still_gyro_max": float(best[2]),
                                      "accepted_gt_still_A": -best[0]}


def evaluate(run, dq_max, gyro_max):
    a = run.accept(dq_max, gyro_max)
    fa = a & ~run.gt & ~run.unk
    by = {c: int(sum(1 for j in np.nonzero(fa)[0] if run.cls[j] == c)) for c in "abcd"}
    res = {"instants": int(len(run.t)), "gt_still_instants": int(run.gt.sum()), "accepted": int(a.sum()),
           "unknown_gt_instants": int(run.unk.sum()), "false_acceptances": int(fa.sum()), "false_by_class": by,
           "accepted_gt_still_fraction": float((a & run.gt).sum() / max(1, run.gt.sum())),
           "nominal_period_s": run.samples["period_s"]}
    for kind, wins in run.holds.items():
        accepted, delays = 0, []
        for w0, w1 in wins:
            m = (run.t >= w0) & (run.t < w1)
            acc = np.nonzero(m & a)[0]
            accepted += bool(len(acc))
            on = np.nonzero(m & run.gt)[0]
            if len(acc) and len(on):
                delays.append(float(run.t[acc[0]] - run.t[on[0]]))
        res[f"{kind}_holds"] = {"n": len(wins), "accepted": accepted, "delay_median_s": float(np.median(delays)) if delays else None,
                                "delay_max_s": max(delays) if delays else None, "delays_s": delays}
    res["useful"] = bool(res["hold_holds"]["n"] == 15 and res["hold_holds"]["accepted"] >= USEFUL_HOLDS
                         and res["iso_holds"]["n"] == 10 and res["iso_holds"]["accepted"] >= USEFUL_ISO)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", required=True, help="run A npz (selection)")
    ap.add_argument("--validate", required=True, help="run B npz (validation, thresholds unchanged)")
    a = ap.parse_args(argv)
    out = {"select": a.select, "validate": a.validate}
    try:
        A, B = Run(a.select), Run(a.validate)
    except ValueError as e:
        out.update(verdict="INCONCLUSIVE", reason=str(e))
        print(json.dumps(out, indent=2))
        return 1
    sel = select_thresholds(A)  # A only
    out["selected"] = sel
    if sel is None:
        out.update(verdict="FAILED: peak check not adopted", reason="no threshold pair with zero false acceptances on A")
    else:
        pair = (sel["still_dq_max"], sel["still_gyro_max"])
        out["A"], out["B"] = evaluate(A, *pair), evaluate(B, *pair)
        ok = (out["A"]["false_acceptances"] == 0 and out["B"]["false_acceptances"] == 0
              and out["A"]["useful"] and out["B"]["useful"])
        out["verdict"] = "PASS" if ok else "FAILED: peak check not adopted"
    print(json.dumps(out, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
