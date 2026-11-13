"""G1 height probe analysis: height command -> pelvis z map, H_default and walking band.

Input is a recorder npz (``sim_gt`` + ``g1_debug`` + ``markers``). Windows are
``<name>:start`` / ``<name>:end`` marker pairs (written by g1_height_probe.py):
``settle:K`` (K=0..2; -1 sentinel in IDLE), ``step:<MODE>:<H>`` (3 s holds),
``stat:<H>`` (3 s stationary reference per walk height, ``-1`` = sentinel) and
``trial:<H>:<DIR>:<K>`` (10 s of one motion + 3 s stationary).

A window that fails ``check_sync`` or has no pelvis data is missing evidence:
a stationary z from it is unusable and a trial in it fails; never a pass.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from gear_sonic.scripts.gates.recorder import StreamReq, _load, _marker_times, align, check_sync

SENTINEL = -1.0
GRID = [round(0.80 - 0.05 * i, 2) for i in range(9)]  # 0.80 .. 0.40
DIRS = ("fwd", "back", "turn")
RUNS = 5
STREAMS = {"sim_gt": StreamReq(0.005), "g1_debug": StreamReq(0.02)}
SENTINEL_SPREAD_M = 0.01  # standing z across the three -1 settles
IDLE_HONORS_M = 0.03
FALL_Z = 0.3
STAT_TOL_M = 0.03
MARGIN_M = 0.03
TOP = 0.80
ASSUMPTION = "interpolated between grid points, step 0.05"
SETTLED_TAIL_S = 1.0
MAX_SYNC_REASONS = 10
MOTION_S = 10.0  # motion part of a trial window
MIN_DISP_M = 1.0  # fwd/back: pelvis xy displacement over the motion part
MIN_DYAW_RAD = 2.0  # turn: |yaw change| over the motion part
EDGE_S = 0.02


def hkey(h: float) -> str:
    return "-1" if h == SENTINEL else f"{h:.2f}"


class _Windows:
    """Marker-window access: pelvis z (aligned once) and per-window sync checks."""

    def __init__(self, d):
        self.d = d
        self.markers = _marker_times(d)
        src = np.asarray(d["sim_gt_src_ns"] if "sim_gt_src_ns" in d else []) / 1e9  # absent stream: z stays None = missing evidence
        self.z = None
        if len(src):
            def yaw(m):
                w, x, y, z = m["pelvis_quat_wxyz"]
                return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            r = align(
                d,
                {"sim_gt": lambda m: m["pelvis_pos"][2], "sim_gt.xy": lambda m: m["pelvis_pos"][:2], "sim_gt.yaw": yaw},
                (src.min(), src.max()),
                hz=100.0,
                angles=("sim_gt.yaw",),
            )
            self.t, self.z, self.xy, self.yaw = r["t"], r["sim_gt"], r["sim_gt.xy"], r["sim_gt.yaw"]
        self.sync_failures: list[str] = []

    def win(self, name):
        s = self.markers.get(f"{name}:start")
        if not s:
            return None
        e = [t for t in self.markers.get(f"{name}:end", []) if t >= s[0]]
        return (s[0], e[0]) if e else None

    def zwin(self, name):
        """Pelvis z samples in the window, or None if missing / failed sync / holes."""
        w = self.win(name)
        if w is None or self.z is None:
            return None
        rep = check_sync(self.d, STREAMS, w)
        if not rep.ok:
            self.sync_failures.append(f"{name}: sync failed: {'; '.join(rep.reasons)}")
            return None
        z = self.z[(self.t >= w[0]) & (self.t < w[1] - EDGE_S)]  # next window starts at w[1]; interp must not blend it in
        # NaN grid points sit inside sim jitter holes that check_sync already bounded (<= 100 ms): skip them
        z = z[~np.isnan(z)]
        return z if len(z) else None

    def motion(self, name):
        """(xy displacement m, yaw change rad) over the first MOTION_S of the window, or None."""
        w = self.win(name)
        if w is None or self.z is None:
            return None
        m = (self.t >= w[0]) & (self.t < w[0] + MOTION_S)
        ok = m & ~np.isnan(self.xy).any(axis=1) & ~np.isnan(self.yaw)
        if ok.sum() < 2:
            return None
        i, j = np.nonzero(ok)[0][[0, -1]]
        return float(np.hypot(*(self.xy[j] - self.xy[i]))), float(self.yaw[j] - self.yaw[i])

    def settled(self, name):
        z = self.zwin(name)
        if z is None:
            return None
        return float(np.median(z[-int(SETTLED_TAIL_S * 100) :]))


def _h_default(zs, cmd_to_z, reasons):
    """Command whose settled z equals the -1 standing z, interpolated; None if calibration fails."""
    if len(zs) != 3 or any(z is None for z in zs):
        reasons.append("h_default: a -1 settle is missing or failed sync")
        return None, None
    if max(zs) - min(zs) > SENTINEL_SPREAD_M:
        reasons.append(f"h_default: standing z varies {max(zs) - min(zs):.3f} m across -1 settles")
        return None, None
    stand = float(np.median(zs))
    cross = [
        (a, b)
        for a, b in zip(cmd_to_z, cmd_to_z[1:])
        if (a[1] - stand) * (b[1] - stand) <= 0 and a[1] != b[1]
    ]
    if len(cross) != 1:
        reasons.append(f"h_default: standing z {stand:.3f} not bracketed by exactly one cmd_to_z pair")
        return None, stand
    (h0, z0), (h1, z1) = cross[0]
    if z1 < z0:
        reasons.append("h_default: cmd_to_z not monotonic over the bracketing pair")
        return None, stand
    return h0 + (stand - z0) * (h1 - h0) / (z1 - z0), stand


def analyze_height(npz) -> dict:
    d = _load(npz)
    W = _Windows(d)
    reasons: list[str] = []
    notes: list[str] = []

    # ---- IDLE honors height? -------------------------------------------
    hi, lo = W.settled("step:IDLE:0.80"), W.settled("step:IDLE:0.40")
    honors = hi is not None and lo is not None and abs(hi - lo) > IDLE_HONORS_M
    if hi is None or lo is None:
        reasons.append("idle_honors_height: IDLE 0.80/0.40 steps missing or failed sync")
    mode = "IDLE" if honors else "IDLE_SQUAT"

    # ---- cmd -> z map in the honoring mode -----------------------------
    cmd_to_z = []
    for h in sorted({float(n.split(":")[2]) for n in {k.rsplit(":", 1)[0] for k in W.markers} if n.startswith(f"step:{mode}:")}):
        z = W.settled(f"step:{mode}:{h:.2f}")
        if z is not None:
            cmd_to_z.append((h, z))

    h_default, stand = _h_default([W.settled(f"settle:{k}") for k in range(3)], cmd_to_z, reasons)
    h_def_r = None if h_default is None else round(h_default, 2)

    # ---- walk table ----------------------------------------------------
    heights = [SENTINEL] + ([h_def_r] if h_def_r is not None and h_def_r not in GRID else []) + GRID
    walk_table: dict[float, dict[str, list[bool]]] = {}
    trials: dict[str, dict[str, list[dict]]] = {}
    for h in heights:
        zstat = W.settled(f"stat:{hkey(h)}")
        row, trow = {}, {}
        for dr in DIRS:
            res, info = [], []
            for k in range(RUNS):
                name = f"trial:{hkey(h)}:{dr}:{k}"
                z = W.zwin(name)
                mv = W.motion(name)
                rec = {"min_z": None, "max_dev": None, "disp_m": None, "dyaw_rad": None, "fail": None}
                if z is None or zstat is None:
                    rec["fail"] = "missing evidence (not run, failed sync or no stationary z)"
                else:
                    rec["min_z"], rec["max_dev"] = float(z.min()), float(np.abs(z - zstat).max())
                    if mv is not None:
                        rec["disp_m"], rec["dyaw_rad"] = mv
                    if mv is None:
                        moved = False
                    elif dr == "turn":
                        moved = abs(mv[1]) >= MIN_DYAW_RAD
                    else:
                        moved = mv[0] >= MIN_DISP_M
                    if not moved:
                        rec["fail"] = "no motion evidence"
                    elif rec["min_z"] <= FALL_Z:
                        rec["fail"] = "fall"
                    elif rec["max_dev"] > STAT_TOL_M:
                        rec["fail"] = "z deviation > 3 cm"
                res.append(rec["fail"] is None)
                info.append(rec)
            row[dr], trow[dr] = res, info
        walk_table[h], trials[hkey(h)] = row, trow
    passes = {h: all(all(v) for v in row.values()) for h, row in walk_table.items()}

    def why(h, label):
        bad = {dr: f"{sum(v)}/{RUNS}" for dr, v in walk_table[h].items() if not all(v)}
        fails = sorted({r["fail"] for row in trials[hkey(h)].values() for r in row if r["fail"]})
        return f"{label} (h={hkey(h)}) fails walk trials: {bad}; causes: {fails}"

    # ---- band ----------------------------------------------------------
    lowest = None
    for h in GRID:
        if not passes[h]:
            reasons_h = why(h, "band stop")
            notes.append(reasons_h)
            break
        lowest = h
    h_walk_min, band_assumption = None, None
    if lowest is not None and round(lowest + MARGIN_M, 2) <= TOP + 1e-9:
        h_walk_min, band_assumption = round(lowest + MARGIN_M, 2), ASSUMPTION
    else:
        notes.append("band unknown: PRD 6.3 applies H_walk_min := H_default")

    # ---- gate ----------------------------------------------------------
    gate = h_default is not None
    if not passes[SENTINEL]:
        gate = False
        reasons.append(why(SENTINEL, "-1 sentinel"))
    if h_def_r is not None and not passes[h_def_r]:
        gate = False
        reasons.append(why(h_def_r, "h_default"))
    for n in {k.split(":")[1] for k in W.markers if k.startswith("fall:")}:
        reasons.append(f"fall recorded at height {n}")
    if W.sync_failures:
        reasons += W.sync_failures[:MAX_SYNC_REASONS]
        if len(W.sync_failures) > MAX_SYNC_REASONS:
            reasons.append(f"... {len(W.sync_failures) - MAX_SYNC_REASONS} more sync failures")

    switch_z = {f"{k}:{m}": W.settled(f"switch:{k}:{m}") for k in range(3) for m in ("IDLE", "IDLE_SQUAT")}
    idle_cmd_to_z = [
        (h, z) for h in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2) if (z := W.settled(f"step:IDLE:{h:.2f}")) is not None
    ]
    return {
        "idle_honors_height": bool(honors),
        "cmd_mode": mode,
        "cmd_to_z": cmd_to_z,
        "idle_cmd_to_z": idle_cmd_to_z,
        "settle_z": [W.settled(f"settle:{k}") for k in range(3)],
        "switch_z": switch_z,
        "trials": trials,
        "standing_z": stand,
        "h_default": h_default,
        "band_assumption": band_assumption,
        "walk_table": walk_table,
        "h_walk_min": h_walk_min,
        "gate_pass": bool(gate and not reasons),  # any missing/failed evidence anywhere is not a pass
        "reasons": reasons,
        "notes": notes,
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: g1_analyze.py g1.npz")
        return 2
    res = analyze_height(argv[0])
    print(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"gate_pass={res['gate_pass']} h_default={res['h_default']} h_walk_min={res['h_walk_min']} "
          f"idle_honors_height={res['idle_honors_height']}", file=sys.stderr)  # stdout stays valid JSON
    return 0 if res["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
