"""G0 twist diagnosis: staged, calibration-validated attribution of hand-reach twist.

``analyze_twist(npz)`` reports two separate results (see PRD section 9, G0):

* ``robot_side``  (hops B, C, BC; decides the gate): ``none`` / ``inconclusive`` /
  ``+``-joined subset of ``pipeline``, ``sonic``, ``chest_residual``.
* ``reference_side`` (hop A; recorded only): never ``"pico"`` without an
  independent operator-motion stream (``operator_ref``).

Every level's comparison is valid only if its calibration segment (operator
keypress markers ``turn_body`` / ``twist_chest``) shows corr >= 0.9, same sign
and a least-squares gain in [0.8, 1.2]. All yaws are degrees relative to their
own 1 s baseline; the baseline removes constant offsets only.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.scripts.gates.recorder import StreamReq, _load, align, check_sync

TWIST_DEG = 2.0  # a hop introduces twist above this
AMBIG_DEG = (1.5, 2.5)  # peak in this band -> inconclusive
REF_NONE_DEG = 1.5
OP_STILL_DEG = 1.0
CAL_CORR_MIN = 0.9
CAL_GAIN = (0.8, 1.2)
CAL_MIN_MOTION_DEG = 10.0  # a calibration segment must move at least this much (ptp)
GUARD_PTP_DEG = 0.5  # target channel flatter than this while the reference moves = placeholder
REACH_START_M, REACH_END_M = 0.10, 0.03
REACH_TAIL_S = 2.0
LAG_MAX_S = 0.3
BASE_S = 1.0
IMU_SIM_MAX_DEG = 1.0
MIN_REACHES = 5
HZ = 100.0
LAG_N = int(round(LAG_MAX_S * HZ))
WRIST = 21  # smpl_joints index of the right wrist
WAIST_YAW = 12  # G1 waist_yaw_joint in MuJoCo joint order
REQUIRED = {"pose": StreamReq(0.02), "g1_debug": StreamReq(0.02), "sim_gt": StreamReq(0.005)}
LABEL = {"B": "pipeline", "C": "sonic", "BC": "chest_residual"}


# ---- Signal helpers ------------------------------------------------------
def yaw_wxyz(q) -> float:
    """Heading (rad, about +z) of a wxyz quaternion."""
    w, x, y, z = np.asarray(q, dtype=float).ravel()[:4]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def twist_about_y(q_xyzw) -> float:
    """Swing-twist (q = swing * twist) twist angle (rad) about +y, wrapped to (-pi, pi]."""
    x, y, z, w = np.asarray(q_xyzw, dtype=float)
    a = 2.0 * np.arctan2(y, w)
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def spine_twist(smpl_pose) -> float:
    """Twist (rad) of R(spine1) R(spine2) R(spine3) about SMPL local up (+y).

    ``smpl_pose`` is the (21, 3) axis-angle body pose; spine1..3 are SMPL joints
    3, 6, 9 = body-pose indices 2, 5, 8.
    """
    p = np.asarray(smpl_pose, dtype=float).reshape(21, 3)
    rot = R.from_rotvec(p[2]) * R.from_rotvec(p[5]) * R.from_rotvec(p[8])
    return twist_about_y(rot.as_quat())


def chest_twist_joints(smpl_joints) -> float:
    """Yaw (rad, about the root-local up axis +z) of the shoulder line (joint 16 L -> 17 R)
    relative to the hip line (1 L -> 2 R), from the streamed root-local ``smpl_joints`` (24, 3).

    Root-local frame is z-up: ``smpl_joints_local`` = z-up world joints rotated by the inverse
    of the z-up root quaternion with the SMPL base rotation removed. Lateral lines are unchanged
    by a forward bend (rotation about the lateral axis), so this is robust to bending.
    """
    j = np.asarray(smpl_joints, dtype=float).reshape(24, 3)
    s, h = j[17] - j[16], j[2] - j[1]
    a = np.arctan2(s[1], s[0]) - np.arctan2(h[1], h[0])
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _last(m, key, shape):
    return np.asarray(m[key], dtype=float).reshape((-1,) + shape)[-1]


def _op_yaw(m) -> float:
    return twist_about_y(np.asarray(m["trackers"], dtype=float).reshape(-1, 7)[int(m["idx"]), 3:7])


RECORDED_ROBOT = ("pose.rp", "pose.rj", "g1_debug.tp", "g1_debug.tw", "g1_debug.mp", "g1_debug.mw", "sim_gt.sp")
SIGNALS = {
    "pose.rp": lambda m: yaw_wxyz(_last(m, "body_quat_w", (4,))),  # ref_pelvis_yaw
    "pose.rj": lambda m: chest_twist_joints(_last(m, "smpl_joints", (24, 3))),  # ref_chest_twist_joints
    "pose.rs": lambda m: spine_twist(_last(m, "smpl_pose", (21, 3))),  # ref_spine_twist (diagnostic only)
    "g1_debug.tp": lambda m: yaw_wxyz(m["base_quat_target"]),  # tgt_pelvis_yaw
    "g1_debug.tw": lambda m: float(np.asarray(m["body_q_target"]).ravel()[WAIST_YAW]),  # placeholder guard
    "g1_debug.mp": lambda m: yaw_wxyz(m["base_quat_measured"]),  # meas_pelvis_yaw
    "g1_debug.mw": lambda m: float(np.asarray(m["body_q_measured"]).ravel()[WAIST_YAW]),  # meas_waist_yaw
    "sim_gt.sp": lambda m: yaw_wxyz(m["pelvis_quat_wxyz"]),  # sim_pelvis_yaw
    "operator_ref.op": _op_yaw,  # op_chest_yaw
}


# ---- Segments and reaches ------------------------------------------------
def _markers(d):
    if "markers_src_ns" not in d:
        return []
    out = [
        ((s if s >= 0 else r) / 1e9, raw["name"])
        for s, r, raw in zip(d["markers_src_ns"], d["markers_recv_ns"], d["markers_raw"])
    ]
    return sorted(out)


def _segment(marks, name, t_last):
    for i, (t, n) in enumerate(marks):
        if n == name:
            return t, next((tt for tt, _ in marks[i + 1 :] if tt > t), t_last)
    return None


def _find_reaches(d, marks, t_last):
    """Reaches of the right wrist after the ``reach`` marker (rest = 1 s before it).

    Returns (t_leave, t_end) per reach: a reach starts at > 10 cm and ends within 3 cm
    of rest; ``t_leave`` is the last time within 3 cm of rest before the start, so the
    1 s baseline before it does not include the onset of the motion.
    """
    rs = next((t for t, n in marks if n == "reach"), None)
    if rs is None:
        return [], "no 'reach' marker"
    te = next((t for t, n in marks if n == "end" and t > rs), t_last)
    src = np.asarray(d["pose_src_ns"]) / 1e9
    order = np.argsort(src, kind="stable")
    w = np.array([_last(d["pose_raw"][i], "smpl_joints", (24, 3))[WRIST] for i in order])
    src = src[order]
    rest_m = (src >= rs - BASE_S) & (src < rs)
    if not rest_m.any():
        return [], "no pose samples in the 1 s before the 'reach' marker"
    dist = np.linalg.norm(w - w[rest_m].mean(0), axis=1)
    reaches, start, leave = [], None, rs
    for t, dd in zip(src, dist):
        if t < rs or t > te:
            continue
        if start is None and dd < REACH_END_M:
            leave = t  # last time at rest; the baseline ends here, before the hand starts moving
        elif start is None and dd > REACH_START_M:
            start = t
        elif start is not None and dd < REACH_END_M:
            reaches.append((leave, t))
            start, leave = None, t
    return reaches, None


# ---- Analysis ------------------------------------------------------------
def _cal(X, Y, spans, sl):
    """Calibration of y(t) ~ g * x(t - L), L in [0, LAG_MAX_S], over baseline-subtracted spans.

    Returns the corr/gain at the lag with the best corr, plus that lag ``L`` (s);
    None when data are missing or contain NaN (including the baselines).
    """
    xs, ys = [], []  # xs[k] = list of x segments shifted by k samples
    xs = [[] for _ in range(LAG_N + 1)]
    for a, b in spans:
        s, sb = sl(a, b), sl(a - BASE_S, a)
        if s is None or sb is None or s.start < LAG_N:
            return None
        bx, by = np.mean(X[sb]), np.mean(Y[sb])  # plain mean: a NaN in the baseline propagates
        y = Y[s] - by
        for k in range(LAG_N + 1):
            xs[k].append(X[s.start - k : s.stop - k] - bx)
        ys.append(y)
        if np.isnan(y).any() or any(np.isnan(xs[k][-1]).any() for k in range(LAG_N + 1)):
            return None
    y = np.concatenate(ys)
    x0 = np.concatenate(xs[0])
    if np.ptp(x0) < CAL_MIN_MOTION_DEG or np.std(y) == 0:
        return {"corr": None, "gain": None, "L": None, "valid": False, "why": "insufficient motion"}
    best = None
    for k in range(LAG_N + 1):
        x = np.concatenate(xs[k])
        corr = float(np.corrcoef(x, y)[0, 1])
        if best is None or corr > best[0]:
            best = (corr, float(np.dot(x, y) / np.dot(x, x)), k / HZ)
    corr, gain, lag = best
    ok = corr >= CAL_CORR_MIN and CAL_GAIN[0] <= gain <= CAL_GAIN[1]
    return {"corr": corr, "gain": gain, "L": lag, "valid": bool(ok), "why": None if ok else "corr/gain out of range"}


def analyze_twist(npz) -> dict:
    d = dict(_load(npz))  # NpzFile re-reads object arrays on every access
    res = {
        "calibration": {},
        "reaches": [],
        "sync": {},
        "robot_side": "inconclusive",
        "reference_side": "inconclusive",
        "target_unobservable": [],
        "operator_ref": False,
        "reasons": [],
        "gate_pass": False,
    }
    rwhy, fwhy = [], []  # robot-side / reference-side reasons

    def done():
        res["reasons"] = rwhy + [f"reference: {w}" for w in fwhy]
        res["gate_pass"] = res["robot_side"] != "inconclusive"
        return res

    for n in ("pose", "g1_debug", "sim_gt", "markers"):
        if f"{n}_src_ns" not in d or len(d[f"{n}_src_ns"]) == 0:
            rwhy.append(f"{n}: missing stream")
    if rwhy:
        return done()
    has_op = "operator_ref_src_ns" in d and len(d["operator_ref_src_ns"]) > 0
    res["operator_ref"] = bool(has_op)

    t_first = max(np.min(d[f"{n}_src_ns"]) for n in ("pose", "g1_debug", "sim_gt")) / 1e9
    t_last = min(np.max(d[f"{n}_src_ns"]) for n in ("pose", "g1_debug", "sim_gt")) / 1e9
    marks = _markers(d)
    tb, tc = _segment(marks, "turn_body", t_last), _segment(marks, "twist_chest", t_last)
    if tb is None or tc is None:
        rwhy.append("missing calibration segment (turn_body and twist_chest markers required)")
        return done()
    reaches, err = _find_reaches(d, marks, t_last)
    if err:
        rwhy.append(err)
    if len(reaches) < MIN_REACHES:
        rwhy.append(f"{len(reaches)} reaches detected (< {MIN_REACHES})")

    # --- sync over the calibration window and every reach window
    windows = {"calibration": (min(tb[0], tc[0]), max(tb[1], tc[1]))}
    windows.update({f"reach_{i}": (a - LAG_MAX_S, b + REACH_TAIL_S) for i, (a, b) in enumerate(reaches)})
    pose_sync_ok = True
    for label, win in windows.items():
        rep = check_sync(d, REQUIRED, win)
        res["sync"][label] = {"ok": rep.ok, "reasons": rep.reasons, "skew_p99_ms": rep.skew_p99_ms,
                              "max_gap_ms": rep.max_gap_ms, "rtf": rep.rtf}
        rwhy += [f"sync {label}: {r}" for r in rep.reasons]
        pose_sync_ok &= not any(r.startswith("pose:") for r in rep.reasons)
    op_sync_ok = False
    if has_op:
        reps = [check_sync(d, {"operator_ref": StreamReq(0.02)}, w) for w in windows.values()]
        op_sync_ok = all(r.ok for r in reps)
        fwhy += [f"operator_ref sync: {x}" for r in reps for x in r.reasons]

    # --- aligned signals in degrees
    sigs = {k: v for k, v in SIGNALS.items() if has_op or not k.startswith("operator_ref")}
    g = align(d, sigs, (t_first, t_last), hz=HZ, angles=list(sigs))
    t = g["t"]
    G = {k: np.degrees(v) for k, v in g.items() if k != "t"}

    def sl(a, b):
        if a < t[0] - 1e-9 or b > t[-1] + 1e-9:
            return None
        return slice(int(np.searchsorted(t, a - 1e-9)), int(np.searchsorted(t, b + 1e-9, side="right")))

    def base(k, a):
        s = sl(a - BASE_S, a)
        return np.nan if s is None else np.mean(G[k][s])

    # --- calibration (pairs: x = reference-side quantity, y = compared quantity)
    pairs = {
        "B": ("rp", "tp", [tb]),
        "C": ("tp", "mp", [tb]),
        "BC": ("rj", "mw", [tc]),  # reference = what the policy can see (joint positions)
    }
    guards = {"tp": ("rp", [tb]), "tw": ("rj", [tc])}  # target channel -> (reference channel, span)
    unobs = set()
    for ch, (ref, spans) in guards.items():
        xs = [G[f"g1_debug.{ch}"][sl(a, b)] for a, b in spans if sl(a, b) is not None]
        rs_ = [G[f"pose.{ref}"][sl(a, b)] for a, b in spans if sl(a, b) is not None]
        if xs and np.ptp(np.concatenate(xs)) < GUARD_PTP_DEG and np.ptp(np.concatenate(rs_)) > 10.0:
            unobs.add(ch)
    names = {"tp": "tgt_pelvis_yaw", "tw": "body_q_target[12]"}
    res["target_unobservable"] = sorted(names[c] for c in unobs)
    key = lambda c: f"{'pose' if c in ('rp', 'rs', 'rj') else 'sim_gt' if c == 'sp' else 'g1_debug'}.{c}"
    for h, (xc, yc, spans) in pairs.items():
        if "tp" in (xc, yc) and "tp" in unobs:
            res["calibration"][h] = {"corr": None, "gain": None, "valid": False, "status": "target_unobservable"}
            rwhy.append(f"hop {h}: target_unobservable (placeholder target channel)")
            continue
        c = _cal(G[key(xc)], G[key(yc)], spans, sl)
        if c is None:
            res["calibration"][h] = {"corr": None, "gain": None, "valid": False, "status": "unmapped"}
            rwhy.append(f"hop {h}: NaN or missing data in calibration window")
            continue
        c["status"] = "valid" if c["valid"] else "unmapped"
        res["calibration"][h] = c
        if not c["valid"]:
            rwhy.append(f"hop {h}: unmapped ({c['why']}, corr={c['corr']}, gain={c['gain']})")
    # diagnostic only: smpl_pose spine twist vs measured waist; never feeds robot_side
    if "pose.rs" in G:
        c = _cal(G["pose.rs"], G["g1_debug.mw"], [tc], sl)
        res["calibration"]["BC_spine_diag"] = None if c is None else c
    op_ok = False
    if has_op and op_sync_ok:
        xa = G["pose.rp"] + G["pose.rj"]
        G["ref_chest"] = xa
        c = _cal(xa, G["operator_ref.op"], [tb, tc], sl)
        if c is not None:
            c["status"] = "valid" if c["valid"] else "unmapped"
            res["calibration"]["A"] = c
            op_ok = c["valid"]
            if not op_ok:
                fwhy.append(f"hop A: unmapped ({c['why']}, corr={c['corr']}, gain={c['gain']})")
        else:
            fwhy.append("hop A: NaN or missing data in calibration window")

    # --- per-reach peaks
    lag_n = LAG_N
    flags = []  # per reach: (set of robot hops, any ambiguous, any unmapped)
    for i, (a, b0) in enumerate(reaches):
        b = b0 + REACH_TAIL_S
        s = sl(a, b)
        if s is None or s.start < lag_n:
            rwhy.append(f"reach {i}: window outside recorded data")
            continue
        rk = [k for k in G if k in RECORDED_ROBOT]
        wins = [G[k][s.start - lag_n : s.stop] for k in rk]
        if any(np.isnan(w).any() for w in wins) or any(np.isnan(base(k, a)) for k in rk):
            rwhy.append(f"reach {i}: NaN in window")
            continue
        dl = lambda k: G[k][s] - base(k, a)
        r = {"t_start": a, "t_end": b0,
             "ref_twist": {"pelvis": float(np.max(np.abs(dl("pose.rp")))), "chest": float(np.max(np.abs(dl("pose.rj")))),
                           "spine_smpl_pose_diag": float(np.max(np.abs(dl("pose.rs"))))},
             "imu_vs_sim_deg": float(np.max(np.abs(dl("g1_debug.mp") - dl("sim_gt.sp")))),
             "L": {}, "op_still": None, "pico_added": None}
        hops, ambig, unmapped = set(), False, False
        for h, (xc, yc, _) in pairs.items():
            cal = res["calibration"][h]
            if not cal["valid"]:
                r[h] = None
                unmapped = True
                continue
            X, Y = G[key(xc)], G[key(yc)]
            y = Y[s] - base(key(yc), a)
            best = None
            for k in range(lag_n + 1):
                rr = y - cal["gain"] * (X[s.start - k : s.stop - k] - base(key(xc), a))
                rms = float(np.sqrt(np.mean(rr**2)))
                if best is None or rms < best[0]:
                    best = (rms, k, float(np.max(np.abs(rr))))
            r[h], r["L"][h] = best[2], best[1] / HZ
            if AMBIG_DEG[0] <= best[2] <= AMBIG_DEG[1]:
                ambig = True
                rwhy.append(f"reach {i}: hop {h} peak {best[2]:.2f} deg in ambiguous band")
            if best[2] > TWIST_DEG:
                hops.add(h)
        if r["imu_vs_sim_deg"] > IMU_SIM_MAX_DEG:
            rwhy.append(f"reach {i}: imu_vs_sim {r['imu_vs_sim_deg']:.2f} deg > {IMU_SIM_MAX_DEG}")
        dop = dl("operator_ref.op") if op_ok else None
        if op_ok and np.isnan(dop).any():
            fwhy.append(f"reach {i}: NaN in operator_ref window")
        elif op_ok:
            added = float(np.max(np.abs(dl("ref_chest") - dop)))
            r["op_still"] = bool(np.max(np.abs(dop)) < OP_STILL_DEG)
            r["pico_added"] = bool(added > TWIST_DEG)
            r["pico_added_peak"] = added
            r["pico_ambiguous"] = AMBIG_DEG[0] <= added <= AMBIG_DEG[1]
        res["reaches"].append(r)
        flags.append((hops, ambig, unmapped))

    # --- robot side
    if not rwhy:
        sets = {tuple(h for h in ("B", "C", "BC") if h in f[0]) for f in flags}
        if len(sets) != 1:
            rwhy.append("reaches disagree on which hops introduce twist")
        else:
            hops = sets.pop()
            res["robot_side"] = "+".join(LABEL[h] for h in hops) if hops else "none"

    # --- reference side (needs pose sync, and every reach analysed)
    if not pose_sync_ok:
        fwhy.append("pose sync failed")
    elif len(res["reaches"]) < len(reaches) or not reaches:
        fwhy.append("reach windows not fully analysable")
    else:
        rt = [max(r["ref_twist"]["pelvis"], r["ref_twist"]["chest"]) for r in res["reaches"]]
        if all(x < REF_NONE_DEG for x in rt):
            res["reference_side"] = "none"
        elif not op_ok or any(r["op_still"] is None for r in res["reaches"]):
            res["reference_side"] = "twist_in_reference_operator_unverified"
        else:
            cls = set()
            for r in res["reaches"]:
                if r["pico_ambiguous"]:
                    cls.add("amb")
                elif not r["op_still"]:
                    cls.add("operator_moved")
                else:
                    cls.add("pico" if r["pico_added"] else "amb")
            if len(cls) == 1 and "amb" not in cls:
                res["reference_side"] = cls.pop()
            else:
                fwhy.append("reaches disagree or fall in the ambiguous band")
    return done()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: g0_analyze.py g0.npz")
        return 2
    res = analyze_twist(argv[0])
    print(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nrobot_side={res['robot_side']}  reference_side={res['reference_side']}  "
          f"gate_pass={res['gate_pass']}  operator_ref={'available' if res['operator_ref'] else 'unavailable'}")
    return 0 if res["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
