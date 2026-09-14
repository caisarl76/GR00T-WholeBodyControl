#!/usr/bin/env python3
"""Read-only per-switch CSV analysis; outputs only adjacent diagnosis directory."""

import csv, json, hashlib, gzip
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT / "live_before"
manifest = {}


def read(path):
    b = gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    manifest[str(path)] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return list(csv.DictReader(b.decode().splitlines()))


def arr(name):
    rows = read(ROOT / "csv" / f"{name}.csv")
    cols = list(rows[0])[5:]
    return np.array([[float(r[c]) for c in cols] for r in rows]), rows


motion = read(ROOT / "csv/motion_name.csv")
t = np.array([float(r["time_monotonic_ms"]) / 1000 for r in motion])
names = np.array([r["motion_name"] for r in motion])
n = len(t)
data = {}
for k in ["encoder_mode", "action", "q", "dq", "motor_torque", "token_state"]:
    a, rows = arr(k)
    assert len(a) == n and all(r["index"] == motion[i]["index"] for i, r in enumerate(rows))
    data[k] = a
pr = read(ROOT / "physics.csv")
pt = np.array([float(r["wall"]) for r in pr])
pz = np.array([float(r["z"]) for r in pr])
tilt = np.array([float(r["tilt_deg"]) for r in pr])
band = np.array([float(r["band"]) for r in pr])
dt = np.r_[np.nan, np.diff(t) * 1000]
deltas = {k: np.r_[np.full((1, a.shape[1]), np.nan), np.diff(a, axis=0)] for k, a in data.items()}
switches = np.flatnonzero(names[1:] != names[:-1]) + 1
switches = [
    i
    for i in switches
    if names[i - 1] in ["planner_motion", "streamed"] and names[i] in ["planner_motion", "streamed"]
]
results = []


def maxabs(a):
    return float(np.nanmax(np.abs(a))) if a.size else None


for si, i in enumerate(switches):
    ts = t[i]
    nxt = t[switches[si + 1]] if si + 1 < len(switches) else t[-1] + 0.02
    r = {
        "switch": si + 1,
        "index": int(i),
        "monotonic_s": ts,
        "relative_s": ts - t[0],
        "from": names[i - 1],
        "to": names[i],
        "encoder_from": float(data["encoder_mode"][i - 1, 0]),
        "encoder_to": float(data["encoder_mode"][i, 0]),
        "next_switch_delta_s": nxt - ts,
        "first_new_policy_action_step_maxabs": maxabs(deltas["action"][i + 1]),
        "first_new_policy_action_l2": float(np.linalg.norm(deltas["action"][i + 1])),
        "first_new_policy_action_argmax": int(np.argmax(abs(deltas["action"][i + 1]))),
        "windows": {},
    }
    for label, lo, hi in [("pre", -0.5, 0), ("early", 0, 1), ("late", 1, 3)]:
        end = min(ts + hi, nxt, t[-1] + 0.02)
        mask = (t >= ts + lo) & (t < end)
        pm = (pt >= ts + lo) & (pt < end)
        w = {
            "actual_end_s": end - ts,
            "count": int(mask.sum()),
            "next_switch_truncated": bool(ts + hi > nxt),
            "tilt_max_deg": float(tilt[pm].max()),
            "tilt_mean_deg": float(tilt[pm].mean()),
            "z_min_m": float(pz[pm].min()),
            "z_mean_m": float(pz[pm].mean()),
            "z_max_m": float(pz[pm].max()),
            "band_values": sorted(set(band[pm].tolist())),
            "intertick_max_ms": float(np.nanmax(dt[mask])),
            "intertick_p99_ms": float(np.nanpercentile(dt[mask], 99)),
            "intertick_over30_count": int(np.sum(dt[mask] > 30)),
        }
        for k in ["action", "q", "dq", "motor_torque"]:
            w[k + "_maxabs"] = maxabs(data[k][mask])
            w[k + "_step_maxabs"] = maxabs(deltas[k][mask])
        for k in ["action", "dq"]:
            w[k + "_l2_max"] = float(np.linalg.norm(data[k][mask], axis=1).max())
        # Raw actions are previous-tick policy output in IsaacLab order, not radians or applied targets.
        w["token_step_l2_max"] = float(np.linalg.norm(deltas["token_state"][mask], axis=1).max())
        if label == "early":
            inds = np.flatnonzero(mask)
            pi = np.flatnonzero(pm)
            aa = np.abs(deltas["action"][mask])
            row, col = np.unravel_index(aa.argmax(), aa.shape)
            r["early_action_peak_delay_s"] = float(t[inds[row]] - ts)
            r["early_action_peak_index"] = int(col)
            r["early_tilt_peak_delay_s"] = float(pt[pi[tilt[pm].argmax()]] - ts)
        r["windows"][label] = w
    results.append(r)
summary = {
    "n_rows": n,
    "time_start": float(t[0]),
    "time_end": float(t[-1]),
    "n_switches": len(results),
    "intertick_global_max_ms": float(np.nanmax(dt)),
    "intertick_global_p99_ms": float(np.nanpercentile(dt, 99)),
    "intertick_over30": int(np.sum(dt > 30)),
    "first_z_under_04_during_logging": float(pt[(pt >= t[0]) & (pt <= t[-1]) & (pz < 0.4)][0])
    if np.any((pt >= t[0]) & (pt <= t[-1]) & (pz < 0.4))
    else None,
}
(OUT / "switch_metrics.json").write_text(
    json.dumps({"summary": summary, "switches": results, "manifest": manifest}, indent=2) + "\n"
)
lines = [
    "# Live A+X handoff metrics",
    "",
    f"{len(results)} switches, {n} control samples; monotonic range {t[0]:.6f}–{t[-1]:.6f} s.",
    "",
    "Intervals use shared monotonic clocks; pre [-0.5,0), early [0,1), late [1,3), truncated before the next switch. Maximums are over all joints. Action is raw decoder output in IsaacLab order (dimensionless), logged one control tick after it was computed; first new-policy action is row switch+1. q is measured position minus default angle in IsaacLab order (radians); dq is measured velocity in the same order (radians per second); motor torque is hardware order. Logging intervals measure control-loop start cadence, not full inference latency. The final uncontrolled shutdown tail is not used for switch-local metrics.",
    "",
    "|#|switch time (relative s)|direction|window|tilt max °|z min m|action Δ max|q Δ max rad|dq max rad/s|tick max ms|",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in results:
    for label, w in r["windows"].items():
        lines.append(
            f"|{r['switch']}|{r['relative_s']:.3f}|{r['from']}→{r['to']}|{label}{'*' if w['next_switch_truncated'] else ''}|{w['tilt_max_deg']:.2f}|{w['z_min_m']:.4f}|{w['action_step_maxabs']:.3f}|{w['q_step_maxabs']:.3f}|{w['dq_maxabs']:.3f}|{w['intertick_max_ms']:.3f}|"
        )
lines += [
    "",
    "*Late window truncated before next switch. See JSON for all metrics and file digests.",
    "",
    "## First newly selected policy action",
    "",
    "|#|direction|action step max|action step L2|action index|",
    "|---|---|---|---|---|",
]
for r in results:
    lines.append(
        f"|{r['switch']}|{r['from']}→{r['to']}|{r['first_new_policy_action_step_maxabs']:.3f}|{r['first_new_policy_action_l2']:.3f}|{r['first_new_policy_action_argmax']}|"
    )
lines += [
    "",
    "## Peak timing",
    "",
    "|#|raw action step peak delay s (logged)|action index|tilt peak delay s|",
    "|---|---|---|---|",
]
for r in results:
    lines.append(
        f"|{r['switch']}|{r['early_action_peak_delay_s']:.3f}|{r['early_action_peak_index']}|{r['early_tilt_peak_delay_s']:.3f}|"
    )
lines += [
    "",
    "Indices 13/14 are left/right ankle pitch; 10 is right knee; 23 is left wrist roll. Subtract one control tick (~20 ms) from action logged delay for the output computation time. The q/dq values use IsaacLab order as verified in GatherRobotStateToLogger.",
    "",
    "## Findings",
    "",
    "All four POSE→PLANNER events have first-second tilt above both their pre-window and subsequent 1–3-second maxima: 5.39°, 10.48°, 9.09°, 4.81°. The strongest is switch 4 (monotonic 267270.220747): 4.00° before, 10.48° early, 3.33° late; minimum height 0.7852→0.7683→0.7863 m; maximum joint speed 0.932→7.360→2.246 rad/s. This substantiates a transient balance disturbance rather than a no-fall pass.",
    "",
    "Eight of nine switches have their first-second maximum raw action step at logged +0.52 to +0.64 seconds, predominantly ankle pitch. The source has a 1-second smoothstep blend of a captured initial joint position with live policy targets; the peak is around mid-ramp, not its endpoint. The temporal pattern is consistent with that path contributing, but this capture alone cannot prove causality. The logged action is before the ramp, so it is not the applied target.",
    "",
    "No switch-local control tick gap exceeds 25.8 ms. The whole capture has one gap above 30 ms (48.31 ms), at monotonic 267332.709077, +14.368 seconds after switch 9; it does not explain the repeatable 1-second disturbances.",
    "",
    "The final POSE interval also contains separate fall/reset-like episodes BEFORE the final shutdown: first z<0.4 at monotonic 267323.766166 (+5.425 s after switch 9), first low episode reaches z=0.202 m and tilt≈68° and is followed by upright height the next second; another at +16 seconds reaches z=0.204 m and tilt≈89°. Simulator.log explicitly reports fallen-height warnings. These later events are not included as evidence that the mode switch itself caused a fall; input/body motion and reset behavior require separate attribution.",
    "",
    "## Scope and interpretation",
    "",
    "This is an observational live capture: operator/body motion, mode switch, and reference changes are not independently controlled. Temporal alignment can establish a repeatable transition-associated disturbance but cannot by itself identify the causal source. No-fall is not a success criterion.",
    "",
    json.dumps(summary, indent=2),
]
(OUT / "switch_metrics.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
