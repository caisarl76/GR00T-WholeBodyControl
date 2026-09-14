from pathlib import Path
import csv, gzip, json
import numpy as np

ROOT = Path(__file__).resolve().parent
summary = {}
for name in ("baseline", "patched", "baseline_stress", "patched_stress"):
    out = ROOT / name
    if not (out / "events.json").exists():
        continue
    events = json.loads((out / "events.json").read_text())
    if events[-1]["kind"] not in ("end", "fall", "error"):
        continue
    physics = np.genfromtxt(out / "physics.csv.gz", delimiter=",", names=True)
    origin = events[0]["wall"]
    end = events[-1]["wall"]
    measured = physics[(physics["wall"] >= origin + 5) & (physics["wall"] <= end)]
    modes = [e for e in events if e["kind"] == "mode" and e["wall"] > origin + 1]
    result = dict(
        completed=events[-1]["kind"] == "end" and not any(e["kind"] in ("fall", "error") for e in events),
        falls=[e for e in events if e["kind"] == "fall"],
        control_seconds=end - origin,
        measured_seconds=end - origin - 5,
        min_root_height_m=float(measured["z"].min()),
        max_root_tilt_deg=float(measured["tilt_deg"].max()),
        physics_wall_ratio=float(np.ptp(measured["sim_time"]) / np.ptp(measured["wall"])),
        transitions=[],
    )
    for e in modes:
        window = physics[(physics["wall"] >= e["wall"]) & (physics["wall"] < e["wall"] + 1)]
        item = dict(
            to=e["mode"],
            at_seconds=e["wall"] - origin,
            min_height_m=float(window["z"].min()),
            max_tilt_deg=float(window["tilt_deg"].max()),
        )
        if (out / "motion_name.csv.gz").exists():
            with gzip.open(out / "motion_name.csv.gz", "rt") as f:
                rows = list(csv.DictReader(f))
            target = "streamed" if e["mode"] == "pose" else "planner_motion"
            nextrow = next(
                (
                    r
                    for r in rows
                    if float(r["time_monotonic_ms"]) / 1000 >= e["wall"] and r["motion_name"] == target
                ),
                None,
            )
            item["destination_latency_ms"] = (
                None if nextrow is None else float(nextrow["time_monotonic_ms"]) - e["wall"] * 1000
            )
        result["transitions"].append(item)
    if (out / "motion_name.csv.gz").exists():
        a = np.genfromtxt(out / "motion_playing.csv.gz", delimiter=",", names=True)
        mask = (a["time_monotonic_ms"] / 1000 >= origin + 5) & (a["time_monotonic_ms"] / 1000 <= end)
        result["paused_control_ticks_after_settling"] = int(np.sum(a["playing_0"][mask] == 0))
        result["logged_control_ticks_after_settling"] = int(mask.sum())
    summary[name] = result
print(json.dumps(summary, indent=2))
