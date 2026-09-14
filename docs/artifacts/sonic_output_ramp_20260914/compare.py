import csv, json, sys, gzip
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
results = {}
for name in ("baseline_nominal", "patched_nominal", "baseline_repeat"):
    p = root / name
    if not (p / "events.json").exists():
        continue
    events = json.loads((p / "events.json").read_text())
    if events[-1]["kind"] != "end":
        continue
    phys = np.genfromtxt(
        p / ("physics.csv" if (p / "physics.csv").exists() else "physics.csv.gz"), delimiter=",", names=True
    )
    with (
        (p / "csv/motion_name.csv").open()
        if (p / "csv/motion_name.csv").exists()
        else gzip.open(p / "csv/motion_name.csv.gz", "rt")
    ) as f:
        names = list(csv.DictReader(f))
    prev = None
    switches = []
    for row in names:
        n = row["motion_name"]
        t = float(row["time_monotonic_ms"]) / 1000
        if n != prev and prev in ("streamed", "planner_motion") and n in ("streamed", "planner_motion"):
            windows = {}
            for label, lo, hi in [("pre", -0.5, 0), ("early", 0, 1), ("late", 1, 3)]:
                v = phys[(phys["wall"] >= t + lo) & (phys["wall"] < t + hi)]
                windows[label] = {"tilt_max": float(v["tilt_deg"].max()), "height_min": float(v["z"].min())}
            switches.append({"to": n, "wall": t, "windows": windows})
        prev = n
    origin = events[0]["wall"]
    end = events[-1]["wall"]
    v = phys[(phys["wall"] >= origin + 5) & (phys["wall"] <= end)]
    results[name] = {
        "switches": switches,
        "min_root_height_m": float(v["z"].min()),
        "max_root_tilt_deg": float(v["tilt_deg"].max()),
        "physics_wall_ratio": float(np.ptp(v["sim_time"]) / np.ptp(v["wall"])),
        "fall_events": [e for e in events if e["kind"] == "fall"],
        "output_ramp_starts": (
            (p / "sonic.log").read_text()
            if (p / "sonic.log").exists()
            else gzip.decompress((p / "sonic.log.gz").read_bytes()).decode()
        ).count("Teleop output ramp started"),
    }
print(json.dumps(results, indent=2))
(root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
