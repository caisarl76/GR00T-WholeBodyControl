"""G1b probe (sim, no operator): restricted walking-height hand-off, PRD section 9 G1b.

Sim (free-standing, no elastic band; .venv_sim, PYTHONPATH=<repo>):
  python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen --no-elastic-band \\
      --publish-ground-truth --gt-bodies pelvis
Per direction (fwd, back, turn) x 5 runs: pre-hold IDLE_SQUAT 0.60 for 3 s, ramp the height 0.60 ->
H_default at 0.1 m/s, hold ``--hold-s`` (walk_settle_s, measured per trial by ``g1b_analyze.py`` as ``settle_s``),
switch to IDLE -1 for 0.5 s, walk 10 s with -1 (turn: IDLE at 0.5 rad/s), IDLE -1 for 0.5 s, then
IDLE_SQUAT H_default for 3 s. Then an isolated switch block x 5: IDLE_SQUAT H_default 3 s -> IDLE -1 3 s ->
IDLE_SQUAT H_default 3 s (``switch:hold|to_sentinel|to_height:K``). Markers follow the protocol in ``g1b_analyze.py``; analyze with it.
Reuses the planner sender and recorder of ``g1_height_probe.py``. Fall (z < 0.3 m): ``fall:<DIR><K>``
marker, save, exit 3. The npz is saved only at the end (and on any exit) to keep the sync clean.
"""

from __future__ import annotations

import argparse
import gc
import signal
import sys

from gear_sonic.scripts.gates.g1_analyze import DIRS, FALL_Z, RUNS
from gear_sonic.scripts.gates.g1_height_probe import (
    HOLD_S, IDLE, IDLE_SQUAT, MOTION_S, RAMP_RATE, SLOW_WALK, TURN_RATE, WALK_SPEED, Fall, Probe,
)
from gear_sonic.scripts.gates.recorder import GateRecorder

RAMP_START = 0.60
HANDOFF_S = 0.5
PREHOLD_S = 3.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--h-default", type=float, required=True, help="H_default (G1-measured, PRD section 11)")
    ap.add_argument("--hold-s", type=float, default=4.5, help="hold at H_default after the ramp (walk_settle_s)")
    ap.add_argument("--iso-hold-s", type=float, default=3.0, help="each isolated-block window (default 3.0)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--preroll", type=float, default=8.0, help="seconds in CONTROL before the first marker")
    a = ap.parse_args(argv)
    hd = a.h_default

    gc.disable()  # a gen-2 GC pass over ~1e6 recorded dicts stalls the receive thread (skew spikes)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rec = GateRecorder({"sim_gt": (a.host, 5560, "sim_gt"), "g1_debug": (a.host, 5557, "g1_debug")})
    p = Probe(rec, a.port)
    rc = 0
    try:
        p.run(a.preroll, IDLE, -1.0, start_cmd=True)  # deploy enters CONTROL on the start message
        print(f"preroll done, pelvis z={p.z()}", flush=True)
        p.run(HOLD_S, IDLE, -1.0, name="settle:0")
        p.fall_z = FALL_Z
        for k in range(RUNS):
            for d in DIRS:
                tag = f"{d}:{k}"
                p.fall_key = f"{d}{k}"
                p.run(PREHOLD_S, IDLE_SQUAT, RAMP_START, name=f"prehold:{tag}")
                p.run((hd - RAMP_START) / RAMP_RATE, IDLE_SQUAT, lambda t: RAMP_START + RAMP_RATE * t, name=f"ramp:{tag}")
                p.run(a.hold_s, IDLE_SQUAT, hd, name=f"hold:{tag}")
                rec.mark(f"trial:{tag}:start")
                p.run(HANDOFF_S, IDLE, -1.0, name=f"sw1:{tag}")
                if d == "turn":
                    p.run(MOTION_S, IDLE, -1.0, yaw_rate=TURN_RATE, name=f"walk:{tag}")
                else:
                    p.run(MOTION_S, SLOW_WALK, -1.0, move=d, speed=WALK_SPEED, name=f"walk:{tag}")
                p.run(HANDOFF_S, IDLE, -1.0, name=f"stop:{tag}")
                p.run(HOLD_S, IDLE_SQUAT, hd, name=f"back:{tag}")
                rec.mark(f"trial:{tag}:end")
            print(f"run {k} done", flush=True)
        for k in range(RUNS):  # isolated switch block
            p.fall_key = f"iso{k}"
            p.run(a.iso_hold_s, IDLE_SQUAT, hd, name=f"switch:hold:{k}")
            p.run(a.iso_hold_s, IDLE, -1.0, name=f"switch:to_sentinel:{k}")
            p.run(a.iso_hold_s, IDLE_SQUAT, hd, name=f"switch:to_height:{k}")
        print("isolated switches done", flush=True)
    except Fall as e:
        rec.mark(f"fall:{e.height_key}:start")
        print(f"FALL: {e}; remaining trials not run", flush=True)
        rc = 3
    finally:
        try:
            rec.save(a.out)
        finally:
            rec.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
