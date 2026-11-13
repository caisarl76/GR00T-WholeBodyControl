"""G1 height probe (sim, no operator): drives the deploy over ZMQ and records sim_gt + g1_debug.

Sim (free-standing, no elastic band; .venv_sim, PYTHONPATH=<repo>):
  python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen --no-elastic-band \\
      --publish-ground-truth --gt-bodies pelvis
Binds a PUB on 5556 (as pico_manager does), starts the deploy with a ``command`` start message,
then runs the PRD section 9 G1 sequence. Window markers follow the protocol in ``g1_analyze.py``.
Analyze with ``g1_analyze.py <out.npz>``. The npz is saved after the sweeps and on any exit (not per height: pickling a long recording
stalls the receive thread and would spoil the sync of the windows around it).

Fall (pelvis z < 0.3 m) in a walk trial: a ``fall:<H>`` marker is recorded, the npz is saved and
the probe exits 3 naming the height; remaining trials are simply absent (analysis: failed).
"""

from __future__ import annotations

import argparse
import gc
import json
import signal
import subprocess
import sys
import time

import numpy as np
import zmq

from gear_sonic.scripts.gates.g1_analyze import DIRS, FALL_Z, GRID, RUNS, SENTINEL, hkey
from gear_sonic.scripts.gates.recorder import GateRecorder
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message

# LocomotionMode values (gear_sonic/scripts/pico_manager_thread_server.py)
IDLE, SLOW_WALK, IDLE_SQUAT = 0, 1, 4
MODES = {"IDLE": IDLE, "IDLE_SQUAT": IDLE_SQUAT}
DT = 0.02  # command rate 50 Hz (deploy resets to IDLE after 1 s without planner messages)
SWEEP_FALL_Z = 0.15  # sweeps legitimately go low; only abort if clearly on the floor
WALK_SPEED = 0.3  # m/s
TURN_RATE = 0.5  # rad/s
MOTION_S, REST_S, HOLD_S = 10.0, 3.0, 3.0
STEP_HEIGHTS = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
RAMP_RATE = 0.1  # m/s of height command


class Fall(Exception):
    def __init__(self, height_key: str, z: float):
        super().__init__(f"pelvis z {z:.3f} m at height {height_key}")
        self.height_key = height_key


class Probe:
    def __init__(self, rec: GateRecorder, port: int):
        self.rec = rec
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        self.yaw = 0.0
        self.fall_z = SWEEP_FALL_Z
        self.fall_key = "sweep"
        time.sleep(1.0)

    def z(self):
        m = self.rec.last("sim_gt")
        return float(m["pelvis_pos"][2]) if m else None

    def run(self, secs, mode, height, *, move=None, speed=-1.0, yaw_rate=0.0, name=None, start_cmd=False):
        """Send planner commands at 50 Hz for ``secs``. ``height`` may be a callable of elapsed time."""
        if name:
            self.rec.mark(f"{name}:start")
        t0 = time.monotonic()
        n = 0
        while (el := time.monotonic() - t0) < secs:
            self.yaw += yaw_rate * DT
            facing = [np.cos(self.yaw), np.sin(self.yaw), 0.0]
            mv = [0.0, 0.0, 0.0]
            if move is not None:
                s = {"fwd": 1.0, "back": -1.0}[move]
                mv = [s * facing[0], s * facing[1], 0.0]
            h = height(el) if callable(height) else height
            if start_cmd:
                self.sock.send(build_command_message(start=True, stop=False, planner=True))
            self.sock.send(build_planner_message(mode, mv, facing, speed=speed, height=h))
            z = self.z()
            if z is not None and z < self.fall_z:
                raise Fall(self.fall_key, z)
            n += 1
            time.sleep(max(0.0, t0 + n * DT - time.monotonic()))
        if name:
            self.rec.mark(f"{name}:end")

    def sweep(self, label):
        m = MODES[label]
        for h in STEP_HEIGHTS:
            self.run(HOLD_S, m, h, name=f"step:{label}:{h:.2f}")
        for tag, a, b in (("down", 0.8, 0.2), ("up", 0.2, 0.8)):
            sgn = 1.0 if b > a else -1.0
            self.run(abs(b - a) / RAMP_RATE, m, lambda t, a=a, sgn=sgn: a + sgn * RAMP_RATE * t,
                     name=f"ramp:{label}:{tag}")

    def settle(self, k):
        self.run(HOLD_S, IDLE, -1.0, name=f"settle:{k}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--preroll", type=float, default=8.0, help="seconds in CONTROL before the first marker")
    a = ap.parse_args(argv)

    gc.disable()  # a gen-2 GC pass over ~1e6 recorded dicts stalls the receive thread (skew spikes)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rec = GateRecorder({"sim_gt": (a.host, 5560, "sim_gt"), "g1_debug": (a.host, 5557, "g1_debug")})
    p = Probe(rec, a.port)
    rc = 0
    try:
        p.run(a.preroll, IDLE, -1.0, start_cmd=True)  # deploy enters CONTROL on the start message
        print(f"preroll done, pelvis z={p.z()}", flush=True)
        p.settle(0)
        p.run(HOLD_S, IDLE, -1.0)  # buffer
        p.sweep("IDLE")
        p.settle(1)
        p.sweep("IDLE_SQUAT")
        for k in range(3):  # IDLE <-> IDLE_SQUAT at 0.5
            for label in ("IDLE", "IDLE_SQUAT"):
                p.run(HOLD_S, MODES[label], 0.5, name=f"switch:{k}:{label}")
        p.settle(2)

        rec.save(a.out)
        # analysis in a child process: it must not hold this process's GIL (recorder thread)
        out = subprocess.run(
            [sys.executable, "-c",
             "import json,sys;from gear_sonic.scripts.gates.g1_analyze import analyze_height as f;"
             "r=f(sys.argv[1]);print(json.dumps({k:r[k] for k in ('idle_honors_height','cmd_mode','cmd_to_z','h_default')}))",
             a.out], capture_output=True, text=True, check=True).stdout
        r = json.loads(out.splitlines()[-1])
        mode = MODES[r["cmd_mode"]]
        hd = None if r["h_default"] is None else round(r["h_default"], 2)
        print(f"sweeps done: idle_honors={r['idle_honors_height']} h_default={r['h_default']} "
              f"cmd_to_z={r['cmd_to_z']}", flush=True)
        heights = [SENTINEL] + ([hd] if hd is not None and hd not in GRID else []) + GRID
        p.fall_z = FALL_Z
        p.run(4.0, IDLE, -1.0)  # commands stopped during the save; re-settle before the first window
        for h in heights:
            hk = hkey(h)
            p.fall_key = hk
            hh = -1.0 if h == SENTINEL else h
            static = IDLE if h == SENTINEL else mode
            p.run(HOLD_S, static, hh, name=f"stat:{hk}")
            for k in range(RUNS):
                for d in DIRS:
                    name = f"trial:{hk}:{d}:{k}"
                    rec.mark(f"{name}:start")
                    if d == "turn":
                        p.run(MOTION_S, static, hh, yaw_rate=TURN_RATE)
                    else:
                        p.run(MOTION_S, SLOW_WALK, hh, move=d, speed=WALK_SPEED)
                    p.run(REST_S, static, hh)
                    rec.mark(f"{name}:end")
            print(f"height {hk} done", flush=True)
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
