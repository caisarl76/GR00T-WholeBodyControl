"""G1b Step 9 probe (sim, no operator): decision-time stress test of the Step 8 still-standing check.

PRD section 9 "Step 9". Sim (free-standing; .venv_sim, PYTHONPATH=<repo>), see outputs/gates/run_g1b_still3.sh:
  python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen --no-elastic-band \\
      --publish-ground-truth --gt-bodies pelvis --disturbance-port 5561
``--mode calibrate``: run D0, the PRD ladders (each value x2, ascending, IDLE_SQUAT H_default), sim_gt only is analysed.
``--mode validate``: run D, part A (walk-stop-walk x 9, 5 requests each) and part B (21 disturbance blocks) at
the magnitudes of the calibration JSON (``--magnitudes``). Analyze with ``g1b_still3.py``.

The planner stream is the G1b probe's (``g1_height_probe.Probe.run``); the wrench is pushed once per planner tick
(zero outside a disturbance, also on exit). Markers reuse the G1b vocabulary (prehold, ramp, hold, sw1, walk, stop,
back, switch:to_sentinel), plus ``dist:<type>[@<N>]:<rep>`` (onset -> window end, written after the block with
explicit times), ``req:<dir>:<rep>:<i>:<delta>`` (walk-request instant) and ``hold:pre:`` (3 s hold before the -1
switch). A disturbance is not a command change: no static marker starts at an onset.
"""

from __future__ import annotations

import argparse
import gc
import json
import signal
import sys
import time

import msgpack
import zmq

from gear_sonic.scripts.gates.g1_analyze import DIRS, FALL_Z
from gear_sonic.scripts.gates.g1_height_probe import (
    HOLD_S, IDLE, IDLE_SQUAT, RAMP_RATE, SLOW_WALK, TURN_RATE, WALK_SPEED, Fall, Probe,
)
from gear_sonic.scripts.gates.recorder import GateRecorder

RAMP_START, PREHOLD_S, HANDOFF_S = 0.60, 3.0, 0.5
ONSET_S = 6.0  # disturbance onset after the block's last static-command change
WINDOW_S = {"nudge": 8.0, "creep": 17.0}
NUDGE_S = 0.1
CREEP_UP_S, CREEP_HOLD_S, CREEP_DOWN_S = 4.0, 3.0, 4.0
TAIL_S = {"calibrate": 2.0, "validate": 6.0}  # static time after the window (recovery); not in the PRD, see report
# type -> (profile kind, unit wrench [fx, fy, fz, tx, ty, tz] in the world frame)
TYPES = {
    "nudge+x": ("nudge", (1, 0, 0, 0, 0, 0)), "nudge-x": ("nudge", (-1, 0, 0, 0, 0, 0)),
    "nudge+y": ("nudge", (0, 1, 0, 0, 0, 0)), "nudge-y": ("nudge", (0, -1, 0, 0, 0, 0)),
    "lateral": ("creep", (0, 1, 0, 0, 0, 0)), "down": ("creep", (0, 0, -1, 0, 0, 0)),
    "yaw": ("creep", (0, 0, 0, 0, 0, 1)),
}
LADDERS = {"nudge": [25, 50, 100, 150, 200], "lateral": [15, 30, 60, 100], "down": [50, 100, 200, 300], "yaw": [5, 10, 20, 40]}
CAL_TYPES = ("nudge+x", "nudge+y", "lateral", "down", "yaw")
DELTAS = {0: (1.0, 3.0, 4.5, 5.0, 6.0), 1: (6.0, 4.5, 1.0, 5.0, 3.0), 2: (3.0, 5.0, 6.0, 1.0, 4.5)}
PART_A_HOLD_S, WALK_S, POST_HOLD_S = 6.0, 5.0, 8.0
REPS = 3
ZERO = (0.0,) * 6


def ladder(typ):
    return LADDERS["nudge" if typ.startswith("nudge") else typ]


def profile_scale(kind, t):
    """Unit-wrench scale ``t`` seconds after onset: 0.1 s pulse, or 4 s up / 3 s hold / 4 s down."""
    if t < 0:
        return 0.0
    if kind == "nudge":
        return 1.0 if t < NUDGE_S else 0.0
    if t < CREEP_UP_S:
        return t / CREEP_UP_S
    if t < CREEP_UP_S + CREEP_HOLD_S:
        return 1.0
    u = t - CREEP_UP_S - CREEP_HOLD_S
    return 1.0 - u / CREEP_DOWN_S if u < CREEP_DOWN_S else 0.0


class _Tick:
    """Socket proxy: every planner message sent also triggers one wrench push (the planner bytes are untouched)."""

    def __init__(self, sock, on_send):
        self.sock, self.on_send = sock, on_send

    def send(self, data, *a, **k):
        self.sock.send(data, *a, **k)
        self.on_send()


class DProbe(Probe):
    """G1b Probe plus a PUSH to the sim's ``--disturbance-port``; ``profile`` = (kind, unit wrench, magnitude, onset ns)."""

    now_ns = staticmethod(time.monotonic_ns)

    def __init__(self, rec, port, host, dist_port):
        super().__init__(rec, port)
        self.push = zmq.Context.instance().socket(zmq.PUSH)
        self.push.setsockopt(zmq.LINGER, 1000)
        self.push.setsockopt(zmq.SNDTIMEO, 1000)
        self.push.connect(f"tcp://{host}:{dist_port}")
        self.profile = None
        self.sock = _Tick(self.sock, self._tick)

    def send_wrench(self, w):
        self.push.send(msgpack.packb({"force": [float(x) for x in w[:3]], "torque": [float(x) for x in w[3:]]}))

    def _tick(self):
        if self.profile is None:
            return self.send_wrench(ZERO)
        kind, vec, mag, t_on = self.profile
        s = mag * profile_scale(kind, (self.now_ns() - t_on) / 1e9)
        self.send_wrench([s * v for v in vec])


def _hold_prefix(p, rec, hd, tag):
    p.run(PREHOLD_S, IDLE_SQUAT, RAMP_START, name=f"prehold:{tag}")
    p.run((hd - RAMP_START) / RAMP_RATE, IDLE_SQUAT, lambda t: RAMP_START + RAMP_RATE * t, name=f"ramp:{tag}")


def dist_block(p, rec, hd, typ, mag, rep, mode):
    """One disturbance block. Static hold (IDLE_SQUAT H_default; validate rep 1: 3 s hold then IDLE -1) of
    ONSET_S + window + tail after the last static-command change; the wrench starts ONSET_S after it."""
    kind, vec = TYPES[typ]
    cal = mode == "calibrate"
    tag = f"{typ}@{mag:g}:{rep}" if cal else f"{typ}:{rep}"
    p.fall_key = tag
    _hold_prefix(p, rec, hd, f"{'C' if cal else 'B'}:{tag}")
    sentinel = not cal and rep == 1
    if sentinel:
        p.run(HOLD_S, IDLE_SQUAT, hd, name=f"hold:pre:B:{tag}")
        name, m, h = f"switch:to_sentinel:B:{tag}", IDLE, -1.0
    else:
        name, m, h = f"hold:{'C' if cal else 'B'}:{tag}", IDLE_SQUAT, hd
    t_on = p.now_ns() + int(ONSET_S * 1e9)
    p.profile = (kind, vec, float(mag), t_on)
    p.run(ONSET_S + WINDOW_S[kind] + TAIL_S[mode], m, h, name=name)
    p.profile = None
    rec.mark(f"dist:{tag}:start", t_on)
    rec.mark(f"dist:{tag}:end", t_on + int(WINDOW_S[kind] * 1e9))


def walk(p, d, tag):
    p.run(HANDOFF_S, IDLE, -1.0, name=f"sw1:{tag}")
    if d == "turn":
        p.run(WALK_S, IDLE, -1.0, yaw_rate=TURN_RATE, name=f"walk:{tag}")
    else:
        p.run(WALK_S, SLOW_WALK, -1.0, move=d, speed=WALK_SPEED, name=f"walk:{tag}")
    p.run(HANDOFF_S, IDLE, -1.0, name=f"stop:{tag}")


def walk_stop_walk(p, rec, hd, d, k):
    """Part A rep: 6 s settled hold, 5 x (hand-off, walk, stop, hold delta, request), then walk, stop, 8 s hold."""
    tag = f"A:{d}:{k}"
    p.fall_key = tag
    _hold_prefix(p, rec, hd, tag)
    p.run(PART_A_HOLD_S, IDLE_SQUAT, hd, name=f"hold:{tag}")
    for i, delta in enumerate(DELTAS[k]):
        walk(p, d, f"{tag}:{i}")
        p.run(delta, IDLE_SQUAT, hd, name=f"back:{tag}:{i}")
        rec.mark(f"req:{d}:{k}:{i}:{delta:.1f}")  # walk request delta after the last static-command change
    walk(p, d, f"{tag}:5")
    p.run(POST_HOLD_S, IDLE_SQUAT, hd, name=f"back:{tag}:post")


def run_calibrate(p, rec, hd, types):
    for typ in types:
        for mag in ladder(typ):
            for rep in (0, 1):
                dist_block(p, rec, hd, typ, mag, rep, "calibrate")
                print(f"{typ} {mag:g} rep {rep} done", flush=True)


def run_validate(p, rec, hd, mags):
    for k in range(REPS):
        for d in DIRS:
            walk_stop_walk(p, rec, hd, d, k)
            print(f"A {d}:{k} done", flush=True)
    for k in range(REPS):
        for typ, (kind, _) in TYPES.items():
            dist_block(p, rec, hd, typ, mags[kind if kind == "nudge" else typ], k, "validate")
            print(f"B {typ}:{k} done", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("calibrate", "validate"), required=True)
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--h-default", type=float, required=True, help="H_default (G1-measured, PRD section 11)")
    ap.add_argument("--disturbance-port", type=int, default=5561)
    ap.add_argument("--magnitudes", help="validate: the calibration JSON (key 'chosen')")
    ap.add_argument("--types", default=",".join(CAL_TYPES), help="calibrate: comma list (D0b: the remaining types)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--preroll", type=float, default=8.0, help="seconds in CONTROL before the first marker")
    a = ap.parse_args(argv)
    mags = None
    if a.mode == "validate":
        if not a.magnitudes:
            ap.error("--magnitudes is required for validate")
        mags = json.load(open(a.magnitudes))["chosen"]
    types = a.types.split(",")
    if not set(types) <= set(CAL_TYPES):
        ap.error(f"--types must be a subset of {CAL_TYPES}")

    gc.disable()  # a gen-2 GC pass over ~1e6 recorded dicts stalls the receive thread (skew spikes)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rec = GateRecorder({"sim_gt": (a.host, 5560, "sim_gt"), "g1_debug": (a.host, 5557, "g1_debug")})
    p = DProbe(rec, a.port, a.host, a.disturbance_port)
    rc = 0
    try:
        p.run(a.preroll, IDLE, -1.0, start_cmd=True)  # deploy enters CONTROL on the start message
        print(f"preroll done, pelvis z={p.z()}", flush=True)
        p.run(HOLD_S, IDLE, -1.0, name="settle:0")
        p.fall_z = FALL_Z
        if a.mode == "calibrate":
            run_calibrate(p, rec, a.h_default, types)
        else:
            run_validate(p, rec, a.h_default, mags)
    except Fall as e:
        rec.mark(f"fall:{e.height_key}:start")
        print(f"FALL: {e}; remaining blocks not run", flush=True)
        rc = 3
    finally:
        p.profile = None
        try:
            p.send_wrench(ZERO)  # never leave a wrench held in the sim
            p.push.close()
        except zmq.ZMQError as e:
            print(f"WARNING: final zero wrench not sent: {e}", flush=True)
        try:
            rec.save(a.out)
        finally:
            rec.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
