"""G0 recorder: pose (5556), g1_debug (5557), sim_gt (5560), operator keypress markers
and, when the backend exposes it, a raw sternum-tracker stream (``operator_ref``).

Keys (one letter, no Enter needed on a tty) start a segment that lasts until the next key:
    s still   b turn_body   c twist_chest   r reach   e end

    python -m gear_sonic.scripts.gates.g0_twist_log --duration 150 --out g0.npz
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time

import numpy as np

from gear_sonic.scripts.gates.recorder import GateRecorder

KEYS = {"s": "still", "b": "turn_body", "c": "twist_chest", "r": "reach", "e": "end"}
OP_HZ = 200.0  # poll rate; samples are pushed only when the device timestamp changes


def _keys(rec: GateRecorder, stop: threading.Event) -> None:
    fd = sys.stdin.fileno()
    old = None
    if sys.stdin.isatty():
        import termios
        import tty

        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    try:
        while not stop.is_set():
            if select.select([fd], [], [], 0.1)[0]:
                data = os.read(fd, 64)
                if not data:
                    return
                for ch in data.decode(errors="ignore"):
                    if ch in KEYS:
                        rec.mark(KEYS[ch])
                        print(f"[marker] {KEYS[ch]}", flush=True)
    finally:
        if old is not None:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _discover_operator_ref(rec: GateRecorder, stop: threading.Event, idx: int):
    """List backend tracker/motion functions; start the raw-tracker thread if data exists."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError:
        print("backend functions mentioning tracker/motion: none (xrobotoolkit_sdk not importable)")
        print("operator_ref unavailable")
        return None
    names = [n for n in dir(xrt) if "tracker" in n.lower() or "motion" in n.lower()]
    print(f"backend functions mentioning tracker/motion: {names}")
    try:
        xrt.init()
        deadline = time.time() + 3.0
        while xrt.num_motion_data_available() <= 0 and time.time() < deadline:
            time.sleep(0.1)
        n = xrt.num_motion_data_available()
        if n <= 0:
            raise RuntimeError("no motion tracker data")
        print(f"motion trackers: {n}, serials: {xrt.get_motion_tracker_serial_numbers()}; "
              f"using index {idx} as the sternum tracker")
    except Exception as e:  # SDK missing service / second init refused / no tracker
        print(f"operator_ref unavailable ({e})")
        return None

    def run():
        last = None
        while not stop.is_set():
            ts = xrt.get_motion_timestamp_ns()
            if ts != last:
                last = ts
                src = time.monotonic_ns()
                trk = np.array(xrt.get_motion_tracker_pose(), dtype=float).reshape(-1, 7)
                hq = np.array(xrt.get_headset_pose(), dtype=float).ravel()
                rec.push("operator_ref", {"trackers": trk, "idx": idx, "dev_ts_ns": int(ts),
                                          "headset_pose": hq, "t_mono_ns": src}, src)
            time.sleep(1.0 / OP_HZ)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to record")
    ap.add_argument("--out", default="g0.npz")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--op-tracker-index", type=int, default=-1,
                    help="index into the backend motion-tracker list of the sternum tracker (default: last)")
    a = ap.parse_args(argv)

    rec = GateRecorder({
        "pose": (a.host, 5556, "pose"),
        "g1_debug": (a.host, 5557, "g1_debug"),
        "sim_gt": (a.host, 5560, "sim_gt"),
    })
    stop = threading.Event()
    _discover_operator_ref(rec, stop, a.op_tracker_index)
    threading.Thread(target=_keys, args=(rec, stop), daemon=True).start()
    print("keys: s=still b=turn_body c=twist_chest r=reach e=end", flush=True)
    t_end = time.monotonic() + a.duration
    try:
        while time.monotonic() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.3)
        rec.close()
        rec.save(a.out)
    d = np.load(a.out, allow_pickle=True)
    for n in ("pose", "g1_debug", "sim_gt", "markers", "operator_ref"):
        k = f"{n}_src_ns"
        print(f"{n}: {len(d[k]) if k in d else 0} messages")
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
