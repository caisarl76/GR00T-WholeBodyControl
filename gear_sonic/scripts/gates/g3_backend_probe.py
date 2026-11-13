"""G3 capture-backend probe: does a PICO backend expose a per-controller signal that tells
"controller tracked and still" apart from "controller lost"?

Writes one JSONL line per sample (~200 Hz poll) plus a ``{"meta": ...}`` first line and
``{"marker": <text>, "t_mono": ...}`` lines for every line the operator types on stdin.
``t_mono`` = ``time.monotonic()``. Ctrl-C ends the run cleanly. SDK missing/unusable -> exit 2.
XRT with no controller data after --connect-timeout: warns and records anyway (all stale).
Isaac with no session after --connect-timeout: exit 2.
Key fields: XRT per-side snapshot ``receipt_age_ns`` (>= 2**62 means never received) and
``binding_generation``; Isaac snapshot ``aim_pose.is_valid`` / ``grip_pose.is_valid``.
The last line of the file may be truncated after a hard kill: skip a non-JSON last line.

Operator run (per backend; type each phase name + Enter at the moment the phase starts):
  XRT (.venv_teleop, headset connected, roboticsservice running):
    /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m \\
        gear_sonic.scripts.gates.g3_backend_probe --input-source xrt --out g3_xrt.jsonl
  Isaac Capture (env with isaacteleop[cloudxr]; bring the CloudXR session / headset up FIRST):
    python -m gear_sonic.scripts.gates.g3_backend_probe --input-source isaac-teleop --out g3_isaac-teleop.jsonl
  Phases (both backends):
    still        hold both controllers still for 30 s
    cover_left   left controller behind your back for 5 s
    off_left     turn the left controller OFF for 5 s, then back on
    cover_right  repeat for the right controller (5 s)
    off_right    right controller OFF for 5 s, then back on
    swing        swing a controller fast (~2 m/s) for 5 s
    end          then Ctrl-C

Analysis (Step 3), per backend:
  1. Window the samples between markers (still / cover_* / off_*).
  2. For each candidate signal (any ``validity`` field e.g. is_valid / tracking-state, or a
     per-controller / reader timestamp that stops advancing), check it: (a) never flags "lost"
     in ``still`` (zero false positives), and (b) does flag it in cover_* and off_* windows.
     Pick ``validity`` (a boolean/state), ``timestamp`` (stops changing), or ``none``.
     stick_control_enabled = (signal != "none").
  3. Swing peak speed: take the controller position in the ``swing`` window (XRT: ``<side>.pose[:3]`` of the snapshot;
     Isaac: ``<side>.snapshot.aim_pose.pose.position`` x/y/z, NOT a fallback to grip),
     speed_i = |p[i+1]-p[i]| / (t_mono[i+1]-t_mono[i]) over samples where the pose changed
     (drop duplicate polls); report max (and p99) and compare with the 6 m/s outlier limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

_SIDES = ("left", "right")


def _jsonable(x, depth=0):
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, depth) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth) for k, v in x.items()}
    if depth < 4:  # generic object: dump every public non-callable attribute
        out = {}
        for a in dir(x):
            if a.startswith("_"):
                continue
            try:
                v = getattr(x, a)
            except Exception as e:  # noqa: BLE001
                out[a] = f"<err {e}>"
                continue
            if not callable(v):
                out[a] = _jsonable(v, depth + 1)
            elif a.startswith(("get_", "is_")) or "time" in a or "valid" in a:
                try:  # zero-arg methods that may carry validity / timestamps
                    out[a + "()"] = _jsonable(v(), depth + 1)
                except Exception as e:  # noqa: BLE001
                    out[a + "()"] = f"<err {e}>"
        if out:
            return out
    return repr(x)


def _die(msg):
    # os._exit: the XRT SDK's native threads abort() (core dump) on a normal interpreter exit
    print(msg, file=sys.stderr, flush=True)
    os._exit(2)


class XrtBackend:
    def __init__(self, timeout):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as e:
            _die(f"xrobotoolkit_sdk not importable ({e}); run in .venv_teleop")
        self.xrt = xrt
        try:
            xrt.init()
        except Exception as e:  # noqa: BLE001
            _die(f"xrt.init() failed ({e})")
        t0 = time.monotonic()
        while not self._connected():
            if time.monotonic() - t0 > timeout:
                print(f"WARNING: no controller data after {timeout}s (headset/roboticsservice?); "
                      "recording anyway, samples will be stale", file=sys.stderr, flush=True)
                break
            time.sleep(0.2)
        self.getters = [
            g for g in ("get_left_controller_snapshot", "get_right_controller_snapshot", "get_time_stamp_ns",
                        "get_body_timestamp_ns", "get_motion_timestamp_ns",
                        "get_left_hand_is_active", "get_right_hand_is_active")
            if hasattr(xrt, g)
        ]
        self.meta = {"dir": sorted(dir(xrt)), "polled_getters": self.getters}

    def _connected(self):
        x = self.xrt
        if x.get_time_stamp_ns() != 0:
            return True
        return any(getattr(x, f"get_{k}_controller_snapshot")()["receipt_age_ns"] < 2**62 for k in _SIDES)

    def sample(self):
        s = {}
        for n in self.getters:
            try:
                s[n] = _jsonable(getattr(self.xrt, n)())
            except Exception as e:  # noqa: BLE001
                s[n] = f"<err {e}>"
        for side in _SIDES:  # snapshot dict: pose, axis, trigger, grip, binding_generation, receipt_age_ns, ...
            s[side] = s[f"get_{side}_controller_snapshot"]
        return s

    def close(self):
        try:
            self.xrt.close()
        except Exception:  # noqa: BLE001
            pass


class IsaacBackend:
    def __init__(self, timeout):
        try:
            from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient
        except ImportError as e:
            _die(f"isaacteleop not importable ({e}); need an env with isaacteleop[cloudxr]")
        self.client = IsaacTeleopClient()
        t0 = time.monotonic()
        while True:
            if self.client._deviceio_session is None:  # noqa: SLF001 (retry: session may come up later)
                self.client.start_streaming()
            raw = self.client._get_tracker_data()  # noqa: SLF001
            if raw and (raw.get("left_controller") is not None or raw.get("right_controller") is not None):
                break
            if time.monotonic() - t0 > timeout:
                _die(f"Isaac: no controller snapshot after {timeout}s (no XR session / headset?)")
            time.sleep(0.5)
        self.meta = {"note": "every public field of left/right ControllerSnapshot, plus head is_valid"}

    def sample(self):
        raw = self.client._get_tracker_data()  # noqa: SLF001
        if raw is None:
            raise RuntimeError("DeviceIO session closed")
        s = {"head": _jsonable(raw.get("head"))}
        for side in _SIDES:
            snap = raw.get(f"{side}_controller")
            d = {"snapshot": _jsonable(snap)}  # all fields, incl. aim_pose/grip_pose is_valid
            if snap is not None:
                inp = snap.inputs
                d["stick"] = [float(inp.thumbstick_x), float(inp.thumbstick_y)]
                d["trigger"] = float(inp.trigger_value)
                d["grip"] = float(inp.squeeze_value)
            s[side] = d
        return s

    def close(self):
        self.client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-source", choices=["xrt", "isaac-teleop"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    a = ap.parse_args()

    backend = (XrtBackend if a.input_source == "xrt" else IsaacBackend)(a.connect_timeout)
    f = open(a.out, "w")
    lock = threading.Lock()
    stopping = False

    def emit(rec):
        with lock:
            if stopping:
                return
            f.write(json.dumps(rec) + "\n")
            f.flush()

    def read_markers():
        for line in sys.stdin:
            if line.strip():
                emit({"marker": line.strip(), "t_mono": time.monotonic()})
                print(f"marker: {line.strip()}", flush=True)

    emit({"meta": {"backend": a.input_source, **backend.meta}, "t_mono": time.monotonic()})
    threading.Thread(target=read_markers, daemon=True).start()
    print("recording; type phase names + Enter, Ctrl-C to stop", flush=True)
    n = 0
    try:
        while True:
            t = time.monotonic()
            emit({"t_mono": t, **backend.sample()})
            n += 1
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            stopping = True
            f.close()
        backend.close()
        print(f"wrote {n} samples to {a.out}")


if __name__ == "__main__":
    main()
