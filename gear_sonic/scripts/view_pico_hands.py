#!/usr/bin/env python3
"""Read-only local viewer of PICO input and recorded Dex3 mapping/feedback.

Uses completed --hand-log-dir shards, not a second XR SDK connection. Robot
geometry is forward kinematics of recorded q; retargeting is never rerun.
"""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.utils.teleop.pico_hand_log import validate_capture
from gear_sonic.utils.teleop.pico_hand_retargeting import _ROOT, _output_names, _verify_assets, _wrist_frame
from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, TrackingState


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class CaptureViewer:
    def __init__(self, directory):
        from dex_retargeting.robot_wrapper import RobotWrapper

        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError("capture directory must be a directory")
        _verify_assets(_ROOT)
        self.robots = []
        self.orders = []
        self.chains = []
        for side in ("left", "right"):
            robot = RobotWrapper(str(_ROOT / f"unitree_hand/unitree_dex3_{side}.urdf"))
            self.robots.append(robot)
            self.orders.append([robot.dof_joint_names.index(name) for name in _output_names("dex3", side)])
            self.chains.append(
                [
                    [robot.get_link_index(f"{side}_hand_thumb_{j}_link") for j in range(3)]
                    + [robot.get_link_index("thumb_tip")],
                    *[
                        [robot.get_link_index(f"{side}_hand_{digit}_{j}_link") for j in range(2)]
                        + [robot.get_link_index(f"{digit}_tip")]
                        for digit in ("index", "middle")
                    ],
                ]
            )

    def files(self):
        return sorted(p.name for p in self.directory.glob("capture_*.npz") if p.is_file())

    def geometry(self, q, side):
        if q.shape != (7,) or not np.isfinite(q).all():
            return None
        robot = self.robots[side]
        full = np.zeros(robot.dof)
        full[self.orders[side]] = q
        robot.compute_forward_kinematics(full)
        return [[robot.get_link_pose(link)[:3, 3].copy() for link in chain] for chain in self.chains[side]]

    @lru_cache(maxsize=2)
    def capture(self, name):
        path = self.directory / name
        if path.name != name or path.resolve().parent != self.directory or name not in self.files():
            raise ValueError("unknown capture filename")
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        if a.get("profile", np.array([""]))[0] != "dex3":
            raise ValueError("This viewer currently supports Dex3 captures only")
        count = validate_capture(a, "dex3")
        for key in ("captured_target", "captured_command", "captured_state", "captured_reason"):
            if key not in a:
                raise ValueError(f"Capture has no {key}; actual recorded mapping is required")
        frames = []
        last_change = [None, None]
        for i in range(count):
            frame = {
                "time_s": (int(a["tick_ns"][i]) - int(a["tick_ns"][0])) * 1e-9,
                "active": a["active"][i],
                "enabled": bool(a["enabled"][i]) if "enabled" in a else None,
                "sent": a["hand_sent"][i] if "hand_sent" in a else [None, None],
                "raw_pose": a["poses"][i],
                # Preserve all uint64 flag bits in JavaScript.
                "flags": [list(map(str, side)) for side in a["flags"][i]],
                "target": a["captured_target"][i],
                "command": a["captured_command"][i],
                "measured": a["measured"][i],
                "pico": [],
                "state": [],
                "reason": [],
                "source_age_ms": [],
                "robot": {},
            }
            for side, name_side in enumerate(("left", "right")):
                try:
                    canonical = _wrist_frame(a["poses"][i, side, 1:, :3], name_side)
                    # The frame only uses wrist/index/middle anchors. Replacing
                    # unused little-tip slot 24 lets the same transform place palm.
                    probe = a["poses"][i, side, 1:, :3].copy()
                    probe[24] = a["poses"][i, side, 0, :3]
                    palm = _wrist_frame(probe, name_side)[24]
                    frame["pico"].append(np.vstack((palm, canonical)))
                except ValueError:
                    frame["pico"].append(None)
                for key, enum in (("state", TrackingState), ("reason", HandReason)):
                    try:
                        label = enum(int(a[f"captured_{key}"][i, side])).name
                    except ValueError:
                        label = "UNAVAILABLE"
                    frame[key].append(label)
                if i and a["source_timestamp_ns"][i, side] != a["source_timestamp_ns"][i - 1, side]:
                    last_change[side] = int(a["tick_ns"][i])
                frame["source_age_ms"].append(
                    None if last_change[side] is None else (int(a["tick_ns"][i]) - last_change[side]) * 1e-6
                )
            for key in ("target", "command", "measured"):
                frame["robot"][key] = [self.geometry(frame[key][side], side) for side in range(2)]
            frames.append(frame)
        return json.dumps(
            finite_json({"name": name, "profile": "dex3", "frames": frames}), allow_nan=False
        ).encode()


def handler_for(viewer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    content = Path(__file__).with_name("pico_hand_viewer.html").read_bytes()
                    kind = "text/html; charset=utf-8"
                elif url.path == "/api/captures":
                    content = json.dumps({"files": viewer.files()}).encode()
                    kind = "application/json"
                elif url.path == "/api/capture":
                    content = viewer.capture(parse_qs(url.query).get("name", [""])[0])
                    kind = "application/json"
                else:
                    self.send_error(404)
                    return
            except (ValueError, OSError, KeyError) as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--port", default=8766, type=int)
    args = parser.parse_args()
    viewer = CaptureViewer(args.capture_dir)
    with HTTPServer(("127.0.0.1", args.port), handler_for(viewer)) as server:
        print(f"Hand viewer: http://127.0.0.1:{args.port}", flush=True)
        print(f"Reading {viewer.directory}; completed shards only, normally ~5 s behind live.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
