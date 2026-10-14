import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Run portable baseline/patched SONIC MuJoCo evaluation")
    p.add_argument("config", type=Path)
    p.add_argument("--variant", choices=("baseline", "patched"), required=True)
    p.add_argument("--scenario", choices=("nominal", "stress"), required=True)
    a = p.parse_args()
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError("refusing to launch children outside the private loopback-only namespace")
    config_path = a.config.resolve()
    c = json.loads(config_path.read_text())
    required = (
        "source_root",
        "baseline_executable",
        "patched_executable",
        "sim_python",
        "models_dir",
        "reference_dir",
        "robot_scene",
        "pose_fixture",
        "output_root",
        "cuda_visible_devices",
        "library_path",
    )
    missing = [k for k in required if not c.get(k)]
    if missing:
        raise ValueError("missing config keys: " + ", ".join(missing))
    path_keys = [key for key in required if key not in ("library_path", "cuda_visible_devices")]
    relative = [key for key in path_keys if not Path(c[key]).is_absolute()]
    if relative:
        raise ValueError("filesystem config paths must be absolute: " + ", ".join(relative))
    out = Path(c["output_root"]) / (a.variant + "_" + a.scenario)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(c["source_root"])
    models = Path(c["models_dir"])
    exe = Path(c[a.variant + "_executable"])
    sys.path.insert(0, str(source))
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        pack_pose_message,
        build_planner_message,
        build_command_message,
    )

    with np.load(c["pose_fixture"]) as f:
        fields = {
            k: np.repeat(f[k][-1:], 5, axis=0)
            for k in ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel")
        }
        for key in ("vr_position", "vr_orientation"):
            fields[key] = f[key].copy()
    fields["joint_vel"][:] = 0
    import zmq

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    pub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    pub.bind("tcp://127.0.0.1:15556")
    sub.connect("tcp://127.0.0.1:15557")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(source),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=str(c["cuda_visible_devices"]),
        LD_LIBRARY_PATH=str(c["library_path"]) + ":" + env.get("LD_LIBRARY_PATH", ""),
    )
    env.pop("CYCLONEDDS_URI", None)
    procs = []
    logs = []
    events = []

    def event(kind, **kw):
        events.append(dict(wall=time.monotonic(), kind=kind, **kw))
        (out / "events.json").write_text(json.dumps(events, indent=2))

    def launch(argv, name):
        h = (out / name).open("w")
        logs.append(h)
        q = subprocess.Popen(argv, cwd=out, env=env, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(q)
        return q

    def wait_for(test, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if test():
                return
            if any(q.poll() is not None for q in procs):
                raise RuntimeError("child exited during initialization")
            time.sleep(0.05)
        raise TimeoutError("initialization deadline")

    try:
        launch(
            [c["sim_python"], str(Path(__file__).with_name("simulator.py")), str(out), str(config_path)],
            "sim.log",
        )
        wait_for(lambda: (out / "sim.ready").exists(), 30)
        args = [
            str(exe),
            "lo",
            str(models / "model_decoder.onnx"),
            c["reference_dir"],
            "--obs-config",
            str(models / "observation_config.yaml"),
            "--encoder-file",
            str(models / "model_encoder.onnx"),
            "--planner-file",
            str(models / "V2" / "planner_sonic.onnx"),
            "--input-type",
            "zmq_manager",
            "--output-type",
            "all",
            "--zmq-host",
            "127.0.0.1",
            "--zmq-port",
            "15556",
            "--zmq-out-port",
            "15557",
            "--disable-crc-check",
        ]
        args += ["--enable-csv-logs", "--logs-dir", str(out / "csv"), "--live-pose-playback"]
        (out / "invocation.json").write_text(
            json.dumps(
                dict(
                    argv=args,
                    fixture=c["pose_fixture"],
                    gpu=c["cuda_visible_devices"],
                    network="private loopback-only namespace",
                ),
                indent=2,
            )
        )
        launch(args, "sonic.log")
        wait_for(lambda: "Init Done" in (out / "sonic.log").read_text(errors="replace"), 60)
        time.sleep(0.5)
        event("start")
        origin = time.monotonic()
        prev = None
        seq = 0
        released = False
        requested = None
        warm_until = 0
        facing = [1.0, 0.0, 0.0]
        duration = 38 if a.scenario == "nominal" else 36
        debug = (out / "debug.bin").open("wb")
        logs.append(debug)
        schedule = (
            [
                (0, "planner"),
                (8, "pose"),
                (13, "planner"),
                (18, "pose"),
                (23, "planner"),
                (28, "pose"),
                (33, "planner"),
            ]
            if a.scenario == "nominal"
            else [(0, "planner"), (12, "pose"), (17, "planner"), (26, "pose"), (31, "planner")]
        )
        while time.monotonic() - origin < duration:
            t = time.monotonic()
            elapsed = t - origin
            desired = next(m for at, m in reversed(schedule) if elapsed >= at)
            if desired != requested:
                requested = desired
                event("request", mode=desired)
                warm_until = t + 0.1 if a.scenario == "stress" and desired == "pose" else 0
                seq = 0 if desired == "pose" else seq
            if elapsed >= 3 and not released:
                (out / "release").touch()
                event("release_band")
                released = True
            if a.scenario == "stress":
                nf = (
                    [float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0]
                    if (6 <= elapsed < 17 or 20 <= elapsed < 31)
                    else [1.0, 0.0, 0.0]
                )
                if nf != facing:
                    facing = nf
                    event("facing", value=facing)
            mode = (
                "planner"
                if desired == "pose" and a.scenario == "stress" and t < warm_until and a.variant == "patched"
                else desired
            )
            if mode == "planner":
                pub.send(build_planner_message(0, [0.0, 0.0, 0.0], facing))
            elif not (a.scenario == "stress" and t < warm_until):
                fields["frame_index"] = np.arange(seq, seq + 5, dtype=np.int64)
                seq += 1
                pub.send(pack_pose_message(fields, "pose", version=3))
            if mode != prev:
                pub.send(build_command_message(start=True, stop=False, planner=mode == "planner"))
                event("mode", mode=mode)
                prev = mode
            while True:
                try:
                    raw = sub.recv(zmq.NOBLOCK)
                    debug.write(len(raw).to_bytes(4, "little") + raw)
                except zmq.Again:
                    break
            rows = (out / "physics.csv").read_text().splitlines()
            if len(rows) > 1 and elapsed > 5:
                last = rows[-1].split(",")
                if len(last) >= 6 and (float(last[2]) < 0.45 or float(last[3]) > 60):
                    event("fall", z=float(last[2]), tilt=float(last[3]))
                    raise RuntimeError("fall threshold exceeded")
            if any(q.poll() is not None for q in procs):
                raise RuntimeError("child exited during test")
            time.sleep(max(0, 0.02 - (time.monotonic() - t)))
        event("end")
    except Exception as exc:
        event("error", message=str(exc))
        raise
    finally:
        for _ in range(3):
            pub.send(build_command_message(start=False, stop=True, planner=True))
            time.sleep(0.03)
        (out / "stop").touch()
        for q in reversed(procs):
            if q.poll() is None:
                os.killpg(q.pid, signal.SIGTERM)
                try:
                    q.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(q.pid, signal.SIGKILL)
                    q.wait()
        for h in logs:
            h.close()
        pub.close()
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
