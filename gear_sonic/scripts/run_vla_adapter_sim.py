"""Bounded headless simulation evidence runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(x) for x in value)
    if isinstance(value, bool) or value is None:
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)


def _command_target(command: Any) -> list[float] | None:
    motors = getattr(command, "motor_cmd", None)
    if motors is None:
        return None
    return [float(getattr(motor, "q")) for motor in motors]


def _snapshot(obs: dict, bridge: Any, elapsed: float, sim_time: float, env: Any = None) -> dict:
    result = {"elapsed": elapsed, "sim_time": sim_time, "monotonic_ns": time.monotonic_ns()}
    if env is not None:
        data = getattr(env, "mj_data", None)
        if data is not None and hasattr(data, "time"):
            result["physics_time"] = float(data.time)
        band = getattr(env, "elastic_band", None)
        if band is not None and hasattr(band, "enable"):
            result["band_enabled"] = bool(band.enable)
    required = (
        "body_q",
        "body_dq",
        "left_hand_q",
        "left_hand_dq",
        "right_hand_q",
        "right_hand_dq",
        "floating_base_pose",
    )
    for key in required:
        if key not in obs:
            raise ValueError(f"missing observation field: {key}")
        value = obs[key]
        result[key] = value.tolist() if hasattr(value, "tolist") else list(value)
    if "floating_base_vel" in obs:
        result["floating_base_vel"] = list(obs["floating_base_vel"])
    flags = {}
    for side, received_attr, command_attr, lock_name in (
        ("body", "low_cmd_received", "low_cmd", "low_cmd_lock"),
        ("left", "left_hand_cmd_received", "left_hand_cmd", "left_hand_cmd_lock"),
        ("right", "right_hand_cmd_received", "right_hand_cmd", "right_hand_cmd_lock"),
    ):
        lock = getattr(bridge, lock_name, None)
        if lock is None:
            flags[side] = bool(getattr(bridge, received_attr, False))
            command = getattr(bridge, command_attr, None)
        else:
            with lock:
                flags[side] = bool(getattr(bridge, received_attr, False))
                command = getattr(bridge, command_attr, None)
                target = _command_target(command) if command is not None else None
        if lock is None:
            target = _command_target(command) if command is not None else None
        if target is not None:
            result[f"{side}_command_target"] = target
    result["command_received"] = flags
    result["finite"] = _finite_tree(result)
    return result


def _plant_diagnostics(env: Any) -> dict:
    if not hasattr(env, "mj_model"):
        return {}
    model, data = env.mj_model, env.mj_data
    floor = model.geom("floor").id
    feet = {side: model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
    contacts = {side: False for side in feet}
    for contact in data.contact[: data.ncon]:
        pair = (int(contact.geom1), int(contact.geom2))
        if floor not in pair or contact.dist > 1e-4:
            continue
        body = int(model.geom_bodyid[pair[1] if pair[0] == floor else pair[0]])
        while body:
            for side, foot in feet.items():
                contacts[side] |= body == foot
            body = int(model.body_parentid[body])
    waist = {}
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        joint = model.joint(name).id
        actuator = next(i for i, pair in enumerate(model.actuator_trnid) if int(pair[0]) == joint)
        dof = int(model.jnt_dofadr[joint])
        waist[name] = {
            "ctrl": float(data.ctrl[actuator]),
            "actuator_force": float(data.actuator_force[actuator]),
            "constraint_force": float(data.qfrc_constraint[dof]),
        }
    return {"foot_contacts": contacts, "waist_plant": waist}


def run_simulation(
    sim: Any,
    output_dir: Path,
    duration_seconds: float,
    release_file: Path | None = None,
    startup_file: Path | None = None,
    wall_clock=time.monotonic,
    sleep_fn=time.sleep,
) -> dict:
    if release_file is not None and startup_file is not None:
        raise ValueError("select either legacy release or unhang/reset startup")
    if (
        not isinstance(duration_seconds, (int, float))
        or isinstance(duration_seconds, bool)
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "observations.jsonl"
    started = wall_clock()
    sim_time = 0.0
    steps = 0
    released = False
    error = None
    release_time = None
    startup_reset_completed = False
    fall_detected = False
    received = {"body": False, "left": False, "right": False}
    try:
        env = sim.sim_env
        if hasattr(env, "mj_model"):
            groups = {}
            for key, indices in (
                ("body", env.body_joint_index),
                ("left", env.left_hand_index),
                ("right", env.right_hand_index),
            ):
                groups[key] = [
                    {"name": env.mj_model.joint(int(i)).name, "range": env.mj_model.jnt_range[int(i)].tolist()}
                    for i in indices
                ]
            (output_dir / "joint_contract.json").write_text(json.dumps(groups, indent=2) + "\n")
        sim_dt = float(sim.sim_dt)
        if not math.isfinite(sim_dt) or sim_dt <= 0:
            raise ValueError("sim_dt must be finite and positive")
        image_dt = float(getattr(sim, "image_dt", 0.0))
        next_image = image_dt
        bridge = getattr(sim, "unitree_bridge", None)
        with records_path.open("w", encoding="utf-8") as stream:
            next_record = 0.0
            deadline = started + duration_seconds
            while sim_time < duration_seconds and wall_clock() < deadline:
                step_started = wall_clock()
                if (
                    not startup_reset_completed
                    and startup_file is not None
                    and startup_file.exists()
                    and bool(getattr(bridge, "low_cmd_received", False))
                ):
                    band = getattr(env, "elastic_band", None)
                    if band is None or not band.enable:
                        raise ValueError("unhang/reset startup must begin with the support band enabled")
                    for key in ("9", "backspace"):
                        sim.handle_keyboard_button(key)
                        stream.write(json.dumps({
                            "event": "keyboard", "key": key, "sim_time": sim_time,
                            "monotonic_ns": time.monotonic_ns(),
                        }) + "\n")
                    if band.enable:
                        raise ValueError("keyboard startup did not leave the support band disabled")
                    released = True
                    release_time = sim_time
                    startup_reset_completed = True
                    reset_state = _snapshot(env.prepare_obs(), bridge, wall_clock() - started, sim_time, env)
                    reset_state.update(keyboard_sequence=["9", "backspace"], elastic_band_enabled=False)
                    temporary = output_dir / "startup_reset.tmp"
                    temporary.write_text(json.dumps(reset_state, allow_nan=False) + "\n")
                    temporary.replace(output_dir / "startup_reset.json")
                    stream.flush()
                sim.sim_env.sim_step()
                sim_time += sim_dt
                steps += 1
                if image_dt > 0 and sim_time >= next_image:
                    sim.sim_env.update_render_caches()
                    next_image += image_dt
                # The minimal simulator's get_privileged_obs hook returns {}.
                # Read the measured joints used by its DDS bridge directly.
                obs = sim.sim_env.prepare_obs()
                if sim_time + 1e-12 >= next_record:
                    record = _snapshot(obs, bridge, wall_clock() - started, sim_time, env)
                    record.update(_plant_diagnostics(env))
                    record["finite"] = _finite_tree(record)
                    if not record["finite"]:
                        raise ValueError("nonfinite observation")
                    if steps == 1:
                        (output_dir / "initial_state.json").write_text(
                            json.dumps(record, allow_nan=False) + "\n"
                        )
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    next_record += 0.02
                    received = {key: received[key] or record["command_received"][key] for key in received}
                low_received = bool(getattr(bridge, "low_cmd_received", False))
                if not released and release_file is not None and release_file.exists() and low_received:
                    band = getattr(sim.sim_env, "elastic_band", None)
                    if band is None:
                        raise ValueError("elastic band unavailable")
                    band.enable = False
                    released = True
                    release_time = sim_time
                    stream.write(
                        json.dumps({"event": "release_elastic_band", "sim_time": sim_time}, sort_keys=True) + "\n"
                    )
                if bool(getattr(env, "fall", False)) and (released or startup_reset_completed):
                    fall_detected = True
                    stream.write(
                        json.dumps(
                            {
                                "event": "fall_detected",
                                "monotonic_ns": time.monotonic_ns(),
                                "sim_time": sim_time,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    stream.flush()
                    raise RuntimeError("fall detected after support release/reset")
                sleep_fn(max(0.0, sim_dt - (wall_clock() - step_started)))
    except Exception as exc:
        error = str(exc)
    finally:
        try:
            sim.close()
        except Exception as exc:
            error = error or str(exc)
    summary = {
        "steps": steps,
        "duration_seconds": sim_time,
        "wall_duration_seconds": wall_clock() - started,
        "released": released,
        "startup_reset_completed": startup_reset_completed,
        "release_time": release_time,
        "body_command_received": received["body"],
        "left_hand_command_received": received["left"],
        "right_hand_command_received": received["right"],
        "finite": error is None,
        "fall_detected": fall_detected,
        "mode": "released" if released else "supported",
        "standing_verified": False,
        "error": error,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-profile", choices=("dex3", "inspire_ftp"), default="dex3")
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--release-file", type=Path)
    parser.add_argument("--startup-file", type=Path)
    parser.add_argument("--offscreen", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or args.output_dir.exists():
        parser.error("duration must be finite and positive; output directory must not exist")
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile=args.hand_profile).load_wbc_yaml()
    sim = BaseSimulator(config=config, onscreen=False, offscreen=args.offscreen, env_name="default")
    summary = run_simulation(
        sim, args.output_dir, args.duration_seconds, args.release_file, args.startup_file
    )
    return 0 if summary["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
