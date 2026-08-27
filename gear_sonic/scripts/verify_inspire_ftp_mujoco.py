"""Headless structural and per-motor verification for the Inspire FTP model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.mujoco_sim.inspire_ftp_hand import (  # noqa: E402
    InspireFtpMujocoPlant,
)
from gear_sonic.utils.teleop.inspire_ftp import (  # noqa: E402
    CLOSED_RADIANS,
    MOTOR_NAMES,
    OPEN,
)

DEFAULT_SCENE = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml"


def publish_scripted_commands(
    *, duration_s: float = 15.0, open_hold_s: float = 5.0, port: int = 5556
) -> dict[str, Any]:
    """Publish a bounded open-hold then smooth close/open cycle locally."""

    if duration_s <= open_hold_s or open_hold_s < 0.0:
        raise ValueError("duration_s must be greater than non-negative open_hold_s")
    import zmq

    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
    )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://127.0.0.1:{port}")
    frame_count = 0
    last_command = OPEN.copy()
    try:
        time.sleep(1.0)
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < duration_s:
            if elapsed < open_hold_s:
                last_command = OPEN.copy()
            else:
                phase = (elapsed - open_hold_s) / (duration_s - open_hold_s)
                value = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))
                last_command = np.full(6, value, dtype=np.float64)
            if frame_count % 50 == 0:
                socket.send(build_command_message(start=True, stop=False, planner=True))
            socket.send(
                build_planner_message(
                    mode=0,
                    movement=[0.0, 0.0, 0.0],
                    facing=[1.0, 0.0, 0.0],
                    upper_body_position=[0.0] * 17,
                    left_hand_position=last_command,
                    right_hand_position=last_command,
                )
            )
            frame_count += 1
            time.sleep(0.02)
        socket.send(build_command_message(start=False, stop=True, planner=True))
    finally:
        socket.close()
        context.term()
    return {
        "passed": True,
        "frames": frame_count,
        "duration_s": duration_s,
        "open_hold_s": open_hold_s,
        "final_hand": last_command.tolist(),
    }


def _constraint_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    maximum = 0.0
    for equality_index in range(model.neq):
        dependent_id = int(model.eq_obj1id[equality_index])
        driver_id = int(model.eq_obj2id[equality_index])
        dependent = data.qpos[int(model.jnt_qposadr[dependent_id])]
        driver = data.qpos[int(model.jnt_qposadr[driver_id])]
        coefficients = model.eq_data[equality_index, :5]
        expected = sum(coefficient * driver**power for power, coefficient in enumerate(coefficients))
        maximum = max(maximum, abs(float(dependent - expected)))
    return maximum


def _hand_joint_ids(model: mujoco.MjModel) -> np.ndarray:
    tokens = ("thumb", "index", "middle", "ring", "little")
    return np.array(
        [
            joint_id
            for joint_id in range(model.njnt)
            if model.joint(joint_id).name.startswith(("left_", "right_"))
            and any(token in model.joint(joint_id).name for token in tokens)
        ],
        dtype=np.int64,
    )


def _joint_limit_error(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: np.ndarray) -> float:
    maximum = 0.0
    for joint_id in joint_ids:
        if not model.jnt_limited[joint_id]:
            continue
        position = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        lower, upper = model.jnt_range[joint_id]
        maximum = max(maximum, float(lower - position), float(position - upper))
    return max(maximum, 0.0)


def _finite(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    del model
    return all(
        np.all(np.isfinite(values))
        for values in (
            data.qpos,
            data.qvel,
            data.qacc,
            data.ctrl,
            data.actuator_force,
        )
    )


def run_contact_cycle(
    scene_path: str | Path = DEFAULT_SCENE,
    *,
    duration_s: float = 15.0,
    open_hold_s: float = 5.0,
) -> dict[str, Any]:
    """Run the scripted bilateral hand cycle with normal contacts enabled."""

    if duration_s <= open_hold_s or open_hold_s < 0.0:
        raise ValueError("duration_s must be greater than non-negative open_hold_s")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.005
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    plant = InspireFtpMujocoPlant.resolve(model, data)
    hand_joint_ids = _hand_joint_ids(model)
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    finite = True
    max_constraint_error = 0.0
    max_limit_error = 0.0
    max_projection_error = 0.0
    max_contacts = 0

    for step in range(steps):
        elapsed = step * model.opt.timestep
        if elapsed < open_hold_s:
            command = OPEN
        else:
            phase = (elapsed - open_hold_s) / (duration_s - open_hold_s)
            value = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))
            command = np.full(6, value, dtype=np.float64)
        plant.write_targets(command, command)
        mujoco.mj_step(model, data)
        finite = finite and _finite(model, data)
        max_constraint_error = max(max_constraint_error, _constraint_error(model, data))
        max_limit_error = max(max_limit_error, _joint_limit_error(model, data, hand_joint_ids))
        plant.read_normalized_state()
        max_projection_error = max(max_projection_error, plant.last_measurement_limit_error_rad)
        max_contacts = max(max_contacts, data.ncon)

    final_left, final_right = plant.read_normalized_state()
    final_open_error = float(max(np.max(np.abs(final_left - OPEN)), np.max(np.abs(final_right - OPEN))))
    passed = (
        finite
        and max_constraint_error < 2e-3
        and max_limit_error <= 1e-6
        and max_projection_error <= 1e-3
        and max_contacts > 0
        and final_open_error < 0.03
    )
    return {
        "passed": bool(passed),
        "contacts_enabled": True,
        "duration_s": float(duration_s),
        "open_hold_s": float(open_hold_s),
        "steps": steps,
        "finite": bool(finite),
        "max_contacts": max_contacts,
        "max_constraint_error_rad": max_constraint_error,
        "max_joint_limit_error_rad": max_limit_error,
        "max_measurement_projection_error_rad": max_projection_error,
        "final_open_error_normalized": final_open_error,
    }


def run_motor_sweep(
    scene_path: str | Path = DEFAULT_SCENE,
    *,
    duration_s: float = 2.0,
) -> dict[str, Any]:
    """Close each motor alone and return a JSON-serializable verdict report."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.005
    model.opt.gravity[:] = 0.0
    # This sweep isolates actuator naming, limits, and mimic kinematics. The
    # bounded SONIC regression below it is responsible for contact behavior.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    hand_joint_ids = _hand_joint_ids(model)
    results = []
    global_constraint_error = 0.0
    global_limit_error = 0.0

    for side in ("left", "right"):
        for motor_index, motor_name in enumerate(MOTOR_NAMES):
            data = mujoco.MjData(model)
            plant = InspireFtpMujocoPlant.resolve(model, data)
            plant.write_targets(OPEN, OPEN)
            finite = True
            for _ in range(steps):
                mujoco.mj_step(model, data)
                finite = finite and _finite(model, data)

            baseline = {
                hand_side: data.qpos[plant.qpos_addresses[hand_side]].copy() for hand_side in ("left", "right")
            }
            command = {"left": OPEN.copy(), "right": OPEN.copy()}
            command[side][motor_index] = 0.0
            plant.write_targets(command["left"], command["right"])

            for _ in range(steps):
                mujoco.mj_step(model, data)
                finite = finite and _finite(model, data)

            maximum_constraint_error = _constraint_error(model, data)
            maximum_limit_error = _joint_limit_error(model, data, hand_joint_ids)

            final = {
                hand_side: data.qpos[plant.qpos_addresses[hand_side]].copy() for hand_side in ("left", "right")
            }
            target = CLOSED_RADIANS[motor_index]
            target_progress = float(
                (final[side][motor_index] - baseline[side][motor_index]) / (target - baseline[side][motor_index])
            )
            unrelated = []
            for hand_side in ("left", "right"):
                for other_index in range(6):
                    if hand_side == side and other_index == motor_index:
                        continue
                    unrelated.append(abs(float(final[hand_side][other_index] - baseline[hand_side][other_index])))
            max_unrelated = max(unrelated, default=0.0)
            passed = (
                finite
                and target_progress > 0.5
                and max_unrelated < 1e-3
                and maximum_constraint_error < 2e-3
                and maximum_limit_error <= 1e-6
            )
            result = {
                "side": side,
                "motor_index": motor_index,
                "motor": motor_name,
                "passed": bool(passed),
                "finite": bool(finite),
                "target_progress": target_progress,
                "final_active_rad": float(final[side][motor_index]),
                "target_active_rad": float(target),
                "max_unrelated_active_rad": max_unrelated,
                "max_constraint_error_rad": maximum_constraint_error,
                "max_joint_limit_error_rad": maximum_limit_error,
            }
            results.append(result)
            global_constraint_error = max(global_constraint_error, maximum_constraint_error)
            global_limit_error = max(global_limit_error, maximum_limit_error)

    return {
        "passed": all(result["passed"] for result in results),
        "model": {"njnt": model.njnt, "nu": model.nu, "neq": model.neq},
        "duration_s": float(duration_s),
        "steps_per_phase": steps,
        "contacts_disabled": True,
        "max_constraint_error_rad": global_constraint_error,
        "max_joint_limit_error_rad": global_limit_error,
        "motors": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor-sweep", action="store_true")
    parser.add_argument("--contact-cycle", action="store_true")
    parser.add_argument("--scripted-publisher", action="store_true")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--open-hold", type=float, default=5.0)
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.scripted_publisher:
        report = publish_scripted_commands(
            duration_s=args.duration,
            open_hold_s=args.open_hold,
            port=args.zmq_port,
        )
    elif args.contact_cycle:
        report = run_contact_cycle(duration_s=args.duration, open_hold_s=args.open_hold)
    elif args.motor_sweep:
        report = run_motor_sweep(duration_s=args.duration)
    else:
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        report = {
            "passed": (model.njnt, model.nu, model.neq) == (54, 41, 12),
            "model": {"njnt": model.njnt, "nu": model.nu, "neq": model.neq},
            "motors": [],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
