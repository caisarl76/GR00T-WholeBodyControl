#!/usr/bin/env python3
"""Verify Dex3 XR thumb-index pinch geometry against the G1 MuJoCo model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

DEX3_API_ORDER = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
SIDE_ORDER = {"left": DEX3_API_ORDER, "right": DEX3_API_ORDER}
DEPLOY_HAND_HARD_MIN = {
    "left": np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75]),
    "right": np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0]),
}
DEPLOY_HAND_HARD_MAX = {
    "left": np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0]),
    "right": np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75]),
}
DISTANCE_TOLERANCE_M = 1e-6
CONTACT_DISTANCE_MIN_M = -0.00625
CONTACT_DISTANCE_MAX_M = -0.00525
MAX_COMMAND_STEP_RAD = 0.25


def validate_pinch_targets(targets: dict[str, np.ndarray]) -> None:
    for side in ("left", "right"):
        target = np.asarray(targets[side], dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid {side} pinch target")
        if not np.allclose(target[3:5], 0.0, rtol=0, atol=1e-12):
            raise ValueError(f"{side} middle joints must remain zero")
        if np.any(target < DEPLOY_HAND_HARD_MIN[side] - 1e-12) or np.any(
            target > DEPLOY_HAND_HARD_MAX[side] + 1e-12
        ):
            raise ValueError(f"{side} target exceeds deploy hard limits")


def load_xr_targets(xr_root: Path) -> dict[str, np.ndarray]:
    module_path = xr_root / "teleop" / "utils" / "dex3_controller_pinch.py"
    spec = importlib.util.spec_from_file_location("xr_dex3_controller_pinch_for_verification", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load XR pinch helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = {
        "left": np.asarray(module.DEX3_LEFT_PINCH, dtype=float),
        "right": np.asarray(module.DEX3_RIGHT_PINCH, dtype=float),
    }
    for side, target in targets.items():
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid {side} target: {target}")
    return targets


def _hand_ids(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, int, int]:
    qpos_addresses = []
    for suffix in SIDE_ORDER[side]:
        joint_name = f"{side}_hand_{suffix}_joint"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing joint: {joint_name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    geom_ids = []
    for suffix in ("thumb_2", "index_1"):
        body_name = f"{side}_hand_{suffix}_link"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        candidates = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_contype != 0))
        if candidates.size != 1:
            raise ValueError(f"expected one collision geom on {body_name}, got {candidates.tolist()}")
        geom_ids.append(int(candidates[0]))
    return np.asarray(qpos_addresses, dtype=int), geom_ids[0], geom_ids[1]


def _geometry_trace(
    model: mujoco.MjModel,
    side: str,
    states: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    qpos_addresses, thumb_geom, index_geom = _hand_ids(model, side)
    distances = []
    contact_counts = []
    for state in states:
        data.qpos[:] = model.qpos0
        data.qpos[qpos_addresses] = state
        mujoco.mj_forward(model, data)
        distance = mujoco.mj_geomDistance(
            model,
            data,
            thumb_geom,
            index_geom,
            1.0,
            np.zeros(6, dtype=float),
        )
        distances.append(float(distance))
        contact_counts.append(
            sum(
                1
                for contact in data.contact
                if {int(contact.geom1), int(contact.geom2)} == {thumb_geom, index_geom}
            )
        )
    return np.asarray(distances, dtype=float), np.asarray(contact_counts, dtype=int)


def idealized_slew_states(target: np.ndarray, max_step: float = MAX_COMMAND_STEP_RAD) -> list[np.ndarray]:
    current = np.zeros_like(target)
    states = []
    for _ in range(100):
        if np.allclose(current, target, rtol=0, atol=1e-12):
            return states
        current = current + np.clip(target - current, -max_step, max_step)
        states.append(current.copy())
    raise AssertionError("idealized slew did not converge within 100 recurrence steps")


def _assert_trace(
    side: str,
    name: str,
    distances: np.ndarray,
    contact_counts: np.ndarray,
    *,
    expected_steps: int,
) -> None:
    if distances.shape != (expected_steps,):
        raise AssertionError(f"{side} {name}: expected {expected_steps} samples, got {distances.size}")
    if contact_counts.shape != (expected_steps,):
        raise AssertionError(
            f"{side} {name}: expected {expected_steps} contact samples, got {contact_counts.size}"
        )
    if not np.all(np.isfinite(distances)):
        raise AssertionError(f"{side} {name}: non-finite trace {distances.tolist()}")
    if not np.all(distances[:-1] > 0.0):
        raise AssertionError(f"{side} {name}: pre-final contact {distances.tolist()}")
    if np.any(np.diff(distances) > DISTANCE_TOLERANCE_M):
        raise AssertionError(f"{side} {name}: non-monotonic {distances.tolist()}")
    if not CONTACT_DISTANCE_MIN_M <= distances[-1] <= CONTACT_DISTANCE_MAX_M:
        raise AssertionError(
            f"{side} {name}: final distance {distances[-1]} is outside "
            f"[{CONTACT_DISTANCE_MIN_M}, {CONTACT_DISTANCE_MAX_M}]"
        )
    if contact_counts[-1] < 1:
        raise AssertionError(f"{side} {name}: no final distal thumb-index contact")


def verify_pinch(model_path: Path, targets: dict[str, np.ndarray]) -> dict:
    validate_pinch_targets(targets)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    report = {"mujoco_version": mujoco.__version__, "sides": {}}
    alphas = np.linspace(0.0, 1.0, 11)
    for side, target in targets.items():
        scalar_states = [alpha * target for alpha in alphas]
        scalar_distances, scalar_contacts = _geometry_trace(model, side, scalar_states)
        _assert_trace(
            side,
            "scalar",
            scalar_distances,
            scalar_contacts,
            expected_steps=len(scalar_states),
        )

        slew_states = idealized_slew_states(target)
        if len(slew_states) != 6:
            raise AssertionError(f"{side} idealized slew: expected 6 recurrence steps, got {len(slew_states)}")
        slew_distances, slew_contacts = _geometry_trace(model, side, slew_states)
        _assert_trace(
            side,
            "idealized_slew",
            slew_distances,
            slew_contacts,
            expected_steps=6,
        )
        report["sides"][side] = {
            "scalar_distances_m": scalar_distances.tolist(),
            "scalar_final_contact_count": int(scalar_contacts[-1]),
            "idealized_slew_steps": len(slew_states),
            "idealized_slew_distances_m": slew_distances.tolist(),
            "idealized_slew_final_contact_count": int(slew_contacts[-1]),
        }
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_xr_targets(args.xr_root.resolve())
    report = verify_pinch(args.model.resolve(), targets)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
