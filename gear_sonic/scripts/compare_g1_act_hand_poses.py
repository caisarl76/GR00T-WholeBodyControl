"""Compute palm-frame FK for ACT, SONIC references/commands and recorded MuJoCo q.

This is offline kinematics, not another policy or physics execution. Every stage
uses the same measured floating base, legs and waist. Issued-target FK is a
hypothetical pose at those PD targets, not a measured robot pose.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from gear_sonic.utils.inference.reference_adapter.g1_act import ACTION_NAMES, BODY_NAMES

DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "data/robot_model/model_data/g1/scene_43dof.xml"
STAGES = ("act_raw", "native_reference", "issued_command", "measured")
LABELS = ("ACT target", "SONIC reference", "Issued target FK", "Measured pose")
COLORS = ("#277da8", "#6f42a5", "#d1495b", "#1b8a5a")
PAIRS = (
    ("act_to_reference", "act_raw", "native_reference"),
    ("reference_to_issued", "native_reference", "issued_command"),
    ("act_to_issued", "act_raw", "issued_command"),
    ("reference_to_measured", "native_reference", "measured"),
    ("act_to_measured", "act_raw", "measured"),
)


def pose_error(position_a, quaternion_a, position_b, quaternion_b):
    """Euclidean distance (m) and shortest SO(3) angle (rad), wxyz quaternions."""
    qa, qb = np.asarray(quaternion_a), np.asarray(quaternion_b)
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    # q and -q denote the same rotation. atan2 avoids acos loss near zero.
    qb = np.where((np.sum(qa * qb, axis=-1) < 0)[..., None], -qb, qb)
    angle = 4 * np.arctan2(np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1))
    distance = np.linalg.norm(np.asarray(position_a) - np.asarray(position_b), axis=-1)
    return distance, angle


def statistics(values):
    values = np.asarray(values)
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def palm_frames(scene):
    """Read virtual palm frames from the original XML mesh placements.

    Compiled geom poses may include MuJoCo mesh recentering. The declared mesh
    frame, attached to wrist_yaw_link, is the reproducible palm frame we want.
    """
    model_xml = scene.parent / "g1_29dof_with_hand.xml"
    root = ET.parse(model_xml).getroot()
    frames = {}
    for side in ("left", "right"):
        body_name = f"{side}_wrist_yaw_link"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"missing palm parent {body_name} in {model_xml}")
        placements = [g for g in body.findall("geom") if g.get("mesh") == f"{side}_hand_palm_link"]
        if not placements:
            raise ValueError(f"missing {side} palm mesh frame")
        poses = [
            (
                tuple(float(v) for v in g.get("pos", "0 0 0").split()),
                tuple(float(v) for v in g.get("quat", "1 0 0 0").split()),
            )
            for g in placements
        ]
        if any(p != poses[0] for p in poses) or poses[0][1] != (1, 0, 0, 0):
            raise ValueError("palm mesh frames must agree and have identity local orientation")
        frames[side] = {
            "parent_body": body_name,
            "offset_m": list(poses[0][0]),
            "local_quaternion_wxyz": list(poses[0][1]),
        }
    return frames, model_xml


class PalmKinematics:
    def __init__(self, scene=DEFAULT_SCENE):
        import mujoco

        self.mujoco = mujoco
        self.scene = Path(scene).resolve()
        self.frames, self.model_xml = palm_frames(self.scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.body_qadr = np.array([self.model.joint(name).qposadr[0] for name in BODY_NAMES])
        self.action_qadr = np.array([self.model.joint(name).qposadr[0] for name in ACTION_NAMES])
        free = self.model.joint("floating_base_joint")
        if free.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("expected free floating base")
        self.base_qadr = int(free.qposadr[0])

    def compute(self, observation, action):
        """FK at action's arms/fingers with the recorded base/legs/waist."""
        body = np.asarray(observation["body_q"], dtype=float)
        base = np.asarray(observation["floating_base_pose"], dtype=float)
        action = np.asarray(action, dtype=float)
        if body.shape != (29,) or base.shape != (7,) or action.shape != (28,):
            raise ValueError("expected body[29], floating base[7], action[28]")
        if not np.all(np.isfinite(np.r_[body, base, action])):
            raise ValueError("nonfinite kinematic state")
        if not np.isclose(np.linalg.norm(base[3:]), 1, atol=1e-5):
            raise ValueError("floating base must have a unit wxyz quaternion")
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.base_qadr : self.base_qadr + 7] = base
        self.data.qpos[self.body_qadr] = body
        self.data.qpos[self.action_qadr] = action
        # mj_kinematics updates world body transforms only; no integration,
        # joint-limit projection, contact solver, or controller is executed.
        self.mujoco.mj_kinematics(self.model, self.data)
        torso = self.data.body("torso_link")
        torso_r = torso.xmat.reshape(3, 3).copy()
        torso_p = torso.xpos.copy()
        result = {}
        for side, frame in self.frames.items():
            body_pose = self.data.body(frame["parent_body"])
            rotation = body_pose.xmat.reshape(3, 3).copy()
            position = body_pose.xpos + rotation @ frame["offset_m"]
            relative_r = torso_r.T @ rotation
            relative_q = np.empty(4)
            self.mujoco.mju_mat2Quat(relative_q, relative_r.reshape(-1))
            result[side] = {
                "world_position_m": position.copy(),
                "world_quaternion_wxyz": body_pose.xquat.copy(),
                "torso_position_m": torso_r.T @ (position - torso_p),
                "torso_quaternion_wxyz": relative_q,
            }
        return result


def analyze_poses(comparison, scene=DEFAULT_SCENE):
    fk = PalmKinematics(scene)
    count = len(comparison["observations"])
    poses = {
        stage: {
            side: {
                key: []
                for key in (
                    "world_position_m",
                    "world_quaternion_wxyz",
                    "torso_position_m",
                    "torso_quaternion_wxyz",
                )
            }
            for side in ("left", "right")
        }
        for stage in STAGES
    }
    for i, observation in enumerate(comparison["observations"]):
        for stage in STAGES:
            computed = fk.compute(observation, comparison[stage][i])
            for side in poses[stage]:
                for key in poses[stage][side]:
                    poses[stage][side][key].append(computed[side][key])
    for stage in poses:
        for side in poses[stage]:
            poses[stage][side] = {k: np.asarray(v) for k, v in poses[stage][side].items()}
    errors, metrics = {}, {}
    for side in ("left", "right"):
        errors[side], metrics[side] = {}, {}
        for name, source, target in PAIRS:
            a, b = poses[source][side], poses[target][side]
            distance, angle = pose_error(
                a["world_position_m"],
                a["world_quaternion_wxyz"],
                b["world_position_m"],
                b["world_quaternion_wxyz"],
            )
            local_distance, local_angle = pose_error(
                a["torso_position_m"],
                a["torso_quaternion_wxyz"],
                b["torso_position_m"],
                b["torso_quaternion_wxyz"],
            )
            # Distances/relative angles must be invariant under the common frame.
            np.testing.assert_allclose(distance, local_distance, atol=1e-12, rtol=0)
            np.testing.assert_allclose(angle, local_angle, atol=1e-12, rtol=0)
            errors[side][name] = {"position_m": distance, "orientation_deg": np.rad2deg(angle)}
            metrics[side][name] = {
                "position_m": statistics(distance),
                "orientation_deg": statistics(np.rad2deg(angle)),
            }
    summary = {
        "samples": count,
        "mujoco_version": fk.mujoco.__version__,
        "model_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (fk.scene, fk.model_xml)
        },
        "frames": fk.frames,
        "metrics": metrics,
        "method": (
            "MuJoCo mj_kinematics on saved joint states, with identical measured base/legs/waist at every stage"
        ),
        "orientation_error": "Shortest rotation angle on SO(3); quaternion sign invariant",
        "scope": "Palm mesh-frame origin and orientation, not a calibrated grasp TCP or fingertip pose",
        "issued_target_caveat": (
            "FK at received PD targets is hypothetical; it is not the physical pose produced by the decoder"
        ),
        "measured_pose": (
            "Reconstructed from recorded MuJoCo q and floating base; no hand-pose sensor was recorded"
        ),
        "timing": comparison["summary"].get("alignment", {}),
        "source": comparison["summary"],
    }
    return poses, errors, summary


def write_outputs(output_dir, comparison, poses, errors, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hand_pose_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "hand_pose_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "hand",
                "comparison",
                "position_rms_m",
                "position_mean_m",
                "position_p95_m",
                "position_max_m",
                "orientation_rms_deg",
                "orientation_mean_deg",
                "orientation_p95_deg",
                "orientation_max_deg",
            ]
        )
        for side, metrics in summary["metrics"].items():
            for name, values in metrics.items():
                writer.writerow([side, name, *values["position_m"].values(), *values["orientation_deg"].values()])
    with (output_dir / "hand_pose_timeseries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "monotonic_ns",
                "reference_frame",
                "source_time_s",
                "hand",
                "stage",
                "world_x_m",
                "world_y_m",
                "world_z_m",
                "world_qw",
                "world_qx",
                "world_qy",
                "world_qz",
                "torso_x_m",
                "torso_y_m",
                "torso_z_m",
                "torso_qw",
                "torso_qx",
                "torso_qy",
                "torso_qz",
            ]
        )
        for i in range(summary["samples"]):
            for side in ("left", "right"):
                for stage in STAGES:
                    p = poses[stage][side]
                    writer.writerow(
                        [
                            int(comparison["monotonic_ns"][i]),
                            int(comparison["reference_frames"][i]),
                            comparison["source_time_s"][i],
                            side,
                            stage,
                            *p["world_position_m"][i],
                            *p["world_quaternion_wxyz"][i],
                            *p["torso_position_m"][i],
                            *p["torso_quaternion_wxyz"][i],
                        ]
                    )
    plot_poses(output_dir, comparison, poses, errors)


def plot_poses(output_dir, comparison, poses, errors):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = comparison["source_time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, side in enumerate(("left", "right")):
        for name, label, color, style in (
            ("act_to_measured", "ACT vs measured", COLORS[3], "-"),
            ("reference_to_measured", "Reference vs measured", COLORS[1], "-"),
            ("act_to_issued", "ACT vs issued-target FK", COLORS[2], "--"),
        ):
            axes[0, col].plot(
                times, 100 * errors[side][name]["position_m"], label=label, color=color, ls=style, lw=1.4
            )
            axes[1, col].plot(times, errors[side][name]["orientation_deg"], color=color, ls=style, lw=1.4)
        axes[0, col].set_title(f"{side.title()} palm")
        axes[0, col].set_ylabel("Position error (cm)")
        axes[1, col].set_ylabel("Orientation error (degrees)")
        axes[1, col].set_xlabel("ACT source time (s)")
        for axis in axes[:, col]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        f"ACT → SONIC v1.1 → G1/Dex3: palm pose error\nSame measured base and waist; {len(times)} saved samples"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "hand_pose_errors.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    source_hz = comparison["summary"]["alignment"]["source_hz"]
    source_frames = comparison["summary"]["scope"]["source_action_frames"]
    for col, side in enumerate(("left", "right")):
        for name, label, color in (
            ("act_to_measured", "ACT vs measured", COLORS[3]),
            ("reference_to_measured", "Reference vs measured", COLORS[1]),
        ):
            axes[0, col].plot(times, 100 * errors[side][name]["position_m"], label=label, color=color, lw=1.5)
            axes[1, col].plot(times, errors[side][name]["orientation_deg"], color=color, lw=1.5)
        axes[0, col].set_title(f"{side.title()} palm")
        axes[0, col].set_ylabel("Position error (cm)")
        axes[1, col].set_ylabel("Orientation error (degrees)")
        axes[1, col].set_xlabel("ACT source time (s)")
        for axis in axes[:, col]:
            axis.grid(alpha=0.2)
            for frame in range(100, source_frames, 100):
                axis.axvline(frame / source_hz, color="gray", ls=":", alpha=0.45, lw=0.8)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Actual palm tracking relative to ACT and the conditioned reference\n"
        "Dotted lines: 100-action chunk boundaries"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "hand_tracking_errors.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(13, 6))
    for col, side in enumerate(("left", "right")):
        axis = fig.add_subplot(1, 2, col + 1, projection="3d")
        combined = []
        for stage, label, color in zip(STAGES, LABELS, COLORS):
            p = poses[stage][side]["torso_position_m"] * 100
            combined.append(p)
            axis.plot(
                *p.T, label=label, color=color, alpha=0.85, lw=1.3, ls="--" if stage == "issued_command" else "-"
            )
        all_p = np.vstack(combined)
        center = (all_p.max(axis=0) + all_p.min(axis=0)) / 2
        radius = max(np.ptp(all_p, axis=0).max() / 2, 1) * 1.08
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set(
            xlabel="Torso X (cm)", ylabel="Torso Y (cm)", zlabel="Torso Z (cm)", title=f"{side.title()} palm path"
        )
        axis.view_init(elev=22, azim=-65)
        if col == 0:
            axis.legend(fontsize=9)
    fig.suptitle("Palm trajectories in the torso frame · issued-target FK is hypothetical")
    fig.tight_layout()
    fig.savefig(output_dir / "hand_pose_trajectories.png", dpi=170)
    plt.close(fig)


def main():
    from gear_sonic.scripts.compare_g1_act_sonic_joints import load_comparison

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--chunk-id", type=int, default=2)
    args = parser.parse_args()
    comparison = load_comparison(args.run_dir, args.session_id, args.chunk_id)
    poses, errors, summary = analyze_poses(comparison, args.scene)
    manifest_path = args.run_dir / "result-video/render-manifest.json"
    if manifest_path.exists():
        summary["recorded_render_mujoco_version"] = json.loads(manifest_path.read_text()).get("mujoco_version")
    write_outputs(args.output_dir, comparison, poses, errors, summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
