"""Render the calibrated right Dex3 thumb-middle pinch trajectory in MuJoCo."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gear_sonic.scripts.pico_manager_thread_server import (
    DEX3_MOTOR_ORDER,
    compute_pinch_joints,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = REPO_ROOT / "gear_sonic_deploy" / "g1" / "meshes"
RIGHT_MODEL_JOINT_ORDER = (
    "thumb0",
    "thumb1",
    "thumb2",
    "index0",
    "index1",
    "middle0",
    "middle1",
)
RIGHT_MODEL_FROM_MOTOR = np.array(
    [DEX3_MOTOR_ORDER.index(name) for name in RIGHT_MODEL_JOINT_ORDER], dtype=np.intp
)
FONT_PATHS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
)
NOMINAL_CAPTION = "nominal joint-space visualization — not contact-force validation"
PANEL_WIDTH = 720
RENDER_HEIGHT = 480


def _mjcf() -> str:
    return f"""
<mujoco model="dex3_right_pinch">
  <compiler angle="radian" meshdir="{MESH_DIR}" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="720" offheight="480"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    <mesh name="palm" file="right_hand_palm_link.STL"/>
    <mesh name="thumb0" file="right_hand_thumb_0_link.STL"/>
    <mesh name="thumb1" file="right_hand_thumb_1_link.STL"/>
    <mesh name="thumb2" file="right_hand_thumb_2_link.STL"/>
    <mesh name="index0" file="right_hand_index_0_link.STL"/>
    <mesh name="index1" file="right_hand_index_1_link.STL"/>
    <mesh name="middle0" file="right_hand_middle_0_link.STL"/>
    <mesh name="middle1" file="right_hand_middle_1_link.STL"/>
  </asset>
  <worldbody>
    <light pos="0.10 -0.35 0.45" dir="0 0.65 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="0.30 0.20 0.20" dir="-0.5 -0.2 -0.4" diffuse="0.55 0.65 0.8"/>
    <body name="wrist">
      <geom type="mesh" pos="0.0415 -0.003 0" mesh="palm" rgba="0.32 0.36 0.43 1"
            contype="0" conaffinity="0" group="1" density="0"/>
      <body name="thumb0_body" pos="0.0695 -0.003 0">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="thumb0" axis="0 1 0" range="-1.0472 1.0472"/>
        <geom type="mesh" mesh="thumb0" rgba="1.0 0.55 0.12 1"
              contype="0" conaffinity="0" group="1" density="0"/>
        <body name="thumb1_body" pos="0 0.0246 0">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="thumb1" axis="0 0 1" range="-1.0472 0.724312"/>
          <geom type="mesh" mesh="thumb1" rgba="1.0 0.55 0.12 1"
                contype="0" conaffinity="0" group="1" density="0"/>
          <body name="thumb2_body" pos="0.0055 0.0528 0">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
            <joint name="thumb2" axis="0 0 1" range="-2.0944 0"/>
            <geom type="mesh" mesh="thumb2" rgba="1.0 0.55 0.12 1"
                  contype="0" conaffinity="0" group="1" density="0"/>
          </body>
        </body>
      </body>
      <body name="index0_body" pos="0.1415 -0.0013 0.0285">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="index0" axis="0 0 1" range="-0.191986 1.8326"/>
        <geom type="mesh" mesh="index0" rgba="0.12 0.82 0.96 1"
              contype="0" conaffinity="0" group="1" density="0"/>
        <body name="index1_body" pos="0.0528 0.0055 0">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="index1" axis="0 0 1" range="0 2.0944"/>
          <geom type="mesh" mesh="index1" rgba="0.12 0.82 0.96 1"
                contype="0" conaffinity="0" group="1" density="0"/>
        </body>
      </body>
      <body name="middle0_body" pos="0.1415 -0.0013 -0.0285">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="middle0" axis="0 0 1" range="-0.191986 1.8326"/>
        <geom type="mesh" mesh="middle0" rgba="0.62 0.66 0.72 1"
              contype="0" conaffinity="0" group="1" density="0"/>
        <body name="middle1_body" pos="0.0528 0.0055 0">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="middle1" axis="0 0 1" range="0 2.0944"/>
          <geom type="mesh" mesh="middle1" rgba="0.62 0.66 0.72 1"
                contype="0" conaffinity="0" group="1" density="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _font(size: int):
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError as error:
        raise RuntimeError("DejaVu Sans Bold is required to render deterministic labels") from error


def _set_right_hand_qpos(model: mujoco.MjModel, data: mujoco.MjData, motor_q: np.ndarray) -> None:
    motor_q = np.asarray(motor_q, dtype=np.float64)
    expected_shape = (len(DEX3_MOTOR_ORDER),)
    if motor_q.shape != expected_shape:
        raise ValueError(f"motor_q must have shape {expected_shape}, got {motor_q.shape}")

    model_q = motor_q[RIGHT_MODEL_FROM_MOTOR]
    qpos_addresses = []
    for joint_name in RIGHT_MODEL_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing right-hand joint {joint_name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"MuJoCo joint {joint_name!r} must be a hinge")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    if len(set(qpos_addresses)) != len(qpos_addresses):
        raise ValueError("right-hand joints must have unique qpos addresses")
    for qpos_address, value in zip(qpos_addresses, model_q, strict=True):
        data.qpos[qpos_address] = value


def _render(model: mujoco.MjModel, renderer: mujoco.Renderer, motor_q: np.ndarray) -> Image.Image:
    data = mujoco.MjData(model)
    _set_right_hand_qpos(model, data, motor_q)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.125, 0.018, 0.0)
    camera.distance = 0.39
    camera.azimuth = 92
    camera.elevation = -28
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render())


def _pose_stages() -> tuple[tuple[str, str, np.ndarray], ...]:
    return (
        ("ALL OPEN", "Grip/Squeeze 0.0", compute_pinch_joints("right", 0.0)),
        (
            "PINCH OPEN",
            "thumb-middle Grip/Squeeze 0.2",
            compute_pinch_joints("right", 0.2),
        ),
        (
            "PINCH MID",
            "thumb-middle Grip/Squeeze 0.6",
            compute_pinch_joints("right", 0.6),
        ),
        (
            "PINCH CLOSED",
            "thumb-middle Grip/Squeeze 1.0",
            compute_pinch_joints("right", 1.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "dex3_thumb_middle_pinch_mujoco.png",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(_mjcf())
    poses = _pose_stages()

    canvas = Image.new("RGB", (PANEL_WIDTH * len(poses), 720), (9, 12, 18))
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=PANEL_WIDTH)
    try:
        for index, (title, subtitle, motor_q) in enumerate(poses):
            panel = Image.new("RGB", (PANEL_WIDTH, 660), (15, 19, 28))
            panel.paste(_render(model, renderer, motor_q), (0, 100))
            draw = ImageDraw.Draw(panel)
            draw.text((28, 18), title, fill=(245, 247, 252), font=_font(30))
            draw.text((28, 55), subtitle, fill=(174, 187, 207), font=_font(21))

            footer_top = 100 + RENDER_HEIGHT
            draw.rectangle((0, footer_top, PANEL_WIDTH, 660), fill=(24, 29, 40), outline=None)
            draw.line((0, footer_top, PANEL_WIDTH, footer_top), fill=(61, 70, 88))
            vector = np.array2string(motor_q, precision=6, separator=", ", max_line_width=1000)
            draw.text(
                (28, footer_top + 10),
                vector,
                fill=(222, 228, 240),
                font=_font(14),
            )
            draw.text(
                (28, footer_top + 42),
                NOMINAL_CAPTION,
                fill=(169, 181, 202),
                font=_font(14),
            )
            canvas.paste(panel, (PANEL_WIDTH * index, 60))
    finally:
        renderer.close()

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        "Dex3 right hand — calibrated two-stage thumb-middle pinch",
        fill=(212, 220, 234),
        font=_font(19),
    )
    draw.text(
        (canvas.width - 645, 18),
        "orange: thumb   cyan: index   gray: middle",
        fill=(154, 166, 186),
        font=_font(19),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
