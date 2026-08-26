"""Build the coupled Inspire FTP MuJoCo model used by GEAR-SONIC.

The existing SONIC G1 body model is retained verbatim at the kinematic level.
Only its Dex3 hand subtrees are replaced with hand geometry converted from a
pinned official Unitree FTP URDF. The generated model is committed so runtime
simulation does not require this conversion step.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URDF = REPO_ROOT / "gear_sonic/data/robots/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
SOURCE_SHA256 = "45ca27df411a41c3788561812153a27010cdb5adb144b9141f84c20f78d68d3d"
BASE_MJCF = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
OUTPUT_MJCF = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_ftp.xml"
OUTPUT_SCENE = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml"

ACTIVE_SUFFIXES = (
    "little_1_joint",
    "ring_1_joint",
    "middle_1_joint",
    "index_1_joint",
    "thumb_2_joint",
    "thumb_1_joint",
)
ACTIVE_LIMITS = (1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641)
MIMIC_RELATIONS = (
    ("index_2_joint", "index_1_joint", 1.0843),
    ("middle_2_joint", "middle_1_joint", 1.0843),
    ("ring_2_joint", "ring_1_joint", 1.0843),
    ("little_2_joint", "little_1_joint", 1.0843),
    ("thumb_3_joint", "thumb_2_joint", 0.8024),
    ("thumb_4_joint", "thumb_3_joint", 0.9487),
)

SCENE_XML = """<mujoco model="g1_41dof_inspire_ftp scene">
  <include file="g1_29dof_with_inspire_ftp.xml"/>

  <statistic center="0 0 0.5" extent="2.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>
  </worldbody>
  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
"""


def _verify_source() -> None:
    actual = hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Unitree FTP URDF hash mismatch: expected {SOURCE_SHA256}, got {actual}")


def _compile_official_urdf() -> ET.Element:
    source = SOURCE_URDF.read_text(encoding="utf-8")
    source = source.replace('meshdir="meshes"', 'meshdir="."')
    mesh_names = set(re.findall(r'filename="([^"]+)"', source))
    assets = {name: (SOURCE_URDF.parent / name).read_bytes() for name in mesh_names}
    model = mujoco.MjModel.from_xml_string(source, assets)

    with tempfile.NamedTemporaryFile(suffix=".xml") as temporary:
        mujoco.mj_saveLastXML(temporary.name, model)
        return ET.parse(temporary.name).getroot()


def _find_required(root: ET.Element, xpath: str) -> ET.Element:
    element = root.find(xpath)
    if element is None:
        raise RuntimeError(f"required model element not found: {xpath}")
    return element


def _replace_hand_side(base_root: ET.Element, official_root: ET.Element, side: str) -> None:
    xpath = f".//body[@name='{side}_wrist_yaw_link']"
    base_wrist = _find_required(base_root, xpath)
    official_wrist = _find_required(official_root, xpath)

    for child in list(base_wrist):
        if child.tag == "body":
            base_wrist.remove(child)
        elif child.tag == "geom" and child.get("mesh") != f"{side}_wrist_yaw_link":
            base_wrist.remove(child)

    base_inertial = _find_required(base_wrist, "inertial")
    insert_at = list(base_wrist).index(base_inertial)
    base_wrist.remove(base_inertial)
    official_inertial = _find_required(official_wrist, "inertial")
    base_wrist.insert(insert_at, deepcopy(official_inertial))

    for child in official_wrist:
        if child.tag == "geom" and child.get("mesh") != f"{side}_wrist_yaw_link":
            base_wrist.append(deepcopy(child))
        elif child.tag == "body":
            copied = deepcopy(child)
            for joint in copied.findall(".//joint"):
                joint.set("class", "inspire_ftp_joint")
            base_wrist.append(copied)


def _sync_mesh_assets(base_root: ET.Element, official_root: ET.Element) -> None:
    asset = _find_required(base_root, "asset")
    referenced = {geom.get("mesh") for geom in base_root.findall(".//geom[@mesh]")}
    referenced.discard(None)

    official_meshes = {mesh.get("name"): mesh for mesh in official_root.findall("./asset/mesh")}
    existing_meshes = {mesh.get("name"): mesh for mesh in asset.findall("mesh")}

    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in referenced:
            asset.remove(mesh)

    for name in sorted(referenced):
        if name in existing_meshes:
            continue
        if name not in official_meshes:
            raise RuntimeError(f"official model does not define referenced mesh {name}")
        copied = deepcopy(official_meshes[name])
        copied.attrib.pop("content_type", None)
        copied.set("file", Path(copied.get("file", "")).name)
        asset.append(copied)


def _replace_actuators(base_root: ET.Element) -> None:
    actuator = _find_required(base_root, "actuator")
    for element in list(actuator):
        joint_name = element.get("joint", "")
        if joint_name.startswith("left_hand_") or joint_name.startswith("right_hand_"):
            actuator.remove(element)

    for side in ("left", "right"):
        for suffix, upper in zip(ACTIVE_SUFFIXES, ACTIVE_LIMITS, strict=True):
            joint_name = f"{side}_{suffix}"
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": joint_name,
                    "joint": joint_name,
                    "kp": "10",
                    "kv": "0.2",
                    "ctrlrange": f"0 {upper}",
                    "forcerange": "-10 10",
                },
            )


def _remove_dex3_sensors(base_root: ET.Element) -> None:
    sensors = _find_required(base_root, "sensor")
    for element in list(sensors):
        joint_name = element.get("joint", "")
        if joint_name.startswith("left_hand_") or joint_name.startswith("right_hand_"):
            sensors.remove(element)


def _add_mimic_constraints(base_root: ET.Element) -> None:
    previous = base_root.find("equality")
    if previous is not None:
        base_root.remove(previous)
    equality = ET.SubElement(base_root, "equality")
    for side in ("left", "right"):
        for dependent, driver, ratio in MIMIC_RELATIONS:
            ET.SubElement(
                equality,
                "joint",
                {
                    "name": f"{side}_{dependent}_mimic",
                    "joint1": f"{side}_{dependent}",
                    "joint2": f"{side}_{driver}",
                    "polycoef": f"0 {ratio} 0 0 0",
                },
            )


def _add_hand_default(base_root: ET.Element) -> None:
    defaults = _find_required(base_root, "default")
    for existing in list(defaults.findall("default")):
        if existing.get("class") == "inspire_ftp_joint":
            defaults.remove(existing)
    hand_default = ET.SubElement(defaults, "default", {"class": "inspire_ftp_joint"})
    ET.SubElement(
        hand_default,
        "joint",
        {"damping": "0.05", "armature": "0.001", "frictionloss": "0.1"},
    )


def build_model_bytes() -> bytes:
    _verify_source()
    official_root = _compile_official_urdf()
    base_root = ET.parse(BASE_MJCF).getroot()
    base_root.set("model", "g1_29dof_with_inspire_ftp")
    compiler = _find_required(base_root, "compiler")
    compiler.set("meshdir", "../../../robots/g1/meshes")

    for side in ("left", "right"):
        _replace_hand_side(base_root, official_root, side)
    _sync_mesh_assets(base_root, official_root)
    _replace_actuators(base_root)
    _remove_dex3_sensors(base_root)
    _add_mimic_constraints(base_root)
    _add_hand_default(base_root)

    ET.indent(base_root, space="  ")
    return ET.tostring(base_root, encoding="utf-8", xml_declaration=False) + b"\n"


def _validate_generated_model(model_bytes: bytes) -> None:
    assets = {}
    mesh_root = REPO_ROOT / "gear_sonic/data/robots/g1/meshes"
    root = ET.fromstring(model_bytes)
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if file_name:
            assets[file_name] = (mesh_root / file_name).read_bytes()
    model = mujoco.MjModel.from_xml_string(model_bytes.decode("utf-8"), assets)
    actual = (model.njnt, model.nu, model.neq)
    expected = (54, 41, 12)
    if actual != expected:
        raise RuntimeError(f"generated model counts {actual} do not match {expected}")


def _write_or_check(path: Path, expected: bytes, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != expected:
            raise RuntimeError(f"generated file is stale: {path}")
        return
    path.write_bytes(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if committed generated files are stale")
    args = parser.parse_args()

    model_bytes = build_model_bytes()
    _validate_generated_model(model_bytes)
    _write_or_check(OUTPUT_MJCF, model_bytes, check=args.check)
    _write_or_check(OUTPUT_SCENE, SCENE_XML.encode("utf-8"), check=args.check)
    action = "verified" if args.check else "wrote"
    print(f"{action} {OUTPUT_MJCF.relative_to(REPO_ROOT)}")
    print(f"{action} {OUTPUT_SCENE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
