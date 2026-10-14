"""Audited Unitree assets and named hardware outputs for optical retargeting."""

from dataclasses import fields
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = Path(__file__).with_name("hand_assets")
_COMMIT = "934361718c5886356e19f9068f4536f04d2b9feb"
_HASHES = {
    "unitree_hand/unitree_dex3.yml": "2c39bc8c123bdc00f299d02d995d9bc4a12179adf4942eabc27349252c1c7f15",
    "unitree_hand/unitree_dex3_left.urdf": "569e4d8ca55b1b76e51cd41df3cd6f655c5475a3e02e908f673f999759c51bd0",
    "unitree_hand/unitree_dex3_right.urdf": "4e36e939466d386524d38079f8e2ff84d9dd4d3a7f8db6319374067970ad7075",
    "inspire_hand/inspire_hand.yml": "94feea4fc387a52bdda57538715f508b51a7704225e102e59db24941de5c05bf",
    "inspire_hand/inspire_hand_left.urdf": "bfd26847a64b3794aabdeaeda6c790866b5d702c919d765fecf56fb584212367",
    "inspire_hand/inspire_hand_right.urdf": "3dc82ee57e8918ce6316b4d0a0572450e719474b437f91dca4fea1253547b8c1",
}
_FILES = {"dex3": "unitree_hand/unitree_dex3.yml", "inspire_ftp": "inspire_hand/inspire_hand.yml"}
_INSPIRE_ORDER = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)


def _asset_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("retargeting asset path escapes vendored root")
    return path


def _verify_assets(root):
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    if (
        manifest.get("source_commit") != _COMMIT
        or manifest.get("source_repository") != "https://github.com/unitreerobotics/xr_teleoperate"
        or manifest.get("license") != "Apache-2.0"
        or manifest.get("sha256") != _HASHES
    ):
        raise ValueError("retargeting asset manifest differs from pinned provenance")
    for relative, digest in _HASHES.items():
        if sha256(_asset_path(root, relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"retargeting asset checksum mismatch: {relative}")
    if sha256(_asset_path(root, "LICENSE").read_bytes()).hexdigest() != manifest.get("license_sha256"):
        raise ValueError("retargeting license checksum mismatch")


def _translate_config(document, side, config_class, root):
    if type(document) is not dict or side not in document or type(document[side]) is not dict:
        raise ValueError("retargeting YAML must contain a mapping for the declared side")
    cfg = dict(document[side])
    if any(not isinstance(key, str) for key in cfg):
        raise ValueError("retargeting option names must be strings")
    kind = cfg.get("type")
    if not isinstance(kind, str) or kind.lower() not in ("dexpilot", "vector"):
        raise ValueError("retargeting type must be vector or dexpilot")
    cfg["type"] = kind.lower()
    selected = "target_link_human_indices_" + cfg["type"]
    if selected not in cfg or "target_link_human_indices" in cfg:
        raise ValueError("missing selected human indices or ambiguous canonical indices")
    cfg["target_link_human_indices"] = cfg[selected]
    for key in tuple(cfg):
        if key.startswith("target_link_human_indices_"):
            del cfg[key]
    cfg["low_pass_alpha"] = -1.0
    unknown = set(cfg) - {field.name for field in fields(config_class)}
    if unknown:
        raise ValueError(f"unknown retargeting options: {sorted(unknown)}")
    indices = np.asarray(cfg["target_link_human_indices"])
    if (
        indices.ndim != 2
        or indices.shape[0] != 2
        or indices.shape[1] == 0
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= 25)
    ):
        raise ValueError("human indices must be integer pairs in [0,24]")
    cfg["urdf_path"] = str(_asset_path(root, cfg["urdf_path"]))
    return cfg


def _output_names(profile, side):
    if profile == "dex3":
        return tuple(
            f"{side}_hand_{digit}_{joint}_joint"
            for digit, joint in (
                ("thumb", 0),
                ("thumb", 1),
                ("thumb", 2),
                ("middle", 0),
                ("middle", 1),
                ("index", 0),
                ("index", 1),
            )
        )
    return tuple(("L_" if side == "left" else "R_") + suffix for suffix in _INSPIRE_ORDER)


def _output_mapping(names, wanted, urdf_path):
    joints = {
        j.attrib["name"]: j
        for j in ET.parse(urdf_path).getroot().findall("joint")
        if j.attrib["type"] in ("revolute", "prismatic", "continuous")
    }
    active = {name for name, joint in joints.items() if joint.find("mimic") is None}
    if len(names) != len(set(names)) or set(names) != set(joints) or active != set(wanted):
        raise ValueError("retargeting joint names must exactly cover active and URDF mimic joints")
    limits = np.asarray(
        [[float(joints[name].find("limit").attrib[bound]) for bound in ("lower", "upper")] for name in wanted]
    )
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("invalid retargeting joint limits")
    return np.asarray([names.index(name) for name in wanted]), limits


def _wrist_frame(canonical, side):
    """Express points in Unitree's anatomical hand axes without optical rotations.

    Unitree televuer/tv_wrapper.py uses +X dorsal, -Y toward middle,
    +Z toward index on the left and toward middle on the right. MCP anchors
    are canonical 6 (index) and 11 (middle); ring/little cannot affect Dex3.

    The wire is native OpenXR, not the Unity rendering frame: XRoboToolkit
    Unity Client commit cdc53166b0bf412efae71046c6a225eb5091605f, TrackingData.cs
    GetHandTrackingData/GetPoseStr(Vector3f, Quatf), serializes native Posef
    unchanged. PXR_HandTracking.ToVector3 flips Z only for Unity rendering.
    Applying that reflection here would reverse the dorsal axis.
    """
    points = np.asarray(canonical, dtype=np.float64)
    if points.shape != (25, 3) or not np.all(np.isfinite(points)):
        raise ValueError("canonical landmarks must be finite shape (25,3)")
    points = points - points[0]
    y_axis = -points[11]
    norm = np.linalg.norm(y_axis)
    if norm < 1e-6:
        raise ValueError("degenerate wrist-to-middle frame")
    y_axis /= norm
    z_axis = points[6] - points[11]
    z_axis -= np.dot(z_axis, y_axis) * y_axis
    norm = np.linalg.norm(z_axis)
    if norm < 1e-6:
        raise ValueError("degenerate index-to-middle frame")
    z_axis *= (1 if side == "left" else -1) / norm
    x_axis = np.cross(y_axis, z_axis)
    return points @ np.column_stack((x_axis, y_axis, z_axis))


class HandRetargeter:
    def __init__(self, profile: str, side: str):
        if profile not in _FILES or side not in ("left", "right"):
            raise ValueError("profile must be dex3/inspire_ftp and side must be left/right")
        if version("dex-retargeting") != "0.4.6":
            raise ImportError("optical retargeting requires official dex-retargeting==0.4.6")
        from dex_retargeting.retargeting_config import RetargetingConfig
        import yaml

        _verify_assets(_ROOT)
        cfg = _translate_config(
            yaml.safe_load((_ROOT / _FILES[profile]).read_text()), side, RetargetingConfig, _ROOT
        )
        self.profile, self.side = profile, side
        self._indices = np.asarray(cfg["target_link_human_indices"], dtype=int)
        wanted = _output_names(profile, side)
        configured = cfg.get("target_joint_names", [])
        if len(configured) != len(set(configured)) or set(configured) != set(wanted):
            raise ValueError("configured active joint names differ from hardware profile")
        self._impl = RetargetingConfig.from_dict(cfg).build()
        # Official 0.4.6 otherwise widens bounds by 1e-3, beyond our 1e-4 tolerance.
        self._impl.optimizer.set_joint_limit(self._impl.joint_limits, epsilon=0.0)
        self._out, limits = _output_mapping(
            tuple(self._impl.joint_names), _output_names(profile, side), cfg["urdf_path"]
        )
        self._radian_lower, self._radian_upper = limits.T
        self.lower = (self._radian_lower if profile == "dex3" else np.zeros(6)).astype(np.float32)
        self.upper = (self._radian_upper if profile == "dex3" else np.ones(6)).astype(np.float32)
        if profile == "dex3":
            robot = self._impl.optimizer.robot
            robot.compute_forward_kinematics(np.zeros(len(self._impl.joint_names)))
            self._palm_rotation = robot.get_link_pose(robot.get_link_index(f"{side}_hand_palm_link"))[
                :3, :3
            ].copy()
            # Physical thumb_1 limits from data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf.
            # The pinned retargeting asset narrows this bound to +/-0.92. Allow
            # measured startup holds in the physical range so recovery can slew
            # toward the unchanged optimizer targets without clipping its baseline.
            if side == "left":
                self.upper[1] = 1.04719755
            else:
                self.lower[1] = -1.04719755

    def _align_open_thumb(self, points, q):
        """Match spread-thumb direction rather than equating human bend to motor angle.

        Dex3's thumb starts across the palm: even a straight human thumb needs
        nonzero thumb_2 to point outward within thumb_1's mechanical range.
        Pinches and bent thumbs retain the fingertip-position solution.
        """
        optimizer = self._impl.optimizer
        segments = np.diff(points[1:5], axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths < 1e-6):
            raise ValueError("degenerate thumb segments")
        segments /= lengths[:, None]
        bend = np.arccos(np.clip(np.sum(segments[:-1] * segments[1:], axis=1), -1, 1))
        distance = np.min(np.linalg.norm(points[[9, 14]] - points[4], axis=1))
        # Fade across 50–80 mm separation and 10–30 degrees of thumb bend.
        blend = np.clip(
            [(distance - optimizer.escape_dist) / 0.03, (np.deg2rad(30) - max(bend)) / np.deg2rad(20)],
            0,
            1,
        )
        weight = np.prod(blend * blend * (3 - 2 * blend))
        # In palm coordinates the left distal axis is Ry(q0) Rz(q1+q2) -Y;
        # the right uses +Y and negative flexion. Read the fixed palm rotation
        # from the audited URDF, including its slightly rounded RPY constants.
        direction = self._palm_rotation.T @ segments[-1]
        # The forward hemisphere has an unambiguous solution. Fade out near
        # its boundary instead of flipping q0 at atan2's rearward branch cut.
        forward = np.clip(direction[0] / 0.1, 0, 1)
        weight *= forward * forward * (3 - 2 * forward)
        if weight == 0:
            return
        sign = 1 if self.side == "left" else -1
        desired = q.copy()
        desired[0] = -np.arctan2(direction[2], direction[0])
        total_flexion = sign * np.arccos(np.clip(-sign * direction[1], -1, 1))
        desired[2] = total_flexion - q[1]
        desired = np.clip(desired, self._radian_lower, self._radian_upper)
        q[:] += weight * (desired - q)

    def retarget(self, canonical: np.ndarray) -> np.ndarray:
        points = _wrist_frame(canonical, self.side)
        ref = points[self._indices[1]] - points[self._indices[0]]
        result = np.asarray(self._impl.retarget(ref), dtype=np.float64)
        if result.shape != (len(self._impl.joint_names),) or not np.all(np.isfinite(result)):
            raise ValueError("retargeting returned invalid joint vector")
        q = result[self._out]
        lo, hi = self._radian_lower, self._radian_upper
        if np.any(q < lo - 1e-4) or np.any(q > hi + 1e-4):
            raise ValueError("retargeting output exceeds URDF limits")
        q = np.clip(q, lo, hi)
        if self.profile == "dex3":
            self._align_open_thumb(points, q)
        if self.profile == "inspire_ftp":
            q = 1 - (q - lo) / (hi - lo)
        return np.ascontiguousarray(q, dtype="<f4")


__all__ = ["HandRetargeter"]
