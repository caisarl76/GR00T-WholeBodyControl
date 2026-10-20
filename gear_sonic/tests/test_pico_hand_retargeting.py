"""Run with the teleop extra for real official-0.4.6 URDF smoke coverage."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gear_sonic.utils.teleop import pico_hand_retargeting as r


@dataclass
class ConfigFields:
    type: str
    urdf_path: str
    target_link_human_indices: object = None
    low_pass_alpha: float = 0.1


def config():
    return {
        "left": {
            "type": "DeXpIlOt",
            "urdf_path": "unitree_hand/unitree_dex3_left.urdf",
            "target_link_human_indices_dexpilot": [[0], [4]],
            "target_link_human_indices_vector": [[0], [9]],
        }
    }


def points():
    # OpenXR right-handed frame: left hand dorsal +X, fingers -Y, index +Z.
    x = np.zeros((25, 3))
    for start, count, lateral in ((1, 4, 0.045), (5, 5, 0.025), (10, 5, 0), (15, 5, -0.020), (20, 5, -0.038)):
        for segment in range(count):
            x[start + segment] = [0.004 * segment, -0.035 - 0.023 * segment, lateral]
    return x


def test_translation_is_copy_and_disables_only_filter():
    doc = config()
    original = deepcopy(doc)
    normalized = r._translate_config(doc, "left", ConfigFields, r._ROOT)
    assert doc == original
    assert normalized["type"] == "dexpilot"
    assert normalized["target_link_human_indices"] == [[0], [4]]
    assert normalized["low_pass_alpha"] == -1
    assert normalized["urdf_path"] == str((r._ROOT / doc["left"]["urdf_path"]).resolve())
    assert not any(k.startswith("target_link_human_indices_") for k in normalized)
    doc["left"]["type"] = "vector"
    assert r._translate_config(doc, "left", ConfigFields, r._ROOT)["target_link_human_indices"] == [[0], [9]]


@pytest.mark.parametrize(
    "change",
    [
        {"type": "position"},
        {"type": None},
        {"unknown": 1},
        {"target_link_human_indices": [[0], [4]]},
        {"urdf_path": "../escape.urdf"},
        {"urdf_path": "/tmp/escape.urdf"},
        {"target_link_human_indices_dexpilot": [[0], [25]]},
        {"target_link_human_indices_dexpilot": [[0.0], [4.0]]},
        {"target_link_human_indices_dexpilot": [0, 4]},
    ],
)
def test_rejects_unsafe_configs(change):
    doc = config()
    doc["left"].update(change)
    with pytest.raises(ValueError):
        r._translate_config(doc, "left", ConfigFields, r._ROOT)


@pytest.mark.parametrize("document", [[], {"right": {}}, {"left": []}, {"left": {"type": "dexpilot"}}])
def test_missing_mapping_or_selected_key(document):
    with pytest.raises(ValueError):
        r._translate_config(document, "left", ConfigFields, r._ROOT)


def test_checksums_and_manifest_are_pinned(tmp_path):
    shutil.copytree(r._ROOT, tmp_path / "assets")
    root = tmp_path / "assets"
    r._verify_assets(root)
    for relative in r._HASHES:
        path = root / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="checksum"):
            r._verify_assets(root)
        path.write_bytes(original)
    manifest_path = root / "PROVENANCE.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_commit"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        r._verify_assets(root)


def test_symlink_escape_is_rejected(tmp_path):
    (tmp_path / "escape").symlink_to(r._ROOT / "unitree_hand/unitree_dex3_left.urdf")
    with pytest.raises(ValueError, match="escapes"):
        r._asset_path(tmp_path, "escape")


@pytest.mark.parametrize("profile", ["dex3", "inspire_ftp"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_joint_permutation_and_normalization(profile, side):
    path = r._ROOT / (
        f"unitree_hand/unitree_dex3_{side}.urdf" if profile == "dex3" else f"inspire_hand/inspire_hand_{side}.urdf"
    )
    joints = [
        j.attrib["name"]
        for j in ET.parse(path).getroot().findall("joint")
        if j.attrib["type"] in ("revolute", "prismatic", "continuous")
    ]
    names = tuple(reversed(joints))  # deliberately unlike config and hardware order
    desired = r._output_names(profile, side)
    order, limits = r._output_mapping(names, desired, path)
    assert tuple(names[i] for i in order) == desired
    if profile == "inspire_ftp":
        np.testing.assert_allclose(limits[:, 0], [0, 0, 0, 0, 0, -0.1])
        np.testing.assert_allclose(limits[:, 1], [1.7, 1.7, 1.7, 1.7, 0.5, 1.3])
        assert len(names) > 6  # actual return includes URDF mimic joints
    for bad in (names[:-1], names + (names[0],), names + ("unexpected",)):
        with pytest.raises(ValueError, match="joint names"):
            r._output_mapping(bad, desired, path)
    adapter = object.__new__(r.HandRetargeter)
    adapter.side, adapter.profile = side, profile
    adapter._indices = np.array([[0], [4]])
    adapter._out = order
    adapter._radian_lower, adapter._radian_upper = limits.T
    output = np.zeros(len(names))
    fractions = np.linspace(0, 1, len(desired))
    output[order] = limits[:, 0] + fractions * (limits[:, 1] - limits[:, 0])
    adapter._impl = SimpleNamespace(joint_names=names, retarget=lambda ref: output.copy())
    adapter._align_open_thumb = lambda points, q: None  # This test isolates output mapping.
    expected = output[order] if profile == "dex3" else 1 - fractions
    np.testing.assert_allclose(adapter.retarget(points()), expected, atol=1e-7)
    assert adapter.retarget(points()).dtype == np.dtype("<f4")
    output[order[0]] = limits[0, 0] - 5e-5
    assert np.isfinite(adapter.retarget(points())).all()
    output[order[0]] = limits[0, 0] - 0.001
    with pytest.raises(ValueError, match="exceeds"):
        adapter.retarget(points())
    output[order[0]] = np.nan
    with pytest.raises(ValueError, match="invalid joint vector"):
        adapter.retarget(points())


def test_anatomical_frame_is_rigid_motion_invariant_and_dex3_ignores_other_digits():
    from scipy.spatial.transform import Rotation

    x = points()
    rotation = Rotation.from_rotvec([0.7, -1.1, 0.3]).as_matrix()
    for side in ("left", "right"):
        expected = r._wrist_frame(x, side)
        np.testing.assert_allclose(r._wrist_frame(x @ rotation.T + [1, 2, 3], side), expected, atol=1e-14)
        modified = x.copy()
        modified[15:] += [0.07, -0.02, 0.03]
        np.testing.assert_array_equal(r._wrist_frame(modified, side)[:15], expected[:15])
        for invalid in (np.zeros((25, 3)), np.full((25, 3), np.nan), np.zeros((26, 3))):
            with pytest.raises(ValueError):
                r._wrist_frame(invalid, side)


@pytest.mark.parametrize("profile,size", [("dex3", 7), ("inspire_ftp", 6)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_real_official_046_smoke(profile, size, side):
    pytest.importorskip("dex_retargeting", reason="install gear_sonic[teleop] for mandatory real smoke")
    adapter = r.HandRetargeter(profile, side)
    assert adapter._impl.filter is None
    np.testing.assert_array_equal(adapter._impl.optimizer.opt.get_lower_bounds(), adapter._impl.joint_limits[:, 0])
    np.testing.assert_array_equal(adapter._impl.optimizer.opt.get_upper_bounds(), adapter._impl.joint_limits[:, 1])
    q = adapter.retarget(points())
    assert q.shape == adapter.lower.shape == adapter.upper.shape == (size,)
    assert q.dtype == np.dtype("<f4") and q.flags.c_contiguous
    assert np.isfinite(q).all() and (q >= adapter.lower).all() and (q <= adapter.upper).all()
    if profile == "inspire_ftp":
        np.testing.assert_array_equal(adapter.lower, np.zeros(6))
        np.testing.assert_array_equal(adapter.upper, np.ones(6))


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_thumb_alignment_preserves_pinch_and_other_fingers(side):
    pytest.importorskip("dex_retargeting")
    fixture = json.loads((Path(__file__).parent / "fixtures/dex3_thumb_poses.json").read_text())
    straight = np.asarray(fixture["straight"][side])
    pinch = np.asarray(fixture["pinch"][side]["points"])
    adapter = r.HandRetargeter("dex3", side)
    baseline = r.HandRetargeter("dex3", side)
    baseline._align_open_thumb = lambda points, q: None
    for raw in [straight] * 20 + [pinch] * 20:
        actual = adapter.retarget(raw)
        expected = baseline.retarget(raw)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[3:], expected[3:])
        if raw is pinch:
            np.testing.assert_array_equal(actual, expected)
        assert (actual >= adapter.lower).all() and (actual <= adapter.upper).all()


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_thumb_alignment_blends_continuously_and_releases_for_bent_thumb(side):
    pytest.importorskip("dex_retargeting")
    fixture = json.loads((Path(__file__).parent / "fixtures/dex3_thumb_poses.json").read_text())
    adapter = r.HandRetargeter("dex3", side)
    points = r._wrist_frame(np.asarray(fixture["straight"][side]), side)
    seed = (adapter._radian_lower + adapter._radian_upper) / 2
    previous = None
    for distance in np.linspace(0.04, 0.09, 501):
        sample = points.copy()
        sample[[9, 14]] = sample[4] + [distance, 0, 0]
        q = seed.copy()
        adapter._align_open_thumb(sample, q)
        if previous is not None:
            assert np.max(np.abs(q - previous)) < 0.01
        previous = q
    sample = points.copy()
    proximal = sample[3] - sample[2]
    perpendicular = np.cross(proximal, [1, 0, 0])
    sample[4] = sample[3] + perpendicular / np.linalg.norm(perpendicular) * np.linalg.norm(proximal)
    sample[[9, 14]] = sample[4] + [0.1, 0, 0]
    q = seed.copy()
    adapter._align_open_thumb(sample, q)
    np.testing.assert_array_equal(q, seed)


def test_native_openxr_dorsal_and_digit_axes_do_not_use_unity_reflection():
    # TrackingData.cs at cdc53166b0bf412efae71046c6a225eb5091605f sends
    # native Posef; PXR_HandTracking.ToVector3 is only the rendering transform.
    x = np.zeros((25, 3))
    x[6] = [0, -0.05, 0.03]
    x[11] = [0, -0.05, 0]
    x[4] = [0.02, -0.03, 0.04]
    np.testing.assert_allclose(r._wrist_frame(x, "left"), x, atol=1e-15)
    x[:, 2] *= -1  # anatomically mirrored right hand
    np.testing.assert_allclose(r._wrist_frame(x, "right"), x, atol=1e-15)


def test_unpinned_dependency_is_rejected(monkeypatch):
    monkeypatch.setattr(r, "version", lambda package: "0.5.0")
    with pytest.raises(ImportError, match="0.4.6"):
        r.HandRetargeter("dex3", "left")


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_dex3_physical_thumb_feedback_preserves_startup_and_slew(side, sign):
    from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, HandTracker

    pytest.importorskip("dex_retargeting", reason="install gear_sonic[teleop] for mandatory real smoke")
    adapter = r.HandRetargeter("dex3", side)
    tracker = HandTracker(adapter, max_rate=0.5)
    # Recorded left feedback and its anatomical mirror are beyond the optimizer's
    # 0.92 rad target envelope, but within the robot's physical thumb_1 range.
    measured = sign * np.array(
        [-0.26132095, 0.96607655, 0.49830964, -0.22344580, -0.22493894, -0.17277034, -0.20647488]
    )
    canonical = points()
    if side == "right":
        canonical[:, 2] *= -1
    pose = np.zeros((26, 7), dtype=np.float64)
    pose[1:, :3] = canonical
    sample = {
        "pose": pose,
        "location_flags": np.full(26, 0xA, dtype=np.uint64),
        "radius": np.full(26, 0.005),
        "scale": 1.0,
        "is_active": 1,
        "binding_generation": 1,
    }
    commands = []
    for tick in range(30):
        sample["source_timestamp_ns"] = tick + 1
        out = tracker.step(sample, np.zeros(3), measured, tick * 20_000_000)
        assert out.reason == HandReason.OK
        commands.append(out.command)
        if out.target is not None:
            assert sign * out.target[1] <= 0.920001
    np.testing.assert_allclose(commands[0], measured, atol=1e-7, rtol=0)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.010001
    assert sign * commands[-1][1] < sign * measured[1]

    # This source has mirrored right-hand limits; supplemental_info.py does not.
    model = r._ROOT.parents[2] / "data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf"
    joint = ET.parse(model).find(f".//joint[@name='{side}_hand_thumb_1_joint']/limit")
    np.testing.assert_allclose(
        [adapter.lower[1], adapter.upper[1]],
        [float(joint.attrib[bound]) for bound in ("lower", "upper")],
    )
    assert sign * (adapter._radian_upper[1] if side == "left" else adapter._radian_lower[1]) == 0.92
    measured[1] = sign * 1.05889785  # Actual transient beyond even the physical limit.
    assert tracker._bounded(measured) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_recorded_open_thumb_tip_points_in_human_direction(side):
    pytest.importorskip("dex_retargeting")
    fixture = json.loads((Path(__file__).parent / "fixtures/dex3_thumb_poses.json").read_text())
    raw = np.asarray(fixture["straight"][side])
    adapter = r.HandRetargeter("dex3", side)
    for _ in range(30):
        q = adapter.retarget(raw)
    robot = adapter._impl.optimizer.robot
    full = np.zeros(len(adapter._impl.joint_names))
    full[adapter._out] = q
    robot.compute_forward_kinematics(full)
    tip = robot.get_link_pose(robot.get_link_index("thumb_tip"))[:3, 3].copy()
    distal = robot.get_link_pose(robot.get_link_index(f"{side}_hand_thumb_2_link"))[:3, 3]
    actual = tip - distal
    actual /= np.linalg.norm(actual)
    points = r._wrist_frame(raw, side)
    expected = points[4] - points[3]
    expected /= np.linalg.norm(expected)
    error_degrees = np.rad2deg(np.arccos(np.clip(actual @ expected, -1, 1)))
    assert error_degrees < 5, f"Open thumb points inward by {error_degrees:.1f} degrees"



def test_feedback_allowance_matches_real_dex3_right_index_order():
    from gear_sonic.utils.teleop.pico_hand_tracking import HandTracker

    retargeter = r.HandRetargeter("dex3", "right")
    assert r._output_names("dex3", "right")[5] == "right_hand_index_0_joint"
    measured = np.clip(np.zeros(7), retargeter.lower, retargeter.upper)
    measured[5] = -0.0007002827478572726
    tracker = HandTracker(retargeter)
    bounded = tracker._bounded(measured, measured=True)
    assert bounded is not None and bounded[5] == 0
    assert tracker._bounded(measured) is None
