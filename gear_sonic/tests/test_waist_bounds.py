"""WAIST_BOUNDS come from the G1 model, and every copy of that model agrees."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from gear_sonic.utils.inference.harness_profile import WAIST_BOUNDS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
COPIES = [
    ROOT / "gear_sonic/data/robots/g1/g1_29dof_with_hand.xml",
    ROOT / "gear_sonic_deploy/g1/g1_29dof_with_hand.xml",
    ROOT / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml",
    ROOT / "motionbricks/assets/skeletons/g1/g1_29dof_with_hand.xml",
]


def waist_ranges(path):
    joints = {j.get("name"): j.get("range") for j in ET.parse(path).iter("joint")}
    return {name: tuple(float(v) for v in joints[name].split()) for name in WAIST_BOUNDS}


def test_waist_bounds_match_the_source_model():
    assert list(WAIST_BOUNDS) == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # Motors 12-14.
    assert waist_ranges(SOURCE) == WAIST_BOUNDS


@pytest.mark.parametrize("copy", COPIES, ids=lambda p: str(p.relative_to(ROOT)))
def test_model_copies_agree_on_waist_ranges(copy):
    assert waist_ranges(copy) == waist_ranges(SOURCE)
