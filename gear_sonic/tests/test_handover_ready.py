from dataclasses import replace

import numpy as np
import pytest
import yaml

from test_harness_control import Hooks, PROFILE, request, state
from gear_sonic.utils.inference.harness_control import HarnessControl
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

TARGET = [-.1, -.08, .3, -.21, 1.7, -.09, -.07]
NAMES = [f"right_{name}_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)]


class ReadyHooks(Hooks):
    def begin_ready_reset(self, feedback, right_arm_target):
        return StandingReset(feedback, feedback["left_hand_q_measured"], np.zeros(7),
                             right_arm_target=right_arm_target)


def make_ready(tmp_path, *, verified=True, reviewed=True, hooks=None):
    data = yaml.safe_load(PROFILE.read_text())
    data["skills"] = {"bottle_handover": dict(prompt="test-only handover",
        completion_criteria="Stable offer", completion_action="hold")}
    data["handover"] = dict(skill_id="bottle_handover", checkpoint_verified=verified,
        ready_pose_reviewed=reviewed, ready_right_arm_joints=dict(zip(NAMES, TARGET)))
    path = tmp_path / "ready.yaml"
    path.write_text(yaml.safe_dump(data))
    c = HarnessControl(path, hooks or ReadyHooks(), "boot")
    c.tick(0., state(), 0.)
    c.observation = {"received_at": 0., "frame_id": "first"}
    assert request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})["error"] is None
    return c


def test_ready_preserves_other_arm_waist_heading_and_left_hand():
    s = state()
    s["body_q_measured_motor"] = np.linspace(-.1, .1, 29).tolist()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET)
    upper = np.asarray(s["body_q_measured_motor"])[G1_UPPER_BODY_JOINT_INDICES]
    assert reset.target[:10] == pytest.approx(upper[:10])
    assert reset.target[10:17] == pytest.approx(TARGET)
    assert reset.target[17:24] == pytest.approx([.3]*7)
    assert reset.target[24:] == pytest.approx([0]*7)
    last = np.r_[reset.command.upper_body_position, reset.command.left_hand_position,
                 reset.command.right_hand_position]
    for _ in range(100):
        cmd = reset.advance(s, .02)
        current = np.r_[cmd.upper_body_position, cmd.left_hand_position, cmd.right_hand_position]
        assert np.max(np.abs(current-last)) <= .0100001
        assert np.max(np.abs(current - np.r_[upper, [.3]*7, [.4]*7])) <= .1500001
        assert tuple(cmd.movement) == (0., 0., 0.)
        last = current
    assert not reset.is_settled(s, .05, .1)


@pytest.mark.parametrize("verified,reviewed,hooks", [(False,True,None), (True,False,None), (True,True,Hooks())])
def test_unverified_or_unsupported_handover_cannot_activate(tmp_path, verified, reviewed, hooks):
    c = make_ready(tmp_path, verified=verified, reviewed=reviewed, hooks=hooks)
    assert request(c, "reset_ready", {"execution_id": ""})["error"] is not None
    assert request(c, "start_manipulation", {"skill_id": "bottle_handover"})["error"] is not None
    assert not c.hooks.enabled


def test_ready_requires_measured_settling_for_half_second(tmp_path):
    c = make_ready(tmp_path)
    result = request(c, "reset_ready", {"execution_id": ""})
    assert result["error"] is None
    assert c.phase == "RESETTING"
    s = state(2)
    body = np.asarray(s["body_q_measured_motor"])
    body[G1_UPPER_BODY_JOINT_INDICES] = c.reset.target[:17]
    s["body_q_measured_motor"] = body.tolist()
    s["right_hand_q_measured"] = [0]*7
    c.tick(.1, s, .1)
    c.tick(.59, {**s, "index":3}, .59)
    assert c.phase == "RESETTING"
    c.tick(.61, {**s, "index":4}, .61)
    assert c.phase == "COMPLETED" and c.status(.61)["right_hand_open"] is True
    c.tick(1.2, None, None)
    assert c.status(1.2)["right_hand_open"] is None
