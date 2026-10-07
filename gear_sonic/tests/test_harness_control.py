import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROFILE = Path(__file__).parent / "fixtures/g1_profile.yaml"


def state(index=1, planner=1):
    return dict(
        index=index,
        body_q=[0.0] * 29,
        body_q_measured_motor=[0.0] * 29,
        harness_planner_hold_enabled=[1],
        left_hand_q=[0.3] * 7,
        right_hand_q=[0.4] * 7,
        left_hand_q_measured=[0.3] * 7,
        right_hand_q_measured=[0.4] * 7,
        base_quat=[1, 0, 0, 0],
        reference_heading_quat=[1, 0, 0, 0],
        planner_reference_active=[planner],
    )


@pytest.mark.parametrize(
    "field",
    ["body_q_measured_motor", "harness_planner_hold_enabled", "left_hand_q_measured", "right_hand_q_measured"],
)
def test_claim_rejects_ambiguous_or_unsafe_controller(field):
    c, _ = make()
    incompatible = state(2)
    del incompatible[field]
    c.tick(0.1, incompatible, 0.1)
    response = request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, now=0.1)
    assert response["error"]["code"] == "NOT_READY"
    assert c.owner is None


@pytest.mark.parametrize("field", ["left_hand_q_measured", "right_hand_q_measured"])
def test_missing_or_stale_hand_feedback_revokes_authority(field):
    c, h = make()
    start(c)
    incompatible = state(2)
    incompatible[field] = []  # C++ emits empty measurements for absent/stale DDS data.
    c.tick(0.1, incompatible, 0.1)
    assert c.phase == "FAULT" and c.owner is None and not h.enabled
    assert not c.status(0.1)["hold_confirmed"]


def test_disabled_cpp_timeout_hold_blocks_claim():
    c, _ = make()
    incompatible = state(2)
    incompatible["harness_planner_hold_enabled"] = [0]
    c.tick(0.1, incompatible, 0.1)
    response = request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, now=0.1)
    assert response["error"]["code"] == "NOT_READY"


def test_controller_capability_loss_revokes_owner_and_cannot_acknowledge_hold():
    c, h = make()
    start(c)
    incompatible = state(2)
    incompatible["body_q_measured_motor"][3] = float("nan")
    c.tick(0.1, incompatible, 0.1)
    assert c.phase == "FAULT" and c.owner is None and not h.enabled
    assert not c.status(0.1)["hold_confirmed"]
    c.tick(0.2, state(3), 0.2)
    assert c.phase == "FAULT" and c.owner is None
    assert not c.status(0.2)["hold_confirmed"]


class Hooks:
    def __init__(self):
        self.epoch = 0
        self.enabled = False
        self.holds = []
        self.prompt = None

    def runtime_facts(self):
        return SimpleNamespace(
            controller_running=True,
            policy_enabled=self.enabled,
            policy_ready=True,
            inference_busy=False,
            mode_requested="PLANNER" if not self.enabled else "POSE",
            operator_busy=False,
        )

    def invalidate_policy_actions(self):
        self.epoch += 1
        return self.epoch

    def set_policy_prompt(self, prompt):
        self.prompt = prompt

    def set_policy_enabled(self, enabled):
        self.enabled = enabled

    def request_planner_hold(self, feedback, open_hands):
        self.holds.append((feedback, open_hands))

    def begin_standing_reset(self, feedback, open_hands):
        from gear_sonic.utils.inference.standing_reset import StandingReset

        return StandingReset(
            feedback,
            np.zeros(7) if open_hands else feedback["left_hand_q"],
            np.zeros(7) if open_hands else feedback["right_hand_q"],
        )

    def set_planner_command(self, command):
        self.command = command

    def stop_planner_motion(self):
        from dataclasses import replace

        if hasattr(self, "command"):
            self.command = replace(self.command, mode=0, movement=(0.0, 0.0, 0.0), speed=0.0)


def make():
    assert importlib.util.find_spec("gear_sonic.utils.inference.harness_control"), "Native harness control missing"
    cls = importlib.import_module("gear_sonic.utils.inference.harness_control").HarnessControl
    h = Hooks()
    c = cls(PROFILE, h, "boot")
    c.tick(0.0, state(), 0.0)
    c.observation = {"received_at": 0.0, "frame_id": "initial"}
    return c, h


def request(c, method, params=None, now=0.0, rid=None, session="s", lease=None):
    if lease is None and c.owner:
        lease = c.owner["lease_id"]
    return c.dispatch(
        dict(
            version=1,
            request_id=rid or f"{method}-{now}",
            runtime_id="boot",
            session_id=session,
            lease_id=lease,
            method=method,
            params=params or {},
        ),
        now,
    )


def start(c):
    assert request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})["error"] is None
    result = request(c, "start_manipulation", {"skill_id": "bottle_to_right_table"})
    assert result["error"] is None
    return result["result"]["execution_id"]


def action():
    return dict(
        motion_token=np.zeros((1, 40, 64)),
        left_hand_joints=np.zeros((1, 40, 7)),
        right_hand_joints=np.zeros((1, 40, 7)),
    )


def test_pause_rejects_late_epoch():
    c, h = make()
    eid = start(c)
    old = h.epoch
    assert c.accept_policy_result(old, 0.0, action(), 0.1)
    request(c, "pause_manipulation", {"execution_id": eid}, 0.2)
    assert not c.accept_policy_result(old, 0.2, action(), 0.3)
    assert not h.enabled


def test_paused_start_requires_new_hold_acknowledgement():
    c, h = make()
    eid = start(c)
    request(c, "pause_manipulation", {"execution_id": eid}, 0.1)
    rejected = request(c, "start_manipulation", {"skill_id": "bottle_to_right_table"}, 0.15)
    assert rejected["error"] and rejected["error"]["code"] == "NOT_READY"
    assert c.phase == "PAUSED" and not h.enabled
    c.tick(0.2, state(2), 0.2)
    started = request(c, "start_manipulation", {"skill_id": "bottle_to_right_table"}, 0.2)
    assert started["error"] is None and h.enabled


def test_paused_start_rejects_lost_planner_ack():
    c, h = make()
    eid = start(c)
    request(c, "pause_manipulation", {"execution_id": eid}, 0.1)
    c.tick(0.2, state(2), 0.2)
    c.tick(0.25, state(3, planner=0), 0.25)
    result = request(c, "start_manipulation", {"skill_id": "bottle_to_right_table"}, 0.3)
    assert result["error"] and result["error"]["code"] == "NOT_READY"
    assert not h.enabled


def test_expired_chunk_never_replays_last_frame():
    c, h = make()
    start(c)
    assert not c.accept_policy_result(h.epoch, 0.0, action(), 0.801)
    assert not h.enabled


def test_keyboard_override_revokes_lease():
    c, h = make()
    start(c)
    old = h.epoch
    c.operator_override("p")
    assert c.status(0.1)["owner_session_id"] is None
    assert not c.accept_policy_result(old, 0.05, action(), 0.1)


def test_fault_hold_preserves_hands():
    c, h = make()
    eid = start(c)
    request(c, "cancel", {"execution_id": eid}, 0.1)
    assert h.holds[-1][1] is False
    assert h.holds[-1][0]["left_hand_q"] == [0.3] * 7
    assert not c.status(0.1)["hold_confirmed"]
    c.tick(0.2, state(2), 0.2)
    assert c.status(0.2)["hold_confirmed"]


def test_reset_requires_new_planner_feedback_and_measured_dwell():
    c, h = make()
    eid = start(c)
    request(c, "pause_manipulation", {"execution_id": eid}, 0.1)
    reset = request(c, "reset_standing", {"execution_id": eid, "open_hands": True}, 0.2)
    assert reset["error"] is None
    target = c.reset.target
    measured = state(2)
    from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

    measured["body_q"] = np.zeros(29)
    measured["body_q"][G1_UPPER_BODY_JOINT_INDICES] = target[:17]
    measured["body_q_measured_motor"] = measured["body_q"].copy()
    measured["left_hand_q"] = target[17:24]
    measured["right_hand_q"] = target[24:]
    measured["left_hand_q_measured"] = target[17:24]
    measured["right_hand_q_measured"] = target[24:]
    c.tick(0.3, measured, 0.3)
    assert c.status(0.3)["phase"] == "RESETTING"
    c.tick(0.9, measured, 0.9)  # A republished index cannot satisfy the dwell.
    assert c.status(0.9)["phase"] != "COMPLETED"


@pytest.mark.parametrize("field", ["motion_token", "left_hand_joints", "right_hand_joints"])
def test_nonfinite_action_rejected(field):
    c, h = make()
    start(c)
    a = action()
    a[field][0, 0, 0] = float("nan")
    assert not c.accept_policy_result(h.epoch, 0.0, a, 0.1)


def test_claim_profile_mismatch_and_unknown_skill():
    c, _ = make()
    assert request(c, "claim_control", {"registry_sha256": "wrong"})["error"]["code"] == "PROFILE_MISMATCH"
    request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, rid="claim-ok")
    assert request(c, "start_manipulation", {"skill_id": "invented"})["error"]["code"] == "UNSUPPORTED_SKILL"


def test_stale_feedback_reports_unconfirmed_fault():
    c, h = make()
    start(c)
    c.tick(0.6, state(), 0.6)
    assert c.status(0.6)["phase"] == "FAULT"
    assert not c.status(0.6)["hold_confirmed"]
    assert not h.enabled


def test_missing_camera_prevents_manipulation():
    c, h = make()
    c.observation = None
    request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})
    response = request(c, "start_manipulation", {"skill_id": "bottle_to_right_table"})
    assert response["error"]["code"] == "NOT_READY"
    assert not h.enabled
