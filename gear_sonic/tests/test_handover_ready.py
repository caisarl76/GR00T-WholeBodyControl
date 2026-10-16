import numpy as np
import pytest
from test_harness_control import PROFILE, Hooks, request, state
import yaml

from gear_sonic.utils.inference.harness_control import HarnessControl
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

TARGET = [-.1, -.08, .3, -.21, 1.7, -.09, -.07]
RAISED = [.011, -.68, -.60, .16, 1.09, -.16, -.49]
NAMES = [f"right_{name}_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)]


class ReadyHooks(Hooks):
    def begin_ready_reset(self, feedback, right_arm_target, right_arm_waypoint):
        return StandingReset(feedback, feedback["left_hand_q_measured"], np.zeros(7),
                             right_arm_target=right_arm_target, right_arm_waypoint=right_arm_waypoint)


def make_ready(tmp_path, *, verified=True, reviewed=True, hooks=None, initially_raised=False):
    data = yaml.safe_load(PROFILE.read_text())
    data["skills"] = {"bottle_handover": dict(prompt="test-only handover",
        completion_criteria="Stable offer", completion_action="hold")}
    data["handover"] = dict(skill_id="bottle_handover", checkpoint_verified=verified,
        ready_pose_reviewed=reviewed, ready_right_arm_joints=dict(zip(NAMES, TARGET)),
        raised_right_arm_joints=dict(zip(NAMES, RAISED)))
    path = tmp_path / "ready.yaml"
    path.write_text(yaml.safe_dump(data))
    c = HarnessControl(path, hooks or ReadyHooks(), "boot")
    initial = state()
    if initially_raised:
        initial["body_q_measured_motor"][22:29] = RAISED
    c.tick(0., initial, 0.)
    c.observation = {"received_at": 0., "frame_id": "first"}
    assert request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})["error"] is None
    return c


def test_ready_preserves_other_arm_waist_heading_and_left_hand():
    s = state()
    s["body_q_measured_motor"] = np.linspace(-.1, .1, 29).tolist()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET)
    upper = np.asarray(s["body_q_measured_motor"])[G1_UPPER_BODY_JOINT_INDICES]
    # Check actual motor semantics, independently of the reset's vector layout.
    target_body = np.asarray(s["body_q_measured_motor"]).copy()
    target_body[G1_UPPER_BODY_JOINT_INDICES] = reset.target[:17]
    assert target_body[:22] == pytest.approx(s["body_q_measured_motor"][:22])
    assert target_body[22:29] == pytest.approx(TARGET)
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
    c = make_ready(tmp_path, initially_raised=True)
    result = request(c, "reset_ready", {"execution_id": ""})
    assert result["error"] is None
    assert c.phase == "RESETTING"
    # Exercise the ramp with synthetic tracking before testing native dwell.
    s = {**c.feedback}
    track_reset(c.reset, s)
    c.tick(.1, s, .1)
    c.tick(.59, {**s, "index":s["index"] + 1}, .59)
    assert c.phase == "RESETTING"
    c.tick(.61, {**s, "index":s["index"] + 2}, .61)
    assert c.phase == "COMPLETED" and c.status(.61)["right_hand_open"] is True
    c.tick(1.2, None, None)
    assert c.status(1.2)["right_hand_open"] is None


def test_handover_claim_requests_measured_hold_before_camera_checks(tmp_path):
    c = make_ready(tmp_path)
    assert len(c.hooks.holds) == 1
    assert c.hooks.holds[0][1] is False
    assert c.phase == "IDLE" and c.reset is None
    c.tick(.1, state(2), .1)
    assert c.hold_confirmed
    assert request(c, "release_control", now=.1)["error"] is None


@pytest.mark.parametrize("boundary,drift", [(1.97222,-.1), (-1.97222,.1)])
def test_ready_tracking_compensation_cannot_cross_joint_bound(boundary, drift):
    target = list(TARGET)
    target[4] = boundary
    s = state()
    body = np.asarray(s["body_q_measured_motor"])
    body[22:29] = target
    s["body_q_measured_motor"] = body.tolist()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7),
                          right_arm_target=target, compensate_tracking_bias=True)
    body[26] += drift  # Native right_wrist_roll motor.
    s["body_q_measured_motor"] = body.tolist()
    for _ in range(30):
        command = reset.advance(s, .05)
        commanded_body = body.copy()
        commanded_body[G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        assert -1.97222 <= commanded_body[26] <= 1.97222


@pytest.mark.parametrize("direction", [-1, 1])
def test_ready_rejects_outward_drift_when_joint_and_lead_intervals_do_not_intersect(direction):
    target = list(TARGET)
    target[4] = direction * 1.97222
    s = state()
    s["body_q_measured_motor"][22:29] = target
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=target)
    before = reset.command
    s["body_q_measured_motor"][26] += direction * .155
    with pytest.raises(ValueError, match="joint and lead bounds"):
        reset.advance(s, .02)
    assert reset.command == before


def lift_reset(*, initially_raised=True):
    s = state()
    if initially_raised:
        s["body_q_measured_motor"][22:29] = RAISED
    return StandingReset(s, s["left_hand_q_measured"], np.zeros(7),
                         right_arm_target=TARGET, right_arm_waypoint=RAISED), s


def track_reset(reset, s):
    for _ in range(250):
        s["index"] += 1
        command = reset.advance(s, .03)
        body = np.asarray(s["body_q_measured_motor"])
        body[G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        s["body_q_measured_motor"] = body.tolist()
        s["left_hand_q_measured"] = command.left_hand_position
        s["right_hand_q_measured"] = command.right_hand_position


def test_raise_retains_fingers_until_distinct_feedback_settles():
    reset, s = lift_reset()
    s["body_q_measured_motor"][22:29] = RAISED
    for index in range(2, 11):
        s["index"] = index
        reset.advance(s, .05)
        assert reset.stage == 0
        assert reset.command.right_hand_position == pytest.approx([.4] * 7)
    for _ in range(20):
        reset.advance(s, .05)
        assert reset.stage == 0
    s["index"] = 11
    reset.advance(s, .05)
    # Allow floating-point summation at exactly the dwell boundary.
    for index in (12, 13):
        s["index"] = index
        reset.advance(s, .05)
    assert reset.stage == 1
    assert reset.command.right_hand_position == pytest.approx([.4] * 7)
    assert not reset.is_settled(s, .05, .1)
    s["index"] = 14
    assert reset.advance(s, .05).right_hand_position == pytest.approx([.375] * 7)
    track_reset(reset, s)
    s["right_hand_q_measured"] = [.4] * 7
    assert not reset.is_settled(s, .05, .1)
    s["right_hand_q_measured"] = [0] * 7
    assert reset.is_settled(s, .05, .1)


def test_raise_first_qualifying_sample_does_not_credit_unsettled_interval():
    reset, s = lift_reset()
    s["body_q_measured_motor"][24] += .1
    reset.advance(s, .025)  # The preceding sample is outside raise tolerance.
    s["body_q_measured_motor"][22:29] = RAISED
    for index in range(2, 22):
        s["index"] = index
        reset.advance(s, .025)
    # Twenty samples span only nineteen intervals: .475 s since first settled.
    assert reset.stage == 0
    s["index"] = 22
    reset.advance(s, .025)
    assert reset.stage == 1


@pytest.mark.parametrize("bad", [None, {"index": True}, {"index": -1},
                                 {"index": 2}, {"index": 1}])
def test_missing_or_regressed_raise_feedback_holds_and_clears_dwell(bad):
    reset, s = lift_reset()
    s["body_q_measured_motor"][22:29] = RAISED
    for index in range(2, 11):
        s["index"] = index
        reset.advance(s, .05)
    before = reset.command
    assert reset.advance(bad, .05) == before
    for index in range(11, 14):
        s["index"] = index
        reset.advance(s, .05)
        assert reset.stage == 0


def test_raise_dwell_requires_heading_and_upper_body_but_not_finger_settling():
    reset, s = lift_reset()
    s["body_q_measured_motor"][22:29] = RAISED
    s["right_hand_q_measured"] = [.3] * 7
    s["base_quat"] = [np.cos(.2), 0, 0, np.sin(.2)]
    for index in range(2, 25):
        s["index"] = index
        reset.advance(s, .05)
    assert reset.stage == 0
    s["base_quat"] = [1, 0, 0, 0]
    for index in range(25, 38):
        s["index"] = index
        reset.advance(s, .05)
    assert reset.stage == 1


def test_lift_transition_keeps_rate_lead_and_preserved_joint_limits():
    reset, s = lift_reset(initially_raised=False)
    previous = np.r_[reset.command.upper_body_position,
                     reset.command.left_hand_position, reset.command.right_hand_position]
    for index in range(2, 250):
        s["index"] = index
        measured = np.r_[np.asarray(s["body_q_measured_motor"])[G1_UPPER_BODY_JOINT_INDICES],
                         s["left_hand_q_measured"], s["right_hand_q_measured"]]
        command = reset.advance(s, .05)
        current = np.r_[command.upper_body_position, command.left_hand_position, command.right_hand_position]
        assert np.max(np.abs(current - previous)) <= .02500001
        assert np.max(np.abs(current - measured)) <= .15000001
        body = np.asarray(s["body_q_measured_motor"])
        body[G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        assert body[:22] == pytest.approx([0] * 22)
        assert command.left_hand_position == pytest.approx([.3] * 7)
        assert tuple(command.movement) == (0., 0., 0.)
        s["body_q_measured_motor"] = body.tolist()
        s["right_hand_q_measured"] = command.right_hand_position
        previous = current
    assert reset.stage == 1 and reset.is_settled(s, .05, .1)


@pytest.mark.parametrize("waypoint", [[0] * 6, [0, 0, 0, 0, 3, 0, 0], [float("nan")] * 7])
def test_lift_rejects_invalid_waypoint(waypoint):
    s = state()
    with pytest.raises(ValueError):
        StandingReset(s, s["left_hand_q_measured"], np.zeros(7),
                      right_arm_target=TARGET, right_arm_waypoint=waypoint)


def test_delayed_requests_from_an_ended_session_cannot_regain_control(tmp_path):
    c = make_ready(tmp_path)  # Session "s" claimed with request "claim_control-0.0".
    c.tick(.1, state(2), .1)
    lease = c.owner["lease_id"]
    assert request(c, "heartbeat", {"sequence": 1}, now=.1)["error"] is None
    release = request(c, "release_control", now=.1)
    assert release["error"] is None and c.owner is None
    # The exact retry of the request that ended the session still gets its reply.
    assert request(c, "release_control", now=.1, lease=lease) == release
    # Delayed claim, heartbeat and motion replays are rejected without a cache lookup.
    claim = {"registry_sha256": c.profile.registry_sha256}
    assert request(c, "claim_control", claim, now=0., rid="claim_control-0.0")["error"]["code"] == "NOT_OWNER"
    assert request(c, "heartbeat", {"sequence": 1}, now=.2, lease=lease)["error"]["code"] == "NOT_OWNER"
    assert request(c, "heartbeat", {"sequence": 2}, now=.2, lease=lease)["error"]["code"] == "NOT_OWNER"
    assert c.owner is None and len(c.hooks.holds) == 1
    assert not [key for key in c.replies if key[0] == "s"] and c.heartbeat is None
    # A new session can claim the same executor.
    assert request(c, "claim_control", claim, now=.3, session="s2")["error"] is None
    assert c.owner["session_id"] == "s2"


def test_lease_expiry_retires_the_session(tmp_path):
    c = make_ready(tmp_path)
    c.tick(.1, state(2), .1)
    lease = c.owner["lease_id"]
    c.tick(2.5, state(3), 2.5)
    assert c.owner is None
    claim = {"registry_sha256": c.profile.registry_sha256}
    assert request(c, "claim_control", claim, now=2.6, rid="claim_control-0.0")["error"]["code"] == "NOT_OWNER"
    assert request(c, "heartbeat", {"sequence": 1}, now=2.6, lease=lease)["error"]["code"] == "NOT_OWNER"
