import numpy as np
import pytest
from test_harness_control import PROFILE, Hooks, request, state
import yaml

from gear_sonic.utils.inference.harness_control import HarnessControl
from gear_sonic.utils.inference.standing_reset import StandingReset, _WRIST_PLANNER_INDICES
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

TARGET = [-.1, -.08, .3, -.21, 1.7, -.09, -.07]
RAISED = [.011, -.68, -.60, .16, 1.09, -.16, -.49]
# Ready returns allow wrists twice the shoulder/elbow/finger lead.
READY_LEAD = np.full(31, .15)
READY_LEAD[_WRIST_PLANNER_INDICES] = .30
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
        assert np.all(np.abs(current - np.r_[upper, [.3]*7, [.4]*7]) <= READY_LEAD + 1e-7)
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
    s["body_q_measured_motor"][26] += direction * .305  # Beyond the 0.30 rad ready wrist lead.
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


LEFT_READY = [.137, .309, -.438, .700]  # Shoulder pitch/roll/yaw, elbow.
LEFT_NAMES = [name.replace("right_", "left_") for name in NAMES[:4]]
LEFT_INDICES = [G1_UPPER_BODY_JOINT_INDICES.index(m) for m in range(15, 19)]
LEFT_WRIST_INDICES = [G1_UPPER_BODY_JOINT_INDICES.index(m) for m in range(19, 22)]
RIGHT_INDICES = [G1_UPPER_BODY_JOINT_INDICES.index(m) for m in range(22, 29)]


@pytest.mark.parametrize("waypoint", [None, RAISED])
def test_left_ready_target_moves_shoulder_and_elbow_and_holds_waist_and_wrist(waypoint):
    s = state()
    s["body_q_measured_motor"] = np.linspace(-.1, .1, 29).tolist()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET,
                          right_arm_waypoint=waypoint, left_arm_target=LEFT_READY)
    measured = np.asarray(s["body_q_measured_motor"])[G1_UPPER_BODY_JOINT_INDICES]
    for target in (reset.target, reset._target):  # Final, and the raise stage when present.
        np.testing.assert_allclose(target[LEFT_INDICES], LEFT_READY)
        np.testing.assert_allclose(target[:3], measured[:3])  # Waist stays as measured.
        np.testing.assert_allclose(target[LEFT_WRIST_INDICES], measured[LEFT_WRIST_INDICES])  # Left wrist too.
    np.testing.assert_allclose(reset.target[RIGHT_INDICES], TARGET)


def test_left_arm_ready_target_is_rate_and_lead_limited():
    s = state()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET,
                          left_arm_target=LEFT_READY)
    command = reset.advance(dict(s, index=2), 0.02)
    left = np.asarray(command.upper_body_position)[LEFT_INDICES]
    assert np.all(np.abs(left) <= 0.5 * 0.02 + 1e-9)
    assert np.any(np.abs(left) > 0)


@pytest.mark.parametrize("kwargs", [
    dict(left_arm_target=LEFT_READY),  # Without a right-arm ready target.
    dict(right_arm_target=TARGET, left_arm_target=[3.0] + LEFT_READY[1:]),
    dict(right_arm_target=TARGET, left_arm_target=LEFT_READY[:3]),
    dict(right_arm_target=TARGET, left_arm_target=LEFT_READY + [0.0, 0.0, 0.0]),  # Wrists cannot be set.
])
def test_left_arm_ready_target_rejects_invalid_targets(kwargs):
    s = state()
    with pytest.raises(ValueError):
        StandingReset(s, s["left_hand_q_measured"], np.zeros(7), **kwargs)


def test_profile_left_arm_ready_pose_is_optional_and_validated(tmp_path):
    c = make_ready(tmp_path)
    assert c.profile.handover.left_arm_target is None
    data = yaml.safe_load((tmp_path / "ready.yaml").read_text())
    data["handover"]["ready_left_arm_joints"] = dict(zip(LEFT_NAMES, LEFT_READY))
    good = tmp_path / "left.yaml"
    good.write_text(yaml.safe_dump(data))
    from gear_sonic.utils.inference.harness_profile import load_profile
    assert load_profile(good).handover.left_arm_target == tuple(LEFT_READY)
    wrists = {n.replace("right_", "left_"): 0.0 for n in NAMES[4:]}
    for bad in (dict(zip(NAMES, LEFT_READY)), dict(zip(LEFT_NAMES, [9.0] * 4)), dict(zip(LEFT_NAMES[:3], LEFT_READY)),
                {**dict(zip(LEFT_NAMES, LEFT_READY)), **wrists}):
        data["handover"]["ready_left_arm_joints"] = bad
        (tmp_path / "bad.yaml").write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_profile(tmp_path / "bad.yaml")


def test_ready_settles_a_wrist_that_lags_beyond_the_shoulder_lead():
    # sweep2-0.20: SONIC held the right wrist roll 0.15+ rad behind the reference;
    # a 0.15 rad lead pinned the reference there and the return timed out.
    s = state()
    s["body_q_measured_motor"][22:29] = RAISED
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET,
                          compensate_tracking_bias=True)
    lag = .2  # Measured right wrist roll stays this far above the command.
    for _ in range(1500):
        command = reset.advance(s, .02)
        body = np.asarray(s["body_q_measured_motor"])
        body[G1_UPPER_BODY_JOINT_INDICES] = command.upper_body_position
        body[26] += lag
        s["body_q_measured_motor"] = body.tolist()
        s["right_hand_q_measured"] = command.right_hand_position
        if reset.is_settled(s, .05, .1):
            break
    assert reset.is_settled(s, .05, .1)
    assert s["body_q_measured_motor"][26] == pytest.approx(TARGET[4], abs=.05)


def test_standing_reset_keeps_the_shoulder_lead_for_wrists():
    s = state()
    s["body_q_measured_motor"][26] = 1.0  # Straight standing returns this wrist toward 0.
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7))
    for _ in range(100):
        command = reset.advance(s, .02)
    wrist = np.asarray(command.upper_body_position)[G1_UPPER_BODY_JOINT_INDICES.index(26)]
    assert 1.0 - wrist == pytest.approx(.15)


@pytest.mark.parametrize("left_offset,right_offset,settled", [
    (.14, 0., True), (.16, 0., False), (0., .06, False)])
def test_ready_settles_the_held_left_wrist_within_its_own_tolerance(left_offset, right_offset, settled):
    s = state()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET,
                          left_arm_target=LEFT_READY)
    body = np.asarray(s["body_q_measured_motor"])
    body[G1_UPPER_BODY_JOINT_INDICES] = reset.target[:17]
    body[19] += left_offset  # Held left wrist roll.
    body[26] += right_offset  # Targeted right wrist roll.
    s["body_q_measured_motor"] = body.tolist()
    s["left_hand_q_measured"], s["right_hand_q_measured"] = reset.target[17:24], reset.target[24:]
    assert reset.is_settled(s, .05, .1) is settled


@pytest.mark.parametrize("motor,bound", [
    (15, 2.6704),  # Left shoulder pitch, a ready target joint (review of 37a742a).
    (19, 1.97222),  # Left wrist roll, held with the 0.30 rad ready wrist trim.
    (19, -1.97222),
])
def test_left_arm_compensation_never_commands_past_its_joint_limit(motor, bound):
    s = state()
    body = np.asarray(s["body_q_measured_motor"])
    left = list(LEFT_READY)
    if motor == 15:
        left[0] = bound
    body[15:19] = left
    body[motor] = bound
    body[22:29] = TARGET
    s["body_q_measured_motor"] = body.tolist()
    reset = StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET,
                          left_arm_target=left, compensate_tracking_bias=True)
    body[motor] = bound - np.sign(bound) * .1  # Steady 0.1 rad tracking lag.
    s["body_q_measured_motor"] = body.tolist()
    index = G1_UPPER_BODY_JOINT_INDICES.index(motor)
    for _ in range(40):
        commanded = reset.advance(s, .05).upper_body_position[index]
        assert abs(commanded) <= abs(bound) + 1e-12


@pytest.mark.parametrize("motor,value", [(15, 2.7), (16, -1.6), (19, 2.0), (21, -1.7)])
def test_measured_left_arm_beyond_joint_limits_is_rejected(motor, value):
    s = state()
    s["body_q_measured_motor"][motor] = value
    with pytest.raises(ValueError, match="Measured arm exceeds G1 joint limits"):
        StandingReset(s, s["left_hand_q_measured"], np.zeros(7), right_arm_target=TARGET)
