import itertools
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_handover_ready import NAMES, RAISED, TARGET, ReadyHooks
from test_harness_control import PROFILE, request, state
import yaml

from gear_sonic.utils.inference.harness_contract import decode_response
from gear_sonic.utils.inference.harness_control import ControlError, HarnessControl
from gear_sonic.utils.inference.lerobot_sonic_policy import TokenStream
from gear_sonic.utils.inference.loop_control import LoopController
from gear_sonic.utils.inference.loop_motion import PoseCommander, ReadyPose
from gear_sonic.utils.inference.loop_records import TrialWriter, next_trial

TICK = 0.02
PLANNER = np.full(64, -0.2, np.float32)
_ROWS = np.arange(83, dtype=np.float32)[:, None]
POSE = ReadyPose(
    token=np.full(64, 0.9, np.float32),
    hands=np.full(14, 0.3, np.float32),
    state=np.zeros(28, np.float32),
    raise_tokens=np.tile(_ROWS / 100, (1, 64)),
    raise_hands=np.tile(_ROWS / 1000, (1, 14)),
    lower_tokens=np.tile(1.0 - _ROWS / 100, (1, 64)),
    lower_hands=np.tile(0.2 - _ROWS / 1000, (1, 14)),
)


def chunk(offset=0.0):
    rows = np.zeros((40, 78), np.float32)
    rows[:, :64] = 0.85 + offset + np.arange(40)[:, None] * 1e-3
    return rows


SERVED = yaml.safe_load(PROFILE.read_text())["checkpoint"]


class LoopHooks(ReadyHooks):
    def __init__(self):
        super().__init__()
        self.calls, self.requests = [], []
        self.served_model = SERVED  # what the policy server's info reports

    def runtime_facts(self):
        return SimpleNamespace(**vars(super().runtime_facts()), served_model=self.served_model)

    def switch_to_pose(self):
        self.calls.append("pose")

    def switch_to_planner(self):
        self.calls.append("planner")

    def request_policy_chunk(self):
        self.requests.append(100 + len(self.requests))
        self.calls.append("chunk")
        return self.requests[-1]


class SpyStream(TokenStream):
    """Records, at each set_chunk, the stream's and the commander's last token."""

    def __init__(self, sink, commander):
        super().__init__(50, blend_s=0.3, max_token_step=0.0, anchor_s=1.0)
        self.sink, self.commander = sink, commander

    def set_chunk(self, rows, t0, now):
        mine = None if self.last_token is None else self.last_token.copy()
        self.sink.append((mine, self.commander.last_token.copy(), t0, now))
        super().set_chunk(rows, t0, now)


class Rig:
    """A LoopController ticked at 50 Hz with scripted feedback; answers chunk requests after `answer_after`
    ticks with t0 = the request time (None: never)."""

    def __init__(self, hooks=None):
        self.hooks = hooks or LoopHooks()
        self.cmd = PoseCommander(0.0)
        self.set_chunks = []
        self.loop = LoopController(
            self.cmd, POSE, self.hooks, make_stream=lambda: SpyStream(self.set_chunks, self.cmd),
            right_order=np.arange(7), trial_args={"test": True}, model="test-model")
        self.i, self.right, self.state_age, self.camera_age = 0, 0.0, 0.0, 0.0
        self.body = np.zeros(29)
        self.torque = None  # motor_torque (29, hardware order) in the feedback, when set
        self.sent = []
        self.answer_after, self.asked, self.answered = 1, {}, set()
        self.rows = lambda epoch: chunk((epoch - 100) * 0.01)  # None: a failed inference

    @property
    def now(self):
        return self.i * TICK

    def feedback(self):
        token = self.sent[-1][0] if self.sent else PLANNER
        msg = dict(index=self.i, body_q=self.body.copy(), left_hand_q=np.zeros(7),
                   right_hand_q=np.full(7, self.right), token_state=token)
        if self.torque is not None:
            msg["motor_torque"] = self.torque.copy()
        return msg

    def step(self, n=1):
        for _ in range(n):
            self.i += 1
            token, hands, label = self.loop.tick(self.now, self.feedback(), self.state_age, self.camera_age)
            self.sent.append((token, hands, label, self.loop.phase, self.loop.stage))
            for epoch in self.hooks.requests:
                self.asked.setdefault(epoch, (self.now, self.i))
            if self.answer_after is not None:
                for epoch, (at, i) in self.asked.items():
                    if epoch not in self.answered and self.i - i >= self.answer_after:
                        self.answered.add(epoch)
                        self.loop.accept_chunk(epoch, at, self.rows(epoch))
        return self

    def until(self, stage, limit=500):
        for _ in range(limit):
            if self.loop.stage == stage:
                return self
            self.step()
        raise AssertionError(f"stage {stage} not reached; at {self.loop.stage}")

    def labels(self, start=0):
        return [label for _, _, label, _, _ in self.sent[start:]]

    def take_over(self):
        self.loop.take_over(self.feedback(), self.now)
        return self.step(50)

    def ready(self):
        self.take_over()
        self.loop.raise_(self.now)
        return self.step(50 + 83 + 50)


def runs(labels):
    return [(k, len(list(g))) for k, g in itertools.groupby(labels)]


def trial_npz(tmp_path, trial=0):
    assert TrialWriter.flush(10)
    with np.load(tmp_path / f"trial_{trial:03d}.npz") as npz:
        return {k: npz[k] for k in npz.files}


def test_take_over_holds_planner_then_switches_to_pose():
    rig = Rig()
    with pytest.raises(ControlError):
        rig.loop.take_over(dict(rig.feedback(), token_state=[0.0] * 3), 0.0)
    rig.loop.take_over(rig.feedback(), rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "stance")
    rig.step(49)
    assert rig.hooks.calls == []
    rig.step(1)
    assert rig.hooks.calls == ["pose"]
    assert runs(rig.labels()) == [("planner token", 50)]
    assert all(np.array_equal(t, PLANNER) and not h.any() for t, h, *_ in rig.sent)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "stance")
    rig.step(3)
    assert rig.labels(50) == ["hold"] * 3 and np.array_equal(rig.sent[-1][0], PLANNER)


def test_raise_plays_blend_path_blend_to_ready():
    rig = Rig().take_over()
    rig.loop.raise_(rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "raise")
    rig.step(50 + 83 + 50)
    assert runs(rig.labels(50)) == [("to raise start", 50), ("raise", 83), ("to ready", 50)]
    raised = [t for t, _, label, _, _ in rig.sent if label == "raise"]
    assert np.array_equal(np.array(raised), POSE.raise_tokens)
    assert np.allclose(rig.sent[-1][0], POSE.token) and np.allclose(rig.sent[-1][1], POSE.hands)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "ready")


def test_trial_sequence_and_handover_end(tmp_path):
    rig = Rig().ready()
    start = len(rig.sent)
    rig.loop.trial(0, str(tmp_path), rig.now)
    assert (rig.loop.phase, rig.loop.stage, rig.loop.trial_no) == ("LOOP_TRIAL", "hold_rest", 0)
    rig.until("episode").step(25)  # open
    rig.right = 0.3
    rig.step(30)  # closed 0.6 s
    rig.right = 0.0
    for _ in range(55):  # open 1.1 s; the detector fires after 1.0 s
        rig.step()
        if rig.loop.stage == "await_take":
            break
    assert rig.loop.trial_end_reason == "handover complete"
    assert rig.loop.phase == "LOOP_HOLD"  # R24: hold the open palm until the bottle is taken
    held = len(rig.sent) - 1  # the end tick already holds the last episode token and hands
    rig.step(100)
    assert rig.labels(held) == ["hold"] * 101 and rig.labels(held - 1)[0] == "episode"
    assert all(np.array_equal(t, rig.sent[held - 1][0]) and np.array_equal(h, rig.sent[held - 1][1])
               for t, h, *_ in rig.sent[held:])
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "await_take")
    seq = runs(rig.labels(start))
    assert seq[0] == ("wait first chunk", 76) and seq[1] == ("blend in", 50)
    assert seq[2][0] == "episode" and seq[3] == ("hold", 101)
    npz = trial_npz(tmp_path)  # closed at the episode end: the hold is not recorded
    assert str(npz["episode_end"]) == "handover complete" and str(npz["model"]) == "test-model"
    assert runs(npz["phase"].tolist()) == seq[:3]
    rig.loop.to_ready(rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "to_ready")
    rig.step(50)
    assert runs(rig.labels(-50)) == [("to ready", 50)]
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "ready")
    episode = npz["phase"] == "episode"
    assert (npz["frame"][episode] == (npz["t_ep"][episode] * 50).astype(np.int64)).all()  # the policy rate
    assert npz["frame"][episode].max() >= 50
    assert npz["t_ep"][episode][0] == 0.0 and (npz["t_ep"][~episode & (npz["phase"] != "blend in")] == -1).all()


def test_stop_during_trial_freezes_and_ends_trial(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode").step(10)
    held, n_requests = rig.sent[-1][0], len(rig.hooks.requests)
    rig.loop.freeze("operator stop", rig.now)
    assert (rig.loop.phase, rig.loop.stage, rig.loop.frozen_stage) == ("LOOP_FROZEN", "frozen", "episode")
    assert rig.loop.trial_end_reason == "operator stop"
    rig.step(30)
    assert rig.labels(-30) == ["frozen"] * 30
    assert all(np.array_equal(t, held) for t, *_ in rig.sent[-30:])
    assert len(rig.hooks.requests) == n_requests  # no replans while frozen
    assert not rig.loop.accept_chunk(rig.hooks.requests[-1], rig.now, chunk())
    npz = trial_npz(tmp_path)
    assert str(npz["episode_end"]) == "operator stop" and "to ready" not in npz["phase"].tolist()
    rig.loop.resume(rig.now)
    assert (rig.loop.phase, rig.loop.stage, rig.loop.frozen_stage) == ("LOOP_MOTION", "to_ready", None)
    rig.step(50)
    assert runs(rig.labels(-50)) == [("to ready", 50)]
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "ready")


def test_stale_state_freezes_not_blends(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode").step(5)
    rig.state_age = 0.25
    rig.step(20)
    assert rig.loop.phase == "LOOP_FROZEN" and rig.loop.frozen_stage == "episode"
    assert rig.loop.reason == rig.loop.trial_end_reason == "robot state 0.25 s old"
    assert rig.labels(-20) == ["frozen"] * 20  # the freeze tick already holds; no blend to ready
    assert str(trial_npz(tmp_path)["episode_end"]) == "robot state 0.25 s old"


def make_hc(tmp_path, *, verified=True, anchor=1.0, with_loop=True, right_order="dataset"):
    data = yaml.safe_load(PROFILE.read_text())
    data["skills"] = {"bottle_handover": dict(prompt="test-only handover",
                                              completion_criteria="Stable offer", completion_action="hold")}
    data["handover"] = dict(skill_id="bottle_handover", checkpoint_verified=verified, ready_pose_reviewed=True,
                            ready_right_arm_joints=dict(zip(NAMES, TARGET)),
                            raised_right_arm_joints=dict(zip(NAMES, RAISED)))
    data["lerobot_policy"] = dict(chunk_blend_s=0.3, max_token_step=0.05, chunk_anchor_s=anchor,
                                  dex3_right_order=right_order)
    path = tmp_path / "loop.yaml"
    path.write_text(yaml.safe_dump(data))
    rig = Rig()
    c = HarnessControl(path, rig.hooks, "boot", loop=rig.loop if with_loop else None)
    c.tick(0.0, hc_state(1), 0.0)
    c.observation = {"received_at": 0.0, "frame_id": "first"}
    assert request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256})["error"] is None
    return c, rig


def hc_state(index):
    return dict(state(index), token_state=PLANNER.tolist())


def test_stray_native_key_in_loop_freezes_without_its_action(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    holds, epoch, owner = len(rig.hooks.holds), c.epoch, dict(c.owner)
    assert c.operator_override("p") is True
    assert c.phase == "LOOP_FROZEN" and rig.loop.reason == "operator override: p"
    assert c.owner == owner and len(rig.hooks.holds) == holds and c.epoch == epoch and not rig.hooks.enabled
    assert c.status(0.1)["reason"] == "operator override: p"
    assert c.operator_override("k") is True and c.phase == "LOOP_FROZEN" and c.owner == owner
    assert rig.loop.reason == "operator override: p"  # already frozen: the first reason stays
    assert request(c, "loop_resume", now=0.2)["error"] is None
    assert c.phase == "LOOP_HOLD" and rig.loop.stage == "stance"


def test_lease_expiry_in_loop_freezes_and_allows_reclaim(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    c.tick(1.0, hc_state(2), 1.0)
    holds = len(rig.hooks.holds)
    c.tick(2.5, hc_state(3), 2.5)  # lease_s 2.0
    assert c.owner is None and c.phase == "LOOP_FROZEN"
    assert rig.loop.reason == "lease expired" and rig.loop.frozen_stage == "stance"
    assert len(rig.hooks.holds) == holds and rig.hooks.calls == ["pose"]
    status = c.status(2.5)
    assert (status["loop_stage"], status["frozen_stage"]) == ("frozen", "stance")
    decode_response(json.dumps(dict(request_id="r", runtime_id="boot", result=status, error=None)).encode())
    claim = request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, now=2.6, session="s2")
    assert claim["error"] is None
    assert c.phase == "LOOP_FROZEN" and rig.loop.frozen_stage == "stance" and len(rig.hooks.holds) == holds
    assert request(c, "loop_resume", now=2.7, session="s2")["error"] is None
    assert c.phase == "LOOP_HOLD" and c.status(2.7)["loop_stage"] == "stance"


def test_telemetry_restart_in_loop_freezes_and_retires_owner(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    c.tick(0.2, hc_state(5), 0.2)
    holds = len(rig.hooks.holds)
    c.tick(0.3, hc_state(2), 0.3)  # index went back: the controller restarted
    assert c.owner is None and c.phase == "LOOP_FROZEN" and len(rig.hooks.holds) == holds
    assert rig.loop.reason == "interrupted: telemetry_stream_restarted"


def test_resume_mid_raise_continues_raise():
    rig = Rig().take_over()
    rig.loop.raise_(rig.now)
    rig.step(50 + 40)  # raise rows 0..39 sent
    rig.loop.freeze("operator stop", rig.now)
    assert rig.loop.frozen_stage == "raise"
    rig.step(5)
    assert all(np.array_equal(t, POSE.raise_tokens[39]) for t, *_ in rig.sent[-5:])
    rig.loop.resume(rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "raise")
    start = len(rig.sent)
    rig.step(43 + 50)
    assert np.array_equal(rig.sent[start][0], POSE.raise_tokens[40])
    assert runs(rig.labels(start)) == [("raise", 43), ("to ready", 50)]
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "ready")


def test_resume_mid_lowering_finishes_to_planner_and_completes():
    rig = Rig().ready()
    rig.loop.end(rig.now)
    rig.step(50 + 41)  # lowering rows 0..40 sent
    rig.loop.freeze("operator stop", rig.now)
    assert rig.loop.frozen_stage == "lower" and rig.hooks.calls == ["pose"]
    rig.step(3)
    rig.loop.resume(rig.now)
    start = len(rig.sent)
    rig.step(42 + 50 + 100 + 100)
    assert np.array_equal(rig.sent[start][0], POSE.lower_tokens[41])
    assert runs(rig.labels(start)) == [
        ("lower", 42), ("blend out", 50), ("hold rest", 100), ("handback blend", 100)]
    assert rig.hooks.calls == ["pose", "planner"] and rig.loop.phase == "COMPLETED"


def test_resume_mid_handback_completes():
    rig = Rig().ready()
    rig.loop.end(rig.now)
    rig.step(50 + 83 + 50 + 100 + 30)
    assert rig.loop.stage == "handback"
    rig.loop.freeze("operator stop", rig.now)
    assert rig.loop.frozen_stage == "handback"
    rig.step(3)
    rig.loop.resume(rig.now)
    start = len(rig.sent)
    rig.step(70)
    assert runs(rig.labels(start)) == [("handback blend", 70)]
    assert rig.hooks.calls == ["pose", "planner"] and rig.loop.phase == "COMPLETED"
    assert np.allclose(rig.sent[-1][0], PLANNER) and not rig.sent[-1][1].any()


def test_first_chunk_bootstrap_requested_after_hold_rest(tmp_path):
    rig = Rig().ready()
    rig.answer_after = None
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.step(74)
    assert rig.hooks.requests == [] and rig.loop.stage == "hold_rest"
    rig.step(1)
    assert rig.hooks.requests == [100] and rig.loop.stage == "first_chunk"
    rig.step(30)
    assert rig.hooks.requests == [100] and rig.labels(-105) == ["wait first chunk"] * 105
    assert all(np.allclose(t, POSE.token) and np.allclose(h, POSE.hands) for t, h, *_ in rig.sent[-105:])
    assert not rig.loop.accept_chunk(101, rig.now, chunk()) and not rig.loop.accept_chunk(99, rig.now, chunk())
    assert rig.loop.stage == "first_chunk"
    assert rig.loop.accept_chunk(100, rig.now, chunk())
    rig.step(1)
    assert rig.labels(-1) == ["blend in"] and rig.loop.stage == "blend_in"
    assert not rig.loop.accept_chunk(100, rig.now, chunk())


def test_invalid_policy_result_freezes(tmp_path):
    rig = Rig().ready()
    rig.answer_after = None
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.step(75)
    bad = chunk()
    bad[3, 5] = np.nan
    assert not rig.loop.accept_chunk(100, rig.now, bad)
    assert rig.loop.phase == "LOOP_FROZEN" and rig.loop.reason == "invalid policy result"


def test_replan_one_outstanding_and_t0_is_the_request_time(tmp_path):
    rig = Rig().ready()
    rig.answer_after = None
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.step(75)
    assert rig.loop.accept_chunk(100, rig.now, chunk())
    rig.until("episode")
    episode_start = rig.now
    assert rig.set_chunks[-1][2:] == (episode_start, episode_start)  # the first chunk's t0 = the episode start
    rig.step(45)  # 0.9 s: the 0.4 s replan was requested once and is still outstanding
    assert rig.hooks.requests == [100, 101]
    assert rig.asked[101][0] == pytest.approx(episode_start + 0.4, abs=TICK + 1e-9)
    assert not rig.loop.accept_chunk(100, rig.now, chunk(0.01))  # stale epoch
    captured = rig.asked[101][0] - 0.05
    assert rig.loop.accept_chunk(101, captured, chunk(0.01))
    assert not rig.loop.accept_chunk(101, captured, chunk(0.01))  # no longer outstanding
    rig.step(1)
    # lerobot: t0 = slot_t, the request time; captured_at (the camera frame time) only gates validity (R23a)
    assert rig.set_chunks[-1][2:] == (rig.asked[101][0], rig.now)
    assert rig.hooks.requests == [100, 101, 102]  # due again: request time + 0.4 has passed
    assert rig.loop.stage == "episode"


def test_anchoring_reference_is_the_commanders_last_token(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode").step(60)
    assert len(rig.set_chunks) >= 3
    for mine, commander, _, _ in rig.set_chunks:  # R1b: set before every set_chunk, the first one included
        assert mine is not None and np.array_equal(mine, commander)
    for _, _, _, now in rig.set_chunks[1:]:  # anchoring: the tick a new chunk lands continues the last token
        k = round(now / TICK) - 1  # rig.sent[k] was sent at (k + 1) * TICK
        assert rig.sent[k][2] == "episode"
        assert np.allclose(rig.sent[k][0], rig.sent[k - 1][0], atol=1e-6)


def test_end_lowers_and_hands_back_to_planner():
    rig = Rig().ready()
    start = len(rig.sent)
    rig.loop.end(rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "lower")
    rig.step(50 + 83 + 50 + 100 + 99)
    assert rig.hooks.calls == ["pose"]
    rig.step(1)
    assert rig.hooks.calls == ["pose", "planner"] and rig.loop.phase == "COMPLETED"
    assert runs(rig.labels(start)) == [("to lower start", 50), ("lower", 83), ("blend out", 50),
                                       ("hold rest", 100), ("handback blend", 100)]
    stages = [stage for *_, stage in rig.sent[start:-1]]
    assert runs(stages) == [("lower", 283), ("handback", 99)]
    hold_rest = [t for t, h, label, _, _ in rig.sent[start:] if label == "hold rest"]
    assert all(np.array_equal(t, PLANNER) for t in hold_rest)  # the stance is the planner token
    blend_out = [h for _, h, label, _, _ in rig.sent[start:] if label == "blend out"]
    assert np.allclose(blend_out[0], 0.98 * POSE.lower_hands[-1])  # from the lowering's end to the stance
    hands = np.array([h for _, h, label, _, _ in rig.sent[start:] if label == "handback blend"])
    assert not hands.any()  # hands -> 0 (the stance hands are already open)
    assert np.array_equal(rig.sent[-1][0], PLANNER) and not rig.sent[-1][1].any()


def test_loop_methods_rejected_outside_their_phase(tmp_path):
    rig = Rig()
    for call in (lambda: rig.loop.raise_(0.0), lambda: rig.loop.trial(0, str(tmp_path), 0.0),
                 lambda: rig.loop.end(0.0), lambda: rig.loop.resume(0.0), lambda: rig.loop.freeze("x", 0.0)):
        with pytest.raises(ControlError):
            call()
    rig.take_over()
    for call in (lambda: rig.loop.take_over(rig.feedback(), 0.0), lambda: rig.loop.trial(0, str(tmp_path), 0.0),
                 lambda: rig.loop.end(0.0), lambda: rig.loop.resume(0.0)):
        with pytest.raises(ControlError):
            call()
    rig.loop.raise_(rig.now)
    with pytest.raises(ControlError):
        rig.loop.raise_(rig.now)
    rig.step(183)
    with pytest.raises(ControlError):
        rig.loop.raise_(rig.now)

    c, _ = make_hc(tmp_path, with_loop=False)
    assert request(c, "loop_take_over", now=0.1)["error"]["code"] == "UNSUPPORTED_SKILL"
    for kwargs in (dict(verified=False), dict(anchor=0.0)):
        c, _ = make_hc(tmp_path, **kwargs)
        assert request(c, "loop_take_over", now=0.1)["error"]["code"] == "NOT_READY"
    c, _ = make_hc(tmp_path)
    assert request(c, "loop_raise", now=0.1)["error"]["code"] == "NOT_READY"
    assert request(c, "loop_freeze", now=0.1)["error"]["code"] == "NOT_READY"
    assert c.phase == "IDLE"


def test_loop_trial_rejects_a_recorded_trial_number(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode").step(5)
    rig.loop.freeze("operator stop", rig.now)
    rig.loop.resume(rig.now)
    rig.step(50)
    with pytest.raises(ControlError):
        rig.loop.trial(0, str(tmp_path), rig.now)
    rig.loop.trial(1, str(tmp_path), rig.now)
    assert rig.loop.stage == "hold_rest"


def drive_to_completed(c, rig):
    """take-over, raise and end via RPC, played by loop.tick only: the HC is never ticked meanwhile."""
    for method in ("loop_take_over", "loop_raise", "loop_end"):
        assert request(c, method, now=0.1)["error"] is None
        rig.step({"loop_take_over": 50, "loop_raise": 183, "loop_end": 383}[method])
    assert rig.loop.phase == "COMPLETED" and c.phase == "LOOP_MOTION"  # the HC mirror is stale until synced


def test_override_after_loop_completes_before_hc_tick_takes_the_normal_path(tmp_path):
    c, rig = make_hc(tmp_path)
    drive_to_completed(c, rig)
    assert not c.operator_override("p")  # no ControlError out of the key handler: the non-loop path ran
    assert (c.phase, c.reason, c.owner) == ("INTERRUPTED", "operator_override:p", None)
    assert rig.loop.phase == "COMPLETED" and rig.loop.reason is None  # not frozen

    c, rig = make_hc(tmp_path)
    drive_to_completed(c, rig)
    c.interrupt("stale_policy_observation", 0.2)  # a direct interrupt in the same window: no ControlError
    assert c.phase == "INTERRUPTED" and rig.loop.phase == "COMPLETED"

    c, rig = make_hc(tmp_path)
    drive_to_completed(c, rig)
    assert request(c, "loop_take_over", now=0.2)["error"] is None  # the RPC sees COMPLETED, not LOOP_MOTION
    assert c.phase == "LOOP_MOTION" and rig.loop.stage == "stance"


def test_freeze_during_take_over_aborts_it_on_resume(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(20)  # still before the POSE switch
    assert request(c, "loop_freeze", now=0.1)["error"] is None
    assert rig.loop.frozen_stage == "stance"
    assert request(c, "loop_resume", now=0.1)["error"] is None
    assert (c.phase, rig.loop.phase, rig.loop.stage, rig.hooks.calls) == ("IDLE", "IDLE", None, [])
    assert not rig.cmd.busy and not rig.cmd.frozen
    assert request(c, "loop_take_over", now=0.2)["error"] is None  # re-captures token_state
    rig.step(50)
    assert rig.hooks.calls == ["pose"] and c.status(0.2)["loop_stage"] == "stance"


def test_release_in_loop_leaves_it_frozen_and_reclaimable(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    holds = len(rig.hooks.holds)
    assert request(c, "release_control", now=0.2)["error"] is None
    assert c.owner is None and c.phase == "LOOP_FROZEN" and rig.loop.reason == "released"
    assert rig.loop.frozen_stage == "stance" and len(rig.hooks.holds) == holds
    claim = request(c, "claim_control", {"registry_sha256": c.profile.registry_sha256}, now=0.3, session="s2")
    assert claim["error"] is None and c.phase == "LOOP_FROZEN"
    assert request(c, "release_control", now=0.4, session="s2")["error"] is None  # already frozen: no-op freeze
    assert c.owner is None and rig.loop.reason == "released"


def test_malformed_state_freezes_with_invalid_robot_state(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode").step(5)
    held = rig.sent[-1][0]
    token, _, label = rig.loop.tick(rig.now + TICK, dict(rig.feedback(), body_q=np.zeros(28)), 0.0, 0.0)
    assert label == "frozen" and np.array_equal(token, held)
    assert rig.loop.reason == rig.loop.trial_end_reason == "invalid robot state"
    assert str(trial_npz(tmp_path)["episode_end"]) == "invalid robot state"

    rig = Rig().take_over()  # outside a trial too
    token, _, label = rig.loop.tick(rig.now + TICK, dict(rig.feedback(), right_hand_q=np.zeros(6)), 0.0, 0.0)
    assert label == "frozen" and np.array_equal(token, PLANNER) and rig.loop.frozen_stage == "stance"


def test_blend_in_arm_motion_ends_the_trial_on_the_first_episode_check(tmp_path):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("blend_in").step(46)
    for _ in range(4):  # the blend in's last 4 ticks move arm joint 3 (body 18) at 10 rad/s
        rig.body[18] += 0.2
        rig.step()
    assert rig.loop.stage == "blend_in"
    rig.step()  # the first episode tick, arm still
    assert rig.loop.stage == "await_take"
    assert rig.loop.trial_end_reason.startswith("watchdog: arm joint 3 measured at")


def test_take_over_refuses_swapped_right_hand_order(tmp_path):
    c, rig = make_hc(tmp_path, right_order="swap")
    response = request(c, "loop_take_over", now=0.1)
    assert response["error"]["code"] == "NOT_READY" and "dex3_right_order" in response["error"]["message"]
    assert c.phase == "IDLE" and rig.loop.phase == "IDLE"


def test_failed_inference_keeps_the_last_chunk_and_requests_again(tmp_path, capsys):
    rig = Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode")
    start, n_chunks, episode_start = len(rig.sent), len(rig.set_chunks), rig.now
    rig.rows = lambda epoch: None  # every replan fails from here on (a worker exception)
    for _ in range(80):
        rig.step()
        if rig.loop.stage != "episode":
            break
    assert rig.loop.stage == "await_take" and rig.loop.trial_end_reason == "no valid chunk for 1 s"
    assert len(rig.set_chunks) == n_chunks  # the last good chunk kept playing
    assert set(rig.labels(start)[:-1]) == {"episode"}  # never frozen
    # each failure cleared the outstanding request: the next replan (request time + 0.4 s) asked again
    assert rig.hooks.requests == [100, 101, 102]
    assert [rig.asked[e][0] - episode_start for e in (101, 102)] == pytest.approx([0.4, 0.8], abs=TICK + 1e-9)
    assert "policy inference failed" in capsys.readouterr().out


def test_failed_first_chunk_freezes(tmp_path):
    rig = Rig().ready()
    rig.answer_after = None
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.step(75)
    assert not rig.loop.accept_chunk(100, rig.now, None)
    assert rig.loop.phase == "LOOP_FROZEN" and rig.loop.frozen_stage == "first_chunk"
    assert rig.loop.reason == "first chunk inference failed"


def test_take_over_refuses_unless_the_served_model_is_the_checkpoint(tmp_path):
    for served in ("/mnt/data/jihun/models/other/checkpoint-20000", None):
        c, rig = make_hc(tmp_path)
        rig.hooks.served_model = served
        response = request(c, "loop_take_over", now=0.1)
        assert response["error"]["code"] == "NOT_READY" and "model" in response["error"]["message"]
        assert c.phase == "IDLE" and rig.loop.phase == "IDLE"
    c, rig = make_hc(tmp_path)
    rig.hooks.served_model = SERVED.replace("/checkpoint-20000", "/./checkpoint-20000/")  # same after normpath
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    assert rig.loop.model == rig.hooks.served_model  # each trial records the served model


def test_os_error_in_a_loop_call_is_an_invalid_request(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    assert request(c, "loop_raise", now=0.1)["error"] is None
    rig.step(183)
    locked = tmp_path / "locked"
    (locked / "session").mkdir(parents=True)
    os.chmod(locked, 0)
    try:
        response = request(c, "loop_trial", {"trial": 0, "session_dir": str(locked / "session")}, now=0.2)
    finally:
        os.chmod(locked, 0o700)
    assert response["error"]["code"] == "INVALID_REQUEST" and "Permission denied" in response["error"]["message"]
    assert c.phase == "LOOP_HOLD" and rig.loop.stage == "ready"


def ended_trial(tmp_path, rig=None):
    """A trial ended by the chunk-age guard (every replan fails): the loop holds at await_take."""
    rig = rig or Rig().ready()
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode")
    rig.rows = lambda epoch: None
    return rig.until("await_take", limit=80)


def test_loop_to_ready_only_from_await_take(tmp_path):
    rig = Rig()
    with pytest.raises(ControlError):
        rig.loop.to_ready(rig.now)
    rig.ready()
    with pytest.raises(ControlError):  # LOOP_HOLD ready
        rig.loop.to_ready(rig.now)
    rig.loop.trial(0, str(tmp_path), rig.now)
    rig.until("episode")
    with pytest.raises(ControlError):  # mid-trial
        rig.loop.to_ready(rig.now)
    rig.rows = lambda epoch: None
    rig.until("await_take", limit=80)
    for call in (lambda: rig.loop.end(rig.now), lambda: rig.loop.trial(1, str(tmp_path), rig.now)):
        with pytest.raises(ControlError):  # end and the next trial wait for the take
            call()
    rig.loop.to_ready(rig.now)
    with pytest.raises(ControlError):  # LOOP_MOTION to_ready
        rig.loop.to_ready(rig.now)


def test_freeze_in_await_take_resumes_to_ready(tmp_path):
    rig = ended_trial(tmp_path)
    held = rig.sent[-1][:2]
    rig.loop.freeze("operator stop", rig.now)
    assert rig.loop.frozen_stage == "await_take" and rig.loop.trial_end_reason == "no valid chunk for 1 s"
    rig.step(5)
    assert all(np.array_equal(t, held[0]) for t, *_ in rig.sent[-5:])
    rig.loop.resume(rig.now)
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_MOTION", "to_ready")
    rig.step(50)
    assert runs(rig.labels(-50)) == [("to ready", 50)]
    assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "ready")
    assert str(trial_npz(tmp_path)["episode_end"]) == "no valid chunk for 1 s"  # closed once, at the episode end


def test_lease_expiry_in_await_take_freezes_and_rpc_to_ready(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    assert request(c, "loop_raise", now=0.1)["error"] is None
    rig.step(183)
    assert request(c, "loop_to_ready", now=0.1)["error"]["code"] == "NOT_READY"  # only from await_take
    assert request(c, "loop_trial", {"trial": 0, "session_dir": str(tmp_path / "s")}, now=0.1)["error"] is None
    rig.until("episode")
    rig.rows = lambda epoch: None
    rig.until("await_take", limit=80)
    c.tick(0.2, hc_state(2), 0.2)
    assert c.phase == "LOOP_HOLD" and c.status(0.2)["loop_stage"] == "await_take"
    assert request(c, "loop_to_ready", now=0.2)["error"] is None
    assert c.phase == "LOOP_MOTION" and rig.loop.stage == "to_ready"
    rig.step(50)
    c.tick(0.3, hc_state(3), 0.3)
    assert c.phase == "LOOP_HOLD" and rig.loop.stage == "ready"

    (tmp_path / "second").mkdir()
    c, rig = make_hc(tmp_path / "second")  # lease expiry while holding for the take
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    assert request(c, "loop_raise", now=0.1)["error"] is None
    rig.step(183)
    assert request(c, "loop_trial", {"trial": 0, "session_dir": str(tmp_path / "s2")}, now=0.1)["error"] is None
    rig.until("episode")
    rig.rows = lambda epoch: None
    rig.until("await_take", limit=80)
    c.tick(2.5, hc_state(2), 2.5)  # lease_s 2.0
    assert c.owner is None and c.phase == "LOOP_FROZEN" and rig.loop.frozen_stage == "await_take"
    assert rig.loop.reason == "lease expired"


def test_session_end_after_loop_to_ready(tmp_path):
    rig = ended_trial(tmp_path)
    rig.loop.to_ready(rig.now)
    rig.step(50)
    start = len(rig.sent)
    rig.loop.end(rig.now)
    rig.step(50 + 83 + 50 + 100 + 100)
    assert runs(rig.labels(start)) == [("to lower start", 50), ("lower", 83), ("blend out", 50),
                                       ("hold rest", 100), ("handback blend", 100)]
    assert rig.loop.phase == "COMPLETED" and rig.hooks.calls[-1] == "planner"


def arm_torque(sh, el):
    tau = np.zeros(29)
    tau[22], tau[25] = sh, el
    return tau


def test_take_detected_in_await_take_and_reset_between_trials(tmp_path):
    c, rig = make_hc(tmp_path)
    assert request(c, "loop_take_over", now=0.1)["error"] is None
    rig.step(50)
    assert request(c, "loop_raise", now=0.1)["error"] is None
    rig.step(183)
    assert c.status(0.1)["take_detected"] is None  # outside await_take
    rig.torque = arm_torque(-8.1, -5.2)  # the bottle rests on the palm
    for trial in (0, 1):
        session, at = str(tmp_path / "s"), 0.2 + 0.1 * trial  # request ids derive from (method, now)
        assert request(c, "loop_trial", {"trial": trial, "session_dir": session}, now=at)["error"] is None
        rig.until("episode")
        assert c.status(at)["take_detected"] is None and c.status(at)["take_hand_empty"] is None
        rig.rows = lambda epoch: None
        rig.until("await_take", limit=80)
        rig.rows = lambda epoch: chunk((epoch - 100) * 0.01)
        rig.step(150)  # 3 s of the loaded hold (reference from the episode's end), no take
        assert rig.loop.take_detected is False and c.status(at)["take_detected"] is False
        assert c.status(at)["take_hand_empty"] is rig.loop.take_hand_empty is True  # R32: this rig's wrist is 0
        rig.torque = arm_torque(-5.45, -3.4)  # taken: the arm unloads
        rig.step(49)
        assert not rig.loop.take_detected
        rig.step(2)
        assert rig.loop.take_detected and c.status(at)["take_detected"] is True
        assert (rig.loop.phase, rig.loop.stage) == ("LOOP_HOLD", "await_take")  # it only reports; the runner acts
        assert request(c, "loop_to_ready", now=at)["error"] is None
        assert rig.loop.take_detected is False and c.status(at)["take_detected"] is None
        assert c.status(at)["take_hand_empty"] is None  # outside await_take
        rig.torque = arm_torque(-8.1, -5.2)  # the next bottle
        rig.step(50)
        assert rig.loop.stage == "ready"


def test_take_detection_resets_on_a_freeze(tmp_path):
    rig = Rig().ready()
    rig.torque = arm_torque(-8.1, -5.2)
    ended_trial(tmp_path, rig)
    rig.step(150)
    rig.torque = arm_torque(-5.45, -3.4)
    rig.step(60)
    assert rig.loop.take_detected
    rig.loop.freeze("operator stop", rig.now)
    assert not rig.loop.take_detected


def test_early_take_detected_from_the_episode_reference_and_logged(tmp_path, capsys):
    rig = Rig().ready()
    rig.torque = arm_torque(-8.1, -5.2)  # loaded through the episode's end
    ended_trial(tmp_path, rig)
    log = capsys.readouterr().out
    assert re.search(r"\[loop\] take reference sh/el = -8\.10/-5\.20 spread 0\.00/0\.00 n \d+ \| wrist yaw", log)
    assert "(unsteady" not in log
    rig.step(25)  # still loaded for the hold's first 0.5 s (R28 also takes its first 0.3 s as the reference)
    rig.torque = arm_torque(-5.45, -3.4)  # taken early (r28: +0.6..1.6 s into the hold)
    rig.step(50)
    assert not rig.loop.take_detected
    rig.step(2)
    assert rig.loop.take_detected
    assert "[loop] take detected: rise sh/el = +2.65/+1.80 after 1.5" in capsys.readouterr().out


def test_take_detector_inert_without_episode_torque_and_ring_cleared(tmp_path, capsys):
    rig = Rig().ready()
    ended_trial(tmp_path, rig)  # no motor_torque during the episode
    log = capsys.readouterr().out
    assert "[loop] take reference sh/el = none n 0 (unsteady: waiting for a steady hold)" in log
    rig.torque = arm_torque(-5.45, -3.4)  # R30: a steady hold becomes the reference; staying there is no take
    rig.step(150)
    assert not rig.loop.take_detected and rig.loop._take.steady_at is not None
    rig.loop.freeze("operator stop", rig.now)
    assert not rig.loop._torque_ring
    rig.loop.resume(rig.now)
    rig.step(50)
    rig.torque = arm_torque(-8.1, -5.2)
    rig.loop.trial(1, str(tmp_path), rig.now)
    assert not rig.loop._torque_ring  # a trial starts with an empty ring
    rig.rows = lambda epoch: chunk((epoch - 100) * 0.01)
    rig.until("episode").step(5)
    assert len(rig.loop._torque_ring) == 6


def test_take_summary_logs_before_and_after_torque_per_hold(tmp_path, capsys):
    rig = Rig().ready()
    rig.torque = arm_torque(-6.3, -4.3)  # loaded through the episode's end (r29 trial 0's reference)
    ended_trial(tmp_path, rig)
    rig.step(25)  # loaded through the hold's first 0.5 s
    rig.torque = arm_torque(-5.7, -4.0)  # taken, but a smaller unload than the 0.8/0.45 thresholds: no detection
    rig.step(120)
    assert not rig.loop.take_detected
    rig.loop.to_ready(rig.now)  # the operator's Enter
    log = capsys.readouterr().out
    assert "[loop] hold +  0.0 s torque sh/el = -6.30/-4.30 | sh roll/yaw = +0.00/+0.00 | wrist r/p/y = " in log
    assert ("[loop] take summary: before sh/el = -6.30/-4.30 | hold end = -5.70/-4.00 | diff = +0.60/+0.30 | "
            "detected no |") in log and "ended by loop_to_ready | wrist yaw +0.00 -> +0.00 (+0.00)" in log
    assert "[loop] episode end torque sh/el/wrist yaw, 0.5 s medians from -3 s: " in log  # R31 diagnostics


def test_a_freeze_during_the_move_to_ready_resumes_to_ready(tmp_path):  # review: the TRIAL_STAGES resume branch
    rig = Rig().ready()
    ended_trial(tmp_path, rig)
    rig.loop.to_ready(rig.now)
    rig.step(5)
    assert rig.loop.stage == "to_ready"
    rig.loop.freeze("operator stop", rig.now)
    assert (rig.loop.phase, rig.loop.frozen_stage) == ("LOOP_FROZEN", "to_ready")
    rig.loop.resume(rig.now)
    rig.until("ready")
    assert rig.loop.phase == "LOOP_HOLD" and not rig.loop.take_detected


def test_harness_shutdown_saves_the_active_trial(tmp_path):  # review: Ctrl-C mid-episode lost the samples
    rig = Rig().ready()
    rig.loop.trial(next_trial(tmp_path), str(tmp_path), rig.now)
    rig.rows = lambda epoch: chunk((epoch - 100) * 0.01)
    rig.until("episode").step(20)
    assert rig.loop._recorder is not None
    rig.loop.close_trial("harness shutdown")  # run_vla_inference's finally, before TrialWriter.flush
    assert TrialWriter.flush(10)
    assert (tmp_path / "trial_000.npz").exists() and (tmp_path / "trial_000_frames.npz").exists()
    assert str(np.load(tmp_path / "trial_000.npz")["episode_end"]) == "harness shutdown"
    rig.loop.close_trial("harness shutdown")  # nothing active: a no-op
    assert next_trial(tmp_path) == 1
