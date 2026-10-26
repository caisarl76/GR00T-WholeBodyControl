"""The autonomous loop driven by run_vla_inference's 50 Hz main loop (Task 7), with the fakes of
test_run_vla_reset.run_keys: one iteration per keyboard read, a scripted RPC client, a deploy that echoes the
pose tokens it receives, and an inference worker that answers each queued epoch after `delay` iterations."""

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_handover_ready import NAMES, RAISED, TARGET
from test_run_vla_reset import run_keys
import yaml

from gear_sonic.scripts import run_vla_inference as runner
from gear_sonic.utils.inference import harness_rpc, lerobot_sonic_policy, loop_records, observation_snapshot
from gear_sonic.utils.inference.loop_control import LoopController
from gear_sonic.utils.inference.loop_records import TrialWriter
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message

TICK = 0.02
PLANNER = np.full(64, -0.2, np.float32)
JPEG_B64 = "anBlZw=="  # b"jpeg"


@pytest.fixture(autouse=True)
def real_writer_thread():
    """run_keys replaces threading.Thread; the TrialWriter's lazily started thread must be a real one."""
    if TrialWriter._thread is None:
        TrialWriter._thread = threading.Thread(target=TrialWriter._run, name="trial-writer", daemon=True)
        TrialWriter._thread.start()


def action(value):
    """A [1, 40, *] policy result whose token rows ramp from `value`."""
    tokens = np.full((1, 40, 64), value, np.float32) + np.arange(40, dtype=np.float32)[None, :, None] * 1e-3
    return dict(motion_token=tokens, left_hand_joints=np.zeros((1, 40, 7)), right_hand_joints=np.zeros((1, 40, 7)))


def at(phase, stage=None):
    return lambda c: c.phase == phase and (stage is None or c.loop.stage == stage)


class Session:
    def __init__(self, monkeypatch, tmp_path, script, delay=5, answer=None, hook=None):
        self.monkeypatch, self.tmp_path, self.script = monkeypatch, tmp_path, list(script)
        self.delay, self.answer, self.hook = delay, answer or (lambda n, c: action(0.1 + 0.03 * n)), hook
        self.clock, self.control, self.sequence = [0.0], None, 0
        self.marks, self.walls, self.stages, self.requests, self.pending, self.replies = [], [], [], [], [], []
        self.cursor, self.mode, self.last_pose = 0, "PLANNER", None
        self.state = dict(
            index=0, body_q=np.zeros(29), body_q_measured_motor=np.zeros(29), harness_planner_hold_enabled=[1],
            left_hand_q=np.zeros(7), right_hand_q=np.zeros(7), left_hand_q_measured=np.zeros(7),
            right_hand_q_measured=np.zeros(7), base_quat=[1, 0, 0, 0], token_state=PLANNER.tolist())

    def profile(self):
        data = yaml.safe_load((Path(__file__).parent / "fixtures/g1_profile.yaml").read_text())
        data["skills"] = {"bottle_handover": dict(prompt="test-only handover", completion_criteria="Stable offer",
                                                  completion_action="hold")}
        data["handover"] = dict(skill_id="bottle_handover", checkpoint_verified=True, ready_pose_reviewed=True,
                                ready_right_arm_joints=dict(zip(NAMES, TARGET)),
                                raised_right_arm_joints=dict(zip(NAMES, RAISED)))
        data["lerobot_policy"] = dict(chunk_blend_s=0.3, max_token_step=0.05, chunk_anchor_s=1.0)
        path = self.tmp_path / "loop.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def dispatch(self, method, params):
        c = self.control
        self.sequence += 1
        return c.dispatch(dict(version=1, request_id=f"r{self.sequence}", runtime_id=c.runtime_id, session_id="s",
                               lease_id=c.owner["lease_id"] if c.owner else None, method=method, params=params),
                          self.clock[0])

    def drain(self, control, now):
        self.control = control
        control.observation = dict(camera_key="ego_view", source_timestamp=now, frame_id=str(now), received_at=now,
                                   age_s=0.0, jpeg_rgb_b64=JPEG_B64)
        if self.script and self.script[0][0](control):
            _, method, params = self.script.pop(0)
            self.replies.append((method, self.dispatch(method, params)))
            assert self.replies[-1][1]["error"] is None, self.replies[-1]
        elif control.owner is not None:
            self.dispatch("heartbeat", {"sequence": self.sequence})
        self.stages.append((control.phase, control.loop.stage))  # what the main loop branches on

    def sleep(self, seconds):  # time passes where the main loop waits out its 20 ms period
        self.clock[0] += seconds

    def on_key(self, index, args):
        if index == 0:  # after run_keys' own no-op sleep
            self.monkeypatch.setattr(runner.time, "sleep", self.sleep)
            args[4]()  # the worker's first observation creates its policy client (the served model is known)
        sent = runner.zmq.Context().socket().send.call_args_list
        for call in sent[self.cursor:]:  # the deploy: echo POSE tokens in token_state, PLANNER's otherwise
            raw = call.args[0]
            if raw.startswith(b"pose"):
                self.last_pose = np.asarray(unpack_bridge_message(raw, topic="pose")["token_state"]).reshape(64)
            elif raw.startswith(b"command"):
                command = unpack_bridge_message(raw, topic="command")
                self.mode = "PLANNER" if command["planner"][0] else "POSE"
        self.cursor = len(sent)
        self.marks.append(len(sent))
        self.walls.append(time.perf_counter())
        self.state["index"] += 1
        echo = self.last_pose if self.mode == "POSE" and self.last_pose is not None else PLANNER
        self.state["token_state"] = echo.tolist()
        requests, results = args[0], args[1]
        for item in [p for p in self.pending if p[0] <= index]:
            self.pending.remove(item)
            if results.full():
                results.get_nowait()
            results.put_nowait(item[1])
        if not requests.empty():
            epoch = requests.get_nowait()
            self.requests.append((index, epoch, self.control.phase if self.control else None))
            self.pending.append((index + self.delay, (self.answer(len(self.requests), self.control), self.clock[0],
                                                      epoch)))
        if self.hook:
            self.hook(self, index)

    def run(self, n, keys=None):
        self.monkeypatch.setattr(runner.time, "monotonic", lambda: self.clock[0])
        cache = SimpleNamespace(start=lambda: None, close=lambda: None, latest_camera=lambda now: None)
        checkpoint = yaml.safe_load((Path(__file__).parent / "fixtures/g1_profile.yaml").read_text())["checkpoint"]
        served = SimpleNamespace(model=checkpoint)  # the policy server's info: serves the profile checkpoint
        self.monkeypatch.setattr(lerobot_sonic_policy, "LeRobotSonicPolicyClient", lambda **kw: served)
        self.monkeypatch.setattr(observation_snapshot, "ObservationSnapshotCache", lambda *a: cache)
        session = self

        class RPC:
            def __init__(self, *args):
                pass

            def start(self):
                pass

            def close(self):
                pass

            def drain(self, control, now):
                session.drain(control, now)

        self.monkeypatch.setattr(harness_rpc, "HarnessRPCServer", RPC)
        keys = keys or {}
        self.messages = run_keys(
            self.monkeypatch, ["k"] + [keys.get(i) for i in range(1, n)], self.state, self.on_key,
            runner.InferenceConfig(harness_endpoint="ipc:///tmp/unused-test.sock",
                                   harness_profile=str(self.profile())))
        return self

    def iterations(self):
        """(phase, stage, messages) per main-loop iteration; phase and stage as the main loop saw them."""
        bounds = self.marks + [len(self.messages)]
        return [(*self.stages[i], self.messages[bounds[i]:bounds[i + 1]]) for i in range(len(self.marks))]

    def first(self, stage):
        return next(i for i, (_, s) in enumerate(self.stages) if s == stage)


def poses(messages):
    return [m for m in messages if m.startswith(b"pose")]


def token(raw):
    return np.asarray(unpack_bridge_message(raw, topic="pose")["token_state"]).reshape(64)


def opening(tmp_path):
    return [
        (lambda c: c.hooks.runtime_facts().controller_running and c.owner is None, "claim_control", "registry"),
        (lambda c: c.owner is not None and c.phase == "IDLE", "loop_take_over", {}),
        (at("LOOP_HOLD", "stance"), "loop_raise", {}),
        (at("LOOP_HOLD", "ready"), "loop_trial", {"trial": 0, "session_dir": str(tmp_path / "session")}),
    ]


def loop_requests(s):
    return [r for r in s.requests if r[2] == "LOOP_TRIAL"]


def session(monkeypatch, tmp_path, steps, **kwargs):
    s = Session(monkeypatch, tmp_path, steps, **kwargs)
    original = s.dispatch

    def dispatch(method, params):
        if params == "registry":
            params = {"registry_sha256": s.control.profile.registry_sha256}
        return original(method, params)

    s.dispatch = dispatch
    return s


def assert_one_pose_per_loop_iteration(s):
    loop = [(p, st, msgs) for p, st, msgs in s.iterations() if p.startswith("LOOP_")]
    assert loop
    bad = [(p, st, len(poses(msgs))) for p, st, msgs in loop if len(poses(msgs)) != 1]
    assert not bad, bad[:5]
    assert not [m for _, _, msgs in loop for m in msgs if m.startswith(b"planner")]
    return loop


def test_loop_phases_publish_every_tick(monkeypatch, tmp_path, capsys):
    steps = opening(tmp_path) + [
        (lambda c: c.loop.stage == "episode" and len(loop_requests(s)) >= 4, "loop_freeze", {}),
        (at("LOOP_FROZEN"), "loop_resume", {}),
    ]
    s = session(monkeypatch, tmp_path, steps, hook=lambda s, i: s.sleep(0.05) if i == 300 else None)
    s.run(650)
    assert not s.script, s.script
    assert capsys.readouterr().out.count("Autonomous Loop Missed") == 1  # iteration 300 took 50 ms
    loop = assert_one_pose_per_loop_iteration(s)
    visited = [st for _, st, _ in loop]
    for stage in ("stance", "raise", "ready", "hold_rest", "first_chunk", "blend_in", "episode", "frozen",
                  "to_ready"):
        assert stage in visited, stage
    assert visited[-1] == "ready" and s.control.phase == "LOOP_HOLD"
    commands = [(i, unpack_bridge_message(m, topic="command")) for i, m in enumerate(s.messages)
                if m.startswith(b"command")]
    pose_switch = next(i for i, command in commands if not command["planner"][0])
    assert not [m for m in s.messages[pose_switch:] if m.startswith(b"planner")]
    assert s.control.loop.trial_end_reason == "operator stop"


def test_first_chunk_bootstrap_through_inference_queue(monkeypatch, tmp_path):
    s = session(monkeypatch, tmp_path, opening(tmp_path), delay=12).run(450)
    hold_rest, first, blend = s.first("hold_rest"), s.first("first_chunk"), s.first("blend_in")
    asked = [(i, epoch) for i, epoch, phase in s.requests if hold_rest <= i <= blend]
    assert len(asked) == 1 and asked[0][0] == first  # taken off inference_queue on the first wait iteration
    assert blend == first + 12 + 1  # delivered on result_queue at first + 12, accepted, blend in on the next tick
    waiting = s.iterations()[first:blend]
    assert len(waiting) == 13 and all(len(poses(msgs)) == 1 for _, _, msgs in waiting)
    assert s.control.loop.stage == "episode"


def test_slow_trial_write_does_not_stall_ticks(monkeypatch, tmp_path):
    saves, original = [], loop_records._save_npz

    def slow_save(path, compress=True, **arrays):
        start = time.perf_counter()
        if not path.name.endswith("_frames.npz"):
            threading.Event().wait(2.0)
        original(path, compress, **arrays)
        saves.append((threading.current_thread().name, start, time.perf_counter()))

    monkeypatch.setattr(loop_records, "_save_npz", slow_save)

    episode = [0]

    def hand(s, index):  # grasp for 0.6 s, then release: the handover detector ends the trial
        if s.stages and s.stages[-1][1] == "episode":
            episode[0] += 1
            closed = 10 <= episode[0] < 40
            s.state["right_hand_q"] = s.state["right_hand_q_measured"] = np.full(7, 0.3 if closed else 0.0)

    s = session(monkeypatch, tmp_path, opening(tmp_path), hook=hand).run(700)
    assert s.control.loop.trial_end_reason == "handover complete"
    assert {name for name, _, _ in saves} == {"trial-writer"}
    trial_npz = next((start, end) for name, start, end in saves)
    during = [i for i, wall in enumerate(s.walls) if trial_npz[0] < wall < trial_npz[1]]
    # the loop kept ticking while the trial npz was being written (the no-gap check below is the real proof;
    # the count only depends on how fast this machine runs an iteration)
    assert len(during) >= 10
    assert all(len(poses(s.iterations()[i][2])) == 1 for i in during)
    assert max(np.diff(s.walls)) < 0.5
    assert (tmp_path / "session/trial_000.npz").exists()


def test_loop_trial_uses_anchoring(monkeypatch, tmp_path):
    steps = opening(tmp_path) + [
        (lambda c: c.loop.stage == "episode" and len(loop_requests(s)) >= 6, "loop_freeze", {}),
    ]
    s = session(monkeypatch, tmp_path, steps)
    s.run(650)
    its = s.iterations()
    episode_requests = [i for i, _, _ in s.requests if s.stages[i][1] == "episode"]
    delivered = [i + s.delay for i in episode_requests if s.stages[i + s.delay][1] == "episode"]
    assert len(delivered) >= 3
    for i in delivered:  # the replan chunk is set on its delivery tick and continues from the token sent before
        np.testing.assert_allclose(token(poses(its[i][2])[0]), token(poses(its[i - 1][2])[0]), atol=1e-6)


def test_failed_inference_in_loop(monkeypatch, tmp_path):
    # a failed first chunk freezes (there is no chunk to keep playing)
    s = session(monkeypatch, tmp_path, opening(tmp_path),
                answer=lambda n, c: None if c.phase.startswith("LOOP_") else action(0.1)).run(400)
    assert s.control.phase == "LOOP_FROZEN" and s.control.loop.reason == "first chunk inference failed"
    assert s.control.loop.frozen_stage == "first_chunk"
    assert_one_pose_per_loop_iteration(s)
    # failed replans: lerobot keeps the last chunk and requests again; the chunk age ends the trial (R23c)
    second = tmp_path / "second"  # a fresh session_dir: the frozen trial 0 above is recorded
    second.mkdir()
    s = session(monkeypatch, second, opening(second),
                answer=lambda n, c: None if c.loop.stage == "episode" else action(0.1)).run(500)
    assert s.control.loop.trial_end_reason == "no valid chunk for 1 s"
    assert s.control.phase == "LOOP_HOLD" and s.control.loop.stage == "ready"
    assert len([r for r in loop_requests(s) if r[0] > s.first("episode")]) == 2  # at +0.4 s and +0.8 s
    assert_one_pose_per_loop_iteration(s)


@pytest.mark.parametrize("key,own_action", [
    ("p", ("Policy prewarm not ready", "Policy loop")),
    ("w", ("Movement ignored", "Manual movement")),
    ("]", ("Initial pose right hand",)),
])
def test_prompt_in_loop_is_dropped_and_motion_keys_freeze(monkeypatch, tmp_path, capsys, key, own_action):
    steps = opening(tmp_path)[:3]
    s = session(monkeypatch, tmp_path, steps)
    s.run(200, keys={120: "prompt:wave hello", 150: key})
    raise_iteration = next(i for i, (_, st) in enumerate(s.stages) if st == "raise")
    assert raise_iteration < 120 and s.first("frozen") == 150  # both keys arrive mid-raise; 'p' freezes at once
    out = capsys.readouterr().out
    assert "Inference prompt changed" not in out and "wave hello" in out
    assert s.stages[121][1] == "raise" and s.stages[149][1] == "raise"  # the prompt did not freeze
    assert s.control.loop.reason == f"operator override: {key}" and s.control.loop.frozen_stage == "raise"
    assert not [text for text in own_action if text in out]  # the key's own action was skipped
    assert len(s.iterations()[150][2]) == 1  # the key's iteration sent only the frozen pose
    assert_one_pose_per_loop_iteration(s)


def test_shutdown_flushes_trial_writer_and_reports_errors(monkeypatch, tmp_path, capsys):
    def broken_save(path, compress=True, **arrays):
        raise OSError("disk full")

    monkeypatch.setattr(loop_records, "_save_npz", broken_save)
    monkeypatch.setattr(TrialWriter, "errors", [])
    flushes, original = [], TrialWriter.flush

    def flush(timeout_s):
        flushes.append(timeout_s)
        return original(timeout_s)

    monkeypatch.setattr(TrialWriter, "flush", flush)
    steps = opening(tmp_path) + [(lambda c: c.loop.stage == "episode", "loop_freeze", {})]
    s = session(monkeypatch, tmp_path, steps).run(450)
    assert not s.script
    assert flushes == [10]
    out = capsys.readouterr().out
    assert "trial write failed" in out and "disk full" in out


def test_no_loop_tick_after_session_end(monkeypatch, tmp_path):
    ticks, original = [], LoopController.tick

    def tick(self, *args):
        ticks.append(self.phase)
        return original(self, *args)

    monkeypatch.setattr(LoopController, "tick", tick)
    steps = opening(tmp_path)[:3] + [(at("LOOP_HOLD", "ready"), "loop_end", {})]
    s = session(monkeypatch, tmp_path, steps).run(750)
    its = s.iterations()
    loop = [i for i, (p, _, _) in enumerate(its) if p.startswith("LOOP_")]
    assert len(ticks) == len(loop) and loop == list(range(loop[0], loop[-1] + 1))
    after = its[loop[-1] + 1:]
    assert len(after) > 50 and s.control.phase == "COMPLETED"
    assert not [m for _, _, msgs in after for m in msgs if m.startswith(b"pose")]
    assert [m for _, _, msgs in after for m in msgs if m.startswith(b"planner")]  # the measured planner hold
    hand_back = [unpack_bridge_message(m, topic="command") for m in its[loop[-1]][2] if m.startswith(b"command")]
    assert [int(c["planner"][0]) for c in hand_back] == [1]
    assert s.control.hold_confirmed


def test_prewarm_lost_at_take_over_rearms_after_the_loop(monkeypatch, tmp_path):
    steps = opening(tmp_path)[:3] + [(at("LOOP_HOLD", "ready"), "loop_end", {})]
    s = session(monkeypatch, tmp_path, steps)
    s.run(720)
    take_over = next(i for i, (p, _) in enumerate(s.stages) if p.startswith("LOOP_"))
    lost = [i for i, _, _ in s.requests if i <= take_over + 1 and s.stages[i + s.delay][0].startswith("LOOP_")]
    assert lost  # the prewarm queued in IDLE is answered during the loop, where accept_chunk drops it
    completed = next(i for i, (p, _) in enumerate(s.stages) if p == "COMPLETED")
    rearmed = [i for i, _, phase in s.requests if i > completed and phase == "COMPLETED"]
    assert rearmed, "no prewarm epoch was queued again after the loop"
    assert s.control.phase == "COMPLETED" and s.control.hooks.runtime_facts().policy_ready
