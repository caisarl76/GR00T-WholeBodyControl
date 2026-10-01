import queue
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from gear_sonic.scripts import run_vla_inference as runner
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message


def run_keys(monkeypatch, keys, state=None, on_key=None, config=None):
    if state is not None:
        state.setdefault("reference_heading_quat", [1, 0, 0, 0])
        state.setdefault("planner_reference_active", [1])
    socket = MagicMock()

    def acknowledge_mode(raw):
        if state is not None and raw.startswith(b"command"):
            command = unpack_bridge_message(raw, topic="command")
            state["planner_reference_active"] = [int(command["planner"][0])]

    socket.send.side_effect = acknowledge_mode
    context = MagicMock()
    context.socket.return_value = socket
    monkeypatch.setattr(runner.zmq, "Context", lambda: context)
    monkeypatch.setattr(runner, "instantiate_g1_robot_model", MagicMock())
    monkeypatch.setattr(runner, "ComposedCameraClientSensor", MagicMock())
    subscriber = MagicMock()
    subscriber.get_msg.return_value = state
    monkeypatch.setattr(runner, "ZMQStateSubscriber", MagicMock(return_value=subscriber))
    keyboard = MagicMock()
    monkeypatch.setattr(runner, "ZMQKeyboardSubscriber", lambda **kw: keyboard)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    thread_factory = MagicMock()
    monkeypatch.setattr(runner.threading, "Thread", thread_factory)
    sequence = iter(enumerate(keys))

    def read_key():
        try:
            index, key = next(sequence)
        except StopIteration:
            raise KeyboardInterrupt
        if on_key:
            on_key(index, thread_factory.call_args.kwargs["args"])
        return key

    keyboard.read_msg.side_effect = read_key
    monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=MagicMock()))
    runner.main(config or runner.InferenceConfig())
    return [call.args[0] for call in socket.send.call_args_list]


def test_stale_camera_blocks_policy_request():
    from gear_sonic.utils.inference.observation_snapshot import SensorFreshness
    freshness = SensorFreshness()
    camera = {'timestamps': {'ego_view': 1.}, 'images': {'ego_view': np.zeros((2, 2, 3))}}
    assert freshness.capture(camera, 0.) == 0.
    assert freshness.capture(camera, .49) == 0.
    assert freshness.capture(camera, .51) is None


def test_harness_prewarm_pause_reset_and_old_actions(monkeypatch):
    from pathlib import Path
    from gear_sonic.utils.inference import harness_rpc, observation_snapshot
    profile = Path(__file__).parent / 'fixtures/g1_profile.yaml'
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    instances = []

    class Snapshots:
        def __init__(self, *args): pass
        def start(self): pass
        def close(self): pass
        def latest(self, now):
            return dict(camera_key='ego_view', source_timestamp=now, frame_id=str(now), received_at=now, age_s=0., jpeg_rgb_b64='unused')

    class RPC:
        def __init__(self, *args): self.count = 0; self.responses = []; instances.append(self)
        def start(self): pass
        def close(self): pass
        def drain(self, control, now):
            self.control = control
            control.observation = Snapshots().latest(now)
            self.count += 1
            method, params = None, {}
            if self.count == 2:
                method, params = 'claim_control', {'registry_sha256': control.profile.registry_sha256}
            elif self.count == 3:
                method, params = 'start_manipulation', {'skill_id': 'bottle_to_right_table'}
            elif self.count == 5:
                method, params = 'pause_manipulation', {'execution_id': control.execution['execution_id']}
            elif self.count == 6:
                method, params = 'reset_standing', {'execution_id': control.execution['execution_id'], 'open_hands': True}
            if method:
                response = control.dispatch(dict(version=1, request_id=str(self.count), runtime_id=control.runtime_id,
                    session_id='s', lease_id=control.owner['lease_id'] if control.owner else None, method=method, params=params), now)
                self.responses.append(response)

    monkeypatch.setattr(harness_rpc, 'HarnessRPCServer', RPC)
    monkeypatch.setattr(observation_snapshot, 'ObservationSnapshotCache', Snapshots)
    measured = dict(index=0, body_q=np.zeros(29), left_hand_q=np.full(7, .3), right_hand_q=np.full(7, .4), base_quat=[1, 0, 0, 0])

    def on_key(index, args):
        measured['index'] += 1
        clock[0] += .1
        if index in {1, 3}:
            epoch = args[0].get_nowait()
            action = dict(motion_token=np.full((1,40,64), .1), left_hand_joints=np.zeros((1,40,7)), right_hand_joints=np.zeros((1,40,7)))
            args[1].put((action, clock[0], epoch))
        if index == 6:
            args[1].put(({'malformed_old': True}, clock[0], 0))

    messages = run_keys(monkeypatch, ['k'] + [None]*8, measured, on_key,
        runner.InferenceConfig(harness_endpoint='ipc:///tmp/unused-test.sock', harness_profile=str(profile)))
    assert all(r['error'] is None for r in instances[0].responses)
    poses = [unpack_bridge_message(raw, topic='pose') for raw in messages if raw.startswith(b'pose')]
    assert len(poses) == 1  # Prewarm and late result never publish.
    assert instances[0].control.phase == 'RESETTING'
    np.testing.assert_allclose(poses[0]['token_state'], .1)


def test_repeated_init_uses_planner_standing_and_resume_switches_to_pose(monkeypatch):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }
    messages = run_keys(monkeypatch, ["k", "i", "p", "p", "i", None, "k"], state)
    assert not any(raw.startswith(b"pose") for raw in messages)
    commands = [unpack_bridge_message(raw, topic="command") for raw in messages if raw.startswith(b"command")]
    assert [(int(c["start"][0]), int(c["planner"][0])) for c in commands] == [
        (1, 1),
        (1, 1),
        (1, 0),
        (1, 1),
        (0, 1),
    ]
    # Every reset primes a measured hold before requesting planner mode.
    for i, raw in enumerate(messages):
        if raw.startswith(b"command") and i > 0:
            command = unpack_bridge_message(raw, topic="command")
            if command["start"][0] and command["planner"][0]:
                assert messages[i - 1].startswith(b"planner")
                hold = unpack_bridge_message(messages[i - 1], topic="planner")
                np.testing.assert_allclose(hold["upper_body_position"], 0)


def test_init_and_resume_do_not_command_robot_while_stopped(monkeypatch):
    assert run_keys(monkeypatch, ["i", "p"]) == []


def test_init_without_feedback_stays_paused_and_sends_no_pose(monkeypatch):
    messages = run_keys(monkeypatch, ["k", "i"])
    assert len(messages) == 1 and messages[0].startswith(b"command")


def test_inflight_result_retains_epoch_captured_before_observation():
    requests, results = queue.Queue(), queue.Queue()
    stop, busy = threading.Event(), threading.Event()
    requests.put(7)

    def observe():
        # A reset can invalidate epoch 7 during sensor capture or inference.
        return "observation"

    def infer(observation):
        stop.set()
        return {"token": observation}

    runner._inference_worker_loop(requests, results, stop, busy, observe, infer)
    action, started, epoch = results.get_nowait()
    assert epoch == 7 and action == {"token": "observation"}
    assert started > 0 and not busy.is_set()


def test_reset_discards_late_results_and_only_publishes_fresh_actions(monkeypatch):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }

    def on_key(index, worker_args):
        requests, results = worker_args[:2]
        if index in (1, 2, 6, 7):
            assert requests.empty(), "Paused policy must not schedule inference"
        if index in (3, 8):
            # Old requests complete after reset/resume; malformed actions must
            # never reach publication or latency compensation.
            results.put(({"stale": True}, runner.time.monotonic(), 2 if index == 3 else 3))
        if index in (4, 9):
            epoch = requests.get_nowait()
            assert epoch == (3 if index == 4 else 6)
            action = {
                "motion_token": np.full((1, 40, 64), 0.1 if index == 4 else 0.2),
                "left_hand_joints": np.zeros((1, 40, 7)),
                "right_hand_joints": np.zeros((1, 40, 7)),
            }
            results.put((action, runner.time.monotonic(), epoch))

    messages = run_keys(monkeypatch, ["k", "i", "p", None, None, "p", "i", "p", None, None], state, on_key)
    poses = [unpack_bridge_message(raw, topic="pose") for raw in messages if raw.startswith(b"pose")]
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0]["token_state"], 0.1)
    np.testing.assert_allclose(poses[1]["token_state"], 0.2)
    assert poses[0]["frame_index"][0] == poses[1]["frame_index"][0] == 0


def test_keyboard_movement_uses_planner_and_policy_stays_paused(monkeypatch):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }

    def check_paused(index, worker_args):
        if 6 <= index <= 14:
            assert worker_args[0].empty()

    messages = run_keys(
        monkeypatch,
        ["k", "i", "p", "p", "i", "m", "w", "s", "a", "d", "q", "e", "p", "m", "i", "p"],
        state,
        check_paused,
    )
    switches = [
        (i, unpack_bridge_message(raw, topic="command"))
        for i, raw in enumerate(messages)
        if raw.startswith(b"command")
    ]
    assert [(int(c["start"][0]), int(c["planner"][0])) for _, c in switches] == [
        (1, 1),
        (1, 1),
        (1, 0),
        (1, 1),
        (1, 1),
        (1, 0),
        (1, 1),
        (1, 0),
    ]
    planners = [
        unpack_bridge_message(raw, topic="planner") for raw in messages[switches[4][0] + 1 : switches[5][0]]
    ]
    assert any(c["mode"][0] == 1 and c["movement"][0] > 0 for c in planners)
    assert any(c["mode"][0] == 1 and c["movement"][0] < 0 for c in planners)
    assert any(c["movement"][1] > 0 for c in planners)
    assert any(c["movement"][1] < 0 for c in planners)
    assert any(c["facing"][1] > 0 for c in planners)
    assert any(c["facing"][1] < 0 for c in planners)
    assert planners[-1]["mode"][0] == 0
    np.testing.assert_allclose(planners[-1]["movement"], 0)
    assert not any(raw.startswith(b"pose") for raw in messages)


def test_planner_toggle_does_not_start_stopped_controller(monkeypatch):
    assert run_keys(monkeypatch, ["m", "w", "s", "m"]) == []


def test_planner_entry_requires_fresh_feedback(monkeypatch):
    messages = run_keys(monkeypatch, ["k", "m", "w", "m"])
    assert len(messages) == 1 and messages[0].startswith(b"command")


def test_failed_initial_reset_after_turn_keeps_current_heading(monkeypatch):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }

    def update_state(index, worker_args):
        if index == 4:
            state["base_quat"] = [np.cos(0.3), 0, 0, np.sin(0.3)]
        if index == 5:
            state["body_q"][0] = np.nan

    messages = run_keys(monkeypatch, ["k", "i", "m", "q", None, "i", None], state, update_state)
    planners = [unpack_bridge_message(raw, topic="planner") for raw in messages if raw.startswith(b"planner")]
    for command in planners[-3:]:
        np.testing.assert_allclose(command["facing"], [np.cos(0.6), np.sin(0.6), 0], atol=1e-7)
        assert command["mode"][0] == 0


def test_posture_reset_after_manual_turn_tracks_new_reference_without_world_turn(monkeypatch):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }

    def update_state(index, worker_args):
        if index == 4:
            state["base_quat"] = [np.cos(0.6), 0, 0, np.sin(0.6)]
        if index == 6:
            state["planner_reference_active"] = [0]
        if index in (7, 8):
            offset = 1.17 if index == 7 else 1.19
            state["planner_reference_active"] = [1]
            state["reference_heading_quat"] = [np.cos(offset / 2), 0, 0, np.sin(offset / 2)]

    messages = run_keys(monkeypatch, ["k", "i", "m", "q", None, "m", "i", None, None], state, update_state)
    planners = [unpack_bridge_message(raw, topic="planner") for raw in messages if raw.startswith(b"planner")]
    for command, offset in zip(planners[-2:], (1.17, 1.19)):
        reference_yaw = np.arctan2(command["facing"][1], command["facing"][0])
        assert abs(reference_yaw + offset - 1.2) < 1e-6
        np.testing.assert_allclose(command["movement"], 0)


def test_reset_does_not_reuse_active_reference_from_before_pose_switch(monkeypatch, capsys):
    state = {
        "body_q": np.zeros(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }

    def delayed_feedback(index, worker_args):
        if index == 4:
            state["planner_reference_active"] = [1]

    messages = run_keys(monkeypatch, ["k", "i", "m", "m", "i"], state, delayed_feedback)
    assert messages[-1].startswith(b"command")
    assert unpack_bridge_message(messages[-1], topic="command")["planner"][0] == 0
    assert "waiting for POSE feedback" in capsys.readouterr().out
