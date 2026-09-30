import queue
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from gear_sonic.scripts import run_vla_inference as runner
from gear_sonic.utils.teleop.xr_upperbody_bridge import unpack_bridge_message


def policy_action(value):
    return {
        "motion_token": np.full((1, 40, 64), value),
        "left_hand_joints": np.zeros((1, 40, 7)),
        "right_hand_joints": np.zeros((1, 40, 7)),
    }


def run_keys(monkeypatch, keys, state=None, on_key=None):
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
    runner.main(runner.InferenceConfig(initial_pose="planner_standing"))
    return [call.args[0] for call in socket.send.call_args_list]


def test_repeated_init_and_resume_hold_planner_until_first_action(monkeypatch):
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


def test_standing_reset_waits_for_measured_joints_before_advancing(monkeypatch):
    state = {
        "body_q": np.ones(29),
        "left_hand_q": np.zeros(7),
        "right_hand_q": np.zeros(7),
        "base_quat": [1, 0, 0, 0],
    }
    clock = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def advance_clock(*_):
        clock[0] += 0.05

    monkeypatch.setattr(runner, "_sleep_remaining", advance_clock)

    def follow_target(index, worker_args):
        assert worker_args[0].empty(), "Standing reset must keep inference paused"
        if index == 14:
            state["body_q"][:] = 0.8

    messages = run_keys(monkeypatch, ["k", "i"] + [None] * 20, state, follow_target)
    positions = np.array([
        unpack_bridge_message(raw, topic="planner")["upper_body_position"]
        for raw in messages if raw.startswith(b"planner")
    ])
    np.testing.assert_allclose(positions[0], 1)
    # The waist target stops 0.15 rad ahead of stalled feedback, then advances
    # again when the measured posture follows, at at most 0.5 rad/s.
    np.testing.assert_allclose(positions[10:13, 0], 0.85)
    np.testing.assert_allclose(positions[-1, 0], 0.65)
    assert np.max(np.abs(np.diff(positions[:, 0]))) <= 0.025 + 1e-7
    assert not any(raw.startswith(b"pose") for raw in messages)


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
            action = policy_action(0.1 if index == 4 else 0.2)
            results.put((action, runner.time.monotonic(), epoch))

    messages = run_keys(monkeypatch, ["k", "i", "p", None, None, "p", "i", "p", None, None], state, on_key)
    poses = [unpack_bridge_message(raw, topic="pose") for raw in messages if raw.startswith(b"pose")]
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0]["token_state"], 0.1)
    np.testing.assert_allclose(poses[1]["token_state"], 0.2)
    assert poses[0]["frame_index"][0] == poses[1]["frame_index"][0] == 0
    commands = [unpack_bridge_message(raw, topic="command") for raw in messages if raw.startswith(b"command")]
    assert [int(command["planner"][0]) for command in commands] == [1, 1, 0, 1, 0]
    for index, raw in enumerate(messages):
        if raw.startswith(b"pose"):
            # Resuming and discarded results keep the planner active. Only a
            # fresh action triggers the POSE command, immediately before it.
            assert messages[index - 1].startswith(b"command")
            assert unpack_bridge_message(messages[index - 1], topic="command")["planner"][0] == 0


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
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    planners = [
        unpack_bridge_message(raw, topic="planner") for raw in messages[switches[3][0] + 1 : switches[4][0]]
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
        if index == 8:
            requests, results = worker_args[:2]
            results.put((policy_action(0.1), runner.time.monotonic(), requests.get_nowait()))
        if index == 9:
            state["planner_reference_active"] = [0]
        if index in (10, 11):
            offset = 1.17 if index == 10 else 1.19
            state["planner_reference_active"] = [1]
            state["reference_heading_quat"] = [np.cos(offset / 2), 0, 0, np.sin(offset / 2)]

    messages = run_keys(
        monkeypatch, ["k", "i", "m", "q", None, "m", "i", "p", None, "i", None, None], state, update_state
    )
    assert sum(raw.startswith(b"pose") for raw in messages) == 1
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
        if index == 3:
            requests, results = worker_args[:2]
            results.put((policy_action(0.1), runner.time.monotonic(), requests.get_nowait()))
        if index == 4:
            state["planner_reference_active"] = [1]

    messages = run_keys(monkeypatch, ["k", "i", "p", None, "i"], state, delayed_feedback)
    assert messages[-2].startswith(b"command")
    assert unpack_bridge_message(messages[-2], topic="command")["planner"][0] == 0
    assert messages[-1].startswith(b"pose")
    assert "waiting for POSE feedback" in capsys.readouterr().out
