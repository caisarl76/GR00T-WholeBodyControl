"""Exercise the merged keyboard handlers without starting robot or policy clients."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gear_sonic.scripts import run_vla_inference as runner


def _control_harness(initial_pose="standing", mode="PLANNER", token=None):
    # Compile the actual nested handlers with in-memory I/O in place of main's
    # client/thread setup. This retains their shared nonlocal control state.
    factory = ast.parse(
        """
def make(config, keyboard_listener, zmq_socket):
    pause_loop = True
    cpp_loop_running = INITIAL_MODE != "OFF"
    cpp_mode = INITIAL_MODE
    initial_pose_ready = False
    pose_start_pending = False
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_epoch = 0
    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    language_prompt_ref = [config.prompt]
    standing_reset = None
    manual_planner = None
    zmq_frame_counter = 0
    last_sent_motion_token = INITIAL_TOKEN
    PROMPT_MSG_PREFIX = "prompt:"
"""
    ).body[0]
    main_node = ast.parse(inspect.getsource(runner.main)).body[0]
    handlers = {
        "_initial_pose_hands",
        "publish_latent_initial_pose",
        "blend_to_initial_pose",
        "send_cpp_control_command",
        "invalidate_policy_actions",
        "check_keyboard_input",
    }
    factory.body.extend(
        node
        for node in main_node.body
        if isinstance(node, ast.FunctionDef) and node.name in handlers
    )
    factory.body.extend(
        ast.parse(
            """
def snapshot():
    return dict(paused=pause_loop, mode=cpp_mode, ready=initial_pose_ready,
                pending=pose_start_pending, token=last_sent_motion_token,
                epoch=inference_epoch)
return check_keyboard_input, send_cpp_control_command, snapshot, inference_queue, result_queue
"""
        ).body
    )
    namespace = dict(vars(runner))
    calib = Mock()
    standing = Mock()
    namespace.update(
        INITIAL_MODE=mode,
        INITIAL_TOKEN=token,
        time=SimpleNamespace(sleep=lambda _: None, monotonic=lambda: 0.0),
        publish_calib_full_pose=calib,
        publish_standing_pose=standing,
        pack_latent_action_message=lambda **data: {"kind": "latent", **data},
        build_command_message=lambda **data: {"kind": "command", **data},
    )
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    exec(compile(module, runner.__file__, "exec"), namespace)
    keyboard, socket = Mock(), Mock()
    config = runner.InferenceConfig(initial_pose=initial_pose, action_publish_rate=2)
    handle, command, snapshot, requests, results = namespace["make"](config, keyboard, socket)

    def press(key):
        keyboard.read_msg.return_value = key
        handle()

    return SimpleNamespace(
        press=press, command=command, snapshot=snapshot, socket=socket,
        calib=calib, standing=standing,
        requests=requests, results=results,
    )


def _tokens(control):
    return [
        call.args[0]["motion_token"]
        for call in control.socket.send.call_args_list
        if call.args[0]["kind"] == "latent"
    ]


def test_default_calib_full_prepares_planner_and_defers_pose_until_action():
    assert runner.InferenceConfig().initial_pose == "calib_full"
    control = _control_harness(initial_pose="calib_full", mode="OFF")
    control.press("i")
    control.calib.assert_called_once_with(send_latent_handoff=False)
    assert control.snapshot()["mode"] == "PLANNER"
    assert control.snapshot()["ready"]
    control.press("p")
    assert not control.snapshot()["paused"]
    assert control.snapshot()["pending"]
    assert control.snapshot()["mode"] == "PLANNER"
    assert not _tokens(control)


def test_standing_from_planner_publishes_initial_token_before_starting_pose():
    control = _control_harness(token=np.zeros(64, dtype=np.float32))
    control.press("i")
    assert len(_tokens(control)) == 1
    np.testing.assert_array_equal(_tokens(control)[0], runner.LATENT_INITIAL_MOTION_TOKEN)
    assert control.socket.send.call_args_list[0].args[0]["kind"] == "latent"
    assert control.snapshot()["mode"] == "POSE"
    assert control.snapshot()["ready"]
    control.calib.assert_not_called()


def test_standing_blends_from_current_pose_and_repeated_i_uses_updated_token():
    start = np.zeros(64, dtype=np.float32)
    control = _control_harness(mode="POSE", token=start)
    control.press("i")
    np.testing.assert_allclose(_tokens(control)[0], runner.LATENT_INITIAL_MOTION_TOKEN / 2)
    np.testing.assert_array_equal(_tokens(control)[-1], runner.LATENT_INITIAL_MOTION_TOKEN)
    control.socket.reset_mock()
    control.press("i")
    for token in _tokens(control):
        np.testing.assert_array_equal(token, runner.LATENT_INITIAL_MOTION_TOKEN)


def test_planner_handoff_and_stop_invalidate_previous_motion_token():
    control = _control_harness(mode="POSE", token=np.zeros(64, dtype=np.float32))
    assert control.command(start=True, planner=True)
    assert control.snapshot()["token"] is None
    assert control.command(start=True, planner=False)
    control.press("i")
    assert len(_tokens(control)) == 1
    assert control.command(start=False)
    assert control.snapshot()["token"] is None


def test_failed_pose_switch_does_not_arm_standing_inference():
    control = _control_harness()

    def fail_command(message):
        if message["kind"] == "command":
            raise RuntimeError("test transport failure")

    control.socket.send.side_effect = fail_command
    control.press("i")
    assert not control.snapshot()["ready"]
    control.press("p")
    assert control.snapshot()["paused"]


def test_standing_pause_resume_keeps_pose_and_stop_uses_standing_ramp():
    control = _control_harness()
    control.press("i")
    control.socket.reset_mock()
    control.press("p")
    assert not control.snapshot()["paused"]
    resume_epoch = control.snapshot()["epoch"]
    control.requests.put_nowait(resume_epoch)
    control.results.put_nowait(({"stale": True}, 0.0, resume_epoch))
    control.press("p")
    assert control.snapshot()["paused"]
    assert control.snapshot()["epoch"] == resume_epoch + 1
    assert control.requests.empty()
    assert control.results.empty()
    assert control.snapshot()["mode"] == "POSE"
    control.socket.send.assert_not_called()
    control.calib.assert_not_called()
    control.requests.put_nowait(resume_epoch)
    control.results.put_nowait(({"stale": True}, 0.0, resume_epoch))
    control.press("k")
    control.standing.assert_called_once_with()
    assert control.snapshot()["mode"] == "OFF"
    assert not control.snapshot()["ready"]
    assert not control.snapshot()["pending"]
    assert control.snapshot()["token"] is None
    assert control.snapshot()["epoch"] == resume_epoch + 2
    assert control.requests.empty()
    assert control.results.empty()


def test_standing_cannot_initialize_or_resume_before_control_starts():
    control = _control_harness(mode="OFF")
    control.press("i")
    control.press("p")
    control.socket.send.assert_not_called()
    assert control.snapshot()["paused"]
    assert not control.snapshot()["ready"]


def test_failed_planner_switch_does_not_ramp_or_report_control_stopped():
    control = _control_harness(initial_pose="standing", mode="POSE")

    def fail_command(message):
        if message["kind"] == "command":
            raise RuntimeError("command transport failed")

    control.socket.send.side_effect = fail_command
    control.press("k")

    assert control.snapshot()["mode"] == "POSE"
    control.standing.assert_not_called()


def test_failed_stop_command_keeps_planner_control_state():
    control = _control_harness(initial_pose="standing", mode="PLANNER")

    def fail_command(message):
        if message["kind"] == "command" and message["stop"]:
            raise RuntimeError("stop transport failed")

    control.socket.send.side_effect = fail_command
    control.press("k")

    control.standing.assert_called_once_with()
    assert control.snapshot()["mode"] == "PLANNER"
