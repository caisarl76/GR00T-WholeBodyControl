"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> prepare the selected initial pose (CALIB_FULL, latent standing, or planner standing)
  m  -> enter / leave manual planner repositioning (policy stays paused)
  w/s, a/d -> toggle forward/back, left/right movement in manual mode
  q/e, z -> turn left/right, stop movement in manual mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start / finish recording (legacy data exporter keyboard mode)
  x  -> discard active recording (legacy data exporter keyboard mode)
"""

from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from typing import Literal
import uuid

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.control_transitions import (
    InferenceControlState,
    plan_i_transition,
    plan_k_transition,
    plan_p_transition,
)
from gear_sonic.utils.inference.initial_pose_ramp import (
    build_calib_full_hold_message,
    build_calib_full_ramp_messages,
    build_standing_ramp_messages,
)
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.manual_planner import ManualPlanner
from gear_sonic.utils.inference.planner_heading_frame import planner_command_in_reference_frame
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.xr_upperbody_bridge import feedback_payload_heading_yaw
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_CALIB_FULL_UPPER_BODY,
    G1_STANDING_UPPER_BODY,
    G1_UPPER_BODY_JOINT_INDICES,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    initial_pose: Literal["calib_full", "standing", "planner_standing"] = "calib_full"
    """Initial pose: CALIB_FULL ramp, latent standing blend, or feedback-paced planner standing."""

    initial_pose_ramp_s: float = 2.0
    """Seconds for the standing-to-CALIB_FULL planner ramp before latent pose handoff."""

    standing_ramp_s: float = 2.0
    """Seconds for the CALIB_FULL-to-standing planner ramp before stopping control."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation with initial_pose=standing. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""

    harness_endpoint: str | None = None
    """Opt-in local IPC control endpoint; pair with harness_profile."""
    harness_profile: str | None = None
    """Shared validated G1 profile. Authoritative for policy settings in harness mode."""
    harness_locomotion: bool = False
    """Enable bounded planner skills after simulation validation."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(f"left_hand_joints must have shape [7], got {left_hand_joints.shape}")
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(f"right_hand_joints must have shape [7], got {right_hand_joints.shape}")
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Training's data exporter uses all measured hand joints unchanged.
    # Preserve those measurements here, including independently measured fingers.
    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(projected_gravity, dtype=np.float32)[
        np.newaxis, np.newaxis
    ]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    close_fn=None,
    report_failures=False,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                epoch = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                inference_start_time = time.monotonic()
                observation = prepare_obs_fn()
                if observation is None:
                    if report_failures:
                        raise ValueError("Policy observation unavailable")
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                if isinstance(observation, dict):
                    inference_start_time = observation.pop("_harness_captured_at", inference_start_time)

                processed_action = inference_fn(observation)

                if processed_action is not None or report_failures:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time, epoch))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time, epoch))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time, epoch))
            except Exception:
                if report_failures:
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    result_queue.put_nowait((None, inference_start_time, epoch))
                else:
                    raise
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()

    if close_fn is not None:
        close_fn()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    pause_loop = True

    harness_enabled = config.harness_endpoint is not None
    if harness_enabled != (config.harness_profile is not None):
        raise ValueError("harness_endpoint and harness_profile must be supplied together")
    harness_control = None
    harness_rpc = None
    snapshots = None
    harness_planner_command = None
    policy_ready = False
    prewarm_epoch = None
    prewarm_started = None
    cached_capture_time = 0.0
    if harness_enabled:
        from gear_sonic.utils.inference.harness_control import HarnessControl, RuntimeFacts, validate_native_action
        from gear_sonic.utils.inference.harness_profile import load_profile
        from gear_sonic.utils.inference.harness_rpc import HarnessRPCServer
        from gear_sonic.utils.inference.observation_snapshot import ObservationSnapshotCache

        profile = load_profile(Path(config.harness_profile))
        config.host, config.port = profile.policy_host, profile.policy_port
        config.embodiment_tag = profile.embodiment
        config.action_horizon, config.action_publish_rate = profile.action_horizon, profile.publish_rate
        config.prompt = next(iter(profile.skills.values())).prompt

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    ready_link_boxes = ready_pelvis_height = None
    if harness_enabled and profile.handover is not None and profile.handover.desk_top_above_floor_m is not None:
        from gear_sonic.utils.inference.ready_clearance import (
            pelvis_height_estimator,
            ready_decision,
            right_arm_link_boxes,
        )

        # Own instance: forward kinematics overwrites the cache robot_model shares with inference.
        clearance_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        ready_link_boxes = right_arm_link_boxes(clearance_model)
        ready_pelvis_height = pelvis_height_estimator(clearance_model)
    ready_yaw_tau = None
    if harness_enabled and profile.handover is not None and profile.handover.ready_gravity_feedforward:
        from gear_sonic.utils.inference.ready_feedforward import right_yaw_gravity_tau

        ready_yaw_tau = right_yaw_gravity_tau(instantiate_g1_robot_model(waist_location="lower_and_upper_body"))

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = None if harness_enabled else PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if harness_enabled:
        print("Harness policy connects and prewarms on its inference worker; outputs stay paused.")
    elif n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = (
        None
        if harness_enabled
        else ZMQStateSubscriber(
            host=config.state_zmq_host,
            port=config.state_zmq_port,
        )
    )
    # Separate socket: the inference worker owns state_subscriber's reads.
    reset_state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    reset_feedback = None
    reset_feedback_index = None
    reset_feedback_time = float("-inf")
    standing_reset = None
    feedforward_reset = None  # The ready return that carries ready_yaw_tau, while it is standing_reset.
    manual_planner = None

    camera_subscriber = (
        None
        if harness_enabled
        else ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"
    initial_pose_ready = False
    pose_start_pending = False
    last_calib_full_hold_time = 0.0

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def _initial_pose_hands() -> tuple[np.ndarray, np.ndarray]:
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        return left_hand, right_hand

    def _feedback_upper_body(default: np.ndarray) -> np.ndarray:
        # The main loop already consumed this socket's latest message.
        state_msg = reset_feedback
        if state_msg is None:
            return default
        body_q = state_msg.get("body_q_measured", state_msg.get("body_q"))
        if body_q is None:
            return default
        body_q = np.asarray(body_q, dtype=np.float32).reshape(-1)
        if body_q.shape[0] >= max(G1_UPPER_BODY_JOINT_INDICES) + 1:
            return body_q[G1_UPPER_BODY_JOINT_INDICES].astype(np.float32)
        if body_q.shape[0] == default.shape[0]:
            return body_q.astype(np.float32)
        return default

    def _publish_messages_at_action_rate(messages: list[bytes]) -> None:
        for message in messages:
            zmq_socket.send(message)
            time.sleep(loop_period)

    def publish_latent_initial_pose(left_hand: np.ndarray | None = None, right_hand: np.ndarray | None = None):
        """Publish the streamed-motion latent initial token."""
        nonlocal last_sent_motion_token
        if left_hand is None or right_hand is None:
            left_hand, right_hand = _initial_pose_hands()
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        last_sent_motion_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)

    def publish_calib_full_pose(*, send_latent_handoff: bool):
        """Ramp planner target to CALIB_FULL and optionally seed streamed-motion mode."""
        print("Moving to CALIB_FULL pose")
        left_hand, right_hand = _initial_pose_hands()
        if config.initial_pose == "calib_full":
            start_upper_body = _feedback_upper_body(G1_STANDING_UPPER_BODY)
            ramp_messages = build_calib_full_ramp_messages(
                duration_s=config.initial_pose_ramp_s,
                rate_hz=config.action_publish_rate,
                start_upper_body=start_upper_body,
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            print(
                "Ramping planner target from standing to teleop CALIB_FULL "
                f"over {config.initial_pose_ramp_s:.2f}s ({len(ramp_messages)} frames)"
            )
            _publish_messages_at_action_rate(ramp_messages)
            print_green("Sent teleop CALIB_FULL planner ramp via ZMQ")

        if send_latent_handoff:
            publish_latent_initial_pose(left_hand=left_hand, right_hand=right_hand)
        print("CALIB_FULL pose done.")

    def publish_standing_pose():
        """Ramp planner target from current/CALIB_FULL upper body back to standing."""
        print("Ramping to standing pose")
        start_upper_body = _feedback_upper_body(G1_CALIB_FULL_UPPER_BODY)
        ramp_messages = build_standing_ramp_messages(
            duration_s=config.standing_ramp_s,
            rate_hz=config.action_publish_rate,
            start_upper_body=start_upper_body,
        )
        print(
            "Ramping planner target from CALIB_FULL to standing "
            f"over {config.standing_ramp_s:.2f}s ({len(ramp_messages)} frames)"
        )
        _publish_messages_at_action_rate(ramp_messages)
        print_green("Sent standing planner ramp via ZMQ")

    def publish_calib_full_hold_pose():
        """Refresh planner CALIB_FULL target to avoid deploy-side planner timeout."""
        left_hand, right_hand = _initial_pose_hands()
        zmq_socket.send(
            build_calib_full_hold_message(
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
        )

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_latent_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def begin_standing_reset():
        """Prime measured joints before switching back to the standing planner."""
        nonlocal standing_reset
        if reset_feedback is None or time.monotonic() - reset_feedback_time > 0.5:
            print("Cannot initialize: no fresh robot state. Policy remains paused.")
            return False
        if cpp_mode == "POSE" and np.array_equal(reset_feedback.get("planner_reference_active"), [1]):
            print("Cannot initialize: waiting for POSE feedback. Retry 'i' after the mode switch settles.")
            return False
        left_hand = (
            _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        )
        try:
            reset = StandingReset(reset_feedback, left_hand, right_hand)
            message = planner_command_in_reference_frame(reset.command, reset_feedback).encode()
        except (TypeError, ValueError) as exc:
            print(f"Cannot initialize: {exc}. Policy remains paused.")
            return False
        zmq_socket.send(message)
        if send_cpp_control_command(start=True, planner=True):
            standing_reset = reset
            print_green("Returning to straight standing; policy paused. Press 'p' when ready.")
            return True
        return False

    def fresh_heading():
        if reset_feedback is None or time.monotonic() - reset_feedback_time > 0.5:
            return None
        return feedback_payload_heading_yaw(reset_feedback)

    def stop_manual_motion():
        """Cancel movement before a reset or mode switch, including stale feedback."""
        heading = fresh_heading()
        if heading is not None:
            manual_planner.handle_key("z", heading)
        command = manual_planner.command(standing_reset.command, None)
        publish_planner_command(command)

    def publish_planner_command(command):
        # Reproject every frame: POSE -> PLANNER reanchors the reference.
        # StandingReset and ManualPlanner retain their desired WORLD heading.
        try:
            message = planner_command_in_reference_frame(command, reset_feedback).encode()
        except ValueError:
            return False
        zmq_socket.send(message)
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode, last_sent_motion_token
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            if not harness_enabled:
                time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            if planner or not start:
                last_sent_motion_token = None
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # LeRobot SONIC-token policies play like lerobot's sonic_policy_streamer.py (cross-fade, token slew limit).
    lerobot_policy = profile.lerobot_policy if harness_enabled else None
    token_stream = None
    if lerobot_policy is not None:
        from gear_sonic.utils.inference.lerobot_sonic_policy import (
            RIGHT_SWAP,
            LeRobotSonicPolicyClient,
            TokenStream,
            lerobot_state,
        )

        token_stream = TokenStream(
            config.action_publish_rate, lerobot_policy.chunk_blend_s, lerobot_policy.max_token_step
        )
        lerobot_right_order = RIGHT_SWAP if lerobot_policy.dex3_right_order == "swap" else np.arange(7)

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_epoch = 0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def invalidate_policy_actions():
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time, inference_epoch
        inference_epoch += 1
        cached_action_chunk = None
        action_chunk_index = 0
        last_inference_time = 0.0
        for pending in (inference_queue, result_queue):
            while True:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    break
        return inference_epoch

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode, initial_pose_ready
        nonlocal pose_start_pending
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token, standing_reset, manual_planner

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if harness_control is not None and (
            key.startswith(PROMPT_MSG_PREFIX)
            or key in {"i", "m", "p", "k", "[", "]"}
            or (manual_planner is not None and key in {"w", "s", "a", "d", "q", "e", "z"})
        ):
            was_owned = harness_control.owner is not None
            harness_control.operator_override(key)
            if key == "p" and was_owned:
                print("Agent ownership revoked; policy remains paused.")
                return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX) :]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (recording toggle -- handled by data exporter)")
        elif key == "x":
            print("Keyboard: 'x' (discard recording -- handled by data exporter)")
        elif key == "i" and config.initial_pose == "planner_standing":
            pause_loop = True
            initial_pose_ready = False
            pose_start_pending = False
            invalidate_policy_actions()
            if manual_planner is not None:
                stop_manual_motion()
            zmq_frame_counter = 0
            if cpp_loop_running:
                if begin_standing_reset():
                    manual_planner = None
                    initial_pose_ready = True
            else:
                print("C++ control loop is not running; press 'k' first.")
        elif key == "i" and config.initial_pose == "standing":
            if not cpp_loop_running:
                print("C++ control loop is not running; press 'k' first.")
                return
            pause_loop = True
            initial_pose_ready = False
            pose_start_pending = False
            invalidate_policy_actions()
            if manual_planner is not None:
                stop_manual_motion()
            manual_planner = None
            standing_reset = None
            zmq_frame_counter = 0
            if (
                cpp_mode == "POSE"
                and config.initial_pose_blend_duration > 0
                and last_sent_motion_token is not None
            ):
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_latent_initial_pose()
            if send_cpp_control_command(start=True, planner=False):
                initial_pose_ready = True
                print("Standing initial pose ready in POSE mode; press 'p' to start inference")
        elif key == "i":
            if manual_planner is not None:
                stop_manual_motion()
            manual_planner = None
            standing_reset = None
            transition = plan_i_transition(
                InferenceControlState(
                    pause_loop=pause_loop,
                    cpp_loop_running=cpp_loop_running,
                    cpp_mode=cpp_mode,
                    initial_pose_ready=initial_pose_ready,
                )
            )
            pause_loop = True
            pose_start_pending = False
            if transition.reset_frame_counter:
                zmq_frame_counter = 0
                print("Reset ZMQ frame counter")
            if transition.clear_action_cache:
                invalidate_policy_actions()
                print("Cleared cached action chunk")
            if transition.start_planner:
                print("Starting/switching C++ control loop in PLANNER mode for CALIB_FULL ramp...")
                send_cpp_control_command(start=True, planner=True)
            if transition.publish_calib_full:
                publish_calib_full_pose(send_latent_handoff=transition.start_pose)
            if transition.start_pose:
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode; press 'p' to start inference")
            pause_loop = transition.next_state.pause_loop
            cpp_loop_running = transition.next_state.cpp_loop_running
            cpp_mode = transition.next_state.cpp_mode
            initial_pose_ready = transition.next_state.initial_pose_ready
            print("Holding CALIB_FULL in PLANNER mode; press 'p' to start inference")
        elif key == "m":
            pause_loop = True
            invalidate_policy_actions()
            pose_start_pending = False
            initial_pose_ready = False
            if not cpp_loop_running:
                print("C++ loop not running - press 'k' to start")
                return
            if manual_planner is not None:
                stop_manual_motion()
                if begin_standing_reset():
                    manual_planner = None
                    print_green("Manual movement stopped; holding PLANNER. Use 'i' then 'p' for the next trial.")
            elif begin_standing_reset():
                facing = standing_reset.command.facing
                manual_planner = ManualPlanner(float(np.arctan2(facing[1], facing[0])))
                print_green(
                    "Manual planner: w/s forward/back, a/d left/right, q/e turn, "
                    "same key stops, z stops all; m exits. Enter after each key."
                )
        elif key in {"w", "s", "a", "d", "q", "e", "z"}:
            if manual_planner is None:
                print("Movement ignored - press 'm' to enter manual planner mode")
                return
            heading = fresh_heading()
            if heading is None:
                print("Movement ignored - no fresh robot heading")
                return
            manual_planner.handle_key(key, heading)
            active = manual_planner.active_key
            print_green(f"Manual movement: {active or 'STOPPED'} (same key or z stops)")
        elif key == "p":
            if harness_enabled and not policy_ready and pause_loop:
                print("Policy prewarm not ready; keep policy paused.")
                return
            if manual_planner is not None:
                print("Policy stays paused during manual repositioning - press 'm' to exit first")
                return
            if config.initial_pose == "planner_standing":
                if not cpp_loop_running or not initial_pose_ready:
                    print("Planner standing pose is not ready; press 'k' then 'i' first.")
                    return
                invalidate_policy_actions()
                pause_loop = not pause_loop
                pose_start_pending = not pause_loop and cpp_mode != "POSE"
                print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
                return

            if config.initial_pose == "standing":
                if not cpp_loop_running or not initial_pose_ready:
                    print("Standing initial pose is not ready; press 'k' then 'i' first.")
                    return
                pause_loop = not pause_loop
                pose_start_pending = False
                invalidate_policy_actions()
                print(f"{'Paused' if pause_loop else 'Resumed'} policy loop in POSE mode")
                return
            transition = plan_p_transition(
                InferenceControlState(
                    pause_loop=pause_loop,
                    cpp_loop_running=cpp_loop_running,
                    cpp_mode=cpp_mode,
                    initial_pose_ready=initial_pose_ready,
                )
            )
            if transition.blocked_reason:
                print(transition.blocked_reason)
                return
            if not transition.next_state.pause_loop:
                if transition.clear_action_cache:
                    invalidate_policy_actions()
                    print("Cleared cached action chunk")
                if transition.publish_latent_initial:
                    publish_latent_initial_pose()
                if transition.start_pose:
                    send_cpp_control_command(start=True, planner=False)
                pose_start_pending = transition.start_pose_on_next_action
                pause_loop = False
                cpp_loop_running = transition.next_state.cpp_loop_running
                cpp_mode = transition.next_state.cpp_mode
                initial_pose_ready = transition.next_state.initial_pose_ready
                if pose_start_pending:
                    print("Policy loop resumed; POSE mode will start on first VLA action")
                else:
                    print("Policy loop resumed")
                return

            pause_loop = True
            pose_start_pending = False
            invalidate_policy_actions()
            print("Policy loop paused; returning to CALIB_FULL pose")
            if transition.start_planner:
                send_cpp_control_command(start=True, planner=True)
            if transition.publish_calib_full:
                publish_calib_full_pose(send_latent_handoff=False)
            cpp_loop_running = transition.next_state.cpp_loop_running
            cpp_mode = transition.next_state.cpp_mode
            initial_pose_ready = transition.next_state.initial_pose_ready
            print("Paused at CALIB_FULL (press 'p' to resume, 'k' to stand and stop)")
        elif key == "k":
            invalidate_policy_actions()
            if manual_planner is not None:
                stop_manual_motion()
            manual_planner = None
            standing_reset = None
            transition = plan_k_transition(
                InferenceControlState(
                    pause_loop=pause_loop,
                    cpp_loop_running=cpp_loop_running,
                    cpp_mode=cpp_mode,
                    initial_pose_ready=initial_pose_ready,
                )
            )
            if not cpp_loop_running:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    pose_start_pending = False
                    pause_loop = transition.next_state.pause_loop
                    cpp_loop_running = transition.next_state.cpp_loop_running
                    cpp_mode = transition.next_state.cpp_mode
                    initial_pose_ready = transition.next_state.initial_pose_ready
                    print("Started C++ control loop in PLANNER mode")
                    print(f"Press 'i' to prepare {config.initial_pose} and arm inference")
                return

            pause_loop = True
            pose_start_pending = False
            print(f"Stopping C++ control loop from {cpp_mode} mode via standing ramp...")
            if transition.start_planner and not send_cpp_control_command(start=True, planner=True):
                print("Cannot stop: PLANNER switch failed; controller state is unchanged.")
                return
            if transition.publish_standing:
                publish_standing_pose()
            if transition.stop_control and not send_cpp_control_command(start=False, planner=True):
                print("Cannot confirm stop: controller remains in its last known mode.")
                return
            cpp_loop_running = transition.next_state.cpp_loop_running
            cpp_mode = transition.next_state.cpp_mode
            initial_pose_ready = transition.next_state.initial_pose_ready
            print("Stopped C++ control loop after standing ramp")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}")
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(f"Initial pose right hand: {'closed' if initial_pose_right_hand_closed else 'open'}")

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    worker_sensors = {}

    def prepare_worker_observation():
        if not harness_enabled:
            return prepare_observation_from_sensors(
                camera_subscriber, state_subscriber, robot_model, language_prompt_ref[0], True
            )
        if not worker_sensors:
            worker_sensors["state"] = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
            client = LeRobotSonicPolicyClient if lerobot_policy is not None else PolicyClient
            options = {} if lerobot_policy is None else {"right_order": lerobot_policy.dex3_right_order}
            worker_sensors["policy"] = client(
                host=config.host, port=config.port, timeout_ms=int(profile.limits.prewarm_deadline_s * 1000), **options
            )
        state_msg = worker_sensors["state"].get_msg()
        now = time.monotonic()
        camera_snapshot = snapshots.latest_camera(now)
        if camera_snapshot is None or state_msg is None or not harness_control._fresh(now):
            return None
        camera_msg, captured = camera_snapshot
        observation = prepare_observation_from_sensors(
            SimpleNamespace(read=lambda: camera_msg),
            SimpleNamespace(get_msg=lambda: state_msg),
            robot_model,
            language_prompt_ref[0],
            True,
        )
        if observation is not None:
            observation["_harness_captured_at"] = captured
            if lerobot_policy is not None:
                observation["lerobot"] = {
                    "state": lerobot_state(state_msg, lerobot_right_order),
                    "image": camera_msg["images"]["ego_view"],
                    "task": language_prompt_ref[0],
                }
        return observation

    def close_worker_sensors():
        for sensor in worker_sensors.values():
            close = getattr(sensor, "close", None) or getattr(sensor, "close_client", None)
            if close:
                close()

    if harness_enabled:

        class LoopHooks:
            def runtime_facts(self):
                operator_busy = manual_planner is not None or (
                    standing_reset is not None
                    and harness_control.owner is None
                    and (
                        reset_feedback is None
                        or not standing_reset.is_settled(
                            reset_feedback,
                            profile.limits.reset_joint_tolerance_rad,
                            profile.limits.reset_yaw_tolerance_rad,
                        )
                    )
                )
                return RuntimeFacts(
                    cpp_loop_running,
                    not pause_loop,
                    policy_ready,
                    inference_busy_event.is_set() or not inference_queue.empty(),
                    cpp_mode,
                    operator_busy,
                )

            def invalidate_policy_actions(self):
                return invalidate_policy_actions()

            def set_policy_prompt(self, prompt):
                language_prompt_ref[0] = prompt

            def set_policy_enabled(self, enabled):
                nonlocal pause_loop, standing_reset, manual_planner, harness_planner_command
                if enabled:
                    if not cpp_loop_running or not send_cpp_control_command(start=True, planner=False):
                        raise ValueError("Controller is not running")
                    standing_reset, manual_planner, harness_planner_command = None, None, None
                    if token_stream is not None:
                        # Slew from the planner token the deploy decodes now, as the streamer's handoff does.
                        planner = np.asarray((reset_feedback or {}).get("token_state", []), np.float32).reshape(-1)
                        token_stream.reset(start_token=planner if planner.shape == (64,) else None)
                pause_loop = not enabled

            def request_planner_hold(self, feedback, open_hands):
                nonlocal standing_reset, manual_planner, harness_planner_command
                reset = StandingReset(
                    feedback,
                    feedback.get("left_hand_q_measured", feedback.get("left_hand_q")),
                    feedback.get("right_hand_q_measured", feedback.get("right_hand_q")),
                )
                standing_reset, manual_planner = None, None
                harness_planner_command = reset.command
                if not publish_planner_command(harness_planner_command):
                    raise ValueError("Cannot encode measured planner hold")
                if not cpp_loop_running or not send_cpp_control_command(start=True, planner=True):
                    raise ValueError("Cannot activate planner hold")

            def begin_standing_reset(self, feedback, open_hands):
                nonlocal standing_reset, manual_planner, harness_planner_command
                left = (
                    np.zeros(7)
                    if open_hands
                    else feedback.get("left_hand_q_measured", feedback.get("left_hand_q"))
                )
                right = (
                    np.zeros(7)
                    if open_hands
                    else feedback.get("right_hand_q_measured", feedback.get("right_hand_q"))
                )
                reset = StandingReset(
                    feedback, left, right, compensate_tracking_bias=True,
                    joint_tolerance_rad=harness_control.profile.limits.reset_joint_tolerance_rad,
                )
                if not publish_planner_command(reset.command):
                    raise ValueError("Cannot encode standing reset")
                if not cpp_loop_running or not send_cpp_control_command(start=True, planner=True):
                    raise ValueError("Cannot activate planner")
                standing_reset, manual_planner, harness_planner_command = reset, None, None
                return reset

            def set_planner_command(self, command):
                nonlocal harness_planner_command, standing_reset
                planner_command_in_reference_frame(command, reset_feedback).encode()
                harness_planner_command, standing_reset = command, None

            def begin_ready_reset(self, feedback, right_arm_target, right_arm_waypoint):
                nonlocal standing_reset, manual_planner, harness_planner_command, feedforward_reset
                if ready_link_boxes is not None:
                    right_arm_waypoint, reason = ready_decision(
                        harness_control.profile.handover, feedback, ready_link_boxes, ready_pelvis_height)
                    print(f"[harness] ready return: {'direct' if right_arm_waypoint is None else 'raise first'}"
                          f" ({reason})", flush=True)
                reset = StandingReset(
                    feedback, feedback["left_hand_q_measured"], np.zeros(7),
                    right_arm_target=right_arm_target, compensate_tracking_bias=True,
                    right_arm_waypoint=right_arm_waypoint,
                    left_arm_target=harness_control.profile.handover.left_arm_target,
                    joint_tolerance_rad=harness_control.profile.limits.reset_joint_tolerance_rad,
                    yaw_tolerance_rad=harness_control.profile.limits.reset_yaw_tolerance_rad,
                    settle_dwell_s=harness_control.profile.limits.settle_dwell_s,
                )
                if not publish_planner_command(reset.command):
                    raise ValueError("Cannot encode ready reset")
                if not cpp_loop_running or not send_cpp_control_command(start=True, planner=True):
                    raise ValueError("Cannot activate ready planner")
                standing_reset, manual_planner, harness_planner_command = reset, None, None
                feedforward_reset = reset if ready_yaw_tau is not None else None
                return reset

            def stop_planner_motion(self):
                nonlocal harness_planner_command, standing_reset, manual_planner
                command = harness_planner_command
                if command is None and standing_reset is not None:
                    command = standing_reset.command
                if command is not None:
                    harness_planner_command = replace(command, mode=0, movement=(0.0, 0.0, 0.0), speed=0.0)
                standing_reset, manual_planner = None, None

        harness_control = HarnessControl(Path(config.harness_profile), LoopHooks(), uuid.uuid4().hex)
        harness_control.locomotion_enabled = config.harness_locomotion
        snapshots = ObservationSnapshotCache(
            lambda: ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port),
            "ego_view",
            profile.limits.camera_max_age_s,
        )
        harness_rpc = HarnessRPCServer(config.harness_endpoint, snapshots)
        snapshots.start()
        harness_rpc.start()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            prepare_worker_observation,
            lambda obs: run_policy_inference_and_process(
                policy=worker_sensors["policy"] if harness_enabled else n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
            close_worker_sensors,
        ),
        kwargs={"report_failures": harness_enabled},
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        last_reset_tick = time.monotonic()
        while True:
            t_start = time.monotonic()
            feedback = reset_state_subscriber.get_msg()
            if feedback is not None:
                if not harness_enabled or reset_feedback is None or feedback.get("index") != reset_feedback_index:
                    reset_feedback = feedback
                    reset_feedback_index = feedback.get("index")
                    reset_feedback_time = t_start
            check_keyboard_input()

            if harness_control is not None:
                harness_control.tick(t_start, reset_feedback, reset_feedback_time)
                harness_rpc.drain(harness_control, t_start)
                if pause_loop:
                    try:
                        warm_action, _, warm_epoch = result_queue.get_nowait()
                        if prewarm_epoch is not None and warm_epoch == prewarm_epoch:
                            policy_ready = (
                                validate_native_action(warm_action)
                                and time.monotonic() - prewarm_started <= profile.limits.prewarm_deadline_s
                            )
                            prewarm_epoch = None
                    except queue.Empty:
                        pass
                    if (
                        not policy_ready
                        and not inference_busy_event.is_set()
                        and inference_queue.empty()
                        and harness_control.observation is not None
                        and harness_control._fresh(t_start)
                    ):
                        if prewarm_started is None:
                            prewarm_started = t_start
                        if t_start - prewarm_started <= profile.limits.prewarm_deadline_s:
                            prewarm_epoch = inference_epoch
                            inference_queue.put_nowait(prewarm_epoch)

            if standing_reset is not None and cpp_loop_running and cpp_mode == "PLANNER":
                fresh_feedback = reset_feedback if t_start - reset_feedback_time <= 0.5 else None
                if (
                    harness_control is not None
                    and harness_control.owner is not None
                    and (harness_control.phase != "RESETTING" or not harness_control.hold_confirmed)
                ):
                    fresh_feedback = None
                try:
                    command = standing_reset.advance(fresh_feedback, t_start - last_reset_tick)
                except ValueError as error:
                    if harness_control is not None and harness_control.owner is not None:
                        harness_control.interrupt("reset_reference_lead_exceeded", t_start)
                    else:
                        pause_loop = True
                        invalidate_policy_actions()
                        try:
                            hold = StandingReset(
                                reset_feedback,
                                reset_feedback.get("left_hand_q_measured", reset_feedback.get("left_hand_q")),
                                reset_feedback.get("right_hand_q_measured", reset_feedback.get("right_hand_q")),
                            ).command
                        except (TypeError, ValueError):
                            hold = standing_reset.command
                        standing_reset, manual_planner, harness_planner_command = None, None, hold
                        publish_planner_command(hold)
                    print(f"Standing reset interrupted: {error}")
                    last_reset_tick = t_start
                    _sleep_remaining(t_start, loop_period)
                    continue
                if manual_planner is not None:
                    command = manual_planner.command(command, fresh_heading())
                if standing_reset is feedforward_reset and fresh_feedback is not None:
                    # Stale feedback sends none; the deploy then applies zero.
                    command = replace(command, upper_body_tau=ready_yaw_tau(fresh_feedback))
                publish_planner_command(command)
            elif harness_planner_command is not None and cpp_loop_running and cpp_mode == "PLANNER":
                publish_planner_command(harness_planner_command)
            last_reset_tick = t_start

            if pause_loop:
                if (
                    config.initial_pose == "calib_full"
                    and cpp_loop_running
                    and cpp_mode == "PLANNER"
                    and initial_pose_ready
                    and standing_reset is None
                    and manual_planner is None
                    # A harness hold owns the planner; its measured targets must not be replaced.
                    and harness_planner_command is None
                ):
                    now = time.monotonic()
                    if now - last_calib_full_hold_time >= 0.2:
                        publish_calib_full_hold_pose()
                        last_calib_full_hold_time = now
                _sleep_remaining(t_start, loop_period)
                continue

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time, result_epoch = result_queue.get_nowait()
                if result_epoch != inference_epoch:
                    # A request already in flight during pause/reset cannot be reused.
                    _sleep_remaining(t_start, loop_period)
                    continue
                if (
                    harness_control is not None
                    and harness_control.owner is not None
                    and not harness_control.accept_policy_result(
                        result_epoch, inference_start_time, processed_action, time.monotonic()
                    )
                ):
                    _sleep_remaining(t_start, loop_period)
                    continue
                if harness_enabled and (
                    processed_action is None
                    or not validate_native_action(processed_action)
                    or not 0
                    <= time.monotonic() - inference_start_time
                    < config.action_horizon / config.action_publish_rate
                ):
                    harness_control.interrupt("invalid_or_failed_policy_result", time.monotonic())
                    _sleep_remaining(t_start, loop_period)
                    continue
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                cached_capture_time = inference_start_time
                if token_stream is not None:
                    rows = np.concatenate(
                        [np.asarray(processed_action[key], np.float32)[0]
                         for key in ("motion_token", "left_hand_joints", "right_hand_joints")],
                        axis=1,
                    )
                    token_stream.set_chunk(rows, t0=inference_start_time, now=time.monotonic())
                # Harness chunks expire from observation capture, so start the
                # next query on that cadence too. Waiting a full interval after
                # acceptance needlessly spends the preceding query's latency
                # a second time and can exhaust otherwise fresh chunks.
                last_inference_time = inference_start_time if harness_enabled else time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                if harness_enabled and (
                    not harness_control._fresh(time.monotonic())
                    or harness_control.observation is None
                    or time.monotonic() - harness_control.observation["received_at"]
                    > profile.limits.camera_max_age_s
                ):
                    harness_control.interrupt("stale_policy_observation", time.monotonic())
                    _sleep_remaining(t_start, loop_period)
                    continue
                try:
                    inference_queue.put_nowait(inference_epoch)
                except queue.Full:
                    pass

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    if harness_enabled and (
                        action_chunk_index >= config.action_horizon
                        or time.monotonic() - cached_capture_time
                        >= config.action_horizon / config.action_publish_rate
                    ):
                        harness_control.interrupt("action_chunk_exhausted", time.monotonic())
                        _sleep_remaining(t_start, loop_period)
                        continue
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]
                    if token_stream is not None:
                        motion_token, hands = token_stream.at(time.monotonic())
                        left_hand_joints, right_hand_joints = hands[:7], hands[7:]

                    if pose_start_pending and cpp_mode != "POSE":
                        print("Starting POSE mode on first VLA action")
                        if not send_cpp_control_command(start=True, planner=False):
                            _sleep_remaining(t_start, loop_period)
                            continue
                        pose_start_pending = False

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - frame: {frame_index[0]}, token shape: {motion_token.shape}"
                        )

                action_chunk_index = (
                    action_chunk_index + 1
                    if harness_enabled
                    else min(action_chunk_index + 1, config.action_horizon - 1)
                )

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(context="VLA Inference Loop Missed", threshold=0.001)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        if harness_rpc is not None:
            harness_rpc.close()
        if snapshots is not None:
            snapshots.close()
        zmq_socket.close()
        zmq_context.term()
        if state_subscriber is not None:
            state_subscriber.close()
        reset_state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
