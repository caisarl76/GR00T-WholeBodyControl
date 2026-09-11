"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --dataset-name my_session
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Literal

# Direct script launches must use this checkout, even with another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.data.pico_hand_features import (
    DEX3_API_TO_MODEL,
    INSPIRE_RANGES,
    hand_episode_features,
    join_hand_frame,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)
from gear_sonic.utils.teleop.pico_recording import RecorderProtocol, RecordingState
from gear_sonic.utils.teleop.zmq.zmq_message_decoder import unpack_pose_message
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = "demo"
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    use_dummy_camera: bool = False
    """Use a black ego-view image instead of connecting to a camera server."""

    hand_profile: Literal["dex3", "inspire_ftp"] = "dex3"
    legacy_recording_controls: bool = False
    recording_status_port: int = 5562
    inspire_status_host: str = "localhost"
    inspire_status_port: int = 5563

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        use_dummy_camera: bool = False,
        hand_profile: str = "dex3",
        legacy_recording_controls: bool = False,
        recording_status_port: int = 5562,
        inspire_status_host: str = "localhost",
        inspire_status_port: int = 5563,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model

        self._episode_state = EpisodeState()
        self.hand_profile = hand_profile
        self.legacy_recording_controls = legacy_recording_controls
        self._legacy_save_failed = False
        if legacy_recording_controls and hand_profile != "dex3":
            raise ValueError("legacy recording supports only Dex3")
        self._keyboard_listener = ZMQKeyboardSubscriber() if legacy_recording_controls else None
        self.recorder = RecorderProtocol(hand_profile)
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-save")
        self._save_future = None
        self._saving_episode_index = None
        self._proprio_received_ns = None
        self._last_status_ns = None
        self._body_packets = {}
        self._hand_diagnostics = None
        self._inspire_status = None
        self._last_recorded_generation = None
        self.dropped_hand_frames = 0
        self._recording_context = zmq.Context()
        self._recording_status_socket = self._recording_context.socket(zmq.PUB)
        self._recording_status_socket.setsockopt(zmq.LINGER, 0)
        self._recording_status_socket.bind(f"tcp://127.0.0.1:{recording_status_port}")
        self._inspire_status_socket = None
        if hand_profile == "inspire_ftp":
            self._inspire_status_socket = self._recording_context.socket(zmq.SUB)
            self._inspire_status_socket.setsockopt(zmq.LINGER, 0)
            self._inspire_status_socket.setsockopt_string(zmq.SUBSCRIBE, "inspire_hand_status")
            self._inspire_status_socket.connect(f"tcp://{inspire_status_host}:{inspire_status_port}")
        self._pc2_session = None
        self._pc2_retired = set()
        self._pc2_seq = -1
        self._pc2_healthy_streak = 0

        self.use_dummy_camera = use_dummy_camera
        self._dummy_image = np.zeros((EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3), dtype=np.uint8)
        if self.use_dummy_camera:
            self._image_subscriber = None
            print("[Camera] Dummy camera enabled — writing black ego_view frames")
        else:
            self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._state_subscriber = ZMQStateSubscriber(
            host=state_zmq_host,
            port=state_zmq_port,
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        try:
            self._sonic_zmq_ctx = zmq.Context()
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "hand_tracking")
            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {sonic_data_zmq_host}:{sonic_data_zmq_port}")
            print("[Sonic] Subscribed to: pose, planner, manager_state")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(max_failures=3, reset_timeout_sec=5, time_delta=0.1)

        self._last_latency_log_time = 0.0
        self._initial_yaw = None

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return (
            self._saving_episode_index
            if self._saving_episode_index is not None
            else self.data_exporter.episode_buffer["episode_index"]
        )

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return
        if not self.legacy_recording_controls:
            for key in ("body_q", "last_action"):
                values = np.asarray(msg.get(key))
                if (
                    values.shape != (29,)
                    or not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all()
                ):
                    return

        if msg.get("ros_timestamp", 0.0) == 0.0:
            msg["ros_timestamp"] = time.time()

        self.latest_proprio_msg = msg
        self._proprio_received_ns = time.monotonic_ns()

    def _read_image_msg(self):
        if self.use_dummy_camera:
            return {
                "images": {"ego_view": self._dummy_image.copy()},
                "timestamps": {"ego_view": time.time()},
            }
        return self._image_subscriber.read()

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        if not self.legacy_recording_controls:
            return
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                self._print_and_say(f"Started recording {self.current_episode_index}", blocking=False)
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save", blocking=False)
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state", blocking=False)
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._save_legacy_episode(discarded=True)
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._print_and_say("Discarded episode", blocking=False)

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for pose, planner, and manager_state messages (non-blocking)."""
        if self._sonic_zmq_socket is None:
            return

        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"hand_tracking"):
                self._handle_hand_tracking(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        managed = "recording_protocol_version" in data
        if self.legacy_recording_controls:
            if managed:
                raise RuntimeError("--legacy-recording-controls cannot consume managed recording fields")
            if "stream_mode" in data:
                self.current_stream_mode = int(data["stream_mode"].flat[0])
            self._manager_toggle_dc |= self._extract_bool(data, "toggle_data_collection")
            self._manager_toggle_da |= self._extract_bool(data, "toggle_data_abort")
            return
        if not managed:
            return
        if data["version"] != 4:
            return
        before = (self.recorder.manager_session_id, self.recorder._mode_epoch)
        event = self.recorder.receive(data, time.monotonic_ns())
        if self.recorder.manager_session_id is not None:
            self.current_stream_mode = self.recorder._mode
        if before != (self.recorder.manager_session_id, self.recorder._mode_epoch):
            self._last_recorded_generation = None
        self._apply_recording_event(event)
        self._publish_recording_status(force=True)

    def _handle_hand_tracking(self, raw):
        try:
            data = unpack_pose_message(raw, topic="hand_tracking")
        except (ValueError, TypeError, KeyError):
            return
        self._hand_diagnostics = {**data, "received_ns": time.monotonic_ns()}

    def _poll_inspire_status(self):
        if self._inspire_status_socket is None:
            return
        from gear_sonic.utils.teleop.pico_inspire_protocol import validate_inspire_status

        for _ in range(20):
            try:
                raw = self._inspire_status_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                fields = validate_inspire_status(unpack_pose_message(raw, topic="inspire_hand_status"))
                session = fields["pc2_session_id"].tobytes()
                sequence = int(fields["status_seq"].item())
                if session in self._pc2_retired:
                    continue
                if session != self._pc2_session:
                    if len(self._pc2_retired) >= 64:
                        self._inspire_status = None
                        continue
                    if self._pc2_session is not None:
                        self._pc2_retired.add(self._pc2_session)
                    self._pc2_session, self._pc2_seq = session, -1
                    self._pc2_healthy_streak = 0
                    self._inspire_status = None
                if sequence <= self._pc2_seq:
                    continue
                now_ns = time.monotonic_ns()
                if (
                    self._inspire_status is not None
                    and now_ns - self._inspire_status["received_ns"] >= 500_000_000
                ):
                    self._pc2_healthy_streak = 0
                self._pc2_seq = sequence
                healthy = bool(fields["feedback_healthy"].item()) and fields["fault_code"].item() == 0
                self._pc2_healthy_streak = self._pc2_healthy_streak + 1 if healthy else 0
                self._inspire_status = {**fields, "received_ns": now_ns, "ready": self._pc2_healthy_streak >= 10}
            except (ValueError, TypeError, KeyError):
                continue

    def _publish_recording_status(self, *, force=False):
        if self.legacy_recording_controls:
            return
        now_ns = time.monotonic_ns()
        if not force and self._last_status_ns is not None and now_ns - self._last_status_ns < 50_000_000:
            return
        status = self.recorder.status_fields(int(self.current_episode_index))
        self._recording_status_socket.send(
            pack_pose_message(status, topic="recording_status", version=1), zmq.NOBLOCK
        )
        self._last_status_ns = now_ns

    def _apply_recording_event(self, event):
        if event == "start":
            self._initial_yaw = None
            self._last_recorded_generation = None
            self._print_and_say(f"Started recording {self.current_episode_index}", blocking=False)
        elif event in ("save", "abort"):
            if self._save_future is not None:
                raise RuntimeError("concurrent episode save")
            self._saving_episode_index = int(self.data_exporter.episode_buffer["episode_index"])
            self._save_future = self._save_executor.submit(self._save_owned_episode, event == "abort")
        if event is not None:
            self._publish_recording_status(force=True)

    def _save_owned_episode(self, discarded):
        """Only this worker touches the exporter until completion reaches the loop."""
        original = copy.deepcopy(self.data_exporter.episode_buffer)
        try:
            if original.get("size", 0) == 0:
                # No episode was produced; successful idle is an explicit no-op.
                return
            if discarded:
                self.data_exporter.save_episode_as_discarded()
            else:
                self.data_exporter.save_episode()

        except Exception:
            self.data_exporter.episode_buffer = original
            raise

    def _service_recording(self):
        if self.legacy_recording_controls:
            return
        if self._save_future is not None and self._save_future.done():
            error = self._save_future.exception()
            self._save_future = None
            self._saving_episode_index = None
            self.recorder.finish_save(error is None)
            if error is not None:
                self._print_and_say(f"SAVE ERROR; original buffer and artifacts preserved: {error}", say=False)
            else:
                self._initial_yaw = None
                self.sonic_timing_monitor.reset()
            self._publish_recording_status(force=True)
        now_ns = time.monotonic_ns()
        feedback_age = None if self._proprio_received_ns is None else now_ns - self._proprio_received_ns
        self._apply_recording_event(self.recorder.check_timeout(now_ns, feedback_age))
        self._publish_recording_status()

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

        self._body_packets["planner"] = {**data, "received_ns": time.monotonic_ns()}
        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
            self._body_packets["pose"] = {**pose_data, "received_ns": time.monotonic_ns()}
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][0],
                "smpl_pose": smpl_pose,
                "body_quat_w": (pose_data["body_quat_w"][0] if "body_quat_w" in pose_data else None),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "receive_timestamp": time.time(),
            }
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    def _extract_hand_joints(self, pose_data: dict, key: str) -> np.ndarray | None:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(7, dtype=np.float32) if self.legacy_recording_controls else None

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        if not self.legacy_recording_controls:
            return True
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self._save_legacy_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                self._print_and_say("Finished saving episode")
            else:
                self._print_and_say("Skipping save: no frames collected", say=False)
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            self._print_and_say(
                f"Waiting for message. "
                f"Avail msg: proprio {self.latest_proprio_msg is not None} | "
                f"image {self.latest_image_msg is not None}",
                say=False,
            )
            return False

        recording = (
            self._episode_state.get_state() == self._episode_state.RECORDING
            if self.legacy_recording_controls
            else self.recorder.capture_active
        )
        if not recording:
            return self._finalize_frame(t_start)

        return self._add_data_frame_sonic(t_start)

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        hand_features = None
        if not self.legacy_recording_controls:
            try:
                topic = "pose" if self.current_stream_mode == 1 else "planner"
                body = self._body_packets.get(topic)
                hand_features = join_hand_frame(
                    body,
                    self._hand_diagnostics,
                    self.hand_profile,
                    time.monotonic_ns(),
                    self.recorder.manager_session_id,
                    self.current_stream_mode,
                    self._inspire_status,
                    manager_epoch=self.recorder._mode_epoch,
                )
                generation = (body["pv"].tobytes(), int(body["sample_generation"].item()))
                if generation == self._last_recorded_generation:
                    return False
            except (ValueError, KeyError, TypeError):
                self.dropped_hand_frames += 1
                return False
        if self.hand_profile == "inspire_ftp":
            measured = [hand_features[f"observation.{side}_inspire_hand_state"] for side in ("left", "right")]
            applied = [hand_features[f"action.{side}_inspire_hand_applied"] for side in ("left", "right")]
            measured = [(1 - q) * INSPIRE_RANGES for q in measured]
            applied = [(1 - q) * INSPIRE_RANGES for q in applied]
        else:
            try:
                measured = [np.asarray(proprio[f"{side}_hand_q"])[DEX3_API_TO_MODEL] for side in ("left", "right")]
                applied = [
                    np.asarray(proprio[f"last_{side}_hand_action"])[DEX3_API_TO_MODEL]
                    for side in ("left", "right")
                ]
            except (KeyError, IndexError, TypeError, ValueError):
                self.dropped_hand_frames += 1
                return False
            if not self.legacy_recording_controls:
                # g1_debug can continue while one physical DDS hand stream freezes.
                try:
                    now_ns = time.monotonic_ns()
                    if (
                        self._proprio_received_ns is None
                        or not 0 <= now_ns - self._proprio_received_ns < 100_000_000
                    ):
                        raise ValueError("stale Dex3 feedback transport")
                    for side in ("left", "right"):
                        age = proprio[f"{side}_hand_feedback_age_ns"]
                        if (
                            proprio[f"{side}_hand_feedback_valid"] is not True
                            or type(age) is not int
                            or age < 0
                            or age + now_ns - self._proprio_received_ns >= 100_000_000
                        ):
                            raise ValueError("stale Dex3 physical feedback")
                    if any(q.shape != (7,) or not np.isfinite(q).all() for q in measured + applied):
                        raise ValueError("invalid Dex3 state/action")
                except (KeyError, ValueError, TypeError):
                    self.dropped_hand_frames += 1
                    return False
        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["body_q"],
            left_hand_actuated_joint_values=measured[0],
            right_hand_actuated_joint_values=measured[1],
        )
        whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["last_action"],
            left_hand_actuated_joint_values=applied[0],
            right_hand_actuated_joint_values=applied[1],
        )

        self.robot_model.cache_forward_kinematics(whole_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(self.robot_model.supplemental_info.hand_frame_names[side])
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        if hand_features is not None:
            frame_data.update(hand_features)
            if self.hand_profile == "inspire_ftp":
                frame_data.pop("teleop.left_hand_joints", None)
                frame_data.pop("teleop.right_hand_joints", None)

        self._add_images_to_frame_data(frame_data)

        self._log_latency_periodic(sonic_latency_ms)

        self.data_exporter.add_frame(frame_data)
        if hand_features is not None:
            self._last_recorded_generation = generation
        return self._finalize_frame(t_start)

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            frame_data["observation.root_orientation"] = base_quat
            frame_data["observation.projected_gravity"] = compute_projected_gravity(base_quat).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            frame_data["observation.root_orientation"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            frame_data["observation.projected_gravity"] = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            frame_data["observation.cpp_rotation_offset"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(proprio["init_base_quat"], dtype=np.float64)
        else:
            frame_data["observation.init_base_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(self, frame_data: dict) -> float | None:
        """Add teleop features based on current stream mode."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode == 5 and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = (
            smpl_msg
            if self.current_stream_mode in (1, 4) and smpl_msg is not None
            else planner_msg
            if planner_msg is not None
            else smpl_msg
        )
        if self.legacy_recording_controls:
            frame_data["teleop.left_hand_joints"] = (
                hand_msg["left_hand_joints"].astype(np.float32)
                if hand_msg is not None and hand_msg.get("left_hand_joints") is not None
                else np.zeros(7, dtype=np.float32)
            )
            frame_data["teleop.right_hand_joints"] = (
                hand_msg["right_hand_joints"].astype(np.float32)
                if hand_msg is not None and hand_msg.get("right_hand_joints") is not None
                else np.zeros(7, dtype=np.float32)
            )

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return sonic_latency_ms

    def _compute_target_body_orientation(self, body_quat_w: np.ndarray, frame_data: dict) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False).as_quat(scalar_first=True).astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def _save_legacy_episode(self, *, discarded=False):
        if self._legacy_save_failed or self.data_exporter.episode_buffer.get("size", 0) == 0:
            return
        # An exception or interrupt may leave partially written artifacts. Never
        # retry that episode from cleanup; successful saves enable the next one.
        self._legacy_save_failed = True
        if discarded:
            self.data_exporter.save_episode_as_discarded()
        else:
            self.data_exporter.save_episode()
        self._legacy_save_failed = False

    def save_and_cleanup(self):
        if self.legacy_recording_controls:
            try:
                self._save_legacy_episode()
            except Exception as error:
                self._print_and_say(f"Error saving episode: {error}", say=False)
        else:
            if self.recorder.capture_active:
                self.recorder.state = RecordingState.SAVING
                self._apply_recording_event("abort")
            while self._save_future is not None:
                self._service_recording()
                time.sleep(0.01)
        self._save_executor.shutdown(wait=True)
        self._state_subscriber.close()
        if self._keyboard_listener is not None:
            self._keyboard_listener.close()
        if self._sonic_zmq_socket is not None:
            self._sonic_zmq_socket.close(linger=0)
        if self._sonic_zmq_ctx is not None:
            self._sonic_zmq_ctx.term()
        if self._inspire_status_socket is not None:
            self._inspire_status_socket.close(linger=0)
        self._recording_status_socket.close(linger=0)
        self._recording_context.term()
        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    self._poll_inspire_status()
                    self._service_recording()
                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    with self.telemetry.timer("poll_image"):
                        img_msg = self._read_image_msg()
                        if img_msg is not None:
                            self.latest_image_msg = img_msg

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(context="Data Exporter Loop Missed", threshold=0.001)

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            if self.legacy_recording_controls:
                self._save_legacy_episode(discarded=True)

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: SonicDataExporterConfig):
    g1_rm = get_g1_robot_model(hand_profile=config.hand_profile)

    dataset_features = get_features_sonic_vla(g1_rm)
    modality_config = get_modality_config_sonic_vla(g1_rm)
    if not config.legacy_recording_controls:
        dataset_features.update(hand_episode_features(config.hand_profile))
    if config.hand_profile == "inspire_ftp":
        for side in ("left", "right"):
            dataset_features.pop(f"teleop.{side}_hand_joints")
            modality_config["action"].pop(f"{side}_hand_joints")
        for name, feature in hand_episode_features(config.hand_profile).items():
            prefix, key = name.split(".", 1)
            group = "state" if prefix == "observation" else "action"
            modality_config[group][key] = {"start": 0, "end": feature["shape"][0], "original_key": name}

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout)

    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "hand_profile": config.hand_profile,
            "legacy_recording_controls": config.legacy_recording_controls,
            "record_wrist_cameras": config.record_wrist_cameras,
            "use_dummy_camera": config.use_dummy_camera,
        },
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        use_dummy_camera=config.use_dummy_camera,
        hand_profile=config.hand_profile,
        legacy_recording_controls=config.legacy_recording_controls,
        recording_status_port=config.recording_status_port,
        inspire_status_host=config.inspire_status_host,
        inspire_status_port=config.inspire_status_port,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
