"""Manager-owned optical hands and fresh physical feedback, without motor I/O."""

import json

import msgpack
import numpy as np
import zmq

from gear_sonic.utils.teleop.pico_hand_retargeting import HandRetargeter
from gear_sonic.utils.teleop.pico_hand_tracking import HandTracker
from gear_sonic.utils.teleop.pico_inspire_protocol import validate_inspire_status
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, unpack_pose_message


class PicoPublisher:
    """Attach this tick's provenance and hands at the single body send boundary."""

    def __init__(self, socket, hand_input, hand_profile, hands=None):
        self.socket, self.hand_input, self.hand_profile, self.hands = socket, hand_input, hand_profile, hands
        self.pv = None
        self.generation = 0
        self.body_sent = False
        self.last_commands = None

    def send(self, message):
        topic = "pose" if message.startswith(b"pose") else "planner" if message.startswith(b"planner") else None
        if topic is None:
            self.socket.send(message)
            return
        data = unpack_pose_message(message, topic)
        version = data.pop("version")
        data.pop("endian")
        data["pv"] = self.pv
        data["sample_generation"] = np.array([self.generation], dtype=np.int64)
        # Keep the fixed body header within its budget; these controller/source
        # diagnostics are not inputs to the robot body decoder.
        for field in (
            "left_trigger",
            "right_trigger",
            "left_grip",
            "right_grip",
            "pico_dt",
            "pico_fps",
            "toggle_data_collection",
            "toggle_data_abort",
        ):
            data.pop(field, None)
        if self.hand_input != "controller":
            for side in ("left", "right"):
                data.pop(f"{side}_hand_joints", None)
            if self.hands is not None and self.hand_profile == "dex3":
                for side, command in zip(("left", "right"), self.hands.commands()):
                    if command is not None:
                        data[f"{side}_hand_joints"] = command
        self.last_commands = [data.get(f"{side}_hand_joints") for side in ("left", "right")]
        self.socket.send(pack_pose_message(data, topic, version))
        self.body_sent = True


class PicoHandRuntime:
    def __init__(self, sdk, context, profile, feedback_endpoint, *, max_rate=None):
        for side in ("left", "right"):
            if not callable(getattr(sdk, f"get_{side}_hand_snapshot", None)):
                raise RuntimeError("Optical mode needs the rebuilt XR atomic-snapshot binding")
        self.sdk = sdk
        self.profile = profile
        self.feedback_endpoint = feedback_endpoint
        self.feedback_error = None
        self.trackers = [
            HandTracker(HandRetargeter(profile, side), profile, max_rate) for side in ("left", "right")
        ]
        self.socket = context.socket(zmq.SUB)
        self.topic = "g1_debug" if profile == "dex3" else "inspire_hand_status"
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(feedback_endpoint)
        self.feedback = None
        self.received_ns = None
        self.feedback_sequence = -1
        self.pc2_session = None
        self.retired_pc2 = set()
        self.healthy_streak = 0
        self.outputs = [None, None]
        self.snapshots = [None, None]
        self._timestamp_sources = [None, None]
        self._last_input_diagnostic_ns = None
        self.body_wrists = [None, None]
        self.measured_inputs = [None, None]
        self.pv = None
        self.generation = 0
        self._inspire_seq = 0
        self._last_pv = None

    def poll_feedback(self, now_ns):
        if (
            self.profile == "inspire_ftp"
            and self.received_ns is not None
            and not 0 <= now_ns - self.received_ns < 500_000_000
        ):
            self.healthy_streak = 0
        try:
            raw = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            if self.profile == "dex3":
                data = msgpack.unpackb(raw[len(self.topic) :], raw=False)
                if not isinstance(data, dict) or type(data.get("index")) is not int:
                    raise ValueError("Invalid Dex3 feedback index")
                sequence = data["index"]
                if not 0 <= sequence <= np.iinfo(np.int64).max:
                    raise ValueError("Invalid Dex3 feedback index")
                if sequence == self.feedback_sequence:
                    return
                if sequence < self.feedback_sequence:
                    self._reset_trackers()
            else:
                data = validate_inspire_status(unpack_pose_message(raw, self.topic))
                session = data["pc2_session_id"].tobytes()
                sequence = int(data["status_seq"][0])
                if session in self.retired_pc2:
                    return
                if session != self.pc2_session:
                    if self.pc2_session is not None:
                        self.retired_pc2.add(self.pc2_session)
                    if len(self.retired_pc2) > 64:
                        raise ValueError("Too many PC2 restarts")
                    self.pc2_session = session
                    self.feedback_sequence = -1
                    self.healthy_streak = 0
                    self._reset_trackers()
                if sequence <= self.feedback_sequence:
                    return
                self.healthy_streak = self.healthy_streak + 1 if data["feedback_healthy"][0] else 0
            self.feedback = data
            self.feedback_error = None
            self.feedback_sequence = sequence
            self.received_ns = now_ns
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, msgpack.UnpackException) as exc:
            self.feedback = None
            self.healthy_streak = 0
            self.feedback_error = f"{type(exc).__name__}: {exc}"

    def _reset_trackers(self):
        self.trackers = [HandTracker(t.retargeter, self.profile, t.max_rate) for t in self.trackers]
        self.outputs = [None, None]

    def measured(self, now_ns):
        limit = 100_000_000 if self.profile == "dex3" else 500_000_000
        if self.feedback is None or self.received_ns is None or not 0 <= now_ns - self.received_ns < limit:
            return [None, None]
        if self.profile == "inspire_ftp":
            if (
                self.healthy_streak < 10
                or not self.feedback["feedback_healthy"][0]
                or self.feedback["bridge_state"][0] not in (1, 2)
            ):
                return [None, None]
            return [self.feedback[f"{side}_angle_act"].astype(np.float64) / 1000 for side in ("left", "right")]
        measured = []
        for side in ("left", "right"):
            valid = self.feedback.get(f"{side}_hand_feedback_valid")
            age = self.feedback.get(f"{side}_hand_feedback_age_ns", -1)
            if valid is True and type(age) is int and 0 <= age + now_ns - self.received_ns < limit:
                # *_q_measured belongs to visualization and can contain the command.
                measured.append(self.feedback.get(f"{side}_hand_q"))
            else:
                measured.append(None)
        return measured

    def step(self, sample, now_ns, *, enabled):
        self.poll_feedback(now_ns)
        measured = self.measured(now_ns)
        body = None
        try:
            if sample is not None and 0 <= now_ns - int(sample.get("timestamp_monotonic", 0) * 1e9) < 100_000_000:
                poses = np.asarray(sample.get("body_poses_np"), dtype=np.float64)
                if poses.shape == (24, 7):
                    body = poses
        except (TypeError, ValueError, AttributeError, OverflowError):
            pass
        self.measured_inputs = measured
        self.body_wrists = [None if body is None else body[20 + index, :3] for index in range(2)]
        for index, side in enumerate(("left", "right")):
            try:
                snapshot = getattr(self.sdk, f"get_{side}_hand_snapshot")()
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError):
                snapshot = None
            self.snapshots[index] = snapshot
            previous = self.outputs[index]
            output = self.trackers[index].step(
                snapshot,
                self.body_wrists[index],
                measured[index],
                now_ns,
                enabled=enabled,
            )
            self.outputs[index] = output
            source = self.trackers[index].timestamp_source
            if source is not None and source != self._timestamp_sources[index]:
                self._timestamp_sources[index] = source
                detail = (
                    "original APK compatibility; unchanged hand data holds after 100 ms"
                    if source == 1
                    else "per-hand device timestamps"
                )
                print(f"[Hands] {side}: {detail}")
            if previous is None or (previous.state, previous.reason) != (output.state, output.reason):
                print(f"{side.upper()} HAND {output.state.name}: {output.reason.name}")
        if enabled and (
            self._last_input_diagnostic_ns is None or now_ns - self._last_input_diagnostic_ns >= 5_000_000_000
        ):
            self._last_input_diagnostic_ns = now_ns
            if any(t.binding_generation == 0 for t in self.trackers):
                for detail in self.input_diagnostics():
                    print(f"[Hands input] {detail}")
            if not self.ready(now_ns):
                for detail in self.feedback_blockers(now_ns):
                    print(f"[Hands feedback] {detail}")

    def input_diagnostics(self):
        getter = getattr(self.sdk, "get_hand_packet_diagnostics", None)
        if not callable(getter):
            return ["No accepted hand samples; restart manager with the rebuilt binding for packet diagnostics"]
        try:
            diagnostic = getter()
            if not diagnostic["packets"]:
                return ["No XR packets received"]
            if not diagnostic["hand_json"]:
                return ["XR packets have no Hand section; enable Tracking > Hand and Send in the headset app"]
            hand = json.loads(diagnostic["hand_json"])
            if not isinstance(hand, dict):
                return [f"Hand section must be an object, received {type(hand).__name__}"]
            result = []
            for side in ("leftHand", "rightHand"):
                data = hand.get(side)
                if not isinstance(data, dict):
                    result.append(f"{side}: no hand object; put controllers down and enable headset hand tracking")
                    continue
                joints = data.get("HandJointLocations")
                joint_count = len(joints) if isinstance(joints, list) else type(joints).__name__
                result.append(
                    f"{side}: keys={list(data)}, isActive={data.get('isActive')!r}, "
                    f"scale={data.get('scale')!r}, joints={joint_count}, "
                    f"first_joint={joints[0] if isinstance(joints, list) and joints else None}"
                )
            return result
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return [f"Hand packet diagnostic unavailable: {type(exc).__name__}: {exc}"]

    def ready(self, now_ns):
        return all(t._bounded(q, measured=True) is not None for t, q in zip(self.trackers, self.measured(now_ns)))

    def feedback_blockers(self, now_ns):
        """Explain admission failure without changing its freshness or limit checks."""
        if self.feedback is None:
            detail = (
                f"Invalid {self.topic}: {self.feedback_error}"
                if self.feedback_error
                else (f"No {self.topic} received from {self.feedback_endpoint}; check deployment output")
            )
            return [detail]
        limit = 100_000_000 if self.profile == "dex3" else 500_000_000
        if self.received_ns is None or not 0 <= now_ns - self.received_ns < limit:
            return [f"{self.topic} transport stale; check whether deployment control stopped"]
        blockers = []
        for side, tracker, measured in zip(("left", "right"), self.trackers, self.measured(now_ns)):
            if self.profile == "dex3":
                required = (f"{side}_hand_feedback_valid", f"{side}_hand_feedback_age_ns")
                missing = [key for key in required if key not in self.feedback]
                if missing:
                    blockers.append(f"{side}: missing {', '.join(missing)}; rebuild deployment in this worktree")
                    continue
                if measured is None:
                    blockers.append(
                        f"{side}: DDS feedback valid={self.feedback[required[0]]}, "
                        f"age_ns={self.feedback[required[1]]}; check simulated hand states and Dex3 enablement"
                    )
                    continue
            elif measured is None:
                blockers.append(
                    f"{side}: PC2 feedback not admitted; healthy_streak={self.healthy_streak}, "
                    f"bridge_state={int(self.feedback['bridge_state'][0])}"
                )
                continue
            if tracker._bounded(measured, measured=True) is None:
                blockers.append(
                    f"{side}: measured joints malformed, nonfinite or outside command limits: {measured}"
                )
        return blockers

    def commands(self):
        return [None if out is None else out.command for out in self.outputs]

    def diagnostics(self, pv, generation, *, body_sent):
        fields = {
            "pv": pv,
            "sample_generation": np.array([generation], dtype=np.int64),
            "hand_profile": np.array([0 if self.profile == "dex3" else 1], dtype=np.int32),
            "body_sent": np.array([body_sent], dtype=bool),
            "hand_input": np.array([0], dtype=np.int32),
        }
        for side, out, tracker in zip(("left", "right"), self.outputs, self.trackers):
            if out is None or out.command is None:
                return None
            fields.update(
                {
                    f"{side}_state": np.array([out.state], dtype=np.int32),
                    f"{side}_source_epoch": np.array([out.source_epoch], dtype=np.int64),
                    f"{side}_source_timestamp_ns": np.array([out.source_timestamp_ns], dtype=np.int64),
                    f"{side}_timestamp_source": np.array(
                        [-1 if tracker.timestamp_source is None else tracker.timestamp_source], dtype=np.int32
                    ),
                    f"{side}_valid": np.array([out.valid], dtype=bool),
                    f"{side}_command": out.command,
                }
            )
        return fields

    def inspire_command(self, pv, generation):
        if self.profile != "inspire_ftp" or any(q is None for q in self.commands()):
            return None
        if self._last_pv != pv.tobytes():
            self._last_pv = pv.tobytes()
            self._inspire_seq = 0
        fields = {
            "pv": pv,
            "message_seq": np.array([self._inspire_seq], dtype=np.int64),
            "sample_generation": np.array([generation], dtype=np.int64),
        }
        for side, q in zip(("left", "right"), self.commands()):
            fields[f"{side}_command"] = q
        for field, attribute, dtype in (
            ("tracking_state", "state", np.int32),
            ("source_epoch", "source_epoch", np.int64),
            ("source_timestamp_ns", "source_timestamp_ns", np.int64),
        ):
            for side, out in zip(("left", "right"), self.outputs):
                fields[f"{side}_{field}"] = np.array([getattr(out, attribute)], dtype=dtype)
        if self._inspire_seq == np.iinfo(np.int64).max:
            raise RuntimeError("Inspire sequence exhausted")
        self._inspire_seq += 1
        return fields

    def close(self):
        self.socket.close()
