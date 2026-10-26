"""Main-loop ownership and lifecycle for the opt-in G1 harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from numbers import Real
import os
from pathlib import Path
import time
from typing import Protocol
import uuid

import numpy as np

from .harness_contract import SkillCall, decode_request
from .harness_profile import load_profile, validate_skill_call
from .harness_hand_pose import classify_right_hand, HAND_FRAME_MAX_SKEW_S
from .standing_reset import StandingReset, _measured_joints


@lru_cache(maxsize=1)
def monotonic_clock_id():
    """Match C++ receipt times only on the same Linux boot and time namespace."""
    try:
        identity = (Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                    + '|' + os.readlink('/proc/self/ns/time'))
        return tuple(ord(char) for char in identity)
    except OSError:
        return ()


@dataclass(frozen=True)
class RuntimeFacts:
    controller_running: bool
    policy_enabled: bool
    policy_ready: bool
    inference_busy: bool
    mode_requested: str | None
    operator_busy: bool = False
    served_model: str | None = None  # the policy server's info "model"; None until its client has asked


class RuntimeHooks(Protocol):
    def runtime_facts(self) -> RuntimeFacts: ...
    def invalidate_policy_actions(self) -> int: ...
    def set_policy_prompt(self, prompt: str) -> None: ...
    def set_policy_enabled(self, enabled: bool) -> None: ...
    def request_planner_hold(self, feedback: dict, open_hands: bool) -> None: ...
    def begin_standing_reset(self, feedback: dict, open_hands: bool) -> StandingReset: ...
    def begin_ready_reset(self, feedback: dict, right_arm_target, right_arm_waypoint) -> StandingReset: ...
    def set_planner_command(self, command) -> None: ...
    def stop_planner_motion(self) -> None: ...
    # Autonomous loop (LoopController): POSE mode start, PLANNER mode start, enqueue one inference epoch.
    def switch_to_pose(self) -> None: ...
    def switch_to_planner(self) -> None: ...
    def request_policy_chunk(self) -> int: ...


def load_harness_profile(path: Path) -> dict:
    return asdict(load_profile(path))


def validate_native_action(action):
    try:
        for key, width in [("motion_token", 64), ("left_hand_joints", 7), ("right_hand_joints", 7)]:
            array = np.asarray(action[key])
            if array.shape != (1, 40, width) or not np.isfinite(array).all():
                return False
        return bool(np.max(np.abs(action["motion_token"])) <= 1.25)
    except (ValueError, TypeError, KeyError):
        return False


class ControlError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class HarnessControl:
    def __init__(self, profile_path: Path, hooks: RuntimeHooks, runtime_id: str, loop=None):
        self.profile = load_profile(profile_path)
        self.hooks, self.runtime_id = hooks, runtime_id
        self.loop = loop  # LoopController: owns the LOOP_* phases, which self.phase mirrors
        self.owner = None
        self.execution = None
        self.phase, self.reason, self.epoch = "IDLE", None, 0
        self.feedback, self.feedback_time, self.feedback_index = None, None, None
        self.observation = None
        self.reset = None
        self.planner = None
        self.hold_confirmed = False
        self.hold_index, self.hold_started, self.dwell = None, None, None
        self.transition_started = 0.0
        self.last_policy_result = None
        # Mutation replies of the owning session, for exact retries. Heartbeats use a
        # per-session sequence instead, so retries are exact without growing this.
        self.replies = {}
        self.heartbeat = None  # (session_id, sequence, canonical request, reply) of the last renewal.
        # Ended sessions -> the request that ended them and its reply (or None). Every
        # other request from them is rejected, so their caches can be dropped.
        self.retired_sessions = {}
        self.locomotion_enabled = False

    def _fresh(self, now):
        return (
            self.feedback is not None
            and self.feedback_time is not None
            and max(0.0, now - self.feedback_time) <= self.profile.limits.feedback_max_age_s
            and self._compatible_feedback(self.feedback)
        )

    @staticmethod
    def _compatible_feedback(feedback):
        try:
            body = np.asarray(feedback["body_q_measured_motor"], dtype=float)
            capability = np.asarray(feedback["harness_planner_hold_enabled"], dtype=float)
            hands = [
                np.asarray(feedback[key], dtype=float)
                for key in ("left_hand_q_measured", "right_hand_q_measured")
            ]
            # The controller reports stale or absent Dex3 feedback through these flags.
            stale_hand = any(
                key in feedback and feedback[key] is not True
                for key in ("left_hand_feedback_valid", "right_hand_feedback_valid")
            )
            return bool(
                body.shape == (29,) and np.isfinite(body).all() and capability.shape == (1,) and capability[0] == 1
                and all(hand.shape == (7,) and np.isfinite(hand).all() for hand in hands) and not stale_hand
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _planner_active(self):
        value = np.asarray(self.feedback.get("planner_reference_active", []) if self.feedback else []).reshape(-1)
        return bool(value.shape == (1,) and value[0] == 1)

    def _hold(self, now):
        self.hold_confirmed = False
        self.hold_index, self.hold_started = self.feedback_index, now
        self.reset, self.planner, self.dwell = None, None, None
        if not self._fresh(now):
            self.phase = "FAULT"
            self.hold_started = None
            return
        try:
            _measured_joints(self.feedback)
            self.hooks.request_planner_hold(self.feedback, False)
        except (TypeError, ValueError):
            self.phase = "FAULT"

    def _in_loop(self):
        return self.loop is not None and self.phase.startswith("LOOP_")

    def _sync_loop(self, now):
        if not self._in_loop():
            return
        self.phase, self.reason = self.loop.phase, self.loop.reason
        if not self._in_loop():  # handed back (COMPLETED) or take-over aborted (IDLE): confirm the planner hold
            self.hold_confirmed, self.hold_index, self.hold_started = False, self.feedback_index, now

    def interrupt(self, reason: str, now: float):
        self._sync_loop(now)
        if self._in_loop():  # a stop freezes the loop; the planner hold is only the session end's hand-back
            self.loop.freeze("lease expired" if reason == "lease_expired" else f"interrupted: {reason}", now)
            self._sync_loop(now)
            return
        self.epoch = self.hooks.invalidate_policy_actions()
        self.hooks.set_policy_enabled(False)
        self.hooks.stop_planner_motion()
        self.phase, self.reason = "INTERRUPTED", reason
        self._hold(now)

    def operator_override(self, reason: str):
        """True when the key was consumed: in a LOOP phase it only freezes the loop (owner kept, and the caller
        must not perform the key's own action)."""
        self._sync_loop(time.monotonic())  # loop.tick may have left the LOOP phases since the last HC tick
        if self._in_loop():
            self.loop.freeze(f"operator override: {reason}", time.monotonic())
            self._sync_loop(time.monotonic())
            return True
        if self.owner is None and self.execution is None and self.phase == "IDLE":
            return
        if self.owner is not None:
            self.interrupt(f"operator_override:{reason}", time.monotonic())
        else:
            self.epoch = self.hooks.invalidate_policy_actions()
        self._retire_owner()
        self.reset, self.planner = None, None
        self.hold_confirmed = False
        self.phase = "IDLE" if reason == "i" else "INTERRUPTED"
        self.reason = f"operator_override:{reason}"

    def _retire_owner(self):
        """End ownership: reject the session from now on, then drop its retry caches."""
        if self.owner is not None:
            session = self.owner["session_id"]
            self.retired_sessions.setdefault(session, None)
            self.replies = {key: value for key, value in self.replies.items() if key[0] != session}
            self.heartbeat = None
        self.owner = None

    def _lease(self, req, now):
        if (
            not self.owner
            or req["session_id"] != self.owner["session_id"]
            or req["lease_id"] != self.owner["lease_id"]
        ):
            raise ControlError("NOT_OWNER", "No matching owner")
        if now >= self.owner["expires_at"]:
            self.interrupt("lease_expired", now)
            self._retire_owner()
            raise ControlError("EXPIRED_LEASE", "Ownership expired")

    def _execution(self):
        return dict(
            runtime_id=self.runtime_id,
            execution_id=self.execution["execution_id"],
            skill_id=self.execution["skill_id"],
            phase=self.phase,
            inference_epoch=self.epoch,
            started_at=self.execution["started_at"],
            reason=self.reason,
        )

    def _handover_enabled(self):
        h = self.profile.handover
        return bool(h and h.checkpoint_verified and h.ready_pose_reviewed
                    and callable(getattr(self.hooks, "begin_ready_reset", None)))

    def _new_execution(self, skill_id, now):
        self.execution = dict(execution_id=uuid.uuid4().hex, skill_id=skill_id, started_at=now)
        self.reason, self.dwell = None, None
        self.transition_started = now

    def dispatch(self, request: dict, now: float) -> dict:
        rid = request.get("request_id", "") if isinstance(request, dict) else ""
        response = dict(request_id=rid, runtime_id=self.runtime_id, result=None, error=None)
        try:
            req = asdict(decode_request(json.dumps(request, allow_nan=False).encode()))
            if req["runtime_id"] not in (None, self.runtime_id):
                raise ControlError("STALE_RUNTIME", "Executor restarted")
            key = (req["session_id"], req["request_id"])
            canonical = json.dumps(req, sort_keys=True)
            if req["session_id"] in self.retired_sessions and req["method"] not in {"get_status", "observe"}:
                # Before any cache lookup: a delayed retry must not reacquire or renew control.
                final = self.retired_sessions[req["session_id"]]
                if final is not None and final[0] == canonical:
                    return final[1]  # Exact retry of the request that ended the session.
                raise ControlError("NOT_OWNER", "Session ended; claim with a new session ID")
            if req["method"] == "heartbeat":
                return self._heartbeat(req, canonical, response, now)
            if key in self.replies:
                previous, cached = self.replies[key]
                if previous != canonical:
                    raise ControlError("REQUEST_CONFLICT", "Request ID reused with a different payload")
                return cached
            result = self._dispatch(req, now)
            response["result"] = result
            # Observe/status can be large and never actuate. Mutations are deduplicated.
            if req["method"] not in {"get_status", "observe"}:
                if req["session_id"] in self.retired_sessions:
                    self.retired_sessions[req["session_id"]] = (canonical, response)
                else:
                    self.replies[key] = (canonical, response)
        except ControlError as exc:
            response["error"] = dict(code=exc.code, message=str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            response["error"] = dict(code="INVALID_REQUEST", message=str(exc))
        return response

    def _heartbeat(self, req, canonical, response, now):
        """Renew once per sequence number; replay the last reply for its exact retry."""
        sequence = req["params"]["sequence"]
        if self.heartbeat is not None and self.heartbeat[0] == req["session_id"]:
            _, last, previous, cached = self.heartbeat
            if sequence == last:
                if previous != canonical:
                    raise ControlError("REQUEST_CONFLICT", "Heartbeat sequence reused with a different request")
                return cached
            if sequence < last:
                raise ControlError("REQUEST_CONFLICT", "Heartbeat sequence must increase")
        response["result"] = self._dispatch(req, now)
        self.heartbeat = (req["session_id"], sequence, canonical, response)
        return response

    def _dispatch(self, req, now):
        self._sync_loop(now)  # loop.tick may have left the LOOP phases since the last HC tick
        method, params = req["method"], req["params"]
        if method == "get_status":
            return self.status(now)
        if method == "observe":
            if (
                self.observation is None
                or now - self.observation["received_at"] > self.profile.limits.camera_max_age_s
            ):
                raise ControlError("UNAVAILABLE", "No fresh ego_view")
            hand = dict(right_hand_state="unknown", right_hand_sample_at=None,
                        right_hand_sample_index=None)
            try:
                receipt = self.feedback.get('right_hand_received_at', []) if self.feedback else []
                clock = tuple(self.feedback.get('controller_clock_id', [])) if self.feedback else ()
                if (self._fresh(now) and monotonic_clock_id() and clock == monotonic_clock_id()
                        and len(receipt) == 1 and isinstance(receipt[0], Real)
                        and not isinstance(receipt[0], bool) and math.isfinite(receipt[0])):
                    at = float(receipt[0])
                    if (at >= 0 and 0 <= now - at <= self.profile.limits.feedback_max_age_s
                            and abs(at - self.observation['received_at']) <= HAND_FRAME_MAX_SKEW_S):
                        hand.update(right_hand_state=classify_right_hand(self.feedback['right_hand_q_measured']),
                                    right_hand_sample_at=at,
                                    right_hand_sample_index=int(round(at * 1_000_000_000)))
            except (TypeError, ValueError, OverflowError):
                pass
            return {
                **self.observation,
                **hand,
                "age_s": max(0.0, now - self.observation["received_at"]),
                "runtime_id": self.runtime_id,
                "execution_id": self.execution["execution_id"] if self.execution else None,
                "inference_epoch": self.epoch,
            }
        if method == "claim_control":
            if params["registry_sha256"] != self.profile.registry_sha256:
                raise ControlError("PROFILE_MISMATCH", "Profile digest mismatch")
            if self.owner:
                raise ControlError("BUSY", "Executor already owned")
            facts = self.hooks.runtime_facts()
            if (
                not facts.controller_running
                or facts.policy_enabled
                or getattr(facts, "operator_busy", False)
                or self.phase not in {"IDLE", "COMPLETED", "PAUSED", "LOOP_FROZEN"}
            ):
                raise ControlError("NOT_READY", "Operator must start control and prepare a paused executor")
            if not self._fresh(now):
                raise ControlError(
                    "NOT_READY", "Fresh measured body/hand telemetry and C++ --harness-planner-hold required"
                )
            self.owner = dict(
                runtime_id=self.runtime_id,
                session_id=req["session_id"],
                lease_id=uuid.uuid4().hex,
                expires_at=now + self.profile.limits.lease_s,
            )
            if self.profile.handover and self._handover_enabled() and not self._in_loop():
                self._hold(now)
                if self.phase == "FAULT":
                    self._retire_owner()
                    raise ControlError("FAULT", "Cannot establish initial handover hold")
            return self.owner.copy()
        self._lease(req, now)
        if method == "heartbeat":
            self.owner["expires_at"] = now + self.profile.limits.lease_s
            return self.status(now)
        if method == "release_control" and self._in_loop():  # R16b: leave the loop frozen, reclaimable
            self.loop.freeze("released", now)
            self._sync_loop(now)
            self._retire_owner()
            return self.status(now)
        if method == "release_control":
            if not self.hold_confirmed or self.phase not in {"IDLE", "PAUSED", "COMPLETED", "INTERRUPTED", "FAULT"}:
                raise ControlError("NOT_READY", "Release requires a confirmed planner hold")
            self._retire_owner()
            return self.status(now)
        if method == "start_manipulation":
            try:
                skill = self.profile.require_skill(params["skill_id"])
            except ValueError as exc:
                raise ControlError("UNSUPPORTED_SKILL", str(exc)) from exc
            if self.profile.handover and skill.skill_id == self.profile.handover.skill_id and not self._handover_enabled():
                raise ControlError("NOT_READY", "Handover checkpoint/ready return not verified")
            facts = self.hooks.runtime_facts()
            if (
                self.phase not in {"IDLE", "COMPLETED", "PAUSED"}
                or not facts.policy_ready
                or facts.inference_busy
                or not facts.controller_running
                or not self._fresh(now)
                or (self.phase == "PAUSED" and (not self.hold_confirmed or not self._planner_active()))
            ):
                raise ControlError("NOT_READY", "Paused prepared executor required")
            if (
                self.observation is None
                or not 0 <= now - self.observation["received_at"] <= self.profile.limits.camera_max_age_s
            ):
                raise ControlError("NOT_READY", "Fresh camera required")
            self.epoch = self.hooks.invalidate_policy_actions()
            self.hooks.set_policy_prompt(skill.prompt)
            self._new_execution(skill.skill_id, now)
            self.phase, self.last_policy_result = "MANIPULATING", None
            self.hold_started, self.hold_index = None, None
            self.reset, self.planner, self.hold_confirmed = None, None, False
            self.hooks.set_policy_enabled(True)
            return self._execution()
        if method in {"walk_for", "turn_by"}:
            return self._begin_locomotion(method, params, now)
        if method.startswith("loop_"):
            try:
                return self._loop_call(method, params, now)
            except OSError as exc:  # e.g. loop_trial's session_dir: a bad request, never out of dispatch
                raise ControlError("INVALID_REQUEST", str(exc)) from exc
        standalone = method in {"reset_standing", "reset_ready"} and params["execution_id"] == ""
        if not standalone and (not self.execution or params["execution_id"] != self.execution["execution_id"]):
            raise ControlError("INVALID_REQUEST", "Execution mismatch")
        if method == "cancel":
            self.interrupt("cancelled", now)
            return self._execution()
        if method == "pause_manipulation":
            if self.phase != "MANIPULATING":
                raise ControlError("NOT_READY", "Manipulation is not running")
            self.interrupt("paused", now)
            if self.phase != "FAULT":
                self.phase, self.reason = "PAUSED", None
            return self._execution()
        if method in {"reset_standing", "reset_ready"}:
            if method == "reset_ready" and not self._handover_enabled():
                raise ControlError("NOT_READY", "Verified handover ready return unavailable")
            if self.phase not in ({"IDLE", "COMPLETED"} if standalone else {"PAUSED"}) or not self._fresh(now):
                raise ControlError("NOT_READY", "Pause and fresh telemetry required before reset")
            if standalone:
                self._new_execution(method, now)
            self.epoch = self.hooks.invalidate_policy_actions()
            self.hooks.set_policy_enabled(False)
            self.reset = (
                self.hooks.begin_ready_reset(self.feedback, self.profile.handover.right_arm_target,
                                             self.profile.handover.right_arm_waypoint)
                if method == "reset_ready"
                else self.hooks.begin_standing_reset(self.feedback, params["open_hands"])
            )
            self.phase, self.transition_started, self.dwell = "RESETTING", now, None
            self.hold_confirmed, self.hold_index, self.hold_started = False, self.feedback_index, now
            return self._execution()
        raise ControlError("INVALID_REQUEST", "Unknown method")

    def _loop_call(self, method, params, now):
        if self.loop is None:
            raise ControlError("UNSUPPORTED_SKILL", "Autonomous loop unavailable")
        if method == "loop_take_over":
            lerobot, handover = self.profile.lerobot_policy, self.profile.handover
            if (
                self.phase not in {"IDLE", "COMPLETED"}
                or not self.hooks.runtime_facts().controller_running
                or not (lerobot and lerobot.chunk_anchor_s > 0)
                or not (handover and handover.checkpoint_verified)
                or not self._fresh(now)
            ):
                raise ControlError(
                    "NOT_READY", "Take-over needs IDLE/COMPLETED, a running controller, chunk anchoring, "
                    "a verified handover checkpoint and fresh telemetry")
            if lerobot.dex3_right_order != "dataset":  # the ready pose and the wire would mix right-hand orders
                raise ControlError("NOT_READY", "The loop supports only lerobot_policy.dex3_right_order: dataset")
            served = getattr(self.hooks.runtime_facts(), "served_model", None)
            if not served or os.path.normpath(served) != os.path.normpath(self.profile.checkpoint):
                raise ControlError(
                    "NOT_READY", f"Policy server model {served!r} is not the profile checkpoint "
                    f"{self.profile.checkpoint!r} (None: not known yet)")
            self.loop.take_over(self.feedback, now, model=served)
            self.execution, self.reset, self.planner, self.dwell = None, None, None, None
            self.hold_confirmed, self.hold_started, self.hold_index = False, None, None
            self.phase = self.loop.phase
        elif not self._in_loop():
            raise ControlError("NOT_READY", "Autonomous loop not taken over")
        elif method == "loop_raise":
            self.loop.raise_(now)
        elif method == "loop_trial":
            self.loop.trial(params["trial"], params["session_dir"], now)
        elif method == "loop_freeze":
            self.loop.freeze("operator stop", now)
        elif method == "loop_resume":
            self.loop.resume(now)
        elif method == "loop_to_ready":
            self.loop.to_ready(now)
        elif method == "loop_end":
            self.loop.end(now)
        self._sync_loop(now)
        return self.status(now)

    def _begin_locomotion(self, method, params, now):
        if not self.locomotion_enabled:
            raise ControlError("UNSUPPORTED_SKILL", "Locomotion disabled")
        validate_skill_call(self.profile, SkillCall(method, params), locomotion_enabled=True)
        facts = self.hooks.runtime_facts()
        if (
            not self.hold_confirmed
            or not self._fresh(now)
            or not self._planner_active()
            or facts.policy_enabled
            or not facts.controller_running
            or self.phase not in {"IDLE", "PAUSED", "COMPLETED"}
        ):
            raise ControlError("NOT_READY", "Confirmed planner hold required")
        from .bounded_planner import BoundedPlanner

        self._new_execution(method, now)
        self.planner = BoundedPlanner(self.feedback, now, self.profile.limits)
        if method == "walk_for":
            self.planner.begin_walk(**params)
            self.phase = "WALKING"
        else:
            self.planner.begin_turn(**params)
            self.phase = "TURNING"
        self.hold_confirmed = False
        self.hold_started, self.hold_index = None, None
        self.reset = None
        return self._execution()

    def tick(self, now, feedback, feedback_received_at):
        self._sync_loop(now)  # LoopController.tick ran since the last tick
        advanced = False
        if feedback is not None:
            index = feedback.get("index")
            if type(index) is int and index >= 0 and (self.feedback_index is None or index > self.feedback_index):
                self.feedback, self.feedback_index, self.feedback_time = feedback, index, feedback_received_at
                advanced = True
                if not self._compatible_feedback(feedback) and (
                    self.owner is not None or self.phase in {"MANIPULATING", "RESETTING", "WALKING", "TURNING"}
                    or self._in_loop()
                ):
                    in_loop = self._in_loop()
                    self.interrupt("incompatible_controller_telemetry", now)
                    self._retire_owner()
                    if not in_loop:  # a loop stays frozen for a reclaim
                        self.phase = "FAULT"
                    return
            elif type(index) is int and self.feedback_index is not None and index < self.feedback_index:
                self.feedback, self.feedback_time, self.feedback_index = None, None, index
                self.interrupt("telemetry_stream_restarted", now)
                self._retire_owner()
                self.hold_started = None
        if self.owner and now >= self.owner["expires_at"]:
            self.interrupt("lease_expired", now)
            self._retire_owner()
        active = self.phase in {"MANIPULATING", "RESETTING", "WALKING", "TURNING"} or self._in_loop()
        if active and not self._fresh(now):
            self.interrupt("stale_feedback", now)
        if not self._fresh(now):
            self.hold_confirmed, self.dwell = False, None
            return
        if not self._planner_active():
            self.hold_confirmed, self.dwell = False, None
        if (
            self.hold_started is not None
            and advanced
            and self.feedback_index > self.hold_index
            and self._planner_active()
        ):
            self.hold_confirmed = True
        if (
            self.hold_started is not None
            and not self.hold_confirmed
            and not self._in_loop()
            and now - self.hold_started > self.profile.limits.planner_deadline_s
        ):
            self.interrupt("planner_activation_timeout", now)
            self.phase = "FAULT"
            return
        lim = self.profile.limits
        if self.phase == "MANIPULATING":
            since = self.last_policy_result if self.last_policy_result is not None else self.transition_started
            if (
                now - self.execution["started_at"] > lim.manipulation_deadline_s
                or now - since > lim.policy_deadline_s
            ):
                self.interrupt("policy_or_task_timeout", now)
        elif self.phase == "RESETTING":
            if now - self.transition_started > lim.reset_deadline_s:
                self.interrupt("reset_timeout", now)
            elif advanced and self.hold_confirmed:
                if self.reset.is_settled(
                    self.feedback, lim.reset_joint_tolerance_rad, lim.reset_yaw_tolerance_rad
                ):
                    self.dwell = now if self.dwell is None else self.dwell
                    if now - self.dwell >= lim.settle_dwell_s:
                        self.phase = "COMPLETED"
                else:
                    self.dwell = None
        elif self.phase in {"WALKING", "TURNING"}:
            if not self._planner_active():
                self.interrupt("planner_inactive", now)
                return
            try:
                self.hooks.set_planner_command(self.planner.advance(self.feedback, now))
            except (TypeError, ValueError) as exc:
                self.interrupt(f"planner_error:{exc}", now)
                return
            if self.planner.finished:
                if self.hold_started is None or self.hold_started < self.transition_started:
                    self.hold_index, self.hold_started = self.feedback_index, now
                    self.hold_confirmed = False
                elif self.hold_confirmed:
                    self.phase = "COMPLETED"

    def accept_policy_result(self, epoch, captured_at, action, now):
        if self.phase != "MANIPULATING" or epoch != self.epoch:
            return False
        valid = (
            0 <= now - captured_at < self.profile.action_horizon / self.profile.publish_rate
            and validate_native_action(action)
        )
        if not valid:
            self.interrupt("invalid_or_stale_policy_action", now)
            return False
        self.last_policy_result = now
        return True

    def status(self, now):
        facts = self.hooks.runtime_facts()
        fresh = self._fresh(now)
        observation_age = None if self.observation is None else max(0.0, now - self.observation["received_at"])
        return dict(
            runtime_id=self.runtime_id,
            phase=self.phase,
            skill_id=self.execution["skill_id"] if self.execution else None,
            execution_id=self.execution["execution_id"] if self.execution else None,
            inference_epoch=self.epoch,
            controller_running=facts.controller_running,
            policy_ready=facts.policy_ready,
            mode_requested=facts.mode_requested,
            planner_reference_active=self._planner_active() if fresh else None,
            hold_confirmed=self.hold_confirmed and fresh,
            telemetry_age_s=max(0.0, now - self.feedback_time) if self.feedback_time is not None else None,
            telemetry_index=self.feedback_index,
            observation_age_s=observation_age,
            frame_id=self.observation["frame_id"] if self.observation else None,
            registry_sha256=self.profile.registry_sha256,
            policy_host=self.profile.policy_host,
            policy_port=self.profile.policy_port,
            checkpoint_expected=self.profile.checkpoint,
            owner_session_id=self.owner["session_id"] if self.owner else None,
            reason=self.reason,
            locomotion_enabled=self.locomotion_enabled,
            ready_return_enabled=self._handover_enabled(),
            right_hand_open=(bool(np.max(np.abs(self.feedback["right_hand_q_measured"])) <= 0.05)
                             if fresh else None),
            loop_stage=self.loop.stage if self.loop else None,
            loop_trial=self.loop.trial_no if self.loop else None,
            trial_end_reason=self.loop.trial_end_reason if self.loop else None,
            frozen_stage=self.loop.frozen_stage if self.loop else None,
        )
