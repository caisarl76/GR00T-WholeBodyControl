"""Main-loop ownership and lifecycle for the opt-in G1 harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Protocol
import uuid

import numpy as np

from .harness_contract import SkillCall, decode_request
from .harness_profile import load_profile, validate_skill_call
from .standing_reset import StandingReset, _measured_joints


@dataclass(frozen=True)
class RuntimeFacts:
    controller_running: bool
    policy_enabled: bool
    policy_ready: bool
    inference_busy: bool
    mode_requested: str | None
    operator_busy: bool = False


class RuntimeHooks(Protocol):
    def runtime_facts(self) -> RuntimeFacts: ...
    def invalidate_policy_actions(self) -> int: ...
    def set_policy_prompt(self, prompt: str) -> None: ...
    def set_policy_enabled(self, enabled: bool) -> None: ...
    def request_planner_hold(self, feedback: dict, open_hands: bool) -> None: ...
    def begin_standing_reset(self, feedback: dict, open_hands: bool) -> StandingReset: ...
    def set_planner_command(self, command) -> None: ...
    def stop_planner_motion(self) -> None: ...


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
    def __init__(self, profile_path: Path, hooks: RuntimeHooks, runtime_id: str):
        self.profile = load_profile(profile_path)
        self.hooks, self.runtime_id = hooks, runtime_id
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
        self.replies = {}
        self.locomotion_enabled = False

    def _fresh(self, now):
        return (
            self.feedback is not None
            and self.feedback_time is not None
            and 0 <= now - self.feedback_time <= self.profile.limits.feedback_max_age_s
        )

    def _planner_active(self):
        value = np.asarray(self.feedback.get("planner_reference_active", []) if self.feedback else []).reshape(-1)
        return bool(value.shape == (1,) and value[0] == 1)

    def _hold(self, now):
        self.hold_confirmed = False
        self.hold_index, self.hold_started = self.feedback_index, now
        self.reset, self.planner, self.dwell = None, None, None
        if not self._fresh(now):
            self.phase = "FAULT"
            return
        try:
            _measured_joints(self.feedback)
            self.hooks.request_planner_hold(self.feedback, False)
        except (TypeError, ValueError):
            self.phase = "FAULT"

    def interrupt(self, reason: str, now: float):
        self.epoch = self.hooks.invalidate_policy_actions()
        self.hooks.set_policy_enabled(False)
        self.hooks.stop_planner_motion()
        self.phase, self.reason = "INTERRUPTED", reason
        self._hold(now)

    def operator_override(self, reason: str):
        if self.owner is None and self.execution is None and self.phase == "IDLE":
            return
        if self.owner is not None:
            self.interrupt(f"operator_override:{reason}", time.monotonic())
        else:
            self.epoch = self.hooks.invalidate_policy_actions()
        self.owner = None
        self.reset, self.planner = None, None
        self.hold_confirmed = False
        self.phase = "IDLE" if reason == "i" else "INTERRUPTED"
        self.reason = f"operator_override:{reason}"

    def _lease(self, req, now):
        if (
            not self.owner
            or req["session_id"] != self.owner["session_id"]
            or req["lease_id"] != self.owner["lease_id"]
        ):
            raise ControlError("NOT_OWNER", "No matching owner")
        if now >= self.owner["expires_at"]:
            self.interrupt("lease_expired", now)
            self.owner = None
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
            if key in self.replies:
                previous, cached = self.replies[key]
                if previous != canonical:
                    raise ControlError("REQUEST_CONFLICT", "Request ID reused with a different payload")
                return cached
            result = self._dispatch(req, now)
            response["result"] = result
            # Observe/status can be large and never actuate. Mutations are deduplicated.
            if req["method"] not in {"get_status", "observe"}:
                self.replies[key] = (canonical, response)
        except ControlError as exc:
            response["error"] = dict(code=exc.code, message=str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            response["error"] = dict(code="INVALID_REQUEST", message=str(exc))
        return response

    def _dispatch(self, req, now):
        method, params = req["method"], req["params"]
        if method == "get_status":
            return self.status(now)
        if method == "observe":
            if (
                self.observation is None
                or now - self.observation["received_at"] > self.profile.limits.camera_max_age_s
            ):
                raise ControlError("UNAVAILABLE", "No fresh ego_view")
            return {
                **self.observation,
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
                or self.phase not in {"IDLE", "COMPLETED", "PAUSED"}
            ):
                raise ControlError("NOT_READY", "Operator must start control and prepare a paused executor")
            if not self._fresh(now):
                raise ControlError("NOT_READY", "Fresh telemetry required")
            self.owner = dict(
                runtime_id=self.runtime_id,
                session_id=req["session_id"],
                lease_id=uuid.uuid4().hex,
                expires_at=now + self.profile.limits.lease_s,
            )
            return self.owner.copy()
        self._lease(req, now)
        if method == "heartbeat":
            self.owner["expires_at"] = now + self.profile.limits.lease_s
            return self.status(now)
        if method == "release_control":
            if not self.hold_confirmed or self.phase not in {"PAUSED", "COMPLETED", "INTERRUPTED", "FAULT"}:
                raise ControlError("NOT_READY", "Release requires a confirmed planner hold")
            self.owner = None
            return self.status(now)
        if method == "start_manipulation":
            try:
                skill = self.profile.require_skill(params["skill_id"])
            except ValueError as exc:
                raise ControlError("UNSUPPORTED_SKILL", str(exc)) from exc
            facts = self.hooks.runtime_facts()
            if (
                self.phase not in {"IDLE", "COMPLETED", "PAUSED"}
                or not facts.policy_ready
                or facts.inference_busy
                or not facts.controller_running
                or not self._fresh(now)
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
        standalone = method == "reset_standing" and params["execution_id"] == ""
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
        if method == "reset_standing":
            if self.phase not in ({"IDLE", "COMPLETED"} if standalone else {"PAUSED"}) or not self._fresh(now):
                raise ControlError("NOT_READY", "Pause and fresh telemetry required before reset")
            if standalone:
                self._new_execution("reset_standing", now)
            self.epoch = self.hooks.invalidate_policy_actions()
            self.hooks.set_policy_enabled(False)
            self.reset = self.hooks.begin_standing_reset(self.feedback, params["open_hands"])
            self.phase, self.transition_started, self.dwell = "RESETTING", now, None
            self.hold_confirmed, self.hold_index, self.hold_started = False, self.feedback_index, now
            return self._execution()
        raise ControlError("INVALID_REQUEST", "Unknown method")

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
        advanced = False
        if feedback is not None:
            index = feedback.get("index")
            if type(index) is int and index >= 0 and (self.feedback_index is None or index > self.feedback_index):
                self.feedback, self.feedback_index, self.feedback_time = feedback, index, feedback_received_at
                advanced = True
            elif type(index) is int and self.feedback_index is not None and index < self.feedback_index:
                self.feedback, self.feedback_time, self.feedback_index = None, None, index
                self.interrupt("telemetry_stream_restarted", now)
                self.owner = None
                self.hold_started = None
        if self.owner and now >= self.owner["expires_at"]:
            self.interrupt("lease_expired", now)
            self.owner = None
        if self.phase in {"MANIPULATING", "RESETTING", "WALKING", "TURNING"} and not self._fresh(now):
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
        )
