"""LoopController: the autonomous handover loop's state machine. It reproduces lerobot a6e267c6's validated
``--trials --ready-pose`` session (stream_backends.py SonicBackend, sonic_policy_streamer.py run_episode):

  take_over  50 ticks of the planner token, POSE switch, hold the stance (the planner token, hands open)
  raise_     blend to the raise start, play the raise, blend to the ready pose
  trial      hold-rest on the ready pose, one bootstrap chunk, blend in, the anchored policy episode with
             guards and the handover detector, then hold the last pose (await_take) until loop_to_ready
  to_ready   blend to the ready pose once the bottle was taken (lerobot: the operator's Enter, then go_ready)
  freeze     hold the last sent token and hands; resume continues or returns to the ready pose
  end        blend to the lowering start, play the lowering, blend to the stance, hold it, blend the hand-back
             to the planner token with the hands to 0, PLANNER mode

Only the PoseCommander produces tokens; policy outputs reach it only through ``accept_chunk`` and the trial's
TokenStream. The main loop publishes what ``tick`` returns.
"""

from collections import deque
import math
import time

import numpy as np

from gear_sonic.utils.inference.harness_control import ControlError
from gear_sonic.utils.inference.lerobot_sonic_policy import lerobot_state
from gear_sonic.utils.inference.loop_episode import (
    TAKE_EPISODE_RING_S,
    TAKE_RING_S,
    EpisodeGuards,
    HandoverDetector,
    TakeDetector,
    take_reference,
    take_torque,
)
from gear_sonic.utils.inference.loop_motion import TOKEN_ABS_MAX
from gear_sonic.utils.inference.loop_records import TrialRecorder, next_trial

BLEND_TICKS = 50  # 1 s blends to and from the ready pose, blend in, blend out
TAKE_OVER_TICKS = 50  # planner token before the POSE switch
HOLD_REST_TICKS = 75  # hold_rest_s 1.5
STANCE_HOLD_TICKS = 100  # lerobot finish(): 2 s "hold rest"
HANDBACK_TICKS = 100  # handoff_blend_s 2.0
REPLAN_S = 0.4
TRIAL_STAGES = ("hold_rest", "first_chunk", "blend_in", "episode", "await_take", "to_ready")
FRAME_STAGES = ("first_chunk", "blend_in", "episode")


def _rows(vector, n):
    return np.tile(np.asarray(vector, np.float32), (n, 1))


class LoopController:
    def __init__(self, commander, ready_pose, hooks, *, make_stream, right_order, trial_args, model):
        self.commander, self.ready, self.hooks = commander, ready_pose, hooks
        self.make_stream, self.right_order = make_stream, np.asarray(right_order)
        self.trial_args, self.model = trial_args, model
        self.phase, self.stage, self.frozen_stage, self.reason = "IDLE", None, None, None
        self.trial_no, self.trial_end_reason = None, None
        self.planner = self.stance = None
        self._then = None  # called once, after the tick on which the commander's queue played out
        self._recorder = self._stream = self._guards = self._detector = self._first = None
        self._outstanding = self._pending = None  # requested epoch; accepted (rows, t0) for the next tick
        self._requested_at = None  # now at the outstanding request: a replanned chunk's t0 (lerobot slot_t)
        self._take = None  # TakeDetector while in await_take (R26)
        self._torque_ring = deque()  # (now, take_torque) over the episode's last TAKE_EPISODE_RING_S
        self._hold_ring = deque()  # the same over the hold's last TAKE_RING_S, for the take summary (diagnostics)
        self._take_before = self._take_at = None  # the reference median (also when inert); when it fired
        self._hold_trace_at = 0.0  # next hold-trace log line
        self._episode_t0 = self._last_chunk_at = self._next_replan = None
        self._chunk_t0 = math.nan

    @property
    def take_detected(self) -> bool:
        """True once the bottle was taken in the current await_take (it only reports; the runner acts)."""
        return self._take is not None and self._take.taken

    @property
    def take_hand_empty(self) -> bool:
        """R32: the wrist carries no bottle (steady low wrist yaw) in the current await_take; the runner acts on it
        only together with the camera."""
        return self._take is not None and self._take.hand_empty

    def _require(self, ok, method):
        if not ok:
            raise ControlError("NOT_READY", f"{method} not allowed in {self.phase} (stage {self.stage})")

    def _motion(self, phase, stage, then):
        self.phase, self.stage, self._then = phase, stage, then

    def _hold(self, stage):
        self._motion("LOOP_HOLD", stage, None)

    # --- RPC entry points -------------------------------------------------------------------------------------

    def take_over(self, feedback, now, model=None):
        """model: the policy server's served model, recorded in each trial (the constructor's otherwise)."""
        self._require(self.phase in ("IDLE", "COMPLETED"), "loop_take_over")
        planner = np.asarray((feedback or {}).get("token_state", []), np.float32).reshape(-1)
        if planner.shape != (64,) or not np.isfinite(planner).all():
            raise ControlError("NOT_READY", "no 64D token_state from the deploy for the take-over")
        zero = np.zeros(14, np.float32)
        if model is not None:
            self.model = model
        self.planner, self.stance = planner.copy(), (planner.copy(), zero)  # --start planner: rest on it
        self.reason = self.frozen_stage = self.trial_no = self.trial_end_reason = None
        self.commander.start(planner, zero)
        self.commander.play(_rows(planner, TAKE_OVER_TICKS), _rows(zero, TAKE_OVER_TICKS), "planner token")
        self._motion("LOOP_MOTION", "stance", self._pose_switched)

    def raise_(self, now):
        self._require(self.phase == "LOOP_HOLD" and self.stage == "stance", "loop_raise")
        p, c = self.ready, self.commander
        c.blend_to(p.raise_tokens[0], p.raise_hands[0], BLEND_TICKS, "to raise start")
        c.play(p.raise_tokens, p.raise_hands, "raise")
        c.blend_to(p.token, p.hands, BLEND_TICKS, "to ready")
        self._motion("LOOP_MOTION", "raise", self._at_ready)

    def trial(self, trial, session_dir, now):
        self._require(self.phase == "LOOP_HOLD" and self.stage == "ready", "loop_trial")
        if trial < next_trial(session_dir):  # the previous recorder is closed here, so its trial counts
            raise ControlError("INVALID_REQUEST", f"trial {trial} is already recorded in {session_dir}")
        self._recorder = TrialRecorder(session_dir, trial, self.trial_args, self.model)
        self._stream, self._guards, self._first = self.make_stream(), None, None
        self.trial_no, self.trial_end_reason = trial, None
        self._torque_ring.clear()
        p = self.ready
        self.commander.play(_rows(p.token, HOLD_REST_TICKS), _rows(p.hands, HOLD_REST_TICKS), "wait first chunk")
        self._motion("LOOP_TRIAL", "hold_rest", self._request_first_chunk)

    def freeze(self, reason, now):
        self._require(self.phase.startswith("LOOP_"), "loop_freeze")
        if self.phase == "LOOP_FROZEN":
            return
        if self._recorder is not None:  # the trial ends here (a trial already ending keeps its reason)
            if self.trial_end_reason is None:
                self.trial_end_reason = reason
            self._close_recorder()
        if self.stage == "await_take":
            self._take_summary(now, f"freeze: {reason}")
        self._outstanding = self._pending = self._take = None
        self._torque_ring.clear()
        self.commander.freeze()
        self.frozen_stage, self.stage, self.phase, self.reason = self.stage, "frozen", "LOOP_FROZEN", reason

    def resume(self, now):
        self._require(self.phase == "LOOP_FROZEN", "loop_resume")
        stage, self.frozen_stage, self.reason = self.frozen_stage, None, None
        if self._then == self._pose_switched:  # R16a: frozen before the POSE switch: abort the take-over
            self.commander.hold()
            self.phase, self.stage, self._then = "IDLE", None, None
            return
        if stage in TRIAL_STAGES or stage == "ready":
            self.commander.hold()
            self.commander.blend_to(self.ready.token, self.ready.hands, BLEND_TICKS, "to ready")
            self._motion("LOOP_MOTION", "to_ready", self._at_ready)
            return
        # stance, raise, lower, handback: continue the queued path from its frame; _then is still pending
        self.commander.resume()
        self.phase, self.stage = ("LOOP_MOTION" if self.commander.busy else "LOOP_HOLD"), stage

    def to_ready(self, now):
        """After a trial end: the bottle was taken (lerobot's operator Enter in finish(to_ready))."""
        self._require(self.phase == "LOOP_HOLD" and self.stage == "await_take", "loop_to_ready")
        self._take_summary(now, "loop_to_ready")
        self._take = None
        self.commander.blend_to(self.ready.token, self.ready.hands, BLEND_TICKS, "to ready")
        self._motion("LOOP_MOTION", "to_ready", self._at_ready)

    def end(self, now):
        self._require(self.phase == "LOOP_HOLD" and self.stage == "ready", "loop_end")
        p, c = self.ready, self.commander
        (stance, stance_hands), zero = self.stance, np.zeros(14, np.float32)
        c.blend_to(p.lower_tokens[0], p.lower_hands[0], BLEND_TICKS, "to lower start")
        c.play(p.lower_tokens, p.lower_hands, "lower")
        c.blend_to(stance, stance_hands, BLEND_TICKS, "blend out")
        c.play(_rows(stance, STANCE_HOLD_TICKS), _rows(stance_hands, STANCE_HOLD_TICKS), "hold rest")
        c.blend_to(self.planner, zero, HANDBACK_TICKS, "handback blend")
        self._motion("LOOP_MOTION", "lower", self._handed_back)

    # --- policy results and camera frames (main loop) -----------------------------------------------------------

    def accept_chunk(self, epoch: int, captured_at: float, rows) -> bool:
        """Take the outstanding request's [T, 78] chunk. Only the outstanding epoch is accepted, in stages
        first_chunk and episode. rows None (a failed inference) in the episode is lerobot's error path: keep
        playing the last chunk and let the next replan request again (the chunk-age guard ends the trial if
        failures persist); with no chunk yet (first_chunk) it freezes. Malformed rows freeze."""
        if self.stage not in ("first_chunk", "episode") or epoch != self._outstanding:
            return False
        self._outstanding = None
        if rows is None:
            print(f"[loop] policy inference failed (epoch {epoch}), stage {self.stage}", flush=True)
            if self.stage == "first_chunk":
                self.freeze("first chunk inference failed", time.monotonic())
            return False
        rows = np.asarray(rows, np.float32)
        if (
            rows.ndim != 2 or rows.shape[1] != 78 or not len(rows)
            or not np.isfinite(rows).all() or np.abs(rows[:, :64]).max() > TOKEN_ABS_MAX
            or not math.isfinite(captured_at)
        ):
            self.freeze("invalid policy result", time.monotonic())
            return False
        if self.stage == "first_chunk":  # lerobot begin(): 1 s blend in to row 0
            self._first, self._guards = rows, EpisodeGuards()  # the arm history starts with the blend in (R20b)
            self.commander.blend_to(rows[0, :64], rows[0, 64:], BLEND_TICKS, "blend in")
            self._motion("LOOP_TRIAL", "blend_in", None)
        else:
            self._pending = (rows, self._requested_at)  # R23a: t0 = the request time, as lerobot's slot_t
        return True

    def frame(self, t_recv, name, jpeg):
        if self._recorder is not None and self.stage in FRAME_STAGES:
            self._recorder.frame(t_recv, name, jpeg)

    # --- 50 Hz ------------------------------------------------------------------------------------------------

    def tick(self, now, feedback, feedback_age_s, camera_age_s):
        """(token 64, hands 14, label) to publish this tick."""
        if not self.phase.startswith("LOOP_"):
            raise RuntimeError(f"LoopController.tick in {self.phase}")
        state = np.full(28, np.nan, np.float32)
        if feedback:
            try:
                state = lerobot_state(feedback, self.right_order)
            except (KeyError, TypeError, ValueError):  # R16c: never stream on a malformed state
                self.freeze("invalid robot state", now)
        if self.stage == "blend_in" and not self.commander.busy:
            self._start_episode(now)
        if self.stage == "blend_in":
            self._guards.record(now, feedback)
        if self.stage == "await_take":
            arm = take_torque(feedback)
            if arm is not None:  # diagnostics: the hold's torque, traced every 0.5 s and summarised at its end
                self._hold_ring.append((now, arm))
                while self._hold_ring[0][0] < now - TAKE_RING_S:
                    self._hold_ring.popleft()
                if now >= self._hold_trace_at:
                    t_hold = now - self._take.start
                    rest = np.asarray(feedback["motor_torque"], np.float64)[[23, 24, 26, 27, 28]]
                    print(f"[loop] hold +{t_hold:5.1f} s torque sh/el = {arm[0]:+.2f}/{arm[1]:+.2f} | "
                          f"sh roll/yaw = {rest[0]:+.2f}/{rest[1]:+.2f} | wrist r/p/y = "
                          f"{rest[2]:+.2f}/{rest[3]:+.2f}/{rest[4]:+.2f}", flush=True)
                    self._hold_trace_at = now + 0.5
        if self.stage == "await_take" and not self._take.taken and self._take.update(feedback, now):
            sh, el, wy = self._take.rise
            self._take_at = now
            print(f"[loop] take detected: rise sh/el = {sh:+.2f}/{el:+.2f} after {now - self._take.start:.2f} s"
                  f" (wrist yaw drop {wy:+.2f})", flush=True)
        if self.stage == "episode":
            arm = take_torque(feedback)
            if arm is not None:
                self._torque_ring.append((now, arm))
            while self._torque_ring and self._torque_ring[0][0] < now - TAKE_EPISODE_RING_S:
                self._torque_ring.popleft()
            self._episode_step(now, feedback, state, feedback_age_s, camera_age_s)
        if self.stage == "first_chunk" and not self.commander.busy:  # lerobot wait_tick: the rest pose, slewed
            self.commander.play(self.ready.token[None], self.ready.hands[None], "wait first chunk")
        token, hands, label = self.commander.tick(now)
        if self.stage in ("lower", "handback"):
            self.stage = "handback" if label == "handback blend" else "lower"
        if self._recorder is not None:
            self._record(now, state, token, hands, label)
        if self._then is not None and not self.commander.busy and not self.commander.frozen:
            then, self._then = self._then, None
            then(now)
        return token, hands, label

    def _start_episode(self, now):
        """The tick after the blend in: the first chunk plays from here (lerobot t_ep 0); guards start (R10)."""
        self._episode_t0 = self._last_chunk_at = self._chunk_t0 = now
        self._next_replan = now + REPLAN_S
        self._set_chunk(self._first, now, now)
        self.commander.follow(self._stream, "episode")
        self._guards.start(now, keep_history=True)  # cap, echo and checks start here; keep the blend in's arms
        self._detector = HandoverDetector()
        self.stage = "episode"

    def _set_chunk(self, rows, t0, now):
        self._stream.last_token = self.commander.last_token.copy()  # R1b: anchor on the token actually sent
        self._stream.set_chunk(rows, t0=t0, now=now)

    def _episode_step(self, now, feedback, state, feedback_age_s, camera_age_s):
        """lerobot's per-tick order: take a finished chunk, the guards, the handover detector, then replan."""
        if self._pending is not None:
            (rows, t0), self._pending = self._pending, None
            self._set_chunk(rows, t0, now)
            self._last_chunk_at, self._chunk_t0 = now, t0
        verdict = self._guards.check(now, feedback, feedback_age_s, camera_age_s, self._last_chunk_at,
                                     self.commander.recent_tokens)
        if verdict is None and feedback and self._detector.update(float(state[21:28].mean()), now):
            verdict = ("end", "handover complete")
        if verdict is not None:
            kind, reason = verdict
            if kind == "freeze":
                self.freeze(reason, now)
            else:
                # R24: hold the last sent pose (the bottle may rest on the open palm) until loop_to_ready, as
                # lerobot's finish(to_ready) holds for the operator before go_ready. The trial log ends here.
                self.trial_end_reason, self._outstanding, self._pending = reason, None, None
                self.commander.hold()
                self._close_recorder()
                self._hold("await_take")
                self._start_take_detector(now)
            return
        if now >= self._next_replan and self._outstanding is None:
            self._outstanding, self._requested_at = self.hooks.request_policy_chunk(), now
            self._next_replan = now + REPLAN_S

    def _start_take_detector(self, now):
        """R27: the reference is the loaded arm just before the hold (the episode's end); if that is unsteady, the
        detector takes a steady hold window instead (R30)."""
        reference, median, spread, n = take_reference(self._torque_ring, now)
        self._take = TakeDetector(now, reference)
        self._take_before, self._take_at, self._hold_trace_at = median, None, now
        self._hold_ring.clear()
        values = ("none" if median is None
                  else f"{median[0]:.2f}/{median[1]:.2f} spread {spread[0]:.2f}/{spread[1]:.2f}")
        wrist = "" if median is None else f" | wrist yaw {median[2]:+.2f} spread {spread[2]:.2f}"
        print(f"[loop] take reference sh/el = {values} n {n}{wrist}"
              f"{' (unsteady: waiting for a steady hold)' if reference is None else ''}", flush=True)
        # R31 diagnostics: was the bottle still loading the arm at the release (1 s before the episode's end)?
        windows = [[a for t, a in self._torque_ring if now - k - 0.5 < t <= now - k]
                   for k in (2.5, 2.0, 1.5, 1.0, 0.5, 0.0)]
        medians = ["none" if not w else "{:+.2f}/{:+.2f}/{:+.2f}".format(*np.median(w, axis=0)) for w in windows]
        print("[loop] episode end torque sh/el/wrist yaw, 0.5 s medians from -3 s: " + " | ".join(medians),
              flush=True)

    def _take_summary(self, now, how):
        """One log line per hold: the loaded arm before (the reference median), the hold's last 0.5 s, and their
        difference (diagnostics only)."""
        end = [arm for t, arm in self._hold_ring if t >= now - 0.5]
        after = np.median(end, axis=0) if end else None
        fmt = lambda v: "none" if v is None else f"{v[0]:+.2f}/{v[1]:+.2f}"  # noqa: E731
        effective = None if self._take is None else self._take.reference  # R28: set after the first 0.3 s
        before = self._take_before if effective is None else effective
        diff = None if after is None or before is None else after - before
        taken_at = None if self._take is None or self._take_at is None else self._take_at - self._take.start
        detected = "no" if taken_at is None else f"at +{taken_at:.2f} s"
        if self._take is not None and self._take.steady_at is not None:
            detected += f" (reference from the steady hold at +{self._take.steady_at - self._take.start:.1f} s)"
        wrist = "" if diff is None else f" | wrist yaw {before[2]:+.2f} -> {after[2]:+.2f} ({diff[2]:+.2f})"
        print(f"[loop] take summary: before sh/el = {fmt(before)} | hold end = {fmt(after)} | "
              f"diff = {fmt(diff)} | detected {detected} | hold {now - self._take.start:.1f} s, ended by {how}"
              f"{wrist}", flush=True)

    def _record(self, now, state, token, hands, label):
        t_ep, frame, chunk_t0 = -1.0, -1, math.nan
        if label == "blend in":
            t_ep, frame = 0.0, 0
        elif label == "episode":
            t_ep = now - self._episode_t0
            # lerobot k = int(t_ep * images.fps), images.fps = --policy-fps: the stream's source rate
            frame, chunk_t0 = int(t_ep * self._stream.source_fps), self._chunk_t0 - self._episode_t0
        self._recorder.tick(time.time(), t_ep, label, frame, token, hands, chunk_t0, state)

    # --- stage ends (``_then``) ---------------------------------------------------------------------------------

    def _pose_switched(self, now):
        self.hooks.switch_to_pose()
        self._hold("stance")

    def _at_ready(self, now):
        self._hold("ready")

    def _request_first_chunk(self, now):  # its chunk's t0 is the blend in's end (R11e), not this time
        self._outstanding, self._requested_at = self.hooks.request_policy_chunk(), now
        self.stage = "first_chunk"

    def _handed_back(self, now):
        self.hooks.switch_to_planner()
        self.phase, self.stage = "COMPLETED", None

    def _close_recorder(self):
        recorder, self._recorder = self._recorder, None
        recorder.close(self.trial_end_reason)
