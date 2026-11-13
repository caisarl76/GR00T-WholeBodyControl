# PRD — PICO Manipulation Mode for GEAR-SONIC Teleoperation

- Date: 2026-11-09
- Owner: caisarl76 (jihun.kim@genon.ai)
- Status: Approved for implementation planning (rev 5 + last-sent hand command note). The 5 % takeover threshold and 0.2 s blend remain unvalidated until contact tests 7b/7c run.
- Scope: `gear_sonic/scripts/pico_manager_thread_server.py` (streamer), `gear_sonic/scripts/run_data_exporter.py`, `gear_sonic_deploy` (C++ deploy). MuJoCo sim first.

Notation: **[decided]** = agreed with the operator. **[my call]** = author's default; change it if wrong. Every number in this document is an **initial sim target or default, not a validated value**.

---

## 1. Problem

We collect manipulation episodes on a Unitree G1 driven by the GEAR-SONIC whole-body policy. Today the operator uses **full-body Pose mode (A+X)**. In that mode, PICO 4 Ultra plus the ankle trackers estimate a full SMPL body and stream it, and SONIC tracks all of it.

Observed failure: when the operator reaches forward with only the right controller (torso and legs still), the robot's right hand moves forward **and the whole body rotates counter-clockwise**. Episodes also show torso sway and oscillation from balancing.

Candidate causes, which gate G0 (§9) must tell apart:
1. PICO's body solver puts pelvis and spine rotation into the streamed reference.
2. SONIC adds torso and leg motion on its own to balance while tracking the reference.
3. The streamer/deploy path between the streamed reference and SONIC's target adds it ("pipeline").

This contaminates the dataset: the recorded body motion isn't what the operator meant, and it isn't repeatable.

## 2. Goal

Add a new teleop mode, **MANIPULATION**, that follows one rule:

> If the operator moves only a hand controller and keeps their torso and lower body still, the robot moves **only that arm's 7 joints and its hand**. The lower body and the 3 waist joints do not move.

The waist and height change **only on purpose, from the joysticks**. Walking happens **only in FROZEN UPPER BODY mode** **[decided]**.

## 3. Non-goals

- Retraining or fine-tuning SONIC or the planner.
- Changing any behavior of a session launched **without** `--manipulation` (POSE, PLANNER, VR_3PT, FROZEN stay as they are today).
- Physical stepping from ankle trackers in the new mode. Stepping is joystick-only **[decided]**.
- Cancelling operator torso motion. See §6.2: operator torso motion moves the controllers in space, so it shows up as **hand** motion, never as waist or leg motion. Keeping the torso still is the operator's job.

In scope: both capture backends, XRoboToolkit and IsaacCapture. IsaacCapture already exposes raw controller poses (`gear_sonic/utils/teleop/isaac_teleop_client.py:200-209`).

## 4. Anti-goals (a technical success that still counts as failure)

- The robot falls or stumbles because legs or waist were locked too stiffly. **Legs are never hard-locked.**
- Any arm, waist or height **jump**: on mode entry, on tracking recovery, when the waist lock engages or releases, or when walking starts or stops.
- Height or waist snap to a default or range boundary on a mode switch or on first stick use.
- The operator's real torso motion, or PICO's estimate of it, reaches the robot **waist or legs** in MANIPULATION.
- The robot lands in full-body POSE or VR_3PT mid-episode.

## 5. Operator experience

### 5.1 Modes and controls

| Mode | Arms (7+7) + hands | Waist (yaw/roll/pitch) | Legs / base | Left stick | Right stick |
|---|---|---|---|---|---|
| **MANIPULATION** (new) | Follow PICO controllers via IK (§6.2); triggers/grips → hands (as VR_3PT) | `waist_cmd` from the stick; roll fixed at 0 **[my call]** | Planner static stand (IDLE / IDLE_SQUAT, §6.3), height from the stick | **Y: height** (X unused) | **Y: waist pitch, X: waist yaw** |
| **FROZEN UPPER BODY** (B+Y) | Hold arm snapshot | Hold `waist_cmd` | Planner locomotion | X/Y: walk | **X: turn heading, Y: height** |

- **B+Y toggles MANIPULATION ↔ FROZEN** mid-episode without restarting the policy **[decided]**.
- `height_cmd` and `waist_cmd` are **one streamer-owned state each**, shared by both modes. They carry over across every toggle **[decided]**.
- Right stick X turns the whole body **only** in FROZEN **[decided]**.
- **Hands:** in MANIPULATION, triggers and grips control the hands. In FROZEN, the hands hold the **last commanded** hand values from MANIPULATION, and triggers and grips are ignored **[my call]**. Today's FROZEN snapshots the *measured* fingers. Under `--manipulation` that's not used, because on a hand closed around an object the measured pose can differ from the commanded one (how much this changes the grip depends on the hand controller).
- **Hand re-entry into MANIPULATION: soft takeover, per hand [my call].** The hand command is a deterministic function of that hand's trigger and grip, `q = f(trigger, grip)` (`compute_hand_joints_from_inputs`, `pico_manager_thread_server.py:754`), so takeover is defined in that **input space**:
  - On MANIPULATION exit, store each hand's last inputs `(trigger_held, grip_held)`. Also store the **actual last-sent hand command** as `q_held`, and use it as the held command and as the blend start. It is not recomputed from the inputs, so the blend starts from exactly what the robot received, even if a blend was in progress at exit. The takeover test still uses the inputs.
  - On re-entry, each hand keeps sending `q_held` until **both** of its live inputs satisfy `|trigger − trigger_held| ≤ 0.05` **and** `|grip − grip_held| ≤ 0.05`, simultaneously. Inputs are normalized to [0, 1], so 0.05 is 5 % of full travel. Each hand takes over independently.
  - **Continuity:** at takeover, the command blends linearly from `q_held` to `f(live)` over 0.2 s. Because both inputs are within 0.05, the gap is already small, and the blend removes any residual step.
  - **Prompt:** until takeover, the console names the direction to match, per input: for example `L trigger: press more`, `L grip: release a bit`, or `L grip: matched`. It says "release" when the live input is above the held one, so the operator always moves toward the held position.
  - A trigger released while frozen can therefore never open the hand on re-entry, and a harder squeeze can't snap it tighter.

### 5.2 State machine under `--manipulation` **[decided: POSE and VR_3PT disabled]**

```
OFF --(A+B+X+Y)--> PLANNER --(B+Y)--> MANIPULATION <--(B+Y)--> FROZEN_UPPER_BODY
MANIPULATION | FROZEN | PLANNER --(A+B+X+Y)--> OFF
```

- New edge: PLANNER --B+Y--> MANIPULATION. PLANNER has no B+Y edge today (`pico_manager_thread_server.py:2022-2029`).
- Disabled under the flag: A+X (→ POSE) and left_axis_click (→ VR_3PT). POSE_PAUSE is unreachable.
- In FROZEN, A+B / X+Y still step the locomotion mode up and down.
- **Entry guard:** if MANIPULATION entry is refused (§6.5), the robot stays in the prior mode, and the streamer prints the reason.
- Without the flag, the state machine and the chord handling are byte-for-byte today's.

### 5.3 Chord arbitration (under `--manipulation`) **[my call]**

The buttons A, B, X and Y form four chords: A+B+X+Y (stop), B+Y (toggle), and A+B / X+Y (locomotion mode). They overlap, so one arbiter decides which one fires:

1. **Stop has priority and fires immediately.** As soon as A, B, X and Y are all pressed, stop fires, whatever else happened before.
2. **Other chords need a stable hold.** A non-stop chord fires only after its exact button set has been held unchanged for `T_chord = 150 ms`. Any change to the set (a button added or released) restarts the timer. A staggered press of A+B+X+Y that passes through B+Y therefore never fires B+Y unless the operator pauses on B+Y for 150 ms.
3. **Release to re-arm.** After any chord fires, nothing else fires until A, B, X and Y have all been seen released. Stop is the exception: while the policy is running, it is never blocked by re-arm.
4. **Startup:** in OFF, A+B+X+Y means **start**, not stop. Start and all other chords are blocked until all four buttons have been seen released at least once after the process starts, so holding buttons at launch can never start the policy. Rule 1 (stop is never blocked) applies only once the policy is running (PLANNER, MANIPULATION, FROZEN).
5. Mode-switch chords are ignored while an entry attempt is pending (§6.5).

A+B+X+Y is a **policy stop, not an emergency stop**. The safety operator's keyboard **O** and the robot remote stay the e-stop.

### 5.4 Joystick semantics **[my call, all tunable]**

- **Rate control.** Deflection sets a rate, and releasing the stick holds the value. Deadzone is `JOYSTICK_DEADZONE = 0.15`, and the rate scales linearly past it.
- **Waist:** yaw 0.5 rad/s, pitch 0.3 rad/s at full deflection. Soft limits: yaw ±0.6 rad, pitch −0.1 … +0.4 rad. Roll is 0.
- **Height:**
  - Range is **[0.2, 0.8]**, the same thresholds squat mode uses today **[decided]** (`gamepad.hpp:371-382`, `zmq_manager.hpp:603`).
  - Rate is 0.1 m/s at full deflection.
  - **First use:** while the stick is untouched, the planner gets −1 (default). On the first deflection, `height_cmd` is set to `H_default`, the command value that G1 (§9) shows reproduces the default standing pose, and integration starts from there. It never starts from a range boundary.
  - **Send policy:** the height field value sent to the planner changes only when it differs from the last sent value by ≥ 1 cm, and never sooner than **200 ms** after the previous change (a hard 5 Hz limit). When the stick returns to the deadzone, the exact final `height_cmd` is sent at the next allowed slot (≤ 200 ms later), so quantization never loses the last command and the rate limit still holds.

## 6. Technical design

### 6.1 MANIPULATION = planner message with a joint-space upper body **[decided]**

Each tick (≥ 20 Hz), the streamer sends `build_planner_message` with:
- `mode` = the static mode from §6.3, `movement` = 0, `facing` = the current heading (unchanged).
- `height` per §5.4.
- `upper_body_position` (17 joints, order `_get_upper_body_joint_indices`; indices 0..2 = waist yaw/roll/pitch): `[waist_cmd, arm IK solution (14)]`.
- `left_hand_position` / `right_hand_position` from `compute_hand_joints_from_inputs`.
- `waist_lock` field (§6.4).
- **No** `vr_3pt_*` fields.

The deploy writes these 17 joints into SONIC's reference motion (`GatherMotionJointPositionsMultiFrame`, `g1_deploy_onnx_ref.cpp:784-790`). FROZEN takes the same path, now with `upper_body_position = [waist_cmd, arm snapshot]` instead of today's full measured snapshot.

### 6.2 Controller → wrist target mapping

**Frames.**
- `W`: PICO tracking world, converted to robot axes (x forward, y left, z up) using the existing Unity→robot conversion from `_process_3pt_pose`.
- `O`: operator frame. It is `W` rotated by the **headset yaw** `ψ0` captured at anchor time (yaw only; pitch and roll are dropped). `O` stays fixed until the next anchor.
- `T`: robot **torso frame** (`torso_link`), computed by FK at the **current** `waist_cmd` with the pelvis fixed.

**Anchor** (taken at MANIPULATION entry and at each per-controller recovery, §6.6). For each side `s`:
- the controller pose in `W`: `p_c0`, `R_c0`;
- the robot wrist pose in `T`: `p_w0`, `R_w0`, from FK of the **currently commanded** arm reference. That's FROZEN's held arm snapshot on re-entry, or the held IK solution on recovery. Only the very first entry from PLANNER, where no arm reference has been sent yet, uses the **measured** arm joints. Anchoring to the commanded reference keeps it continuous.

**Per tick.**
```
Δp_O = Rz(ψ0)ᵀ · (p_c − p_c0)                       # translation delta, operator axes
ΔR_O = Rz(ψ0)ᵀ · (R_c · R_c0ᵀ) · Rz(ψ0)             # rotation delta, operator axes
p_target_T = p_w0 + k · Δp_O                          # operator fwd/left/up → torso fwd/left/up
R_target_T = ΔR_O · R_w0
```
- `k = 1.0` (scale knob) **[my call]**. The target is the full 6-DoF wrist pose **[my call]**.
- **Hands move with the torso [decided]:** targets live in `T`, so a waist change carries both hands, and the arm joints stay unchanged when the controllers are still.
- **Response to operator torso motion:** controllers that move in `W` because the operator leaned or twisted produce **hand** motion. That is expected and accepted (§3). It never reaches the waist or legs.
- **Re-centering:** toggle to FROZEN and back, which re-anchors **[my call]**.

**IK.**
- Solve the **14 arm joints only**, against `T`-frame targets, with the torso as the fixed base. Add a posture task pulling toward the anchor-time arm joints, plus joint position limits.
- The candidate solver is `decoupled_wbc/control/teleop/solver/body/body_ik_solver.py` (`BodyIKSolver`, pink). It solves whatever model it is configured with, so it **must be adapted** to a reduced arm-only model (torso as root, arm chains only). A solver that can move non-arm joints fails this requirement.
- **Budget:** p99 < 10 ms per tick on the actual teleop PC (record the CPU model), so the streamer can hold 20 Hz. The budget covers **everything per tick**: both arms, torso-frame FK at the current `waist_cmd`, target transforms, and any solver setup. Benchmark (gate G2):
  - Targets: ≥ 1,000 controller trajectories replayed from existing recorded episodes, plus random reachable targets sampled across the arm workspace.
  - 10,000 ticks after 100 warm-up ticks.
  - Report p50 / p99 / max.

  If the budget is missed, pick another solver.
- Raw controller poses: XRoboToolkit `xrt.get_left/right_controller_pose()` (`PicoReader._run` reads only `get_body_joints_pose()` today, so add them to the sample); IsaacCapture `get_pose_by_name`.

### 6.3 Height and planner mode

- G1 (§9) decides whether IDLE honors `height`. If not, MANIPULATION sends IDLE_SQUAT whenever `height_cmd` is set, and IDLE while it is −1. The IDLE↔IDLE_SQUAT switch is part of G1's transition test.
- **Walking height interlock [my call]:**
  - **Band.** G1 measures `[H_walk_min, 0.8]`. Validation in sim: at each candidate height, walk forward, backward and turning for 10 s, × 5 runs. A height passes with no fall and pelvis z within ±3 cm. `H_walk_min` = lowest passing height + 3 cm margin.
  - **Band unknown** (G1 couldn't establish it, or it isn't yet validated on the real robot): `H_walk_min := H_default`. Walking is then allowed only at the −1 sentinel or at `height_cmd ≥ H_default − 1 cm`. On the real robot, this rule applies until the sim band is confirmed on the gantry at `H_walk_min`.
  - **−1 sentinel** (height never touched) counts as the default standing height. See the current walking scope below.
  - **Walk request while `height_cmd < H_walk_min`:** the walk stick is ignored and a warning is printed, until the operator raises the height with right stick Y.
  - **Lower request while walking:** `height_cmd` is clamped at `H_walk_min`. The stick request below it is ignored with a warning. The height is **never** changed automatically: no snap, no oscillation.
  - **Walk-start hysteresis:** walking is requested when stick magnitude > 0.20 and released below 0.12. The actual locomotion start also waits for lock release (§6.4).
  - **Walking at standing height uses −1 [decided 2026-11-11; G1 measured that 0.80 fails 1 of 15 walk trials while −1, at the same standing height, passes 15/15].** This applies only while the band is unknown.
    - **−1-only scope (applies while `still_check_adopted` is false: on hardware until the precondition below holds, and wherever the check isn't adopted).** Decided 2026-11-11, after G1b Step 7 FAILED. Walking is allowed only at the untouched −1 sentinel.
      1. **Height never changed:** `height_cmd` has been −1 since the streamer session started. After any height change, walking is refused for the rest of the session (`walking disabled after height change`).
         - **Sending −1 after a height change does not restore walking eligibility [clarified 2026-11-12, user review].** Once the height has changed, walking stays disabled for the rest of the session under the current design, whatever is sent afterwards. Returning to −1 is itself a transition that would need the unvalidated check.
      2. **Static at −1 for ≥ `walk_settle_s` = 4.5 s** since the last static-command change, including a walk stop, before a new walk starts **[my call]**.
         - Evidence: G1's −1 trials passed 15/15 when started after a 3 s static rest at −1 following the previous walk, plus the initial hold.
         - No still-standing check runs at −1. This path rests on elapsed time and G1's −1 evidence only.
      3. **Every walking interlock still applies at −1 [clarified 2026-11-12, user review].** −1 relaxes no interlock. It doesn't bypass:
         - the controlled stop;
         - the stick re-arm;
         - lock release (§6.4);
         - stick-control enablement;
         - controller validity (§6.6: a lost or stale controller's stick counts as centered, so it can't start or hold a walk);
         - the watchdog latch (§6.6).

         A new walk needs, in every scope:
         1. the previous walk ended in a controlled stop: stick released, the 0.5 s hand-off out completed, and the robot back in the static mode;
         2. the stick returned to neutral and deflected again;
         3. condition 2's timer, restarted at that stop.

         **Watchdog recovery counts as a stop.** While `watchdog_latched` is true, walking is refused (`watchdog latched`). After it clears, the timer restarts at the clear, and a new walk still needs a fresh deflection.
    - **Walking after a height change (lifts the restriction above):** this is allowed only once a still-standing check is adopted. **G1b Step 8 PASSED in sim (2026-11-11, run C), but Step 9 FAILED (2026-11-12, run D, §11), so the adoption is withdrawn everywhere: `still_check_adopted = False` and the −1-only scope applies in sim and on hardware. No still-standing check is adopted.**
      - **A replacement check [2026-11-12, user review]** needs its own separately approved protocol and fresh validation; the Step 8 pass doesn't carry over. Detecting motion inside the window addresses the Step 9 cancellation failure (nudge out-and-back). A displaced but stationary posture (Step 9 `down:1`, held compression under load) needs additional evidence beyond that.
      - When adopted, all of these must hold:
        1. `height_cmd ≥ H_default − 1 cm`, or −1;
        2. the static command (mode and height field, as sent) has been unchanged for ≥ 4.5 s (tested minimum: G1b run A slowest settle 3.51 s + 0.5 s, rounded up; verified in run B);
        3. the adopted check passes. Missing, stale or partial feedback means **not still**.
      - Elapsed time alone never authorizes walking after a height change. Sending −1 during a transition restarts the timer.
      - G1b Step 7, a peak leg-`body_dq` + gyro check, **FAILED** and isn't adopted (§11).
      - **Limits of the Step 8 evidence:**
        - At the deployed scope, run C had no non-still static instants and no instants after a walk stop. Its zero false acceptances therefore came from walking instants alone.
        - The safety evidence is the ≥ 0.5 s diagnostic: 0 false acceptances over 11190 non-still instants, worst margin 1.04× the thresholds, from one run. Condition 2 provides the rest.
      - **Hardware precondition:**
        - `G_max` = 0.00375 rad/s (0.215 °/s) is a 1 s mean, so a constant gyro bias doesn't average out. Settled sim G is already 0.0045 rad/s at p99.
        - Before walking after a height change is enabled on hardware:
          1. **G1b Step 9 (§9) must PASS in sim [required 2026-11-12, user review].** Step 8 never tested rejection of residual motion at the actual decision time;
          2. measure the stationary mean of `base_ang_vel` on the robot;
          3. if it uses up the headroom, walking after height changes stays disabled (−1-only scope);
          4. any change to `G_max` or any bias subtraction is a new check that needs re-validation.
        - Hardware validation must check both false refusals and false acceptances.
    - **Otherwise refused:** the robot stays standing in the static mode and a warning names the reason (`walking disabled after height change`, `raise height to walk`, `height settling`, `watchdog latched` or `not still: <signal>`). Nothing changes automatically. A refused request needs the stick released and deflected again, as for a lock-release timeout (§6.4). The operator raises the robot first if needed.
    - **Changing `walk_settle_s`** (for example to the 5.0 s fallback) requires a new G1b run B at the new value. It is a candidate to validate, not a guarantee.
    - **Hand-off:** on an accepted walk request, the planner gets −1 in the static mode for **0.5 s**, then locomotion starts with −1. After the stick is released, the planner gets −1 in the static mode for 0.5 s, then `height_cmd` again.
    - **`height_cmd` itself never changes.** B+Y switches modes and preserves it.
    - **While walking,** lowering stays disabled (clamp above). After the stop, the height stays at the standing height until the operator lowers it explicitly.
    - G1 measured z 0.787 m at −1 (IDLE) and 0.788 m at 0.80 (IDLE_SQUAT), so the hand-off should move the pelvis by about 1 mm. Gate G1b (§9) verifies that.

### 6.4 Waist hard-lock (A/B option) **[decided: test both]**

**Enabling takes two keys.** Both must hold, or the lock is off:
1. The deploy is started with `--waist-lock` (default off).
2. Each incoming planner message carries `waist_lock = true`. This is a new optional ZMQ planner field; if it's absent, the value is false. The streamer sets it **only** under `--manipulation` with its own `--waist-lock` flag. A session without `--manipulation` never sends the field, so its FROZEN mode is never locked.

**Freshness.**
- If no planner message arrives for **100 ms**, the deploy starts the normal 0.2 s ramp-out. The lock fades smoothly and never drops in a single step.
- If messages resume during or after ramp-out, the lock ramps back in over 0.5 s. Gaps shorter than 100 ms have no effect.
- The deploy counts freshness misses and logs them, so recurring short gaps are visible.

**Permitted planner modes:** IDLE and IDLE_SQUAT only. Kneel, lying and boxing are never locked, even though `is_static_motion_mode` includes them.

**Lock release before walking: sequential, confirmed.**
1. When walk-start hysteresis trips (§6.3), the streamer sends `waist_lock = false` and **keeps sending the static mode with movement = 0**.
   Each lock-off request carries a new, monotonically increasing `lock_req_seq` (uint32, new optional planner field).
2. The deploy publishes, in the g1_debug feedback stream (port 5557), its current blend `waist_lock_alpha` together with `lock_ack_seq`, the latest `lock_req_seq` it has processed. Both come from the same control tick.
3. The streamer starts locomotion only on a feedback sample where `lock_ack_seq == current lock_req_seq` **and** `waist_lock_alpha == 0`. Samples carrying an older `lock_ack_seq` are ignored. So a stale `α == 0` from before the request, or a delayed, reordered or duplicated sample, can never start walking.
4. If that isn't confirmed within **0.5 s** (stale feedback, or α stuck), the streamer does not walk. It prints an error and keeps standing (fail-safe = no walk). The operator retries by releasing and re-deflecting the stick.

After walking stops (stick below 0.12 and the planner back in a static mode), the streamer sends `waist_lock = true` again, and the lock ramps in over 0.5 s.

**Blend, not switch.** In `CreatePolicyCommand` (`g1_deploy_onnx_ref.cpp:3155-3160`), for MuJoCo joints 12/13/14:
```
q_target = (1 − α) · q_policy + α · q_waist_cmd
```
- `α` ramps 0 → 1 over **0.5 s** when the lock becomes active, and 1 → 0 over **0.2 s** when it becomes inactive.
- `q_waist_cmd` is additionally rate-limited to 1 rad/s.
- kp/kd stay the policy's.
- `last_action` keeps the policy's raw output, so the observation history is untouched.

### 6.5 Mode entry and waist handoff

**Entry into MANIPULATION is refused** (stay in the prior mode, print why) unless all of these hold:
- the robot feedback (`body_q_measured`) is valid and fresh, received < **100 ms** ago (`FeedbackReader` must timestamp what it receives; today `_process_upper_body_position_targets` can return `None`);
- both controller poses are valid (§6.6);
- the IK converges for both anchor poses.

The streamer **never anchors to zeros** or to a default pose. The existing zero-pose fallback in `recalibrate_for_vr3pt` (`pico_manager_thread_server.py:1770-1776`) must not be reused.

**Waist handoff: `waist_cmd` wins.**
- `waist_cmd` is created **once**, on the first entry into MANIPULATION or FROZEN from PLANNER, from the measured waist. After that, only the stick changes it.
- On later toggles, both modes send the existing `waist_cmd`; there is no re-initialization from measurement.
- If the measured waist differs from `waist_cmd` at entry (SONIC tracking error), the reference stays at `waist_cmd`. With the lock on, the §6.4 ramp handles the difference; with it off, SONIC tracks the unchanged reference. Either way, the commanded value never jumps.

**Other transitions:**

| Transition | Action |
|---|---|
| → MANIPULATION | Guard as above → anchor (§6.2) → start sending. |
| MANIPULATION → FROZEN | Snapshot the arms from **the last sent IK solution**, not from measurement, so the reference doesn't step; send `waist_cmd`; carry the height over. **Heading is not reset.** Today, entering FROZEN from anything but VR_3PT calls `reset_yaw()` (`pico_manager_thread_server.py:2086-2089`); skip that under `--manipulation`. |
| PLANNER → FROZEN | Not reachable under `--manipulation`. |
| Any → OFF | Unchanged. |

### 6.6 Tracking loss, stale feedback, IK failure

**Per-controller loss.** A controller that is legitimately held still is **never** treated as lost. Detection uses only these signals:
- **Per-device validity or sample freshness from the backend** (a tracking or validity flag, or a per-controller sample timestamp). Gate G3 (§9) probes what XRoboToolkit and IsaacCapture actually expose. Where it exists, it is the primary signal.
- **Outlier guard:** the per-sample speed, |Δp| / Δt measured between consecutive **backend samples** (actual Δt, not the streamer tick), exceeds **6 m/s**. That equals 0.3 m at 20 Hz; a fast legitimate swing is ≈ 2 m/s. The outlier sample is discarded and that side holds.
  - **Re-acquisition** after an outlier: 3 consecutive samples that are each under the speed limit relative to the previous one. Then the normal recovery below applies.
- **Backend without a reliable per-controller signal:** if G3 finds no reliable per-controller validity or freshness signal for a backend, that backend gets **no stick-driven control at all**: no walking, no turning, no waist rate control, no height rate control. The streamer ignores both sticks, keeps `waist_cmd` and `height_cmd` fixed at their current values, refuses walk requests, and prints why. The reader-wide timestamp can keep advancing while one controller's pose and stick are stale, so it can't stop a stale deflected stick from driving a command to its limit. On that backend only arm and hand tracking remain, which is safe because a stale pose only makes the arm hold. Stick control is re-enabled for that backend only when a reliable loss signal is found and passes test 9f. This is written into the operator docs.

| Event | Response |
|---|---|
| One controller lost | That arm holds its last IK solution. That hand holds its last command. That controller's stick is treated as centered, so `height_cmd` / `waist_cmd` hold. The other arm keeps tracking. MANIPULATION has no locomotion, so nothing else changes. |
| Controller recovers | Must be valid for 0.5 s continuously, then **re-anchor that side** (§6.2). The robot-side anchor is the **held commanded** arm pose, `p_w0, R_w0 = FK(held IK solution, current waist_cmd)`, **not** the measured joints. The reference continues from exactly what was being sent, and the delta accumulated during the loss is discarded. Any measured-vs-commanded tracking error stays SONIC's job, as it was before the loss. |
| Whole reader stalls, streamer crash, or network loss | Streamer: stop sending (existing guard). **Deploy input watchdog (new):** if no planner message arrives for **200 ms**, the deploy itself commands a controlled stop. It switches the planner to IDLE with movement = 0, keeps the current height and upper-body references, and the waist lock ramps out (§6.4). The 200 ms bound replaces the existing 1 s `PLANNER_TIMEOUT` (`zmq_manager.hpp:581`) **only** when the deploy is started with `--input-watchdog-ms 200`, which the manipulation launch sets. Default sessions are unchanged. **The stop latches.** After a timeout, the deploy rejects locomotion (any non-static mode or nonzero movement or facing change) until it has received **≥ 150 ms of consecutive messages with a neutral walk command** (static mode, movement = 0), and only then accepts a new walk request. After the timeout, the applied upper-body reference is rate-limited (≤ 1 rad/s per joint) toward the incoming target. It stays limited **until every joint of the applied reference is within 0.01 rad of the incoming target** (catch-up), however long that takes, and it follows any target changes that arrive during catch-up at the same limit. The limit is released only after catch-up, so it can never end with a jump. The deploy reports `watchdog_latched` in feedback, and the streamer prints `release walk stick to re-arm`. |
| Feedback stale during MANIPULATION | Not needed after anchoring. Only blocks **entry**. Recovery re-anchors to the held commanded pose (row above), so it doesn't need feedback. |
| IK fails or misses budget on a tick | Send the last valid solution for that arm. After > 0.5 s of continuous failure, print a warning and treat it as a loss for that side (recovery = re-anchor). |
| IK solves but the hand is far from its target (tracking-error check, required; G2 found a stall that `solver_ok` doesn't report) | Every tick, compute the FK error of the solution against its target. If it exceeds **3 cm or 10°**, solve again in the same tick from the posture seed (`q_posture14`). **Stall** (the re-seeded error is ≤ 0.5 × the original and within 3 cm / 10°): the sent arm moves toward the re-seeded solution at **≤ 1 rad/s per joint**, never in a step, and later ticks warm-start from it. **Reach limit** (re-seeding doesn't help): send the closest solution as today. This is not a failure and doesn't count toward the 0.5 s loss rule. Errors, re-seeds and their outcome are logged in `manip_debug` **[my call: thresholds and re-seed rule]**. |
| Controller lost in FROZEN | The left controller drives walking: treat its stick as centered, so walking stops. The right controller drives turn and height: they hold. |

### 6.7 Data recording

`run_data_exporter.py` records `teleop.stream_mode` and `teleop.planner_*` today, but **not** `upper_body_position`. Add per frame:
- `teleop.upper_body_position` (17), `teleop.waist_cmd` (3), `teleop.height_cmd` (the value actually sent, including −1), `teleop.waist_lock` (sent value);
- per-controller tracking state (valid/lost/recovering);
- the new stream mode as a distinct `stream_mode` value.

Also record per episode: the deploy `--waist-lock` flag and the capture backend.

## 7. Success criteria (initial sim targets, not validated)

### 7.1 Measurement definitions

- **Pelvis:** the MuJoCo floating-base body (`pelvis`) pose, from sim state. The deploy's `base_trans_measured` is a fixed default and can't be used. On the real robot, yaw comes from `base_quat_measured` only.
- **Height:** pelvis z in world. Command→pelvis-z mapping is calibrated in G1.
- **Wrist:** origin and orientation of `left/right_wrist_yaw_link`, from FK of `body_q_measured`.
- **Baseline:** mean over the 1.0 s before the test motion starts, with the operator still.
- **Peak:** max |value − baseline| from motion start to the end of the settle window.
- **Final:** mean of |value − baseline| over the last 0.5 s of a 2.0 s settle window after the motion ends.
- **Oscillation:** peak-to-peak over the final 1.0 s of the settle window.
- Every test runs 5 times. A criterion passes only if all 5 runs pass.

### 7.2 Criteria

Each test runs in MANIPULATION with lock off and with lock on, unless noted.

| # | Test | Criterion |
|---|---|---|
| 1 | Forward reach 30 cm, right / left / both | Pelvis yaw: peak < 3°, final < 2°. Pelvis xy: peak < 2 cm, final < 1 cm. Waist joints vs `waist_cmd`: final < 2° (lock off), < 0.5° (lock on). Oscillation: < 1 cm and < 1° |
| 2 | Each translation axis ±15 cm (fwd, left, up), per arm | Same as #1. Wrist position error vs target, final < 2 cm |
| 3 | Controller rotations ±30° roll / pitch / yaw, per arm | Same pelvis/waist limits as #1. Wrist orientation error, final < 10° |
| 4 | Waist-only: controllers still, stick through the full yaw and pitch range | Arm joints change < 1° (hands ride the torso). Wrist pose in `T` changes < 1 cm. Waist tracks `waist_cmd`, final < 2° |
| 5 | Entry: toggle into MANIPULATION with the operator still | Wrist displacement peak over **1.0 s** after entry < 2 cm. Waist peak deviation < 1° |
| 6 | Height sweep [0.2, 0.8] and back, incl. first use from −1 | Pelvis z within ±3 cm of the calibrated target, final. No fall. **First use, no extra discontinuity:** the first commanded value equals `H_default`, and pelvis z speed in the first 0.5 s stays ≤ 1.5 × the commanded rate (≤ 0.15 m/s at full deflection) |
| 7 | MANIPULATION↔FROZEN toggles ×20 | No fall. `waist_cmd`, `height_cmd` and heading unchanged across toggles. Wrist peak < 2 cm per toggle |
| 7b | Grasp an object in sim, toggle MANIPULATION→FROZEN→MANIPULATION ×10; on half the runs change trigger/grip while frozen (release fully, squeeze harder) | Hand command identical immediately before and after each toggle. Object slip < 1 cm. On re-entry, the hand command stays held until takeover (no open or tighten step) |
| 7c | Takeover rules, per hand: (a) trigger matched, grip off by 0.2 → no takeover; (b) both within 0.04 → takeover; (c) one input at 0.06 → no takeover; approach from above and from below; left and right independently | Takeover exactly per the §5.1 rule. Hand command continuous at takeover (no step beyond the 0.2 s blend). Prompt direction correct in every case |
| 8 | Walk start/stop ×10 in FROZEN (lock on and off), incl. at `H_walk_min`; height-lower request while walking; walk request below `H_walk_min` | No fall. Locomotion never starts before feedback shows `α == 0`. Waist peak deviation from `waist_cmd` < 5° at lock release/engage. No height command change caused by walking. Lowering is clamped at `H_walk_min`. Walk below the band is refused. Current scope: a walk request after any height change is refused, including after −1 is sent again, and a −1 restart within 4.5 s of a stop is refused; the robot stays standing. Once a check is adopted: a walk during a height transition, or while the check fails (feedback motion injected), is refused |
| 8b | Lock-release confirmation fails (feedback blocked during walk request) | No walk; robot keeps standing; error printed |
| 8d | Lock-release ack: inject a stale `α == 0` sample from before the request, delay feedback 300 ms and 700 ms, reorder and duplicate samples | Walking starts only on a sample with the matching `lock_ack_seq` and `α == 0`. 300 ms delay (below the 0.5 s timeout): walking starts once the matching ack arrives. 700 ms delay (above): timeout, no walk, error printed |
| 8c | Brief message gaps (50 ms, 150 ms) while locked | 50 ms: α unchanged. 150 ms: smooth ramp-out, then ramp-in; waist reference has no step |
| 9 | Tracking loss: occlude one controller 2 s, then recover; repeat with a deliberate measured-vs-commanded arm offset (e.g., external load in sim) | Lost arm holds (wrist peak < 1 cm during loss). After recovery: **commanded** arm reference continuous (no step > 1 cm), wrist peak < 2 cm. Other arm unaffected |
| 9b | Stationary controller held still for 30 s | Never classified as lost; arm keeps tracking when it moves again with no re-anchor jump |
| 9c | Fast legitimate swing (≈ 2 m/s) and an injected 0.5 m single-sample spike | Swing: no outlier trip. Spike: discarded, that arm holds, re-acquires after 3 good samples |
| 9d | Left controller lost while walking in FROZEN | Walking stops (stick treated as centered); no fall |
| 9e | Kill the streamer process, and separately drop the network, mid-walk | Deploy watchdog stops locomotion within 200 ms of the last message, robot comes to a controlled stand with no fall. Report time to standstill |
| 9e2 | Drop the network mid-walk for 1 s, then restore it **with the walk stick still deflected** throughout | No locomotion after reconnect until the stick is released (≥ 150 ms neutral) and deflected again. Arm references rate-limited, no step |
| 9e3 | Outage during which the arm target moves ≥ 2 rad on one joint (cannot close in 1 s at 1 rad/s); keep changing the target during catch-up | Applied reference moves at ≤ 1 rad/s until within 0.01 rad of the target, follows the changing target, and the limit releases with no step |
| 9f | Per backend: freeze **each** controller's sample (pose and stick, with the stick deflected) while the rest of the reader keeps updating; left during FROZEN walking and during MANIPULATION height control, right during FROZEN turning/height and during MANIPULATION waist control | Backend with a reliable per-controller signal: the frozen stick is treated as centered, so walking stops and `height_cmd` / `waist_cmd` / heading hold. Backend without one: all stick control was already disabled; verify the sticks have no effect and walk requests are refused |
| 10 | Stale feedback / IK failure: block feedback, then request entry; inject IK failures | Entry refused, robot stays in prior mode. IK failure → hold, no jump |
| 11 | Chords: staggered A+B+X+Y in every press order; B+Y with staggered release; all four held at launch; A+B+X+Y pressed immediately after B+Y fires without releasing | Running: stop always fires, including right after B+Y without re-arm. B+Y never fires inside a staggered A+B+X+Y without a 150 ms pause. At launch, held buttons never start the policy; start works after all are released and pressed again |
| 12 | #1, #5, #9 on **both** capture backends | Same criteria |

A log/analysis script that computes these from sim runs is part of the deliverable.

## 8. Risks

| Risk | Mitigation |
|---|---|
| SONIC still sways or rotates with a clean reference | Waist hard-lock A/B; legs stay with the policy by design; report residuals as measured |
| Hard-lock removes a balance DOF | Two-key enable, off by default, released while walking, blended ramps, gantry first |
| Planner ignores height in IDLE, or walking at low height is unstable | G1 decides; IDLE_SQUAT fallback; walking height band |
| Height replans cause jitter | ≥ 1 cm quantization, ≤ 5 Hz, exact final send on release |
| IK too slow or flips the elbow | Arm-only reduced model, posture task, p99 budget gate, hold-last on failure |
| Controller tracking loss | Per-device validity (G3), outlier guard, hold, re-anchor to the commanded reference (§6.6) |
| Streamer crash or network loss while walking | Deploy input watchdog, 200 ms (§6.6) |
| Chord misfire on overlapping buttons | Arbiter (§5.3), stop-first |

## 9. Gates and rollout

**Gate order:**
1. **G0, G1 and G3** run first and need **no feature code**. They use logging and probe scripts only.
2. **G2** then runs as an **isolated, non-integrated prototype**: a standalone benchmark script that adapts the IK solver to the arm-only model and is not wired into the streamer or the deploy.
3. **Main feature implementation starts only after all four pass.**
4. Each gate's outcome (pass/fail, measured values, chosen parameters) is recorded in §11 before moving on.

- **G0, twist diagnosis (log only).** In today's POSE mode, record **time-synchronized**:
  1. the streamed SMPL (input reference: pelvis/spine/wrist);
  2. the deploy's `base_quat_target` (the streamed root after the deploy's heading correction). In POSE mode `body_q_target` is the streamer's placeholder `joint_pos` (zeros except the wrists), so no waist target is observable: the chest level is compared end to end, the streamed chest twist (shoulder line vs hip line in the root-local `smpl_joints`, which is what the policy reads; `smpl_pose` isn't a policy input in this configuration) vs measured waist yaw. The deploy's input path from `smpl_pose` to the policy observation is inspected and recorded, not assumed;
  3. measured motion: `body_q_measured`, `base_quat_measured`, plus MuJoCo base pose in sim.

  Do a hand-only reach, after a calibration segment (whole-body turn, chest twist) that validates the frame and joint mapping between each compared pair. Results are recorded in two parts:
  - **Robot side** (decides the gate): `none` / `pipeline` (twist appears between the streamed root and the deploy's heading-corrected target) / `sonic` (between that target and the measured pelvis) / `chest_residual` (an end-to-end policy/robot-side residual between the streamed chest twist and the measured waist; it doesn't isolate SONIC's internal target or the cause) / a `+`-joined combination / `inconclusive`.
  - **Reference side** (recorded, doesn't block): `none` / `pico` (only with independent operator-motion evidence, such as a chest-mounted tracker read raw, showing the operator was still) / `operator_moved` / `twist_in_reference_operator_unverified` / `inconclusive`. Twist in the PICO reference is never attributed to the PICO solver without that independent evidence.
- **G1, height probe (sim).**
  - Sweep `height` over [0.2, 0.8] in IDLE and IDLE_SQUAT, both as steps and as ramps, including IDLE↔IDLE_SQUAT switches.
  - Measure pelvis z, record the command→z map, and record `H_default`.
  - Walk at several heights to find `H_walk_min`.
  - Validate the walking band as in §6.3.
- **G1b, restricted walking height (sim), added 2026-11-11.** G1 failed and stays recorded as failed. G1b assesses only the restricted §6.3 behaviour, with the support band off:
  1. Hold 0.60 for 3 s, ramp up to `H_default` at the stick rate, then hold for `walk_settle_s`.
  2. Hand-off to −1 in the static mode for 0.5 s, then walk 10 s with −1.
  3. Stop: −1 in the static mode for 0.5 s, then back to `H_default` (IDLE_SQUAT), the MANIPULATION stance, and hold 3 s.
  4. Repeat forward, backward and turning, × 5 each.
  5. **Isolated switch block, × 5:** settled `H_default` 3 s → IDLE −1 3 s → `H_default` 3 s. The walk trials can't isolate the switch, because the robot is still settling from the ramp and the stop. A no-switch control from G1 data showed 7–10 mm in the same windows. **[revised 2026-11-11 after review]**

  Pass:
  - no fall;
  - pelvis z within ±3 cm of the settled pre-walk z (last 1 s of the `walk_settle_s` hold) throughout each trial;
  - **isolated switches:**
    - both 1 s windows must be stationary: z slope < 2 mm/s, xy speed < 2 cm/s, yaw rate < 0.05 rad/s;
    - the steady step (median of the last 1 s after the switch vs the last 1 s before it) must be < 1 cm;
    - the transient peak over the 3 s after the switch must be < 1 cm.
  - The in-walk switch steps are reported as diagnostics only.
  - **Step 7 (historical, FAILED, not adopted):** a peak leg-`body_dq` + gyro check, thresholds selected on run A and validated on run B. The result is in §11.
  - **Step 8: a new still-standing check, pre-registered 2026-11-11 before any validation data exists.**
    - **Feature.** Computed exactly as deployed: trailing 1 s window, samples with `recv` ≤ t, newest age < 0.1 s, no gap > 2 periods.
      - `D` = max over the 12 leg joints of |`body_q`(newest) − `body_q`(oldest in window)| [rad];
      - `G` = ‖mean of `base_ang_vel` over the window‖ [rad/s].
      - The check passes iff `D < D_max` and `G < G_max`.
    - **Ground truth:** the Step 7 definition, with missing-data instants marked unknown.
    - **Development: runs A and B pooled. Both have been looked at, so neither is validation.**
      - Use static instants ≥ 0.5 s after a static-command change. This is a superset of the deployed scope, so thresholds come out conservative.
      - Find the boundary: the grid pair with zero false acceptances that maximizes accepted ground-truth-still instants.
      - **Final thresholds = 0.8 × boundary values.**
      - **Development gate:** zero false acceptances and ≥ 26/30 pre-walk holds accepted, in-sample. If it fails, record FAILED and don't record run C.
    - **Freeze.** Before run C is recorded, write the sha256 of the script and of the thresholds JSON into the plan ledger. Check them unchanged after.
    - **Validation: run C, a new recording, validated once and never re-tuned.**
      - Content: G1b protocol with 8 s pre-walk holds and 10 isolated holds of 8 s.
      - Evaluate at deployed-scope instants: ≥ 4.5 s since the last static-command change.
      - **PASS** = zero false acceptances, overall and by class (after height or mode change, after walk stop, other static), **and** ≥ 13/15 pre-walk and ≥ 9/10 isolated holds accepted within 1.5 s of the 4.5 s mark.
      - Otherwise FAILED. INCONCLUSIVE applies only to missing data.
    - **Outcome:** PASS adopts the check and enables walking after height changes, per §6.3. Anything else keeps the −1-only scope.
    - In every case, the check is validated in sim only. Hardware validation must check both false refusals and false acceptances.
  - **Step 9: decision-time stress test of the Step 8 check. Pre-registered 2026-11-12 (user review), before any Step 9 data exists.**
    - **Purpose and scope.** Step 8 never put residual motion in front of the check at the moment it decides (≥ 4.5 s after the last static-command change). Step 9 does.
      - Required before walking after height changes is enabled on hardware (§6.3).
      - **Thresholds unchanged:** `D_max` = 0.0135184 rad and `G_max` = 0.0037496 rad/s, from the Step 8 frozen JSON (sha256 `678b6e2f…`, checked unchanged). The feature, ground truth and window rules are unchanged from Step 8.
    - **Disturbance injection (sim only).** A new sim option, `--disturbance-port`, is off by default.
      - When set, the sim applies a world-frame wrench (force N, torque N·m) to the pelvis body. The wrench comes from a PUSH/PULL channel, and the last received value is held.
      - The applied wrench is published in `sim_gt` as `pelvis_wrench`. It is never visible to the check.
    - **Motion confirmation (per disturbance, from `sim_gt` only).**
      - A disturbance is **effective** if, inside its window, the G1b `_stationary` criterion is violated for ≥ 0.5 s in total (nudge) or ≥ 1.0 s (creep). That criterion is z slope ≥ 2 mm/s, xy speed ≥ 2 cm/s or |yaw rate| ≥ 0.05 rad/s, on the trailing 1 s window, whole or either half.
      - "Total" counts the 100 Hz ground-truth instants × 0.01 s.
      - The 3 mm z band doesn't count towards effectiveness. Non-still instants caused by the band alone are reported separately.
    - **Calibration run D0 (magnitudes only; reads `sim_gt` only, the D/G feature isn't computed).**
      - Ladders, each value × 2, ascending, at IDLE_SQUAT `H_default`. Each disturbance starts 6.0 s after a settled hold begins.

        | Disturbance | Ladder | Profile |
        |---|---|---|
        | nudge | 25, 50, 100, 150, 200 N | 0.1 s pulse, along world +x and along world +y |
        | lateral creep | 15, 30, 60, 100 N | world +y force |
        | down creep | 50, 100, 200, 300 N | world −z force |
        | yaw creep | 5, 10, 20, 40 N·m | torque about world z |

        - Every creep: a linear ramp up over 4 s, hold 3 s, linear ramp down over 4 s.
        - Windows: onset → onset + 8 s (nudge), onset → onset + 17 s (creep).
      - **Chosen magnitude per type** = the smallest ladder value effective in 2/2 without a fall. For nudges, the larger of the +x and +y choices.
      - **A fall** makes that value and the rest of its ladder unusable. The remaining types are rerun in D0b.
      - **If no value of a type is effective,** stop and report before run D. The ladder isn't extended without approval.
    - **Freeze (after D0, before run D).** `g1b_still3.py freeze` writes a manifest of sha256 values covering:
      - the analysis script and every repo-local module it imports, as found in `sys.modules` after import;
      - the run-D probe;
      - the sim files with the disturbance option;
      - the Step 8 frozen JSON;
      - the calibration JSON.

      The manifest's own sha256 goes into the ledger line `Step 9 FREEZE:`. `validate` recomputes every entry and refuses on any mismatch. **This applies to all future frozen runs (2026-11-12).**
    - **Run D (new recording, validated once, never re-tuned).** Run D has two parts.
      - **A. Walk-stop-walk:** forward, backward and turning, × 3 reps each.
        - Pre-hold 0.60 for 3 s, ramp to `H_default`, then a settled hold of 6 s.
        - Then 5 cycles of:
          1. hand-off −1 for 0.5 s;
          2. walk 5 s at −1 (turn = IDLE at 0.5 rad/s);
          3. stop: −1 for 0.5 s;
          4. IDLE_SQUAT `H_default` for δ;
          5. a **walk request** at δ after that last static-command change.
        - δ order per rep:
          - rep 0: 1.0, 3.0, 4.5, 5.0, 6.0 s;
          - rep 1: 6.0, 4.5, 1.0, 5.0, 3.0 s;
          - rep 2: 3.0, 5.0, 6.0, 1.0, 4.5 s.
        - After the 5th request, one more walk and stop, then an 8 s post-stop hold.
        - The probe walks after every request regardless of the decision. A decision depends only on data before it.
        - Requests at δ ≥ 4.5 s are deployed decisions (27 of them). Requests at δ < 4.5 s are refused by condition 2 in deployment and reported as diagnostics only.
      - **B. Disturbances at the 4.5 s mark:** 4 nudges (world ±x, ±y) and 3 creeps (lateral, down, yaw) at the chosen magnitudes, × 3 reps each (21 blocks).
        - Each block: pre-hold 0.60 for 3 s, ramp to `H_default`, then a settled hold. In rep 1, the hold is 3 s followed by IDLE −1. Reps 0 and 2 stay at IDLE_SQUAT `H_default`.
        - Onset is 6.0 s after the block's last static-command change. The static command stays unchanged through the window.
        - Window: 8 s (nudge) or 17 s (creep) after onset.
      - Marker names reuse the G1b vocabulary, so the Step 8 code labels static segments, walks and command changes unchanged. A disturbance isn't a command change.
    - **Evaluation (`g1b_still3.py validate D`, once).**
      - **Deployed decision at instant t:** ≥ 4.5 s since the last static-command change, and the check accepts.
      - **False acceptance:** a deployed decision while ground truth isn't still. Unknown ground truth is excluded and counted. Reported by class:
        - post-stop (≤ 10 s after a walk stop);
        - nudge window;
        - creep window;
        - other static;
        - walking (counted, as in Step 8).
        - Also reported: the onset sub-class, the first 0.25 s of each disturbance.
      - **Amendments before freeze (2026-11-12, after the code review; no run-D data exists; based on D0 ground truth only):**
        1. Grid instants that fall in no marker window are ground-truth **unknown**: excluded and counted as `marker_gap_instants`. These are the µs gaps between one segment's `end` marker and the next one's `start`.
        2. In a part-B segment, instants **before onset** use `z_final` = median z over the last 0.5 s before onset, for the 3 mm band. From onset to the segment end, the band is violated only if z is ≥ 3 mm from **both** the pre-onset and the segment-end `z_final`: the robot standing at either settled height is in the band. Stationarity is unchanged. This amendment was added after fix round 1: with only the segment-end reference, the instants just after onset fail the band before anything has moved. Reason: a disturbance that leaves z shifted by ≥ 3 mm would otherwise label the settled pre-onset period not still. D0b yaw 10 and 20 N·m did exactly that.
        3. Each disturbance window's peak `pelvis_wrench` must match the chosen magnitude and axis. A mismatch counts as missing data.
        4. Each false acceptance is listed with:
           - time;
           - window;
           - `since`;
           - D/`D_max` and G/`G_max`;
           - the ground-truth cause (band, stationarity, or no segment).
        5. A sync summary is reported (information only).
        6. Times to first acceptance are reported both raw (check only) and as deployed decisions.
      - **Request table** (all 45 requests): δ, ground truth, raw check result, deployed decision.
      - **Settled holds:** the 9 part-A 6 s holds plus the 21 part-B pre-onset holds, accepted within [4.5 s, 6.0 s) after their change.
      - **Also reported:**
        - accepted share of ground-truth-still instants at deployed scope (the false-refusal rate is the remainder);
        - time to first acceptance in the 9 post-stop holds;
        - after each disturbance, the time from ground truth returning to still until first acceptance.
    - **Verdict, in order:**
      1. any false acceptance (any class, onset included) → **FAILED**;
      2. otherwise, fewer than 2/3 effective in any of the 7 disturbance types, or missing data → **INCONCLUSIVE**;
      3. otherwise, fewer than 26/30 settled holds accepted → **FAILED**;
      4. otherwise → **PASS**.
    - **Outcome:**
      - **PASS**, together with the §6.3 gyro precondition, allows walking after height changes on hardware, for gantry validation that checks both false refusals and false acceptances.
      - **FAILED** withdraws the Step 8 adoption everywhere: `still_check_adopted = False`, and the −1-only scope applies.
      - **INCONCLUSIVE** keeps the sim adoption, and hardware stays −1-only.
  - `walk_settle_s` comes from G1b run A (6 s holds): the per-trial settle time (z slope < 2 mm/s, xy speed < 2 cm/s, within 3 mm of the hold's final z, through the end of the hold); max + 0.5 s, rounded up to 0.5 s. Run B, the gated run, uses that value, and every pre-walk reference window must be stationary.
  - the sync contract holds and motion is evidenced, as in G1.
- **G2, IK benchmark (isolated prototype).** The arm-only solver meets the p99 budget, using the method in §6.2.
- **G3, backend probe.** For XRoboToolkit and IsaacCapture, record whether each exposes per-controller validity and/or a per-controller sample timestamp. Hold a controller still, then cover or turn it off, and log what changes. The result picks the §6.6 loss signal per backend, or disables locomotion for that backend.

After the gates:
1. MANIPULATION, lock off, sim: tests 1–5, 7b, 7c, 9–9d, 10.
2. Lock on, sim: tests 1–5, 8–8d.
3. FROZEN height/turn, toggles, chords and watchdog, sim: tests 6–8, 9e, 9e2, 9e3, 9f, 11.
4. Both backends: test 12.
5. Real robot on the gantry, then free-standing.

## 10. Left to the builder

- IK solver adaptation details and tuning, within the §6.2 requirements.
- Plumbing of `--waist-lock` and the `waist_lock` ZMQ field, following the existing optional-field pattern in `zmq_manager.hpp`.
- Exact rates, limits and timeouts within the stated defaults.
- Structure of the sim-metric script.

## 11. Gate results

Fill this in before moving past each gate.

| Gate | Date | Result | Measured values / chosen parameters | Notes |
|---|---|---|---|---|
| G0 twist diagnosis | | | Robot side: none / pipeline / sonic / chest_residual (+ combinations) / inconclusive. Input-path inspection result. Reference side: none / pico / operator_moved / twist_in_reference_operator_unverified / inconclusive. Calibration gains; `target_unobservable` channels; `operator_ref` available? | |
| G1 height probe | 2026-11-10 | **FAIL** (gate_pass=false) | IDLE honors height: **no** (IDLE pelvis z 0.787 m at every command 0.2..0.8; IDLE_SQUAT is used, so MANIPULATION sends IDLE_SQUAT whenever `height_cmd` is set). Standing z at −1 sentinel (IDLE): 0.787 / 0.786 / 0.787 m (3 settles, spread < 1 cm). **`H_default` = 0.7984** (IDLE_SQUAT; rounds to 0.80). `cmd_to_z` (IDLE_SQUAT, settled z): 0.2→0.453, 0.3→0.449, 0.4→0.494, 0.5→0.542, 0.6→0.621, 0.7→0.713, 0.8→0.788 m (non-monotonic below 0.4: 0.2/0.3 saturate). IDLE↔IDLE_SQUAT switches at 0.5 (3 cycles): IDLE 0.786–0.787 m, IDLE_SQUAT 0.540–0.542 m, no fall. **`H_walk_min` = unknown** → PRD §6.3 fallback `H_walk_min := H_default` (walk only at the −1 sentinel or `height_cmd ≥ H_default − 1 cm`). Band assumption: none established (`band_assumption` = null; the interpolation assumption applies only when a band exists). walk_table (passes of 5 runs, fwd/back/turn): −1: 5/5/5; 0.80: 4/5/5; 0.75: 0/0/5; 0.70–0.40: 0/0/0 (all grid heights fail below 0.80). `reasons`: h_default (0.80) fails walk trials: fwd 4/5 (run 4: pelvis z 0.744 m vs stationary 0.783 m, dev 3.9 cm > 3 cm). Walking (SLOW_WALK 0.3 m/s) returns pelvis z to ~0.786 m whatever `height` is commanded (dev ≈ +3.4 cm at 0.75, +7.5 cm at 0.70, +15 cm at 0.60, +29 cm at 0.40), so a lowered height is not held while walking; turning in place (static mode) holds within 3 cm only down to 0.75. No falls; every analysis window passed check_sync (incl. rtf ≥ 0.9; settle windows 0.975–0.984). Data: `outputs/gates/g1_height.npz` (gitignored), analysis `outputs/gates/g1_analyze.json` (per-trial min_z / max_dev / disp / Δyaw, settle_z, idle_cmd_to_z, switch_z). All 150 trials show motion (min xy displacement 1.17 m fwd/back, min |Δyaw| 4.49 rad turn). Height ramps were recorded but not evaluated. Sim (free-standing): `run_sim_loop.py --no-enable-onscreen --no-elastic-band --publish-ground-truth --gt-bodies pelvis` (the recorded run used an equivalent monkeypatch; smoke check with the flag: pelvis z settles at 0.787 m). |  |
| G1b restricted walking height | 2026-11-11 (fix round 2, run B) | **PASS** (gate_pass=true) | Per direction passes/5: fwd 5, back 5, turn 5 (15/15). Every pre-walk reference window (last 1 s of the hold) stationary. Max abs Δz over each whole trial window: **2.05 cm vs the settled pre-walk z** (fwd 1.39–1.97, back 1.16–2.05, turn 1.07–1.47 cm) and **2.03 cm vs `settle:0`** (limit 3 cm). Isolated switch block (×5; hold 3 s → IDLE −1 3 s → IDLE_SQUAT `H_default` 3 s; both 1 s windows stationary in all 10): steady step to −1: −0.46..−0.29 mm (max abs 0.46 mm); back to `H_default`: +0.52..+0.79 mm (max abs 0.79 mm); transient peak over the 3 s after the switch: to −1 0.97–1.08 mm, to `H_default` 2.23–2.94 mm (limits < 1 cm). Falls: none. Sync: every window passed check_sync. Motion evidenced in all 15 trials (xy fwd 1.97–2.39 m, back 1.43–1.62 m, turn |Δyaw| 4.51–4.63 rad). `H_default` = 0.7984 (G1). **`walk_settle_s` = 4.5 s** (post-ramp hold at `H_default`), measured in run A (hold 6.0 s, same sequence, all 15 holds settled): per-trial `settle_s` = earliest time after ramp end from which z changes < 2 mm/s (trailing 0.25 s), xy speed < 2 cm/s and \|z − final\| < 3 mm (final = median of the hold's last 0.5 s) for the rest of the hold. Run A max settle_s: fwd 1.61 s (1.36–1.61), back 2.69 s (1.50–2.69), turn 3.51 s (1.41–3.51; one outlier, turn:4); overall max 3.51 s → 3.51 + 0.5 = 4.01 → rounded up to 0.5 s = 4.5 s. Run B (hold 4.5 s) settle_s: fwd max 2.27, back max 3.90, turn max 2.25 s; all within 4.5 s (margin 0.6 s; settle time varies run to run, outliers 2.7–3.9 s). Run A also gate_pass=true (max dev 2.43 cm vs hold ref). Walk trial: pre-hold IDLE_SQUAT 0.60 3 s, ramp 0.60→`H_default` 0.1 m/s, hold 4.5 s, IDLE −1 0.5 s, walk 10 s (SLOW_WALK 0.3 m/s; turn = IDLE at 0.5 rad/s) with −1, IDLE −1 0.5 s, IDLE_SQUAT `H_default` 3 s. Diagnostic only (not gated): in-walk 0.5 s steps to −1 −0.8..+0.5 mm, back to `H_default` −1.0..+5.6 mm; G1 no-switch control after the walk: +7.0..+10.0 mm fwd, +3.1..+4.1 back, −1.1..+0.5 turn. Run B 415 s. Sim: `run_sim_loop.py --no-enable-onscreen --no-elastic-band --publish-ground-truth --gt-bodies pelvis`; probe `g1b_walk_handoff_probe.py --out outputs/gates/g1b_runB.npz --h-default 0.7984 --hold-s 4.5` (default); analysis `g1b_analyze.py outputs/gates/g1b_runB.npz outputs/gates/g1_height.npz` → `outputs/gates/g1b_runB_analyze.json` (run A: `g1b_runA.npz`, `g1b_runA_analyze.json`; gitignored). | Assesses §6.3 restricted behaviour only; G1 FAIL above stays as recorded Margin note: run B back:4 reference slope 1.98 mm/s vs the 2.0 limit (next worst 0.59); the settle estimate is estimator-sensitive (a least-squares slope gives turn:4 4.13 s → 5.0 s). If a rerun trips "pre-walk hold not settled", 5.0 s is the fallback. **Still-standing thresholds: FAILED, peak check not adopted.** Offline, `g1b_still_thresholds.py --select outputs/gates/g1b_runA.npz --validate outputs/gates/g1b_runB.npz` (A = selection, B = validation, thresholds unchanged; JSON `outputs/gates/g1b_still.json`). Check: `g1_debug` samples with recv ≤ t only; window [t−1.0 s, t] covered, newest sample age < 0.1 s, no gap > 2 nominal periods (nominal 0.0200 s), max leg |`body_dq[0:12]`| < `still_dq_max` and max ‖`base_ang_vel`‖ < `still_gyro_max`. Ground truth: G1b `_stationary` (z slope < 2 mm/s, xy < 2 cm/s, |yaw rate| < 0.05 rad/s, whole window and both halves) on the trailing 1 s of `sim_gt`, plus |z − z_final| < 3 mm (z_final = median of the last 0.5 s of the static segment); false outside static segments (ramp, walk); a first version labelled a window moving whenever one `sim_gt` grid point was NaN (a 10 ms hole), fixed by skipping NaN points as `g1_analyze.py` does and marking a window unknown only if a hole exceeds 0.1 s (unknown instants: A 0, B 0). Selected on A (zero false acceptances, max accepted gt-still): `still_dq_max` = 0.03443 rad/s, `still_gyro_max` = 0.04669 rad/s. False acceptances (a: ≤ 5 s after a command change / b: ≤ 5 s after a walk stop / c: other / d: walking): A 0 (0/0/0/0), **B 32 (32/0/0/0)**. Fraction of gt-still instants accepted: A 30.0 % (3076/10268), B 9.2 % (734/7938 true accepts). Delay from gt-still onset to first acceptance: A pre-walk holds median 0.84 s, max 1.98 s (11 accepted), A isolated holds median 0.0 s, max 2.73 s (6 accepted); B pre-walk holds median 1.43 s, max 2.23 s (4), B isolated median 2.55 s, max 2.57 s (3). Usefulness (pre-walk holds ≥ 13/15, isolated holds ≥ 9/10 accepted before the hold ends): A 11/15 and 6/10, B 4/15 and 3/10, **not met in either run**. Verdict: **FAILED: peak check not adopted** (false acceptances in B, and usefulness not met), so no still thresholds are adopted and walking after a height transition stays disabled. Validated in sim only; hardware validation must check both false refusals and false acceptances. **Step 8 (new still-standing check, pre-registered): PASS.** Feature (trailing 1 s, recv ≤ t, newest age < 0.1 s, no gap > 2 periods): `D` = max over the 12 leg joints of |`body_q`(newest) − `body_q`(oldest sample used by the window, the last at or before t−1.0 s)| [rad]; `G` = ‖mean `base_ang_vel`‖ over the window [rad/s]; still iff `D` < `D_max` and `G` < `G_max`. Development (A+B pooled, static instants ≥ 0.5 s after a command change, `g1b_still2.py develop`): boundary `D` = 0.016898 rad, `G` = 0.004687 rad/s (8516 accepted gt-still instants); final = 0.8 × boundary: **`D_max` = 0.0135184 rad, `G_max` = 0.0037496 rad/s**; in-sample false acceptances 0 (also 0 incl. walking); pre-walk holds accepted 30/30 (isolated holds, info only: A 3/10, B 1/10 within their 3 s). Frozen before run C: script sha256 `3f88900447a8cf7e5d06f70419734ba40941a549a7e0fee760a9135f3656ae37`, frozen JSON sha256 `678b6e2fa3f84a1cce5784eb733dd36ff82d34dcb95d2697b7fcc94b41a6f685` (ledger line `Step 8 FREEZE:` in progress.md; validate refused-if-changed, unchanged). Validation: run C = `outputs/gates/g1b_runC.npz` (`--hold-s 8 --iso-hold-s 8`, 541 s, G1b analyzer on C for information: gate_pass true, max dev 2.41 cm), validated once at ≥ 4.5 s since the last command change (`outputs/gates/g1b_still2_runC.json`): false acceptances 0 overall and by class (a 0, b 0, c 0, d 0); pre-walk holds accepted within [4.5 s, 6.0 s] 15/15 (first acceptance 4.50–5.25 s after the change), isolated holds 10/10 (all 4.50 s); accepted share of ground-truth-still instants 96.8 % (10500 gt-still, 0 unknown). Diagnostic at ≥ 0.5 s scope: 0 false acceptances, 73.2 % accepted, holds first accepted 3.0–5.25 s (pre-walk) and 3.3–3.9 s (isolated) after the change. Verdict: **PASS**, the check is adopted and walking after height changes follows §6.3. Validated in sim only; hardware validation must check both false refusals and false acceptances. **Step 8 review caveats (2026-11-11):** at the deployed scope, run C contained 0 non-still static instants and 0 post-walk-stop instants, so its zero false acceptances carry little safety information. The safety evidence is the ≥ 0.5 s diagnostic: 0 FA over 11190 non-still instants, nearest at 1.04× the thresholds (`prehold:fwd:0` +2.75 s). The freeze hash covers `g1b_still2.py` and the frozen JSON, not the imported modules (all modified ≤ 17:58:18, before the freeze). Hardware precondition: measure the stationary gyro mean (PRD §6.3). **Step 9 (decision-time stress test, pre-registered 2026-11-12, §9): FAILED, so the Step 8 adoption is withdrawn everywhere (`still_check_adopted = False`, −1-only scope).** Calibration (D0 + D0b, `sim_gt` only; falls at lateral 100 N rep 1 and yaw 40 N·m rep 0): chosen nudge 25 N (0.1 s), lateral creep 15 N, down creep 200 N, yaw creep 5 N·m (JSON `outputs/gates/g1b_still3_calibration.json` sha256 `ec8e6fcf…`). Amendments 1–6 were made before the freeze after code review (§9). Freeze: manifest `outputs/gates/g1b_still3_manifest.json` sha256 `6afa30bafe18de251df4bf9af6a23e4e530f90d2a4f0f23c9d5d74aab2d5e0be` (19 files incl. shared helpers; ledger `Step 9 FREEZE:`, 2026-11-12T14:54:32+09:00), unchanged at validate. Run D = `outputs/gates/g1b_D.npz` (9 walk-stop-walk + 21 disturbance blocks, no fall, sync ok, rtf 0.98), validated once (`outputs/gates/g1b_still3_runD.json`, sha256 `5b3516e9…`), thresholds unchanged (`D_max` 0.0135184 rad, `G_max` 0.0037496 rad/s). **False acceptances 449 instants in 13 runs:** nudge 105 (8 of 12 nudge blocks; ground truth violated stationarity 0.2–1.3 s after onset while D and G were 0.55–0.99× the thresholds), creep 290 (one run, `down:1` at IDLE −1 under the held 200 N force; cause = 3 mm band, D 0.997× `D_max`), other 54 (`hold:B:nudge+x:0`, 13–14 s after onset, stationarity), post-stop 0, walking 0, onset (first 0.25 s) 2. Walk-stop-walk: 27 deployed requests (δ ≥ 4.5 s), 15 accepted, ground truth still at all 27 (0 false acceptances, 12 false refusals); 18 early requests (δ < 4.5 s) all refused by the raw check (8 of them ground-truth still). Settled holds accepted within [4.5, 6.0) s: 30/30. Accepted share of ground-truth-still instants: 76.8 %. Post-stop holds, deployed first acceptance 4.50–7.35 s (one never). Effective disturbances per type: nudge +x 2/3, −x 3/3, +y 3/3, −y 3/3, lateral 3/3, down 3/3, **yaw 1/3** (the run would have been INCONCLUSIVE without the false acceptances). Reading: residual motion after stops had died out by 4.5 s, so it never tested the check. Nudges did produce motion the 1 s net-displacement + mean-gyro feature doesn't see. **Independent review of run D (2026-11-12): the FAILED is genuine, with no analysis artefact (sync p99 0.18 ms, 0 gap/unknown instants, D/G and ground truth recomputed).** A nudge pushes the pelvis 10–12 mm and back within about 1.2 s. The check re-accepts 0.7–1.0 s after onset, mid-return, at an xy speed of 2.0–3.2 cm/s: D (endpoint joint displacement) is 0.26–0.95×, while the in-window joint excursion is up to 1.99× and the gyro peak 12–40× `G_max`. The out-and-back cancels in both terms. The `down:1` run is a quasi-static compression 4.1–4.9 mm under load (band). The 'other' run is a 1.2 mm, 1.3 s-period wobble. Any future check must see oscillation inside the window, not only net change, and needs a new pre-registration. **Task 3b closed (2026-11-12, user review):** the check stays disabled; the Step 8 pass is insufficient to authorize its use. A replacement needs a separately approved protocol and fresh validation, and a displaced but stationary posture needs additional evidence (§6.3). |
| G3 backend probe | | | Loss signal per backend; locomotion enabled per backend | |
| G2 IK benchmark | 2026-11-10 | **PASS on this host (i7-11700); teleop-PC run pending** (p99 < 10 ms) | CPU: 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz. Target source: **no recorded episodes found; random FK targets only** (1,000 timed trajectories of 10 ticks, linear sweeps between random in-limit q, controller poses derived by inverting the §6.2 anchor-delta mapping, k=1.0; the bench asserts the mapped target equals the FK target). 100 warm-up + 10,000 timed ticks; each tick = anchor-delta transforms + solve, plus in-tick both-arm torso-frame FK as conservative overhead. Latency over 4 runs, ms: p50 1.47-1.68, **p99 2.03-3.51, max 2.83-13.0** (max exceeded 10 ms once in 40,000 ticks (shared host, no pinning); that run's p99 was 3.51). Solver: pink, quadprog, reduced 14-joint model (all other joints locked at q0, torso_link frame), dt 0.05, 10 steps/tick, costs/posture weights from `BodyIKSolverSettings`; warm-started from the previous solution, posture toward the anchor q. Joint limits: robot-model limits incl. g1_supplemental_info shoulder_roll narrowing (arms cannot cross midline). `solver_ok` is False only on QP failure. Tracking: sweep tracking p99 8.7 cm is a branch stall under ~1 m/s random sweeps (persists after convergence; not flagged by solver_ok; Task 13 risk: detect stall via FK-error check); smooth 20 Hz streams (|dq| 0.03 rad/tick, 5x300 ticks) track < 2.2 mm position (max 2.12 mm) and < 0.12 deg. | Measured on this host (11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz); not confirmed to be the teleop PC. Default threads, no pinning, host load avg ~1.6 (21 users logged in). Recorded-episode replay (`teleop.smpl_joints`) not done. Bench: `gear_sonic/scripts/gates/g2_ik_bench.py`. |
