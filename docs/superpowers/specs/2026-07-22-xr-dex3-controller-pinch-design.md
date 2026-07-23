# XR Dex3 Controller Dual-Primitive Design

Status: Reduced feature scope approved in conversation on 2026-07-23; pending
written-spec review before implementation.

## Problem

The first controller-pinch implementation replaced the existing controller
close pose. Real-robot testing then exposed two separate requirements:

1. the index trigger must retain the previous analog close primitive; and
2. the controller grip/squeeze input must independently command a calibrated
   thumb-index pinch.

The same test also showed that the right-hand pinch target used the middle
finger slots. The error came from confusing the right retargeter's internal
order with the symmetric Unitree hand API order used by exported commands.

This change fixes those controller semantics and adds a bridge-side
command-space slew limiter. It deliberately does not redesign deploy/DDS
safety, feedback, failure handling, or source-loss behavior.

## Current Feature Scope

This task implements only the following:

- symmetric physical/API Dex3 joint ordering for controller-generated
  `dual_hand_joints` and the bridge hand arrays used by this feature;
- analog index-trigger control of the existing legacy close target;
- analog grip/squeeze control of the calibrated thumb-index pinch target;
- independent per-hand arbitration, with active index-trigger priority;
- safe normalization of both raw controller channels;
- a best-effort `0.25 rad` per-joint, per-published-frame limiter in the XR
  upper-body bridge;
- preservation of the bridge's current tucked-thumb stop preset and stop
  lifecycle; and
- automated checks plus guarded real-robot acceptance tests.

The XR mapping applies only when all three conditions hold:

- end effector is Dex3 (`--ee dex3`);
- input mode is controller (`--input-mode controller`); and
- the XR process is exporting hand targets to GEAR-SONIC.

Hand-tracking retargeting math and policy target generation are unchanged.
The bridge has no controller-origin metadata, so its best-effort limiter
applies to every included hand command that passes through this bridge,
regardless of whether the target originated from controller export, hand
tracking, or replay. Adding provenance metadata or provenance-specific gating
is outside this task.

The earlier trigger-only implementation plan at
`docs/superpowers/plans/2026-07-22-xr-dex3-controller-pinch.md` is superseded
and must not be executed. A replacement test-driven plan will be written only
after this revised design is approved.

## Explicitly Deferred DDS-Hardening Task

The following work is a separate TODO and must not be implemented as part of
this controller feature:

- deploy-side last-successful-write limiting when hand feedback is absent;
- measured-feedback validity, freshness, age, sequence, clock, and session
  metadata;
- feedback-inconsistency detection, persistence, and recovery;
- two-phase hand preparation/commit and DDS write-failure escalation;
- a shared publication gate and atomic first-fault safety-stop state;
- coordinator shutdown ordering and a body-only damping path;
- source-loss latching, XR session/replay protection, and deploy-session
  rebinding;
- bounded, nonblocking repeated stop delivery; and
- corresponding deploy concurrency and failure-injection tests.

The bridge limiter in this feature is defense in depth. It does not establish
an end-to-end DDS increment guarantee: ZMQ PUB/SUB may drop intermediate
frames, and the current deploy driver can publish a requested target directly
when valid hand feedback is unavailable.

## Controller Input Contract

TeleVuer owns raw WebXR sanitation. Each controller event first normalizes all
four per-side controller values into local variables, then refreshes both the
trigger and squeeze channels before pose, thumbstick, or other unrelated
payload parsing can return or raise.

For each side, TeleVuer stores:

- `trigger`: a sanitized Boolean indicating index-trigger activation;
- `triggerValue`: a sanitized raw analog value in `[0, 1]`;
- `squeeze`: a sanitized Boolean indicating grip/squeeze activation; and
- `squeezeValue`: a sanitized raw analog value in `[0, 1]`.

Sanitation is identical for both analog channels:

1. accept a Boolean only when its raw value is actually `bool`;
2. accept an analog value only when it is a real number and not `bool`;
3. clamp finite analog values to `[0, 1]`;
4. reset a missing, malformed, or unsupported analog value to `0.0` rather
   than retaining a value from a previous event; and
5. for an explicitly non-finite numeric value, fall back to `1.0` when the
   sanitized Boolean is active and `0.0` otherwise.

For both trigger and squeeze, TeleVuer publishes the sanitized analog shared
value before its corresponding sanitized Boolean shared value. This preserves
analog-first behavior while preventing stale squeeze commands and avoiding a
new active Boolean being paired with an old full-depth analog value.
For example, a valid `squeezeValue=0.8` followed by `squeezeValue="bad"` must
publish squeeze depth zero, not the previous `0.8`.

The wrapper preserves the existing downstream trigger convention:

```text
trigger_exposed = 10 * (1 - triggerValue)  # 10=open, 0=full
squeeze_exposed = squeezeValue             # 0=open, 1=full
```

The sanitized Booleans remain separate from these amplitudes because branch
selection and command depth have different responsibilities.

## Dex3 Physical/API Joint Order

Controller-generated `dual_hand_joints`, the bridge hand arrays, and the
Dex3 API/DDS interpretation used by this feature use the same order on both
sides:

| Side | Indices 0-6 |
| --- | --- |
| Left | `thumb0, thumb1, thumb2, middle0, middle1, index0, index1` |
| Right | `thumb0, thumb1, thumb2, middle0, middle1, index0, index1` |

The right retargeter configuration has an asymmetric internal target order,
but `HandRetargeting.right_dex_retargeting_to_hardware` explicitly permutes
that result into the symmetric API order above. The right-hand enum labels in
`teleop/robot_control/robot_hand_unitree.py` currently imply a different
order, but their sequential numeric values perform no permutation. Those
labels and nearby documentation will be corrected together with the target
and tests.

This is not a cleanup of the separate hand-tracking retargeting path. Its
retargeting code and permutations remain outside the current feature even
where they deserve separate investigation.

## Hand Targets

All values below are in physical/API order.

### Open

```text
left  = [0, 0, 0, 0, 0, 0, 0]
right = [0, 0, 0, 0, 0, 0, 0]
```

### Legacy Close

```text
left  = [0, 0,  1.75, -1.57, -1.75, -1.57, -1.75]
right = [0, 0, -1.75,  1.57,  1.75,  1.57,  1.75]
```

These are restored compatibility targets. They are called `legacy close`,
not a calibrated or validated full grasp. Hardware acceptance must verify
their usefulness separately.

### Calibrated Thumb-Index Pinch

```text
left  = [-0.379616,  0.516712,  0.121406, 0, 0, -1.273903, -0.419393]
right = [-0.379617, -0.516714, -0.121407, 0, 0,  1.273907,  0.419395]
```

The corrected right target leaves the middle-finger slots `3:5` open and
uses the index-finger slots `5:7`.

## Per-Hand Arbitration

For each side, compute the two analog depths:

```text
trigger_depth = clamp(1 - trigger_exposed / 10, 0, 1)
pinch_depth   = clamp(squeeze_exposed, 0, 1)
```

Then select one primitive from the sanitized activation Booleans:

```text
if trigger_active:
    desired = trigger_depth * legacy_close
elif squeeze_active:
    desired = pinch_depth * calibrated_pinch
else:
    desired = open
```

The left and right hands arbitrate independently. An active index trigger
always selects legacy close, even if grip/squeeze is deeper. If its analog
value is missing or malformed, safe-open depth zero still suppresses the
pinch branch; this satisfies priority without commanding an unsafe close.
Rest noise in a finite analog value cannot select a primitive because branch
selection uses the sanitized Boolean, not `depth > 0`.

A pure helper owns these targets and arbitration math. The XR main loop calls
it only inside the exact Dex3/controller/export gate. The resulting
`dual_hand_joints` JSON field remains a 14-value left-then-right vector.

## Bridge Best-Effort Slew Limiter

The bridge applies a stateful limiter immediately before encoding and locally
enqueuing each planner publication that includes hand fields. It runs after
all hand transformations, including live-source selection, pause/resume
handling, both stop-release publication paths, stop final-hold selection, and
bridge stop-preset selection.

The hand limiter uses a dedicated fixed constant
`MAX_HAND_JOINT_STEP_RAD = 0.25`. It is independent from the existing
`--max-joint-step` option, which continues to limit upper-body joints and is
unchanged by this feature.

For each included hand and joint:

```text
candidate = last_emitted + clip(desired - last_emitted, -0.25, 0.25)
```

For each planner frame, the bridge follows one transaction-like sequence:

1. validate and preview candidates for both included hands without mutating
   limiter state;
2. if either included hand is malformed or non-finite, reject the complete
   planner frame and commit neither hand;
3. encode and send the complete planner message;
4. if validation, encoding, or planner send raises or fails, commit neither
   hand; and
5. after successful local planner enqueue, atomically commit all included
   hand candidates as their new `last_emitted` states.

An omitted hand is neither validated nor committed and retains its state. A
manager-state message is a later, separate send: its failure cannot roll back
a planner command that was already locally enqueued.

Invalid source normalization, hand validation, or planner encoding is caught
at the live-loop publication boundary. The bridge skips the complete planner
frame, commits neither limiter state, emits a rate-limited warning, and
continues processing later frames; it does not publish the body or the other
hand from the rejected frame and does not terminate solely for this input
error.

Additional limiter behavior is:

- maintain independent seven-value state for each hand;
- seed each state from command-space zero; this is only a command reference
  and is not evidence that the physical robot hand is open;
- while `ramp_phase == "pause"`, re-emit `last_emitted` for every included
  hand and do not advance limiter state, even if the XR-frozen raw target is
  farther ahead;
- resume changes the desired target without resetting limiter state and slews
  from the frozen `last_emitted` command; and
- route normal live frames, both stop-release paths, and stop final-hold frames
  through the same limiter instance.

`0.25 rad` is a numerical command increment per bridge publication, not a
motion-rate guarantee. The bridge and deploy processes run at different
frequencies, local ZMQ acceptance is not downstream receipt, and intermediate
frames can be dropped.

This task does not add feedback-based seeding, wait for physical hand
convergence, modify deploy limiting, or change bridge source-loss behavior.

## Lifecycle and Stop Behavior

The existing XR ramp-in, pause/resume, ramp-out, bridge stop-release, and
final-hold ownership remains in place. The bridge's current configured stop
preset is retained. In the documented real-robot workflow that preset is
`tucked-thumb`, because opening the hand before stop can collide with the
robot's leg.

Explicit graceful exit therefore continues to slew toward the selected
bridge preset under the existing lifecycle, now with the per-frame bridge
limiter applied. This feature does not introduce a new source-loss latch,
emergency-stop protocol, convergence watchdog, or stop-delivery contract;
those belong to the deferred DDS-hardening task.

The limiter does not extend stop final-hold duration to await hand convergence.
If the existing stop schedule ends before the limited command reaches the
preset, the hand stop target remains best effort under the current lifecycle.

## Geometry Verification

The MuJoCo verifier imports the targets from the XR helper and interprets both
seven-value hand vectors in the symmetric physical/API order. It maps those
slots to named model joints before evaluating geometry, so the model's
actuator declaration order is not treated as the exported API contract.

At full pinch, the distal thumb/index collision-geometry signed distances are
approximately:

- left: `-8.7e-8 m`;
- right: `-5.3e-8 m`.

The verifier requires:

- all ten pre-final samples from `alpha = numpy.linspace(0, 1, 11)` to remain
  separated;
- signed distance to be non-increasing within `1e-6 m`;
- final absolute signed distance to be at most `1e-6 m`;
- middle-finger joint slots to remain zero throughout pinch; and
- every target to remain inside the current deploy hard limits.

It also exercises the idealized command-space recurrence
`q_next = q + clip(q_target - q, -0.25, 0.25)`, which reaches both pinch
targets in six recurrence steps. This is not a production-driver tick or DDS
guarantee. Numerical tangency is defined by signed-distance tolerance, not by
`MjData.ncon > 0`.

Modeled tangency verifies geometry under the corrected mapping; it does not
prove the physical DDS ordering or pad contact. Both hands still require
guarded hardware acceptance.

## Test-Driven Verification

### External XR Tests

The external tests will cover:

1. trigger and squeeze Boolean/type sanitation;
2. clamp, missing, malformed, and non-finite analog cases;
3. stale squeeze clearing after a valid sample;
4. analog-before-Boolean publication and torn-read regression coverage for
   both channels;
5. wrapper trigger and squeeze output domains;
6. open, half-depth, and full-depth legacy-close targets;
7. open, half-depth, and full-depth pinch targets;
8. explicit trigger priority over squeeze at all amplitude combinations;
9. an active trigger with safe-open depth suppressing squeeze;
10. independent left/right primitive selection;
11. values around the previous `6.5` trigger threshold remaining analog;
12. the exact Dex3/controller/export scope gate;
13. raw WebXR payload through TeleVuer, wrapper, and helper;
14. finite 14-value left-then-right output; and
15. independent ordering oracles that pin both API orders as middle-first,
    the right pinch slots `3:5` as zero and `5:7` as
    `(1.273907, 0.419395)`, the right retargeter permutation from internal
    index-first to API middle-first, and the corrected right enum labels and
    numeric values.

Named files:

- `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_dex3_controller_pinch.py`
- `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_televuer_controller_payload.py`
- `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_teleop_keyboard_controls.py`

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

### Bridge and Geometry Tests

Repository tests will cover:

- exact 14-value JSON split using the corrected symmetric right-hand order;
- per-hand `0.25 rad` limiting on live targets;
- independent limiter state for the two hands;
- pause before target convergence freezing the emitted command, followed by
  resume slewing from that exact command;
- tucked-thumb stop-preset limiting;
- omitted-hand behavior and complete-frame rejection, rate-limited logging,
  and loop continuation for invalid hand fields;
- preview/commit atomicity across both included hands;
- no commit after validation, encoding, or planner-send failure;
- planner commit remaining valid after a later manager-state failure;
- all live, stop-release, and final-hold hand paths sharing one limiter;
- independence from the existing upper-body `--max-joint-step` option; and
- the symmetric-order MuJoCo geometry checks above.

Named files:

- `gear_sonic/tests/test_xr_upperbody_bridge.py`
- `gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py`
- `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py

.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate
```

Numeric target assertions use `rtol=0` and `atol=1e-6`.

## Guarded Real-Robot Acceptance

Only after all automated and MuJoCo checks pass:

1. secure the robot, keep a spotter at the emergency stop, and begin with both
   hands clear of the body and environment;
2. pull each index trigger slowly and independently, confirming analog legacy
   close and correct release;
3. squeeze each grip slowly and independently, confirming that only the thumb
   and physical index finger move and that their pads meet without crossing;
   this contact check assumes the deploy process is using its existing
   `--max-close-ratio 1.0`, because lower values may intentionally prevent
   tangency;
4. on the right hand, explicitly confirm the middle finger remains open;
5. hold trigger and squeeze together at varied depths, confirming trigger
   priority without a discontinuous unsafe jump;
6. verify left/right independence; and
7. request graceful exit and confirm the bridge moves toward the configured
   tucked-thumb stop preset without leg contact.

Passing the model verifier is not a substitute for these hardware checks.
