# SONIC POSE / PLANNER handoff

## Problem and scope

Switching between PICO full-body POSE and PLANNER was reported to destabilize
both MuJoCo and the physical G1. Hardware-free reproductions identified reference
handoff defects; they do not establish a measured fall rate or prove physical
balance has been restored.

The manager selected POSE before its five-frame producer buffer was ready
(about 100 ms at 50 Hz). Deployment discarded prebuffered pose data and paused
reference playback. Returning to PLANNER also paused the reference and polled
initialization in 100 ms intervals. Pausing repeats one reference position over
the policy's future horizon and zeros reference velocities; inference and motor
command publishing continue. It is not a motor power interruption.

## Manager behavior

- A+X / `t` requests a transition. The current source keeps publishing while
  the destination prepares its first packet.
- The destination packet is held locally until its new manager state and
  provenance can be published. It is sent before the mode command.
- A second A+X / `t` cancels pending preparation. A+B+X+Y still stops immediately.
- Preparation expires after one second without a packet; the previous mode
  continues. Release and press the chord again to retry.
- B+Y's transition from POSE to frozen-upper-body PLANNER also prepares its
  destination. A second B+Y cancels that pending transition.
- Recording acknowledgements and measured-hand admission checks remain in force.

Separate ZMQ subscribers can receive the command before the packet even when
Python sends the packet first. The deployment-side change is therefore required;
updating the manager alone is insufficient.

## Deployment behavior

Managed handoffs do not run the legacy pause/reset sequence. Deployment retains
a fresh first pose packet in either subscriber delivery order, keeps the outgoing
reference active during preparation, and checks planner readiness on subsequent
input ticks instead of blocking for 100 ms. Invalid or stale destination packets
do not commit a pose transition. The existing five-second deployment preparation
failure deadline stops control rather than waiting indefinitely.

The reference switch rebases heading so that an outgoing trajectory that has
already turned does not snap back to its earlier heading. Explicit operator
heading resets still take precedence. Planner inference carries a generation
identifier so output from an abandoned transition cannot take back ownership.
Regular startup retains the original neutral planner initialization.

This is not a blend between different learned encoder modes. A delayed planner
can outlast the outgoing five-frame pose window; deployment then holds its last
valid reference frame while waiting. The patch removes the forced pause and
polling delay, but does not manufacture future body motion or reconstruct the
planner's missing motion history.

## Validation before PC2

Use SONIC **v1.1**, the same decoder/encoder/planner files and simulator model as
the previous working run. Build deployment from the patched source; restarting
an old executable does not apply C++ changes. PC2 is not updated by a local build.

Test repeated PLANNER → POSE → PLANNER transitions first at standstill, then with
small upper-body motions. Include B+Y freeze, cancellation while source data is
unavailable, and stop during preparation. Record both manager and deployment
logs plus simulated root height/orientation and reference timing. Check for
falls, abrupt heading changes, reference pauses, dropped first packets, and
unexpected stops. Compare with the unchanged build using the same input trace.

A passing input-handler regression is necessary but cannot qualify the learned
policy's response to different encoder modes or arbitrary operator poses.
MuJoCo balance qualification and physical confirmation remain required before
claiming the reported instability resolved on hardware. No PC2 installation is
part of this local patch.

## Initial local verification (2026-09-14, before live follow-up)

- 202 focused Python tests passed, including real `PoseStreamer` warmup,
  both transition directions, freeze, cancellation, timeout, recording and hand
  provenance checks.
- Release `g1_deploy_onnx_ref` built successfully using the existing local
  TensorRT/ONNX Runtime/Unitree SDK libraries.
- Both registered CTest tests passed: `zmq_mode_handoff_test` and
  `dex3_hand_field_decoder_test`. The latter's LeakSanitizer required execution
  outside the sandbox's tracing restrictions; the rerun retained sanitizer checks.
- Independent review findings on reference/control ownership, encoder selection,
  generation capture, world-heading continuity and legacy startup were resolved.
- Four isolated MuJoCo runs compared the baseline and patch with SONIC v1.1.
  Each build completed ten switches without a fall. In the stress comparison,
  paused reference ticks fell from 21 to zero. The baseline also stayed upright,
  so those initial runs did not reproduce or establish a resolution of the live
  PICO balance loss. See the [measurements and reproduction instructions](../artifacts/sonic_mode_handoff_20260914/README.md)
  and the subsequent live follow-up below.
- At this initial stage, live and physical confirmation were pending. No PC2
  installation was performed.

See the [C++ regression instructions](../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/README_mode_handoff.md)
for the hardware-free handoff test command.

## Live follow-up: preserve policy balance feedback

The live A+X test reproduced a separate one-second balance disturbance. Auxiliary
VR fields in SMPL packets toggled an existing motor-output ramp, even though the
SMPL encoder does not use those fields. That ramp blended all 29 policy targets
with fixed measured joint positions for one second, attenuating balance
corrections while action history retained the unmodified decoder output.

The follow-up removes this global motor-output ramp. Standing initialization,
VR entry checks and upstream reference smoothing remain intact. A compiled
regression reproduces the old target/history mismatch; the repaired binary also
improves root-height and tilt transients in the captured-pose comparison with VR
fields retained. See the [live diagnosis and controlled replay](../artifacts/sonic_output_ramp_20260914/README.md).
A subsequent live MuJoCo retest completed ten switches with finger input disabled;
the user confirmed the balance loss was gone. The subsequent optical Dex3 retest
recorded four PLANNER -> POSE and four POSE -> PLANNER switches, and the user
confirmed A+X switching works. See the [optical integration confirmation](../artifacts/sonic_hand_mode_ownership_20260914/README.md)
and its [manager log and confirmation](../artifacts/sonic_hand_mode_ownership_20260914/live_confirmation.json).
Physical G1 qualification, formal timing/recording qualification, and the
separate investigation of later sustained-POSE falls remain pending. PC2 has
not been updated.

## PR #11 review corrections (2026-09-14)

A reverse POSE command now invalidates an uncommitted planner generation in
`update()`, under the same motion mutex used by reference commit. It preserves
the outgoing reference, including a planner that already committed. Regression
tests reproduce cancellation during and after inference, both at startup and
after POSE, before `handle_input()` or a replacement pose packet arrives. They
also check that restarting cannot revive the abandoned result.

Optical Dex3 recording is now admitted only in POSE: native-fist VR_3PT packets
have no optical hand action fields. The manager explains the refusal and the
recorder independently rejects START/TOGGLE-to-START from older managers. Save
and abort remain available. Controller Dex3 and Inspire VR_3PT recording retain
their episode fields and pass the recording join checks. See the
[recording restriction](pico_optical_hand_tracking.md).

Verification: 297 focused Python tests, 33 exporter tests, and both C++
deployment regressions. The targeted cancellation and recording reproductions
failed before the corrections and passed afterward. These review corrections
were checked locally; the earlier live confirmations above do not qualify this
new revision on physical G1 hardware.
