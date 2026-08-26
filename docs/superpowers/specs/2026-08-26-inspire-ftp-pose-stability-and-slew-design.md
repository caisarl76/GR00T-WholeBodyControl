# Inspire FTP POSE Stability and Hand Slew Design

Date: 2026-08-26

## Objective

Keep PICO full-body `POSE` streaming alive when an elbow has an exact zero
axis-angle rotation, and slow simulated Inspire FTP hand targets to a maximum
physical active-joint speed of 1 rad/s in both directions.

This change remains simulation-only. It does not start or modify the PC2 FTP
DDS driver and does not add a second hardware command owner.

## Evidence and root causes

### POSE termination

`PoseStreamer.run_once()` decomposes each SMPL elbow rotation into swing and
twist. `decompose_rotation_aa()` currently computes
`sin(angle / 2) * rotation_aa / angle`. An exact zero axis-angle vector makes
that expression divide by zero, producing NaN quaternions. SciPy then raises
`ValueError: Found zero norm quaternions in quat`, and the manager shuts down.

A direct zero-vector reproduction produces non-finite twist and swing
quaternions. The same implementation is present in the original upstream
snapshot, so this is not caused by the Inspire integration.

### Hand speed

Fresh Inspire commands currently replace the applied normalized target in one
simulation step. The official `xr_teleoperate` Dex3 controller publishes at
100 Hz and relies on compliant motor control (`kp=1.5`, `kd=0.2`); its gripper
controller also limits the requested target relative to measured state and
uses a short weighted moving filter. The Inspire MuJoCo model uses position
actuators with `kp=10`, so lowering actuator gains would conflate command speed
with contact stiffness.

## Considered approaches

1. **Rate-limit the target before the MuJoCo actuator (selected).** This gives
   deterministic travel time, preserves contact stiffness, and can later be
   reused by the PC2 hardware bridge.
2. **Reduce MuJoCo position-actuator gains.** This appears slower in free space
   but makes timing load-dependent and weakens grasp/contact behavior.
3. **Apply only a moving-average filter to PICO trigger/grip input.** This
   reduces noise but does not bound the speed of a full input step.

## Design

### Stable axis-angle decomposition

`decompose_rotation_aa()` will use the analytic zero-angle limit
`sin(angle/2)/angle -> 1/2` instead of dividing by zero. Batch elements are
handled independently so a mixture of zero and nonzero rotations remains
finite. Twist normalization will also handle its degenerate zero-norm case by
selecting the identity twist, leaving the original rotation in swing.

The function continues to return scalar-first unit quaternions and preserves
existing results for ordinary nonzero rotations.

### Symmetric Inspire slew

`InspireCommandState` remains the single receiver-side owner of the applied
six-motor command. Its existing per-motor fail-open rates will become
bidirectional slew limits:

- a fresh valid command moves the current output toward the latest target by
  at most `rate * dt` per simulation step;
- a stale or absent command moves toward the open vector by the same bound;
- invalid or partial hand pairs remain atomically rejected;
- the initial state remains fully open;
- targets and applied outputs remain in normalized `[0, 1]` space.

The configured normalized rates correspond to 1 rad/s for each active model
joint:

| Motors | Active range | Normalized rate | Full travel |
| --- | ---: | ---: | ---: |
| pinky/ring/middle/index | 1.4381 rad | 0.69536/s | 1.4381 s |
| thumb bend | 0.5864 rad | 1.70532/s | 0.5864 s |
| thumb rotation | 1.1641 rad | 0.85903/s | 1.1641 s |

The public constructor will use `max_slew_speed`. Its previous
`max_open_speed` keyword remains an alias; specifying both is an error. The
simulator reads `INSPIRE_HAND_MAX_SLEW_SPEED` first and falls back to the old
`INSPIRE_HAND_MAX_OPEN_SPEED` key. All in-repository configuration and
documentation will use the new names.

### Scope boundaries

- PICO continues to publish the raw normalized target, so the existing ZMQ
  logger shows operator intent.
- MuJoCo applies the slew-limited target, so the visible hand motion is slow.
- SONIC body observations/actions remain 29-DoF and unchanged.
- Dex3 behavior and DDS endpoints remain unchanged.
- Real FTP publishing on PC2 remains disabled.

## Error handling

The rotation conversion must never emit NaN or a zero-norm quaternion for a
finite axis-angle input. The Inspire state rejects malformed/non-finite/out-of-
range pairs before changing its latest target. Stale input always converges to
open at the configured speed.

## Verification

Implementation will follow red-green TDD with focused tests for:

1. exact zero and mixed zero/nonzero axis-angle batches;
2. unit, finite twist/swing quaternions and unchanged nonzero decomposition;
3. bounded close and open motion at 1 rad/s physical-joint equivalents;
4. no target overshoot, correct `dt=0`, and direction reversal;
5. stale fail-open using the same bounded slew;
6. existing atomic ZMQ validation and watchdog behavior;
7. the complete Inspire test subset, formatter/linter, motor sweep, and contact
   cycle.

Live PICO-to-MuJoCo validation remains operator-run. Success is sustained POSE
streaming through neutral elbow frames plus visibly gradual hand travel that
matches the durations above.
