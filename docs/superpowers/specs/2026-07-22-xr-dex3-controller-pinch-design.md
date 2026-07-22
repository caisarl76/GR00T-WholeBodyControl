# XR Dex3 Controller Pinch Design

Status: Approved by the user on 2026-07-22.

## Problem

Controller-based XR teleoperation currently maps each controller trigger to a
binary, full-hand Dex3 close pose in
`/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`.
That pose curls both non-thumb fingers to their limits while leaving the thumb
opposition joints neutral. On the real hand, the fingers pass one another
without establishing a useful grasp.

The controller is a one-degree-of-freedom input, so its hand primitive should
be a one-degree-of-freedom thumb-index pinch: open at zero pull, fingertip
contact at full pull, and a continuous approach between them.

## Scope

This change applies only when all of the following are true:

- end effector is Dex3 (`--ee dex3`);
- input mode is controller (`--input-mode controller`); and
- the XR process is exporting hand targets to GEAR-SONIC.

Hand-tracking retargeting, arm inverse kinematics, planner intent, the
`xr_upperbody_bridge`, and the GEAR-SONIC Dex3 driver are unchanged.

## Ownership and Data Flow

The mapping belongs in `xr_teleoperate`, not in the bridge or deploy process:

1. TeleVuer provides a per-controller analog trigger value, where `10.0` is
   open and `0.0` is fully pressed.
2. `teleop_hand_and_arm.py` converts that value to a pull ratio in `[0, 1]`.
3. It interpolates each hand from the all-zero open pose to the calibrated
   thumb-index pinch pose.
4. The existing `dual_hand_joints` JSON field carries the resulting 14 values.
5. The bridge and deploy process forward and execute the values unchanged.

A bridge-side conversion was rejected because the bridge receives only the
already-generated joint target and therefore cannot preserve the analog
trigger signal. A deploy-side conversion was rejected because it would alter
Dex3 commands from every input source, including hand tracking and policy
control.

## Dex3 Joint Order

Unitree DDS motor order is asymmetric after the three thumb joints:

| Side | Indices 0-6 |
| --- | --- |
| Left | `thumb0, thumb1, thumb2, middle0, middle1, index0, index1` |
| Right | `thumb0, thumb1, thumb2, index0, index1, middle0, middle1` |

The pinch therefore leaves left indices `3:5` and right indices `5:7` at
zero. Only the thumb and actual index finger move.

## Trigger Mapping

For each controller, compute:

```text
pull_ratio = clamp(1 - trigger_value / 10, 0, 1)
hand_target = open + pull_ratio * (pinch - open)
```

The numeric trigger value is authoritative when it is finite. The trigger
boolean is used only as a safe fallback for a missing or non-finite numeric
value. Values outside the documented range are clamped. This preserves smooth
analog motion instead of allowing the boolean threshold to snap the hand
closed.

## Calibrated Pinch Targets

Full-pull targets in DDS order are:

```text
left  = [-0.379616,  0.516712,  0.121406, 0, 0, -1.273903, -0.419393]
right = [-0.379617, -0.516714, -0.121407, 1.273907, 0.419395, 0, 0]
```

These targets were derived and checked against
`gear_sonic_deploy/g1/g1_29dof_with_hand.xml`. In the model:

- the thumb and index collision surfaces first meet at full pull;
- the interpolated path approaches monotonically and remains separated before
  full pull;
- the unused middle finger remains open; and
- every target remains inside the GEAR-SONIC Dex3 command limits.

The target assumes the deploy-side default `--max-close-ratio 1.0`. A lower
ratio intentionally limits closure and may prevent fingertip contact; the XR
mapping must not bypass that deploy-side safety limit.

## Safety and Lifecycle Behavior

- Both hands are computed independently from their own trigger values.
- Open, ramp-in, pause, resume, ramp-out, and exit-hold continue using the
  existing hand-target lifecycle.
- Releasing a trigger returns that hand continuously toward all-zero open.
- The middle finger never moves as part of controller pinch.
- Non-finite input cannot emit non-finite joint commands.
- Existing uncommitted work in the external `xr_teleoperate` checkout must be
  preserved; implementation edits are limited to the controller primitive and
  focused tests.

## Verification

Test-driven implementation will first add failing tests for:

1. zero pull produces two open hands;
2. full left pull produces only the calibrated left thumb-index target;
3. full right pull uses the right-side index slots and leaves its middle slots
   zero;
4. half pull is exactly the midpoint of open and pinch;
5. left and right triggers remain independent;
6. out-of-range values clamp safely; and
7. non-finite values fall back to the trigger boolean without emitting NaN.

After the unit tests pass, verification will rerun the MuJoCo contact check and
the existing XR controller/keyboard test suite. Real-robot acceptance remains
a guarded hardware check: start open, pull one trigger slowly, confirm the
middle finger stays open, and confirm thumb-index pad contact without crossing.
