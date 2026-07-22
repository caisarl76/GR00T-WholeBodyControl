# XR Dex3 Controller Pinch Design

Status: Revised after user review on 2026-07-22; pending re-approval.

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
2. A pure `teleop/utils/dex3_controller_pinch.py` helper converts that value
   to a pull ratio in `[0, 1]` and owns the calibrated targets.
3. `teleop_hand_and_arm.py` invokes that helper inside the explicit
   Dex3/controller/GEAR-SONIC-export scope gate.
4. The helper interpolates each hand from the all-zero open pose to the
   calibrated thumb-index pinch pose.
5. The existing `dual_hand_joints` JSON field carries the resulting 14 values.
6. During active tracking, the bridge splits and forwards those values
   unchanged. The deploy driver then applies its configured max-close limit
   and its `0.25 rad` per-tick slew limit before publishing DDS commands.

Exit handling is a separate bridge-owned phase. With the bridge default
`--stop-hand-preset tucked-thumb`, stop-release and final-hold frames replace
the live hand target and move both non-thumb finger pairs. Operators who need
open fingers throughout exit must launch the bridge with
`--stop-hand-preset open`.

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

The numeric trigger value is authoritative when it is finite. Values outside
the documented range are clamped. A missing raw WebXR `triggerValue` follows
the existing TeleVuer behavior: it becomes raw `0.0`, the wrapper exposes it
as `10.0`, and the pinch remains safely open even if the raw trigger Boolean is
true. The Boolean is used only when the exposed numeric value is genuinely
non-finite. This preserves smooth analog motion instead of allowing the
Boolean threshold to snap normal finite input closed.

## Calibrated Pinch Targets

Full-pull targets in DDS order are:

```text
left  = [-0.379616,  0.516712,  0.121406, 0, 0, -1.273903, -0.419393]
right = [-0.379617, -0.516714, -0.121407, 1.273907, 0.419395, 0, 0]
```

These targets were derived and checked against
`gear_sonic_deploy/g1/g1_29dof_with_hand.xml`. In MuJoCo 3.8.1:

- the distal thumb and index collision-geometry signed distance is within
  `1e-6 m` of zero at full pull;
- the interpolated path approaches monotonically and remains separated before
  full pull;
- the unused middle finger remains open; and
- every target remains inside the GEAR-SONIC Dex3 command limits.

The contact contract uses signed distance, not `MjData.ncon`: numerical
tangency may leave `ncon == 0`. With the rounded targets above, measured final
signed distances are approximately `-8.7e-8 m` on the left and `-5.3e-8 m` on
the right.

The target assumes the deploy-side default `--max-close-ratio 1.0`. A lower
ratio intentionally limits closure and may prevent fingertip contact; the XR
mapping must not bypass that deploy-side safety limit. The deploy driver's
`0.25 rad` per-tick clamp also means executed intermediate poses are not in
general scalar multiples of the target. The reproducible geometry verifier
therefore checks both the scalar controller path and the clamped driver path.

## Safety and Lifecycle Behavior

- Both hands are computed independently from their own trigger values.
- Open, ramp-in, pause, resume, ramp-out, and exit-hold continue using the
  existing hand-target lifecycle.
- Releasing a trigger returns that hand continuously toward all-zero open.
- The middle finger never moves as part of controller pinch during active
  tracking. Bridge-managed exit presets may move it as described above.
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
6. out-of-range values clamp safely;
7. missing raw analog input remains safe-open even when the Boolean is true;
8. non-finite values fall back to the trigger Boolean without emitting NaN;
9. values around the old binary threshold (`6.49`, `6.50`, and `6.51`) remain
   continuous analog samples;
10. the mapping is enabled only for the exact Dex3/controller/export scope;
11. a raw WebXR controller payload passes through TeleVuer and
    `TeleVuerWrapper` into the pinch helper with the expected 14-value result;
    and
12. the resulting 14-value JSON payload splits into the exact left/right
    targets in `xr_upperbody_bridge`.

### Named Test Files and Commands

External XR tests live in:

- `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_dex3_controller_pinch.py`
- `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_televuer_controller_payload.py`

Run them, together with the existing controller regression test, using:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

The exact bridge split assertion is added to:

- `gear_sonic/tests/test_xr_upperbody_bridge.py`

Run it using:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py
```

All numeric joint assertions use `numpy.testing.assert_allclose` with
`rtol=0` and `atol=1e-6`.

### Reproducible MuJoCo Verification

Implementation adds:

- `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`

Run it using MuJoCo 3.8.1 from the repository teleop environment:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate
```

The verifier imports the targets from the XR helper so there is one source of
truth, loads `gear_sonic_deploy/g1/g1_29dof_with_hand.xml`, and selects the
collision geoms (`contype != 0`) attached to these body pairs:

- `left_hand_thumb_2_link` and `left_hand_index_1_link`;
- `right_hand_thumb_2_link` and `right_hand_index_1_link`.

It evaluates `mujoco.mj_geomDistance(..., distmax=1.0)` at the fixed scalar
schedule `alpha = numpy.linspace(0, 1, 11)`. It requires all pre-final
distances to be positive, the sampled distances to be non-increasing within
`1e-6 m`, and the absolute final distance to be at most `1e-6 m`.

It separately checks an idealized command-space slew trajectory assuming
perfect one-write tracking. Starting at the open pose, it repeatedly applies
the recurrence `q_next = q + clip(q_target - q, -0.25, 0.25)`. Both hands must
reach the exact target in six recurrence steps; the first five signed
distances must be positive and non-increasing, and the final absolute signed
distance must be at most `1e-6 m`.

Those six recurrence steps are not a production-driver tick guarantee. On
each write, the driver recalculates its delta from fresh measured DDS state;
feedback lag may require more writes. If valid seven-motor state feedback is
absent, the driver bypasses the per-tick slew clamp and publishes the
max-close-limited target directly. The verifier documents the geometry of the
idealized clamp recurrence; it does not model feedback latency and does not
use `MjData.ncon` as a pass criterion.

After these checks pass, real-robot acceptance remains a guarded hardware
check: start open, pull one trigger slowly, confirm the middle finger stays
open during active tracking, and confirm thumb-index pad contact without
crossing. If the same invariant is required during exit, run the bridge with
`--stop-hand-preset open`.
