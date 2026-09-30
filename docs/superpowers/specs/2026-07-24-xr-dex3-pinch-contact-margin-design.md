# XR Dex3 Pinch Contact-Margin Design

Status: Approved for implementation on 2026-07-24.

## Problem

The controller dual-primitive feature correctly routes the index trigger to
legacy close and grip/squeeze to thumb-index pinch. Live inspection confirms
that both the XR exporter and upper-body bridge reach the intended seven-joint
pinch targets without changing the middle-finger slots.

The current pinch geometry, however, was calibrated to numerical tangency:

- left signed distance: approximately `-8.7e-8 m`;
- right signed distance: approximately `-5.3e-8 m`; and
- MuJoCo thumb-index contact count: zero on both sides.

That target has effectively no contact margin. Mesh tolerance, position error,
or hardware/model mismatch can therefore leave a visible gap between the
thumb and index pads even when every upstream command is correct.

## Scope

This change adds a small symmetric contact margin to the existing calibrated
pinch primitive. It changes only:

- the full-depth left and right pinch constants in the XR helper;
- independent target oracles in the external XR tests;
- the repository geometry verifier's final-contact contract and report; and
- the associated verifier tests and guarded hardware acceptance procedure.

The following remain unchanged:

- the legacy-close targets;
- trigger/squeeze sanitation and arbitration;
- the analog interpolation contract;
- the symmetric Dex3 API order;
- middle-finger targets, which remain zero during pinch;
- XR ramp behavior;
- bridge hand limiting and lifecycle behavior; and
- all deploy/DDS code and the previously deferred DDS-hardening task.

## Target Adjustment

Both hands retain API order:

```text
thumb0, thumb1, thumb2, middle0, middle1, index0, index1
```

Only the proximal index joint (`index0`, slot 5) moves an additional
`0.005 rad` in the closing direction. Thumb and distal-index calibration are
unchanged.

### Existing Targets

```text
left  = [-0.379616,  0.516712,  0.121406, 0, 0, -1.273903, -0.419393]
right = [-0.379617, -0.516714, -0.121407, 0, 0,  1.273907,  0.419395]
```

### Contact-Margin Targets

```text
left  = [-0.379616,  0.516712,  0.121406, 0, 0, -1.278903, -0.419393]
right = [-0.379617, -0.516714, -0.121407, 0, 0,  1.278907,  0.419395]
```

The adjustment is symmetric in magnitude and sign. It preserves the modeled
pad alignment more closely than changing the distal index joint and creates a
smaller compression than adjusting both index joints.

## MuJoCo Contact Contract

Verification uses the existing distal thumb/index collision meshes in
`gear_sonic_deploy/g1/g1_29dof_with_hand.xml` and the named-joint mapping in
`gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`.

At the proposed targets, MuJoCo 3.8.1 produces:

- left signed distance: approximately `-0.000524027 m`;
- right signed distance: approximately `-0.000524052 m`; and
- exactly one distal thumb/index contact on each side.

The verifier must require, for each hand:

1. every target value is finite and inside the pinned deploy hard limits;
2. middle slots `3:5` are exactly zero within `1e-12 rad`;
3. scalar interpolation samples at `alpha = 0.0, 0.1, ..., 0.9` remain
   separated;
4. signed distance is non-increasing within the existing `1e-6 m` numerical
   monotonicity tolerance;
5. the full-depth signed distance lies in the bounded contact band
   `[-0.00075 m, -0.00025 m]`;
6. the full-depth distal thumb/index pair contributes at least one MuJoCo
   contact; and
7. the idealized `0.25 rad` command-space recurrence still converges in six
   steps and satisfies the same final contact band and contact-count check.

The bounded band rejects both accidental loss of contact and excessive model
penetration. MuJoCo penetration is a geometry acceptance proxy, not a claim of
physical pad compression or force.

## Analog Behavior

The pinch remains analog. For squeeze depth `r` in `[0, 1]`, the XR helper
continues to publish:

```text
desired = r * contact_margin_pinch_target
```

Existing open, half-depth, and full-depth behavior remains covered. Contact is
required only at full depth; the verifier does not promise contact at partial
squeeze values.

## Test-Driven Verification

Tests must fail against the old tangency targets before implementation.

### External XR Checkout

`tests/test_dex3_controller_pinch.py` must independently pin both new vectors
and verify:

- full squeeze returns the exact new target for each hand;
- half squeeze returns exactly half of the new target;
- trigger priority and legacy close remain unchanged;
- left and right arbitration remain independent; and
- both middle-finger slots remain zero.

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

### Repository Geometry Tests

`gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py` must pin the new
targets and exercise the contact band, contact count, middle-zero invariant,
hard-limit rejection, and symmetric API order.

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py

PYTHONPATH=. .venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate
```

The verifier report must show a full-depth signed distance within the contact
band and at least one distal thumb/index contact for both sides.

## Guarded Hardware Acceptance

MuJoCo contact does not prove physical contact. Real-robot acceptance must be
performed with the robot stable, the hands clear of the legs and environment,
an operator ready to stop, and initially at low squeeze depth.

For each hand independently:

1. confirm the index trigger still produces only legacy close;
2. increase grip/squeeze slowly from open to full depth;
3. confirm the thumb and index pads meet without moving the middle finger;
4. release and confirm the hand opens normally; and
5. stop immediately for chatter, excessive force, wrong-finger motion, or
   unexpected arm/body motion.

The hardware criterion is visible pad contact at full squeeze without
crossing, binding, or sustained oscillation. This task does not add force
control, contact-force estimation, or a claim about achieved grasp force.
