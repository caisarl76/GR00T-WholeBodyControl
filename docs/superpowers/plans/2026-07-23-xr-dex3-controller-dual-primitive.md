# XR Dex3 Controller Dual-Primitive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve analog legacy close on each index trigger, add analog thumb-index pinch on each grip/squeeze control, correct the right Dex3 finger order, and apply one shared best-effort hand limiter to every hand-bearing bridge publication.

**Architecture:** TeleVuer sanitizes trigger and squeeze as separate Boolean/amplitude pairs, then a pure XR helper performs per-hand priority arbitration and emits a 14-value left-then-right command. The bridge converts every replay, live, stop-release, and final-hold target to a common `PlannerCommand`, and one `BridgePlannerPublisher` owns the hand preview, planner encode/send, and atomic limiter-state commit sequence. Deploy/DDS behavior is unchanged.

**Tech Stack:** Python 3.10, NumPy, pytest, multiprocessing shared values, JSON/ZMQ, MuJoCo 3.8.1, Unitree Dex3 hand API ordering.

---

## Execution Constraints and Current State

The approved design is
`docs/superpowers/specs/2026-07-22-xr-dex3-controller-pinch-design.md`.
Do not execute the superseded plan at
`docs/superpowers/plans/2026-07-22-xr-dex3-controller-pinch.md`.

Use the existing isolated GR00T worktree:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
git merge --no-edit vr3pt-cleanup-minimal-official
git status --short
```

Expected: the feature branch retains commits `dce6873` and `ba598bc`, receives
the approved design/plan commits, and still shows the intentional untracked
`gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py` until Task 4.

The external checkout at
`/home/jihun/work/unitree_official/xr_teleoperate` is intentionally dirty and
contains the uncommitted GEAR-SONIC export path that this feature extends. Do
not reset it, create a worktree from its upstream `HEAD`, stage broad files, or
commit the pre-existing changes. Tasks 1-3 use file-scoped diffs as
checkpoints. GR00T tasks use normal atomic commits in the isolated worktree.

The user's unrelated untracked
`docs/option_b_vla_sonic_pipeline_spec.md` must remain untouched.

## File Map

### External XR checkout

- Modify `teleop/utils/dex3_controller_pinch.py`: legacy-close and pinch
  constants, analog depth conversion, and per-hand priority arbitration.
- Modify `teleop/televuer/src/televuer/televuer.py`: strict, stale-safe raw
  trigger/squeeze normalization and analog-before-Boolean publication.
- Modify `teleop/televuer/src/televuer/tv_wrapper.py`: preserve distinct
  trigger/squeeze domains and expose actual Booleans.
- Modify `teleop/teleop_hand_and_arm.py`: pass all eight per-hand controller
  fields through the exact Dex3/controller/export gate.
- Modify `teleop/robot_control/robot_hand_unitree.py`: correct right enum labels
  without changing their numeric values.
- Modify `tests/test_dex3_controller_pinch.py`: pure target, analog, priority,
  and independence contract.
- Modify `tests/test_televuer_controller_payload.py`: raw payload sanitation,
  stale clearing, publication ordering, and wrapper domains.
- Modify `tests/test_teleop_keyboard_controls.py`: exact scope and main-loop
  wiring.
- Create `tests/test_dex3_joint_order.py`: independent API, YAML permutation,
  enum, and right-target ordering oracle.

### GR00T feature worktree

- Create `gear_sonic/utils/teleop/bridge_planner_publisher.py`: immutable
  planner command, per-hand limiter preview, and shared send/commit owner.
- Create `gear_sonic/tests/test_bridge_planner_publisher.py`: transaction and
  limiter unit tests.
- Modify `gear_sonic/utils/teleop/xr_upperbody_bridge.py`: command builders and
  routing of replay, live, both stop-release branches, and final hold through
  one publisher instance.
- Modify `gear_sonic/tests/test_xr_upperbody_bridge.py`: corrected JSON order,
  route integration, pause, invalid-frame, stop, and manager-send behavior.
- Create `gear_sonic/tests/test_xr_upperbody_bridge_routes.py`: production-loop
  routing, stop-branch, invalid-frame, encoding, and transport-failure tests.
- Modify `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`: symmetric
  named-joint order, middle-slot invariant, and deploy-limit validation.
- Add the existing untracked
  `gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py`: independent
  verifier-order regression.

## Stable Interfaces

The implementation uses these exact interfaces:

```python
controller_gripper_targets(
    *,
    left_trigger: bool,
    left_trigger_value: object,
    left_squeeze: bool,
    left_squeeze_value: object,
    right_trigger: bool,
    right_trigger_value: object,
    right_squeeze: bool,
    right_squeeze_value: object,
) -> list[float]

HandCommandLimiter.preview(
    left: object | None,
    right: object | None,
    *,
    pause: bool = False,
) -> HandCommandPreview

BridgePlannerPublisher.prepare(
    command: PlannerCommand,
    *,
    ramp_phase: str,
) -> PreparedPlannerPublication

BridgePlannerPublisher.send(
    prepared: PreparedPlannerPublication,
) -> bytes
```

`PlannerCommand` is the only planner representation accepted by the production
publisher. Compatibility helpers may still return encoded bytes for existing
inspection tests, but production replay/live/stop code must build a command,
catch only `PlannerPreparationError` around `prepare()`, and call `send()`
outside that catch so transport failures propagate.

## Task 1: Pure Dual-Primitive XR Contract

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_dex3_controller_pinch.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/utils/dex3_controller_pinch.py`

- [ ] **Step 1: Replace trigger-only expectations with failing dual-primitive tests**

Keep the current target-order test and set its right oracle to:

```python
expected_right = (
    -0.379617,
    -0.516714,
    -0.121407,
    0.0,
    0.0,
    1.273907,
    0.419395,
)
```

Then add these exact tests:

```python
from teleop.utils.dex3_controller_pinch import (
    DEX3_LEFT_LEGACY_CLOSE,
    DEX3_RIGHT_LEGACY_CLOSE,
    squeeze_pull_ratio,
)


def test_full_trigger_selects_legacy_close() -> None:
    actual = controller_gripper_targets(
        left_trigger=True,
        left_trigger_value=0.0,
        left_squeeze=False,
        left_squeeze_value=0.0,
        right_trigger=True,
        right_trigger_value=0.0,
        right_squeeze=False,
        right_squeeze_value=0.0,
    )
    assert_targets_close(
        actual,
        DEX3_LEFT_LEGACY_CLOSE + DEX3_RIGHT_LEGACY_CLOSE,
    )


def test_full_squeeze_selects_calibrated_pinch() -> None:
    actual = controller_gripper_targets(
        left_trigger=False,
        left_trigger_value=10.0,
        left_squeeze=True,
        left_squeeze_value=1.0,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=True,
        right_squeeze_value=1.0,
    )
    assert_targets_close(actual, DEX3_LEFT_PINCH + DEX3_RIGHT_PINCH)


def test_half_depth_squeeze_returns_half_calibrated_pinch() -> None:
    actual = controller_gripper_targets(
        left_trigger=False,
        left_trigger_value=10.0,
        left_squeeze=True,
        left_squeeze_value=0.5,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=True,
        right_squeeze_value=0.5,
    )
    expected = [0.5 * value for value in DEX3_LEFT_PINCH + DEX3_RIGHT_PINCH]
    assert_targets_close(actual, expected)


def test_active_squeeze_with_missing_analog_is_safe_open() -> None:
    actual = controller_gripper_targets(
        left_trigger=False,
        left_trigger_value=10.0,
        left_squeeze=True,
        left_squeeze_value=None,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=True,
        right_squeeze_value=None,
    )
    assert_targets_close(actual, DEX3_DUAL_OPEN)


@pytest.mark.parametrize("trigger_depth", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("squeeze_depth", [0.0, 0.5, 1.0])
def test_active_trigger_has_priority_at_every_depth(
    trigger_depth: float,
    squeeze_depth: float,
) -> None:
    actual = controller_gripper_targets(
        left_trigger=True,
        left_trigger_value=10.0 * (1.0 - trigger_depth),
        left_squeeze=True,
        left_squeeze_value=squeeze_depth,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=False,
        right_squeeze_value=0.0,
    )
    expected_left = [value * trigger_depth for value in DEX3_LEFT_LEGACY_CLOSE]
    assert_targets_close(actual, expected_left + list(DEX3_OPEN))


def test_active_trigger_with_missing_analog_suppresses_squeeze_safely() -> None:
    actual = controller_gripper_targets(
        left_trigger=True,
        left_trigger_value=None,
        left_squeeze=True,
        left_squeeze_value=1.0,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=False,
        right_squeeze_value=0.0,
    )
    assert_targets_close(actual, DEX3_DUAL_OPEN)


def test_left_and_right_select_primitives_independently() -> None:
    actual = controller_gripper_targets(
        left_trigger=True,
        left_trigger_value=5.0,
        left_squeeze=True,
        left_squeeze_value=1.0,
        right_trigger=False,
        right_trigger_value=10.0,
        right_squeeze=True,
        right_squeeze_value=0.25,
    )
    expected_left = [0.5 * value for value in DEX3_LEFT_LEGACY_CLOSE]
    expected_right = [0.25 * value for value in DEX3_RIGHT_PINCH]
    assert_targets_close(actual, expected_left + expected_right)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_squeeze_depth_is_clamped_without_inversion(value, expected) -> None:
    assert squeeze_pull_ratio(False, value) == pytest.approx(expected)


@pytest.mark.parametrize(("active", "expected"), [(False, 0.0), (True, 1.0)])
def test_nonfinite_squeeze_uses_boolean_fallback(active, expected) -> None:
    assert squeeze_pull_ratio(active, math.nan) == expected
```

Update every pre-existing trigger-only call with these safe-open keyword
arguments unless that test explicitly supplies a squeeze value:

```python
left_squeeze=False,
left_squeeze_value=0.0,
right_squeeze=False,
right_squeeze_value=0.0,
```

Delete the obsolete expectations that a trigger produces pinch:

- `test_full_pull_returns_both_calibrated_targets`
- `test_half_pull_returns_half_of_both_calibrated_targets`
- `test_left_and_right_pull_are_independent`

Their full/half/independence coverage is replaced by the legacy-close,
priority, and mixed left/right tests above. Retain the finite trigger sample
table at `6.49`, `6.50`, and `6.51` because the numeric mapping remains
continuous.

- [ ] **Step 2: Run the pure tests and verify the red state**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_dex3_controller_pinch.py
```

Expected: collection fails because `DEX3_LEFT_LEGACY_CLOSE`,
`DEX3_RIGHT_LEGACY_CLOSE`, and `squeeze_pull_ratio` do not exist, or calls fail
because the helper does not yet accept squeeze arguments.

- [ ] **Step 3: Implement the complete pure helper**

Replace the helper contents with this contract, retaining the exact scope gate:

```python
"""Pure controller-to-Dex3 primitives for the XR export path.

Both hands use API order:
thumb0, thumb1, thumb2, middle0, middle1, index0, index1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


DEX3_OPEN = (0.0,) * 7
DEX3_DUAL_OPEN = DEX3_OPEN + DEX3_OPEN
DEX3_LEFT_LEGACY_CLOSE = (0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75)
DEX3_RIGHT_LEGACY_CLOSE = (0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75)
DEX3_LEFT_PINCH = (
    -0.379616,
    0.516712,
    0.121406,
    0.0,
    0.0,
    -1.273903,
    -0.419393,
)
DEX3_RIGHT_PINCH = (
    -0.379617,
    -0.516714,
    -0.121407,
    0.0,
    0.0,
    1.273907,
    0.419395,
)


def _finite_ratio(
    active: bool,
    value: object,
    *,
    scale: float,
    inverted: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 1.0 if active is True else 0.0
    ratio = numeric / scale
    if inverted:
        ratio = 1.0 - ratio
    return min(1.0, max(0.0, ratio))


def trigger_pull_ratio(trigger_active: bool, trigger_value: object) -> float:
    """Convert wrapper trigger domain 10=open, 0=full to [0, 1]."""

    return _finite_ratio(
        trigger_active,
        trigger_value,
        scale=10.0,
        inverted=True,
    )


def squeeze_pull_ratio(squeeze_active: bool, squeeze_value: object) -> float:
    """Convert wrapper squeeze domain 0=open, 1=full to [0, 1]."""

    return _finite_ratio(
        squeeze_active,
        squeeze_value,
        scale=1.0,
        inverted=False,
    )


def _scaled_target(target: Sequence[float], ratio: float) -> list[float]:
    return [float(value) * ratio for value in target]


def _one_hand_target(
    *,
    trigger_active: bool,
    trigger_value: object,
    squeeze_active: bool,
    squeeze_value: object,
    legacy_close: Sequence[float],
    pinch: Sequence[float],
) -> list[float]:
    if trigger_active is True:
        return _scaled_target(
            legacy_close,
            trigger_pull_ratio(trigger_active, trigger_value),
        )
    if squeeze_active is True:
        return _scaled_target(
            pinch,
            squeeze_pull_ratio(squeeze_active, squeeze_value),
        )
    return list(DEX3_OPEN)


def controller_gripper_targets(
    *,
    left_trigger: bool,
    left_trigger_value: object,
    left_squeeze: bool,
    left_squeeze_value: object,
    right_trigger: bool,
    right_trigger_value: object,
    right_squeeze: bool,
    right_squeeze_value: object,
) -> list[float]:
    """Return 14 left-then-right targets in symmetric Dex3 API order."""

    left = _one_hand_target(
        trigger_active=left_trigger,
        trigger_value=left_trigger_value,
        squeeze_active=left_squeeze,
        squeeze_value=left_squeeze_value,
        legacy_close=DEX3_LEFT_LEGACY_CLOSE,
        pinch=DEX3_LEFT_PINCH,
    )
    right = _one_hand_target(
        trigger_active=right_trigger,
        trigger_value=right_trigger_value,
        squeeze_active=right_squeeze,
        squeeze_value=right_squeeze_value,
        legacy_close=DEX3_RIGHT_LEGACY_CLOSE,
        pinch=DEX3_RIGHT_PINCH,
    )
    return left + right


def dex3_controller_pinch_enabled(
    *,
    ee: str,
    input_mode: str,
    gear_sonic_export: bool,
) -> bool:
    return ee == "dex3" and input_mode == "controller" and gear_sonic_export is True
```

- [ ] **Step 4: Run the pure tests and verify green**

Run the Step 2 command again.

Expected: all tests in `test_dex3_controller_pinch.py` pass.

- [ ] **Step 5: Record a file-scoped XR checkpoint without staging**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
git diff --check -- teleop/utils/dex3_controller_pinch.py tests/test_dex3_controller_pinch.py
git diff --stat -- teleop/utils/dex3_controller_pinch.py tests/test_dex3_controller_pinch.py
```

Expected: no whitespace errors. Do not run `git add` in this checkout.

## Task 2: Raw WebXR Trigger/Squeeze Sanitation

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_televuer_controller_payload.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/televuer/src/televuer/televuer.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/televuer/src/televuer/tv_wrapper.py`

- [ ] **Step 1: Add failing raw-payload and ordering tests**

Replace `controller_targets_from_tele_data()` with:

```python
def controller_targets_from_tele_data(tele_data):
    return controller_gripper_targets(
        left_trigger=tele_data.left_ctrl_trigger,
        left_trigger_value=tele_data.left_ctrl_triggerValue,
        left_squeeze=tele_data.left_ctrl_squeeze,
        left_squeeze_value=tele_data.left_ctrl_squeezeValue,
        right_trigger=tele_data.right_ctrl_trigger,
        right_trigger_value=tele_data.right_ctrl_triggerValue,
        right_squeeze=tele_data.right_ctrl_squeeze,
        right_squeeze_value=tele_data.right_ctrl_squeezeValue,
    )
```

Then add:

```python
from contextlib import nullcontext


def test_raw_squeeze_reaches_wrapper_in_direct_zero_to_one_domain() -> None:
    tv = make_controller_televuer_stub()
    identity = np.eye(4).reshape(-1, order="F").tolist()
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    "left": identity,
                    "right": identity,
                    "leftState": {"squeeze": True, "squeezeValue": 0.75},
                    "rightState": {},
                }
            ),
            None,
        )
    )
    wrapper = object.__new__(TeleVuerWrapper)
    wrapper.use_hand_tracking = False
    wrapper.return_hand_rot_data = False
    wrapper.tvuer = tv

    data = wrapper.get_tele_data()

    assert data.left_ctrl_squeeze is True
    assert data.left_ctrl_squeezeValue == pytest.approx(0.75)


def test_malformed_squeeze_clears_previous_depth_and_boolean_is_strict() -> None:
    tv = make_controller_televuer_stub()
    identity = np.eye(4).reshape(-1, order="F").tolist()
    base = {"left": identity, "right": identity, "rightState": {}}
    asyncio.run(
        tv.on_controller_move(
            Event({**base, "leftState": {"squeeze": True, "squeezeValue": 0.8}}),
            None,
        )
    )
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    **base,
                    "leftState": {"squeeze": "true", "squeezeValue": "bad"},
                }
            ),
            None,
        )
    )

    assert tv.left_ctrl_squeeze_shared.value == 0
    assert tv.left_ctrl_squeezeValue_shared.value == 0.0


@pytest.mark.parametrize(
    ("active", "raw_value", "expected"),
    [
        (True, math.nan, 1.0),
        (False, math.nan, 0.0),
        (True, math.inf, 1.0),
        (False, -math.inf, 0.0),
    ],
)
def test_nonfinite_squeeze_publishes_finite_boolean_fallback(
    active,
    raw_value,
    expected,
) -> None:
    tv = make_controller_televuer_stub()
    identity = np.eye(4).reshape(-1, order="F").tolist()
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    "left": identity,
                    "right": identity,
                    "leftState": {"squeeze": active, "squeezeValue": raw_value},
                    "rightState": {},
                }
            ),
            None,
        )
    )

    assert tv.left_ctrl_squeezeValue_shared.value == expected
    assert math.isfinite(tv.left_ctrl_squeezeValue_shared.value)


class RecordingValue:
    def __init__(self, name: str, value, writes: list[str]):
        self.name = name
        self._value = value
        self.writes = writes

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.writes.append(self.name)
        self._value = value

    def get_lock(self):
        return nullcontext()


def test_trigger_and_squeeze_publish_analog_before_boolean() -> None:
    tv = make_controller_televuer_stub()
    writes = []
    for channel in ("trigger", "squeeze"):
        setattr(
            tv,
            f"left_ctrl_{channel}Value_shared",
            RecordingValue(f"{channel}Value", 0.0, writes),
        )
        setattr(
            tv,
            f"left_ctrl_{channel}_shared",
            RecordingValue(channel, False, writes),
        )
    identity = np.eye(4).reshape(-1, order="F").tolist()
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    "left": identity,
                    "right": identity,
                    "leftState": {
                        "trigger": True,
                        "triggerValue": 1.0,
                        "squeeze": True,
                        "squeezeValue": 1.0,
                    },
                    "rightState": {},
                }
            ),
            None,
        )
    )

    assert writes.index("triggerValue") < writes.index("trigger")
    assert writes.index("squeezeValue") < writes.index("squeeze")


def test_malformed_pose_cannot_prevent_squeeze_refresh() -> None:
    tv = make_controller_televuer_stub()
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    "left": [1.0],
                    "right": [1.0],
                    "leftState": {"squeeze": True, "squeezeValue": 0.6},
                    "rightState": {},
                }
            ),
            None,
        )
    )

    assert tv.left_ctrl_squeeze_shared.value == 1
    assert tv.left_ctrl_squeezeValue_shared.value == pytest.approx(0.6)
```

Change the three pre-existing raw-trigger output oracles from pinch to legacy
close:

```python
from teleop.utils.dex3_controller_pinch import (
    DEX3_DUAL_OPEN,
    DEX3_LEFT_LEGACY_CLOSE,
    DEX3_LEFT_PINCH,
    DEX3_OPEN,
)


def test_raw_analog_trigger_value_scales_left_dex3_legacy_close() -> None:
    targets = controller_targets_from_raw({"trigger": True, "triggerValue": 0.25})
    expected = [value * 0.25 for value in DEX3_LEFT_LEGACY_CLOSE] + list(DEX3_OPEN)
    np.testing.assert_allclose(targets, expected, rtol=0, atol=1e-6)


def test_raw_nan_trigger_value_uses_pressed_boolean_fallback() -> None:
    targets = controller_targets_from_raw(
        {"trigger": True, "triggerValue": float("nan")}
    )
    np.testing.assert_allclose(
        targets,
        DEX3_LEFT_LEGACY_CLOSE + DEX3_OPEN,
        rtol=0,
        atol=1e-6,
    )


def test_raw_half_depth_squeeze_scales_calibrated_pinch() -> None:
    targets = controller_targets_from_raw(
        {"squeeze": True, "squeezeValue": 0.5}
    )
    expected = [0.5 * value for value in DEX3_LEFT_PINCH] + list(DEX3_OPEN)
    np.testing.assert_allclose(targets, expected, rtol=0, atol=1e-6)


def test_raw_active_squeeze_with_missing_analog_is_safe_open() -> None:
    targets = controller_targets_from_raw({"squeeze": True})
    np.testing.assert_allclose(targets, DEX3_DUAL_OPEN, rtol=0, atol=1e-6)
```

In `test_extreme_finite_raw_trigger_values_clamp_without_wrapper_overflow`,
keep `trigger=False` and expect `DEX3_DUAL_OPEN` for both raw `1e308` and raw
`-1e308`. The finite amplitude is clamped, but an inactive Boolean never
selects the legacy-close branch. The separate active-trigger tests cover the
full-close amplitude.

- [ ] **Step 2: Run the payload tests and verify the red state**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_televuer_controller_payload.py
```

Expected: squeeze stale-clearing, strict Boolean, publication-order, and
malformed-pose tests fail against the current direct `bool()`/`float()` code.

- [ ] **Step 3: Normalize all channels into locals, then publish analog first**

Add this module-level helper near the existing matrix helpers:

```python
def _normalize_controller_channel(
    controller_state,
    *,
    active_key: str,
    value_key: str,
) -> tuple[bool, float]:
    if not isinstance(controller_state, dict):
        return False, 0.0
    raw_active = controller_state.get(active_key, False)
    active = raw_active if isinstance(raw_active, bool) else False
    raw_value = controller_state.get(value_key, 0.0)
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        return active, 0.0
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return active, 0.0
    if not math.isfinite(value):
        return active, 1.0 if active else 0.0
    return active, min(1.0, max(0.0, value))
```

At the start of `on_controller_move()`, after extracting `leftState` and
`rightState`, normalize all eight values before any shared write:

```python
normalized_channels = {}
for prefix, controller_state in (
    ("left", left_controller),
    ("right", right_controller),
):
    for channel in ("trigger", "squeeze"):
        normalized_channels[(prefix, channel)] = _normalize_controller_channel(
            controller_state,
            active_key=channel,
            value_key=f"{channel}Value",
        )

for (prefix, channel), (active, value) in normalized_channels.items():
    value_shared = getattr(self, f"{prefix}_ctrl_{channel}Value_shared")
    with value_shared.get_lock():
        value_shared.value = value
    active_shared = getattr(self, f"{prefix}_ctrl_{channel}_shared")
    with active_shared.get_lock():
        active_shared.value = active
```

Delete the nested trigger normalizer/publisher and delete the squeeze writes
from `extract_controllers()`. Keep thumbstick/buttons there. This guarantees
both controller channels refresh before matrix or thumbstick parsing.

In `TeleVuerWrapper.get_tele_data()`, make all four activation fields actual
Booleans while preserving the numeric domains:

```python
left_ctrl_trigger=bool(self.tvuer.left_ctrl_trigger),
left_ctrl_triggerValue=10.0 - self.tvuer.left_ctrl_triggerValue * 10.0,
left_ctrl_squeeze=bool(self.tvuer.left_ctrl_squeeze),
left_ctrl_squeezeValue=self.tvuer.left_ctrl_squeezeValue,
right_ctrl_trigger=bool(self.tvuer.right_ctrl_trigger),
right_ctrl_triggerValue=10.0 - self.tvuer.right_ctrl_triggerValue * 10.0,
right_ctrl_squeeze=bool(self.tvuer.right_ctrl_squeeze),
right_ctrl_squeezeValue=self.tvuer.right_ctrl_squeezeValue,
```

- [ ] **Step 4: Run the payload and pure tests**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_televuer_controller_payload.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Record the sanitation checkpoint without staging**

```bash
git diff --check -- \
  teleop/televuer/src/televuer/televuer.py \
  teleop/televuer/src/televuer/tv_wrapper.py \
  tests/test_televuer_controller_payload.py
```

Expected: no whitespace errors.

## Task 3: Main Wiring and Independent Dex3 Order Oracle

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_teleop_keyboard_controls.py`
- Create: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_dex3_joint_order.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/robot_control/robot_hand_unitree.py`

- [ ] **Step 1: Add failing main-wiring and ordering tests**

Extend `_controller_tele_data()` so callers can select both channels:

```python
def _controller_tele_data(
    *,
    left_trigger=False,
    left_trigger_value=10.0,
    left_squeeze=False,
    left_squeeze_value=0.0,
    right_trigger=False,
    right_trigger_value=10.0,
    right_squeeze=False,
    right_squeeze_value=0.0,
):
    return SimpleNamespace(
        left_ctrl_trigger=left_trigger,
        left_ctrl_triggerValue=left_trigger_value,
        left_ctrl_squeeze=left_squeeze,
        left_ctrl_squeezeValue=left_squeeze_value,
        right_ctrl_trigger=right_trigger,
        right_ctrl_triggerValue=right_trigger_value,
        right_ctrl_squeeze=right_squeeze,
        right_ctrl_squeezeValue=right_squeeze_value,
    )
```

Replace the scope test's enabled setup/assertion with this left-trigger,
right-squeeze case, and keep all three existing negative gate assertions:

```python
tele_data = _controller_tele_data(
    left_trigger=True,
    left_trigger_value=0.0,
    right_squeeze=True,
    right_squeeze_value=1.0,
)
enabled = module._controller_gripper_payload_if_enabled(
    ee="dex3",
    input_mode="controller",
    gear_sonic_export=True,
    tele_data=tele_data,
)
np.testing.assert_allclose(
    enabled,
    pinch.DEX3_LEFT_LEGACY_CLOSE + pinch.DEX3_RIGHT_PINCH,
    rtol=0,
    atol=1e-6,
)
```

Create `tests/test_dex3_joint_order.py` with an oracle that does not import the
MuJoCo verifier's `SIDE_ORDER`:

```python
import ast
from pathlib import Path

import yaml

from teleop.utils.dex3_controller_pinch import DEX3_RIGHT_PINCH


ROOT = Path(__file__).resolve().parents[1]
API_SUFFIXES = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)


def _attribute_literal(path: Path, attribute: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == attribute:
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment for {attribute}")


def _enum_literals(path: Path, class_name: str) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.targets[0].id: ast.literal_eval(item.value)
                for item in node.body
                if isinstance(item, ast.Assign)
                and isinstance(item.targets[0], ast.Name)
            }
    raise AssertionError(f"missing enum {class_name}")


def test_both_api_name_lists_are_middle_first() -> None:
    source = ROOT / "teleop" / "robot_control" / "hand_retargeting.py"
    for side in ("left", "right"):
        names = _attribute_literal(source, f"{side}_dex3_api_joint_names")
        expected = [f"{side}_hand_{suffix}_joint" for suffix in API_SUFFIXES]
        assert names == expected


def test_right_internal_retargeting_order_permutes_to_middle_first_api() -> None:
    config = yaml.safe_load(
        (ROOT / "assets" / "unitree_hand" / "unitree_dex3.yml").read_text()
    )
    internal = config["right"]["target_joint_names"]
    api = [f"right_hand_{suffix}_joint" for suffix in API_SUFFIXES]
    assert [internal.index(name) for name in api] == [0, 1, 2, 5, 6, 3, 4]


def test_right_enum_labels_match_middle_first_numeric_slots() -> None:
    values = _enum_literals(
        ROOT / "teleop" / "robot_control" / "robot_hand_unitree.py",
        "Dex3_1_Right_JointIndex",
    )
    assert values == {
        "kRightHandThumb0": 0,
        "kRightHandThumb1": 1,
        "kRightHandThumb2": 2,
        "kRightHandMiddle0": 3,
        "kRightHandMiddle1": 4,
        "kRightHandIndex0": 5,
        "kRightHandIndex1": 6,
    }


def test_right_pinch_uses_index_slots_not_middle_slots() -> None:
    assert DEX3_RIGHT_PINCH[3:5] == (0.0, 0.0)
    assert DEX3_RIGHT_PINCH[5:7] == (1.273907, 0.419395)
```

- [ ] **Step 2: Run the wiring/order tests and verify red**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_teleop_keyboard_controls.py \
  tests/test_dex3_joint_order.py
```

Expected: the main selector does not pass squeeze yet, and the right enum
labels report index in slots 3-4.

- [ ] **Step 3: Wire squeeze into the exact existing gate**

Update `_controller_gripper_targets()` in `teleop_hand_and_arm.py`:

```python
def _controller_gripper_targets(tele_data):
    return dex3_controller_gripper_targets(
        left_trigger=tele_data.left_ctrl_trigger,
        left_trigger_value=tele_data.left_ctrl_triggerValue,
        left_squeeze=tele_data.left_ctrl_squeeze,
        left_squeeze_value=tele_data.left_ctrl_squeezeValue,
        right_trigger=tele_data.right_ctrl_trigger,
        right_trigger_value=tele_data.right_ctrl_triggerValue,
        right_squeeze=tele_data.right_ctrl_squeeze,
        right_squeeze_value=tele_data.right_ctrl_squeezeValue,
    )
```

Do not move or weaken `_controller_gripper_payload_if_enabled()`; it remains
the exact Dex3/controller/GEAR-Sonic-export gate.

Correct only the right enum labels; numeric values stay sequential:

```python
class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandMiddle0 = 3
    kRightHandMiddle1 = 4
    kRightHandIndex0 = 5
    kRightHandIndex1 = 6
```

- [ ] **Step 4: Run all external XR feature tests**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_dex3_joint_order.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Audit only the XR files touched by this feature**

```bash
git diff --check -- \
  teleop/utils/dex3_controller_pinch.py \
  teleop/televuer/src/televuer/televuer.py \
  teleop/televuer/src/televuer/tv_wrapper.py \
  teleop/teleop_hand_and_arm.py \
  teleop/robot_control/robot_hand_unitree.py \
  tests/test_dex3_controller_pinch.py \
  tests/test_dex3_joint_order.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

Expected: no whitespace errors. Do not stage or commit the dirty XR checkout.

## Task 4: Correct the Repository Geometry and JSON Order Oracles

**Files:**
- Modify: `gear_sonic/tests/test_xr_upperbody_bridge.py`
- Modify: `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`
- Add: `gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py`

- [ ] **Step 1: Make repository oracles assert order, middle slots, and hard bounds**

In `test_calibrated_controller_pinch_json_splits_exact_dex3_sides()`, set:

```python
right = [-0.379617, -0.516714, -0.121407, 0.0, 0.0, 1.273907, 0.419395]
```

Expand the existing untracked verifier test into this independent contract.
The Python hard-limit arrays are deliberately pinned independently to
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/dex3_hands.hpp`; do not
import them from the XR helper:

```python
import numpy as np
import pytest

from gear_sonic.scripts.verify_xr_dex3_controller_pinch import (
    DEPLOY_HAND_HARD_MAX,
    DEPLOY_HAND_HARD_MIN,
    SIDE_ORDER,
    validate_pinch_targets,
)


CALIBRATED = {
    "left": np.array(
        [-0.379616, 0.516712, 0.121406, 0.0, 0.0, -1.273903, -0.419393]
    ),
    "right": np.array(
        [-0.379617, -0.516714, -0.121407, 0.0, 0.0, 1.273907, 0.419395]
    ),
}


def test_verifier_uses_symmetric_physical_dex3_dds_order() -> None:
    expected = (
        "thumb_0",
        "thumb_1",
        "thumb_2",
        "middle_0",
        "middle_1",
        "index_0",
        "index_1",
    )
    assert SIDE_ORDER["left"] == expected
    assert SIDE_ORDER["right"] == expected


def test_verifier_pins_current_deploy_hard_limits() -> None:
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MIN["left"],
        [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MAX["left"],
        [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MIN["right"],
        [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        DEPLOY_HAND_HARD_MAX["right"],
        [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75],
        rtol=0,
        atol=0,
    )


def test_calibrated_targets_pass_middle_and_deploy_limit_contract() -> None:
    validate_pinch_targets(CALIBRATED)


def test_verifier_rejects_nonzero_middle_slot() -> None:
    bad = {side: values.copy() for side, values in CALIBRATED.items()}
    bad["right"][3] = 0.01
    with pytest.raises(ValueError, match="middle"):
        validate_pinch_targets(bad)


def test_verifier_rejects_target_outside_deploy_hard_limits() -> None:
    bad = {side: values.copy() for side, values in CALIBRATED.items()}
    bad["left"][0] = -1.051
    with pytest.raises(ValueError, match="hard limits"):
        validate_pinch_targets(bad)
```

- [ ] **Step 2: Run both focused tests and verify red**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py::test_calibrated_controller_pinch_json_splits_exact_dex3_sides \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py
```

Expected: the JSON split or order test exposes the old right mapping, while
the invariant tests fail because the verifier has no hard-limit constants or
target-validation function.

- [ ] **Step 3: Correct the named-joint verifier order**

Use one shared tuple for both sides:

```python
DEX3_API_ORDER = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
SIDE_ORDER = {"left": DEX3_API_ORDER, "right": DEX3_API_ORDER}
```

Add independent deploy limits and the validation function next to the order:

```python
DEPLOY_HAND_HARD_MIN = {
    "left": np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75]),
    "right": np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0]),
}
DEPLOY_HAND_HARD_MAX = {
    "left": np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0]),
    "right": np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75]),
}


def validate_pinch_targets(targets: dict[str, np.ndarray]) -> None:
    for side in ("left", "right"):
        target = np.asarray(targets[side], dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid {side} pinch target")
        if not np.allclose(target[3:5], 0.0, rtol=0, atol=1e-12):
            raise ValueError(f"{side} middle joints must remain zero")
        if np.any(target < DEPLOY_HAND_HARD_MIN[side] - 1e-12) or np.any(
            target > DEPLOY_HAND_HARD_MAX[side] + 1e-12
        ):
            raise ValueError(f"{side} target exceeds deploy hard limits")
```

Call `validate_pinch_targets(targets)` at the start of `verify_pinch()`, before
loading the model or running the six-step recurrence. Do not derive order or
limits from the XR helper; the verifier remains an independent oracle.

- [ ] **Step 4: Run the focused tests and MuJoCo verifier**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py::test_calibrated_controller_pinch_json_splits_exact_dex3_sides \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py

/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate \
  --model /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_with_hand.xml
```

Expected: tests pass; MuJoCo reports six idealized recurrence steps and final
signed distances within `1e-6 m` for both hands.

- [ ] **Step 5: Commit the corrected repository oracles**

```bash
git add \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
git commit -m "fix(teleop): correct Dex3 right-hand API order"
```

Expected: one commit containing only the JSON/geometry order correction.

## Task 5: Hand Preview and Planner Send/Commit Owner

**Files:**
- Create: `gear_sonic/utils/teleop/bridge_planner_publisher.py`
- Create: `gear_sonic/tests/test_bridge_planner_publisher.py`

- [ ] **Step 1: Write failing transaction and limiter tests**

Create the test file with these fixtures and cases:

```python
import struct

import numpy as np
import pytest

import gear_sonic.utils.teleop.bridge_planner_publisher as publisher_module

from gear_sonic.utils.teleop.bridge_planner_publisher import (
    BridgePlannerPublisher,
    HandCommandLimiter,
    MAX_HAND_JOINT_STEP_RAD,
    PlannerCommand,
    PlannerPreparationError,
)


class RecordingSocket:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.sent = []

    def send(self, message: bytes):
        if self.fail:
            raise RuntimeError("send failed")
        self.sent.append(message)


def command(left=None, right=None) -> PlannerCommand:
    return PlannerCommand(
        mode=0,
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=0.0,
        height=-1.0,
        upper_body_position=(0.0,) * 17,
        upper_body_velocity=(0.0,) * 17,
        left_hand_position=left,
        right_hand_position=right,
    )


def test_preview_limits_each_included_hand_from_command_zero() -> None:
    assert MAX_HAND_JOINT_STEP_RAD == 0.25
    limiter = HandCommandLimiter()
    preview = limiter.preview([1.0] * 7, [-1.0] * 7)
    np.testing.assert_allclose(preview.left, [0.25] * 7)
    np.testing.assert_allclose(preview.right, [-0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.0] * 7)


def test_commit_updates_both_included_hands_together() -> None:
    limiter = HandCommandLimiter()
    preview = limiter.preview([1.0] * 7, [-1.0] * 7)
    limiter.commit(preview)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [-0.25] * 7)


def test_omitted_hand_is_not_committed() -> None:
    limiter = HandCommandLimiter()
    limiter.commit(limiter.preview([1.0] * 7, None))
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [0.0] * 7)


@pytest.mark.parametrize("bad", [[1.0] * 6, [1.0] * 6 + [float("nan")], "bad"])
def test_invalid_included_hand_rejects_preview_without_state_change(bad) -> None:
    limiter = HandCommandLimiter()
    with pytest.raises(ValueError):
        limiter.preview([1.0] * 7, bad)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(limiter.last_emitted("right"), [0.0] * 7)


def test_pause_reemits_last_command_without_advancing() -> None:
    limiter = HandCommandLimiter()
    limiter.commit(limiter.preview([1.0] * 7, None))
    paused = limiter.preview([1.0] * 7, None, pause=True)
    np.testing.assert_allclose(paused.left, [0.25] * 7)
    limiter.commit(paused)
    np.testing.assert_allclose(limiter.last_emitted("left"), [0.25] * 7)
    resumed = limiter.preview([1.0] * 7, None)
    np.testing.assert_allclose(resumed.left, [0.5] * 7)


def test_successful_planner_send_commits_limited_state() -> None:
    socket = RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    publisher.publish(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    assert len(socket.sent) == 1
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.25] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [-0.25] * 7)


def test_failed_planner_send_commits_neither_hand() -> None:
    publisher = BridgePlannerPublisher(RecordingSocket(fail=True))
    with pytest.raises(RuntimeError, match="send failed"):
        publisher.publish(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)


def test_encoding_failure_is_preparation_error_without_commit(monkeypatch) -> None:
    def fail_encode(*args, **kwargs):
        raise struct.error("float out of range")

    monkeypatch.setattr(publisher_module, "build_planner_message", fail_encode)
    socket = RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    with pytest.raises(PlannerPreparationError, match="invalid planner command"):
        publisher.prepare(command([1.0] * 7, [-1.0] * 7), ramp_phase="track")
    assert socket.sent == []
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)


def test_send_transport_failure_propagates_without_commit() -> None:
    publisher = BridgePlannerPublisher(RecordingSocket(fail=True))
    prepared = publisher.prepare(
        command([1.0] * 7, [-1.0] * 7),
        ramp_phase="track",
    )
    with pytest.raises(RuntimeError, match="send failed"):
        publisher.send(prepared)
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.0] * 7)
    np.testing.assert_allclose(publisher.last_emitted("right"), [0.0] * 7)
```

- [ ] **Step 2: Run the new tests and verify import failure**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_bridge_planner_publisher.py
```

Expected: collection fails because `bridge_planner_publisher.py` does not yet
exist.

- [ ] **Step 3: Implement the shared preview/send/commit abstraction**

Create `bridge_planner_publisher.py` with these definitions:

```python
from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


HAND_DOF = 7
MAX_HAND_JOINT_STEP_RAD = 0.25


class PlannerPreparationError(ValueError):
    """The planner command could not be validated or encoded."""


@dataclass(frozen=True)
class PlannerCommand:
    mode: int
    movement: Sequence[float]
    facing: Sequence[float]
    speed: float
    height: float
    upper_body_position: Sequence[float]
    upper_body_velocity: Sequence[float]
    left_hand_position: Sequence[float] | None = None
    right_hand_position: Sequence[float] | None = None

    def encode(self) -> bytes:
        try:
            return build_planner_message(
                self.mode,
                self.movement,
                self.facing,
                speed=self.speed,
                height=self.height,
                upper_body_position=self.upper_body_position,
                upper_body_velocity=self.upper_body_velocity,
                left_hand_position=self.left_hand_position,
                right_hand_position=self.right_hand_position,
            )
        except (TypeError, ValueError, OverflowError, struct.error) as exc:
            raise PlannerPreparationError(
                f"invalid planner command: {exc}"
            ) from exc


@dataclass(frozen=True)
class HandCommandPreview:
    left: np.ndarray | None
    right: np.ndarray | None


@dataclass(frozen=True)
class PreparedPlannerPublication:
    message: bytes
    hand_preview: HandCommandPreview


class HandCommandLimiter:
    def __init__(self, max_step: float = MAX_HAND_JOINT_STEP_RAD):
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be finite and positive")
        self._max_step = float(max_step)
        self._last = {
            "left": np.zeros(HAND_DOF, dtype=np.float64),
            "right": np.zeros(HAND_DOF, dtype=np.float64),
        }

    @staticmethod
    def _validated(value, side: str) -> np.ndarray | None:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlannerPreparationError(f"invalid {side} hand command") from exc
        if array.shape != (HAND_DOF,) or not np.all(np.isfinite(array)):
            raise PlannerPreparationError(f"invalid {side} hand command")
        return array

    def preview(self, left, right, *, pause: bool = False) -> HandCommandPreview:
        desired_left = self._validated(left, "left")
        desired_right = self._validated(right, "right")

        def one(side: str, desired: np.ndarray | None) -> np.ndarray | None:
            if desired is None:
                return None
            current = self._last[side]
            if pause:
                return current.copy()
            return current + np.clip(
                desired - current,
                -self._max_step,
                self._max_step,
            )

        return HandCommandPreview(
            left=one("left", desired_left),
            right=one("right", desired_right),
        )

    def commit(self, preview: HandCommandPreview) -> None:
        next_state = {side: value.copy() for side, value in self._last.items()}
        if preview.left is not None:
            next_state["left"] = preview.left.copy()
        if preview.right is not None:
            next_state["right"] = preview.right.copy()
        self._last = next_state

    def last_emitted(self, side: str) -> np.ndarray:
        if side not in self._last:
            raise ValueError(f"unsupported hand side: {side}")
        return self._last[side].copy()


class BridgePlannerPublisher:
    def __init__(self, socket, limiter: HandCommandLimiter | None = None):
        self._socket = socket
        self._limiter = limiter or HandCommandLimiter()

    def prepare(
        self,
        command: PlannerCommand,
        *,
        ramp_phase: str,
    ) -> PreparedPlannerPublication:
        preview = self._limiter.preview(
            command.left_hand_position,
            command.right_hand_position,
            pause=ramp_phase == "pause",
        )
        limited = replace(
            command,
            left_hand_position=(
                None if preview.left is None else preview.left.tolist()
            ),
            right_hand_position=(
                None if preview.right is None else preview.right.tolist()
            ),
        )
        return PreparedPlannerPublication(
            message=limited.encode(),
            hand_preview=preview,
        )

    def send(self, prepared: PreparedPlannerPublication) -> bytes:
        result = self._socket.send(prepared.message)
        if result is False:
            raise RuntimeError("planner socket rejected message")
        self._limiter.commit(prepared.hand_preview)
        return prepared.message

    def publish(self, command: PlannerCommand, *, ramp_phase: str) -> bytes:
        return self.send(self.prepare(command, ramp_phase=ramp_phase))

    def last_emitted(self, side: str) -> np.ndarray:
        return self._limiter.last_emitted(side)
```

The class performs no DDS work, uses command-space zero only as its initial
reference, and commits both included hands only after the planner socket send
returns successfully. `prepare()` owns all hand validation and binary
encoding, normalizes those failures to `PlannerPreparationError`, and performs
no send or commit. `send()` performs transport and commit only; it never
swallows transport exceptions. `publish()` remains a convenience wrapper for
isolated tests, but production loops use the explicit prepare/send boundary.

- [ ] **Step 4: Run the publisher tests and verify green**

Run the Step 2 command again.

Expected: all new publisher tests pass.

- [ ] **Step 5: Commit the isolated publisher unit**

```bash
git add \
  gear_sonic/utils/teleop/bridge_planner_publisher.py \
  gear_sonic/tests/test_bridge_planner_publisher.py
git commit -m "feat(teleop): add shared hand-limited planner publisher"
```

## Task 6: Route Every Hand-Bearing Bridge Path Through One Publisher

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
- Modify: `gear_sonic/tests/test_xr_upperbody_bridge.py`
- Create: `gear_sonic/tests/test_xr_upperbody_bridge_routes.py`

- [ ] **Step 1: Add failing builder tests and production-loop route tests**

Import `BridgePlannerPublisher`, `PlannerCommand`, and the new command builders
in the bridge test. Add these focused contracts:

```python
class _RecordingSocket:
    def __init__(self):
        self.sent = []
        self.fail_manager = False

    def send(self, message):
        if self.fail_manager and message.startswith(b"manager_state"):
            raise RuntimeError("manager send failed")
        self.sent.append(message)


def test_frame_and_stop_builders_return_planner_commands() -> None:
    frame = normalize_live_source_payload(
        {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}
    )
    live = build_frame_planner_command(frame, BridgeConfig())
    stop = build_stop_standing_command(frame, alpha=1.0)
    assert isinstance(live, PlannerCommand)
    assert isinstance(stop, PlannerCommand)


def test_live_pause_freezes_before_target_convergence_then_resume_slews() -> None:
    socket = _RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    frame = normalize_live_source_payload(
        {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}
    )
    command = build_frame_planner_command(frame, BridgeConfig())
    publisher.publish(command, ramp_phase="track")
    publisher.publish(command, ramp_phase="pause")
    publisher.publish(command, ramp_phase="resume")
    first = unpack_bridge_message(socket.sent[0], topic="planner")
    paused = unpack_bridge_message(socket.sent[1], topic="planner")
    resumed = unpack_bridge_message(socket.sent[2], topic="planner")
    np.testing.assert_allclose(first["left_hand_position"], [0.25] * 7)
    np.testing.assert_allclose(paused["left_hand_position"], [0.25] * 7)
    np.testing.assert_allclose(resumed["left_hand_position"], [0.5] * 7)


def test_manager_failure_cannot_roll_back_successful_planner_commit() -> None:
    socket = _RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    frame = normalize_live_source_payload(
        {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}
    )
    command = build_frame_planner_command(frame, BridgeConfig())
    publisher.publish(command, ramp_phase="track")
    socket.fail_manager = True
    with pytest.raises(RuntimeError, match="manager send failed"):
        socket.send(build_manager_state_message())
    np.testing.assert_allclose(publisher.last_emitted("left"), [0.25] * 7)


def test_stop_release_and_final_hold_continue_from_live_limiter_state() -> None:
    socket = _RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    frame = normalize_live_source_payload(
        {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14}
    )
    publisher.publish(
        build_frame_planner_command(frame, BridgeConfig()),
        ramp_phase="track",
    )
    release = build_stop_standing_command(
        frame,
        alpha=0.5,
        hand_preset="tucked-thumb",
    )
    final_hold = build_stop_standing_command(
        frame,
        alpha=1.0,
        hand_preset="tucked-thumb",
    )
    publisher.publish(release, ramp_phase="stop-release")
    publisher.publish(final_hold, ramp_phase="final-hold")
    decoded = [unpack_bridge_message(raw, topic="planner") for raw in socket.sent]
    for previous, current in zip(decoded, decoded[1:]):
        for side in ("left_hand_position", "right_hand_position"):
            assert np.max(np.abs(current[side] - previous[side])) <= 0.25 + 1e-7
```

Add this replay-sequence contract using the same production publisher:

```python
def test_replay_sequence_uses_one_limiter_state() -> None:
    socket = _RecordingSocket()
    publisher = BridgePlannerPublisher(socket)
    frames = load_frames(
        [
            {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [1.0] * 14},
            {"dual_arm_position": [0.0] * 14, "dual_hand_joints": [-1.0] * 14},
        ]
    )
    config = BridgeConfig()
    for frame in frames:
        publisher.publish(
            build_frame_planner_command(frame, config),
            ramp_phase=frame.ramp_phase,
        )
    first = unpack_bridge_message(socket.sent[0], topic="planner")
    second = unpack_bridge_message(socket.sent[1], topic="planner")
    np.testing.assert_allclose(first["left_hand_position"], [0.25] * 7)
    np.testing.assert_allclose(second["left_hand_position"], [0.0] * 7)
```

These builder-level tests stay as fast behavioral checks, but they do not count
as proof of production routing.

Create `test_xr_upperbody_bridge_routes.py` with a reusable `ScriptedZmq`
fixture that installs a fake `zmq` module in `sys.modules`. Its context returns
one scripted SUB socket and one recording PUB socket, its `recv()` script can
yield bytes or raise `FakeAgain`, and every `send()` records both the message
and the current receive-step number. Provide:

```python
class FakeAgain(Exception):
    pass


class FakeTransportError(RuntimeError):
    pass


class EndOfScript(RuntimeError):
    pass


def bridge_args(**overrides) -> argparse.Namespace:
    values = dict(
        bind_host="127.0.0.1",
        port=5556,
        source_host="127.0.0.1",
        source_port=5560,
        source_topic="xr_teleop",
        source_timeout_s=0.001,
        feedback_host="127.0.0.1",
        feedback_port=5557,
        feedback_topic="g1_debug",
        feedback_prime_timeout_s=0.001,
        no_feedback_prime=True,
        allow_unseeded_start_control=True,
        start_control=False,
        start_command_repeat_s=0.0,
        start_command_interval_s=0.2,
        pub_warmup_s=0.0,
        hz=50.0,
        max_abs_joint=3.14,
        max_joint_step=0.03,
        stream_mode=5,
        stop_release_s=1.0,
        stop_final_hold_s=1.0,
        stop_hand_preset="tucked-thumb",
        stop_upper_body_preset="straight",
        send_stop_on_exit=False,
        anchor_planner_heading=False,
        once=True,
        debug_live=False,
        debug_interval=0.5,
        dry_run=False,
        loop=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)
```

Use `unpack_bridge_message(..., topic="planner")` for all numerical
assertions. Add these exact production-boundary tests:

1. `test_send_loop_replay_routes_consecutive_frames_through_one_publisher`:
   call `_send_loop()` with two loaded frames whose hand targets are `+1` then
   `-1` and `bridge_args(once=False, loop=False)`; assert the actual PUB
   planner messages contain `+0.25` then `0.0`.
2. `test_zmq_live_loop_routes_valid_frame_through_publisher`: feed one valid
   topic-prefixed object payload to `_send_zmq_json_loop()` with `once=True`;
   assert the actual planner hand fields are limited to `0.25`.
3. `test_zmq_stop_release_routes_immediate_timeout_and_final_hold_branches`:
   feed a tracking frame, then an `out` frame, then scripted `FakeAgain`
   timeouts while a deterministic monotonic clock advances through release and
   hold. Spy on `BridgePlannerPublisher.prepare()` to record `ramp_phase` and
   record receive-step numbers in the PUB. Assert at least one stop planner send
   occurs on the immediate-source step, another occurs after a timeout step,
   both `stop-release` and `final-hold` phases occur, and every adjacent actual
   hand command differs by at most `0.25 + 1e-7`. Run with `once=False`; end
   the receive script with `EndOfScript`, catch that sentinel, and then inspect
   the recorded production sends.
4. `test_zmq_invalid_frames_continue_and_warning_is_rate_limited`: feed three
   invalid frames less than one fake second apart—valid JSON `[]`, a mapping
   with an invalid hand shape, and `{"mode": 1e309}` whose parsed `inf` raises
   `OverflowError` during `int()` normalization—followed by a valid frame.
   Assert exactly one invalid-frame warning, one planner send, and a zero
   return code.
5. `test_zmq_encoding_failure_does_not_commit_before_next_valid_frame`: feed a
   valid object whose planner float is `1e308` so preparation raises the
   dedicated encoding exception, then a normal frame with hand targets `+1`.
   Assert the loop continues, only one planner message is sent, and its hand
   fields are `0.25` from command-space zero rather than `0.5`.
6. `test_zmq_transport_failure_propagates_and_does_not_commit`: configure the
   recording PUB to raise `FakeTransportError` on its first planner send; assert
   `_send_zmq_json_loop()` raises that exact error. Spy on the constructed
   publisher and assert both `last_emitted()` vectors remain zero.

For test 3, make `build_due_stop_release_commands()` return a phase string as
described in Step 3. For tests 4 and 5, set `once=True`; invalid frames do not
count as the one successful publication. The fake sockets implement all no-op
`bind`, `connect`, `setsockopt`, `setsockopt_string`, and `close` methods used
by the production functions so the test calls the real loops rather than a
copied loop body.

- [ ] **Step 2: Run focused bridge tests and verify red**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py -k \
  "planner_command or replay_sequence or pause_freezes or manager_failure or stop_release_and_final_hold or send_loop or zmq"
```

Expected: collection fails because the command builders do not exist or tests
show that production replay, live, immediate stop, timeout stop, and final hold
still call `pub.send()` directly. Non-object JSON and encoding overflow also
escape the current loop.

- [ ] **Step 3: Introduce command builders while retaining byte wrappers**

Import the publisher types, then split each existing builder:

```python
from gear_sonic.utils.teleop.bridge_planner_publisher import (
    BridgePlannerPublisher,
    PlannerCommand,
)


def build_frame_planner_command(
    frame: UpperBodyFrame,
    config: BridgeConfig,
    filt: UpperBodyFilter | None = None,
    stop_hand_preset: str | None = None,
) -> PlannerCommand:
    filtered_position = (
        filt.apply_ramped(
            frame.upper_body_position,
            ramp_phase=frame.ramp_phase,
            ramp_alpha=frame.ramp_alpha,
        )
        if filt is not None
        else np.clip(
            frame.upper_body_position,
            -config.max_abs_joint,
            config.max_abs_joint,
        )
    )
    left = None if frame.left_hand_joints is None else frame.left_hand_joints.tolist()
    right = None if frame.right_hand_joints is None else frame.right_hand_joints.tolist()
    if (
        stop_hand_preset is not None
        and stop_hand_preset != "none"
        and (frame.stop or frame.ramp_phase in {"out", "hold"})
    ):
        left, right = _stop_hand_targets(
            frame,
            float(np.clip(frame.ramp_alpha, 0.0, 1.0)),
            stop_hand_preset,
        )
    return PlannerCommand(
        mode=frame.mode,
        movement=frame.movement,
        facing=frame.facing,
        speed=frame.speed,
        height=frame.height,
        upper_body_position=filtered_position.tolist(),
        upper_body_velocity=frame.upper_body_velocity.tolist(),
        left_hand_position=left,
        right_hand_position=right,
    )


def build_frame_planner_message(*args, **kwargs) -> bytes:
    return build_frame_planner_command(*args, **kwargs).encode()
```

Add the complete stop command and compatibility wrapper:

```python
def build_stop_standing_command(
    frame: UpperBodyFrame,
    alpha: float = 1.0,
    hand_preset: str = "tucked-thumb",
    upper_body_preset: str = "straight",
) -> PlannerCommand:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    start = np.asarray(frame.upper_body_position, dtype=np.float32)
    target = _stop_upper_body_target(upper_body_preset)
    upper_body_position = (1.0 - smooth_alpha) * start + smooth_alpha * target
    left, right = _stop_hand_targets(frame, smooth_alpha, hand_preset)
    return PlannerCommand(
        mode=0,
        movement=(0.0, 0.0, 0.0),
        facing=frame.facing,
        speed=0.0,
        height=-1.0,
        upper_body_position=upper_body_position.tolist(),
        upper_body_velocity=(0.0,) * UPPER_BODY_DOF,
        left_hand_position=left,
        right_hand_position=right,
    )


def build_stop_standing_message(*args, **kwargs) -> bytes:
    return build_stop_standing_command(*args, **kwargs).encode()
```

Add the command-returning scheduler helper and keep the byte wrapper for
inspection tests:

```python
def build_due_stop_release_commands(
    schedule: StopReleaseSchedule,
    *,
    now: float,
    hz: float,
    stream_mode: int,
    hand_preset: str,
    upper_body_preset: str,
) -> tuple[PlannerCommand, bytes, str] | None:
    alpha = schedule.next_alpha(now=now, hz=hz)
    if alpha is None:
        return None
    phase = "final-hold" if schedule.final_frame_sent else "stop-release"
    return (
        build_stop_standing_command(
            schedule.frame,
            alpha=alpha,
            hand_preset=hand_preset,
            upper_body_preset=upper_body_preset,
        ),
        build_manager_state_message(stream_mode=stream_mode),
        phase,
    )


def build_due_stop_release_messages(*args, **kwargs) -> tuple[bytes, bytes] | None:
    commands = build_due_stop_release_commands(*args, **kwargs)
    if commands is None:
        return None
    planner_command, manager_message, _phase = commands
    return planner_command.encode(), manager_message
```

Production loops call `build_due_stop_release_commands()`; only compatibility
tests call the byte-returning wrapper. `final_frame_sent` becomes true on the
first exact-alpha-one publication, so the phase identifies the final-hold path
without changing the existing schedule timing.

- [ ] **Step 4: Route replay and all live stop branches through shared prepare/send**

In `_send_loop()`, create exactly one publisher after the replay socket is
created:

```python
planner_publisher = None if socket is None else BridgePlannerPublisher(socket)
```

For every replay frame:

```python
command = build_frame_planner_command(frame, config, filt)
if planner_publisher is not None:
    prepared = planner_publisher.prepare(command, ramp_phase=frame.ramp_phase)
    planner_publisher.send(prepared)
    socket.send(build_manager_state_message(stream_mode=args.stream_mode))
else:
    command.encode()
```

Dry-run encoding does not commit limiter state because no planner message was
locally enqueued.

In `_send_zmq_json_loop()`, create one
`planner_publisher = BridgePlannerPublisher(pub)` before entering the loop.
Replace the normal live send using the validation boundary in Step 5.

In both stop-release branches—timeout polling and immediate source-frame
handling—call `build_due_stop_release_commands()`, then:

```python
planner_command, manager_state_message, stop_phase = commands
try:
    prepared = planner_publisher.prepare(
        planner_command,
        ramp_phase=stop_phase,
    )
except PlannerPreparationError as exc:
    warn_invalid_frame("invalid stop planner command", exc)
    continue
planner_publisher.send(prepared)
pub.send(manager_state_message)
```

`prepare()` is inside the dedicated preparation-error catch; `send()` and the
manager transport are outside it. Because the schedule emits repeated
alpha-one commands during final hold, the same publisher continues slewing
toward tucked-thumb without extending the existing hold timer. No direct
hand-bearing planner byte may bypass this publisher in replay, live, either
stop-release branch, or final hold.

- [ ] **Step 5: Separate input validation, preparation, and transport**

Import `Mapping` and `PlannerPreparationError`. Initialize a dedicated warning
clock and one local rate-limited reporter before the loop:

```python
last_invalid_frame_warn_time = -math.inf


def warn_invalid_frame(context: str, exc: Exception) -> None:
    nonlocal last_invalid_frame_warn_time
    now = time.monotonic()
    if now - last_invalid_frame_warn_time >= 1.0:
        print(f"[xr_upperbody_bridge] {context} skipped: {exc}", flush=True)
        last_invalid_frame_warn_time = now
```

Decode and validate the JSON object explicitly. This turns valid JSON such as
`[]` into a handled invalid frame instead of an `AttributeError`:

```python
try:
    payload_bytes = (
        raw[len(args.source_topic) :]
        if raw.startswith(args.source_topic.encode())
        else raw
    )
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("live payload must be a JSON object")
    frame = normalize_live_source_payload(payload)
except (
    UnicodeDecodeError,
    json.JSONDecodeError,
    TypeError,
    ValueError,
    OverflowError,
) as exc:
    warn_invalid_frame("invalid live frame", exc)
    continue
```

Build the normal live command, then catch only the dedicated validation and
encoding error around `prepare()`. Keep `send()` and the manager send outside
the catch:

```python
command = build_frame_planner_command(
    frame,
    config,
    filt,
    stop_hand_preset=args.stop_hand_preset,
)
try:
    prepared = planner_publisher.prepare(
        command,
        ramp_phase=frame.ramp_phase,
    )
except PlannerPreparationError as exc:
    warn_invalid_frame("invalid planner frame", exc)
    continue

planner_publisher.send(prepared)
pub.send(
    build_manager_state_message(
        stream_mode=args.stream_mode,
        toggle_data_collection=frame.toggle_data_collection,
        toggle_data_abort=frame.toggle_data_abort,
    )
)
```

Use the same `prepare()` catch and outside-catch `send()` ordering in both stop
branches from Step 4. `PlannerCommand.encode()` normalizes `struct.error`,
`OverflowError`, `TypeError`, and `ValueError` to
`PlannerPreparationError`. Do not catch ZMQ send exceptions: a transport
failure must leave limiter state uncommitted and propagate through cleanup.

- [ ] **Step 6: Run bridge unit and integration tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_bridge_planner_publisher.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py
```

Expected: all tests pass, including calls to the real replay/live loops, both
production stop branches, final hold, rate-limited invalid-frame continuation,
encoding failure without commit, transport-failure propagation, pause freeze,
atomic hand commit, and manager-send non-rollback.

- [ ] **Step 7: Commit the bridge integration**

```bash
git add \
  gear_sonic/utils/teleop/xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py
git commit -m "feat(teleop): limit all bridge hand publications"
```

## Task 7: Full Cross-Repository Verification

**Files:**
- Verify all feature files from Tasks 1-6.

- [ ] **Step 1: Run the complete external XR test set**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_dex3_joint_order.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

Expected: all selected XR tests pass.

- [ ] **Step 2: Run the complete repository bridge/geometry test set**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_bridge_planner_publisher.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py
```

Expected: all selected repository tests pass.

- [ ] **Step 3: Run MuJoCo geometry verification**

```bash
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate \
  --model /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_with_hand.xml
```

Expected: MuJoCo 3.8.1, six idealized recurrence steps for both sides, middle
slots fixed at zero during pinch, and final signed distance magnitude at most
`1e-6 m`.

- [ ] **Step 4: Run style and static checks on changed Python files**

```bash
ruff check --no-cache \
  gear_sonic/utils/teleop/bridge_planner_publisher.py \
  gear_sonic/utils/teleop/xr_upperbody_bridge.py \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  gear_sonic/tests/test_bridge_planner_publisher.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py

ruff format --check --no-cache \
  gear_sonic/utils/teleop/bridge_planner_publisher.py \
  gear_sonic/utils/teleop/xr_upperbody_bridge.py \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  gear_sonic/tests/test_bridge_planner_publisher.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py \
  gear_sonic/tests/test_xr_upperbody_bridge_routes.py \
  gear_sonic/tests/test_verify_xr_dex3_controller_pinch.py

git diff --check
```

Expected: all checks pass.

- [ ] **Step 5: Audit final scopes and preserve unrelated work**

In the GR00T worktree:

```bash
git status --short
git log --oneline --decorate -8
git diff vr3pt-cleanup-minimal-official...HEAD --stat
```

Expected: only the approved spec/plan plus geometry, publisher, bridge, and
focused test changes appear.

In the external XR checkout:

```bash
git status --short
git diff --check -- \
  teleop/utils/dex3_controller_pinch.py \
  teleop/televuer/src/televuer/televuer.py \
  teleop/televuer/src/televuer/tv_wrapper.py \
  teleop/teleop_hand_and_arm.py \
  teleop/robot_control/robot_hand_unitree.py \
  tests/test_dex3_controller_pinch.py \
  tests/test_dex3_joint_order.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

Expected: no whitespace errors; pre-existing unrelated dirty files remain
untouched and uncommitted.

- [ ] **Step 6: Request a final two-lane code and architecture review**

The code-review lane checks exact behavior, test coverage, and dirty-checkout
preservation. The architecture lane checks that every hand-bearing replay,
live, stop-release, and final-hold planner path uses the same
`BridgePlannerPublisher`, and confirms that no deploy/DDS hardening entered
the diff.

Expected: both lanes return APPROVE/CLEAR before hardware testing.

## Task 8: Guarded Real-Robot Acceptance Handoff

**Files:**
- No code changes. Produce operator commands and collect observations.

- [ ] **Step 1: Start the bridge with tucked-thumb stop behavior**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-dex3-controller-pinch
source /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/activate

PYTHONPATH=. python gear_sonic/scripts/xr_upperbody_bridge.py \
  --source zmq-json \
  --source-host 127.0.0.1 \
  --source-port 5560 \
  --source-topic xr_teleop \
  --bind-host 0.0.0.0 \
  --port 5556 \
  --hz 50 \
  --start-control \
  --start-command-repeat-s 5.0 \
  --start-command-interval-s 0.2 \
  --feedback-host 127.0.0.1 \
  --feedback-port 5557 \
  --feedback-topic g1_debug \
  --feedback-prime-timeout-s 2.0 \
  --anchor-planner-heading \
  --max-joint-step 0.03 \
  --stop-release-s 2.0 \
  --stop-final-hold-s 2.0 \
  --stop-upper-body-preset straight \
  --stop-hand-preset tucked-thumb \
  --debug-live
```

This pre-integration hardware acceptance intentionally launches the reviewed
feature-worktree implementation, not the main checkout. The bridge uses the
main repository's `.venv_teleop`; `--max-joint-step 0.03` remains the
upper-body limit and is independent of the fixed `0.25 rad` hand limiter.

- [ ] **Step 2: Start XR export without bridge-only or simulation flags**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
conda activate tv

python teleop/teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode controller \
  --gear-sonic-export \
  --gear-sonic-export-port 5560 \
  --gear-sonic-export-topic xr_teleop \
  --gear-sonic-ramp-in-s 2.0 \
  --gear-sonic-ramp-out-s 2.0 \
  --gear-sonic-exit-hold-s 2.0 \
  --gear-sonic-debug \
  --image-source gr00t-zmq \
  --gr00t-image-host 192.168.0.223 \
  --gr00t-image-port 5556 \
  --gr00t-image-mount ego_view
```

Do not pass `--stop-hand-preset` to the XR command; it belongs to the bridge.
Do not pass `--sim` for the real-robot export workflow.

- [ ] **Step 3: Perform guarded controller checks**

With the robot secured, hands clear of its legs/environment, and a spotter at
the emergency stop:

1. Press `r` and wait for ramp-in completion.
2. Pull each index trigger slowly and independently; verify analog legacy
   close and release.
3. Pull each grip/squeeze slowly and independently; verify only thumb and
   physical index move, and the right middle finger stays open.
4. Verify thumb/index pads meet without crossing while deploy uses
   `--max-close-ratio 1.0`.
5. Hold trigger and squeeze together at several depths; verify trigger
   priority and bounded bridge-frame transitions.
6. Verify left/right controls remain independent.
7. Press `q`; verify the bridge moves toward tucked-thumb and avoids leg
   contact during the existing stop-release/final-hold schedule.

Expected: all seven observations pass. Any unexpected physical ordering,
crossing, discontinuity, or leg approach aborts acceptance immediately and is
recorded with the corresponding controller and bridge debug lines.

- [ ] **Step 4: Report the guarantee boundary explicitly**

The final handoff states:

```text
The bridge limits each locally enqueued hand-bearing planner frame to 0.25 rad
per joint. This is best-effort command-space defense in depth, not a hard DDS
or physical-motion guarantee. Deploy/DDS feedback, write-failure, source-loss,
and safety-latch hardening remain a separate deferred task.
```

Hardware acceptance is not marked complete until the operator reports all
Step 3 observations.
