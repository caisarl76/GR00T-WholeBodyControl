# XR Dex3 Controller Pinch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary, crossing Dex3 controller close primitive with an analog thumb-index pinch whose calibrated pads reach tangency at full trigger pull.

**Architecture:** A pure helper in the external `xr_teleoperate` checkout owns trigger normalization, asymmetric Dex3 targets, interpolation, and the exact scope gate. `teleop_hand_and_arm.py` calls the helper only for Dex3/controller/GEAR-SONIC export; the existing JSON and bridge path remain unchanged during active tracking. A repository-side MuJoCo verifier imports the helper targets and checks both the scalar controller path and an idealized `0.25 rad` command-space slew recurrence.

**Tech Stack:** Python 3.10, NumPy, pytest, TeleVuer/WebXR controller payloads, JSON/ZMQ, MuJoCo 3.8.1, Unitree Dex3 DDS joint order.

---

## Execution Constraints and File Map

The external checkout at `/home/jihun/work/unitree_official/xr_teleoperate` is intentionally dirty: the GEAR-SONIC export and GR00T image-source code being extended are uncommitted. Do not create a clean worktree from its `HEAD`, reset it, or commit broad tracked-file diffs. Preserve every pre-existing change and keep pinch edits limited to the files named below.

The workspace repository is clean except for the user's unrelated untracked `docs/option_b_vla_sonic_pipeline_spec.md`; do not add, modify, or commit that file.

### External XR checkout

- Create `teleop/utils/dex3_controller_pinch.py`: pure constants and controller-pinch math.
- Create `tests/test_dex3_controller_pinch.py`: unit coverage for targets, analog mapping, fallback, independence, and gating.
- Modify `tests/test_televuer_controller_payload.py`: raw WebXR payload through TeleVuer and `TeleVuerWrapper` into the helper.
- Modify `tests/test_teleop_keyboard_controls.py`: main-module selection wiring and exact three-way scope coverage.
- Modify `teleop/teleop_hand_and_arm.py`: replace binary close constants/function with the helper and route the live export call through the scope-aware selector.

### GR00T-WholeBodyControl workspace

- Modify `gear_sonic/tests/test_xr_upperbody_bridge.py`: calibrated 14-value JSON split characterization.
- Create `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`: reproducible MuJoCo signed-distance verifier.

### Stable interfaces

- `DEX3_OPEN`, `DEX3_DUAL_OPEN`, `DEX3_LEFT_PINCH`, and
  `DEX3_RIGHT_PINCH` are immutable tuples of floats.
- `trigger_pull_ratio(trigger_pressed: bool, trigger_value: object) -> float`
  returns a finite value in `[0, 1]`.
- `controller_gripper_targets(*, left_trigger: bool,
  left_trigger_value: object, right_trigger: bool,
  right_trigger_value: object) -> list[float]` returns exactly 14 values in
  left-then-right DDS order.
- `dex3_controller_pinch_enabled(*, ee: str, input_mode: str,
  gear_sonic_export: bool) -> bool` is true only for the approved three-way
  scope.

## Task 1: Pure Pinch Contract and Raw Controller Pipeline

**Files:**
- Create: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/utils/dex3_controller_pinch.py`
- Create: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_dex3_controller_pinch.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_televuer_controller_payload.py`

- [ ] **Step 1: Capture external-checkout ownership before edits**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
git status --short
git diff -- teleop/teleop_hand_and_arm.py > /tmp/xr_teleop_before_pinch.diff
```

Expected: the existing modified/untracked files remain visible, and `/tmp/xr_teleop_before_pinch.diff` records the pre-pinch tracked-file state without changing it.

- [ ] **Step 2: Write the failing pure-helper tests**

Create `tests/test_dex3_controller_pinch.py` with:

```python
import math

import numpy as np
import pytest

from teleop.utils.dex3_controller_pinch import (
    DEX3_DUAL_OPEN,
    DEX3_LEFT_PINCH,
    DEX3_OPEN,
    DEX3_RIGHT_PINCH,
    controller_gripper_targets,
    dex3_controller_pinch_enabled,
    trigger_pull_ratio,
)


ATOL = 1e-6


def _targets(
    *,
    left_value=10.0,
    right_value=10.0,
    left_pressed=False,
    right_pressed=False,
):
    return np.asarray(
        controller_gripper_targets(
            left_trigger=left_pressed,
            left_trigger_value=left_value,
            right_trigger=right_pressed,
            right_trigger_value=right_value,
        ),
        dtype=float,
    )


def test_calibrated_targets_use_actual_index_slots_only():
    np.testing.assert_allclose(DEX3_LEFT_PINCH[3:5], [0.0, 0.0], rtol=0, atol=ATOL)
    np.testing.assert_allclose(DEX3_RIGHT_PINCH[5:7], [0.0, 0.0], rtol=0, atol=ATOL)
    assert np.any(np.abs(np.asarray(DEX3_LEFT_PINCH)[5:7]) > ATOL)
    assert np.any(np.abs(np.asarray(DEX3_RIGHT_PINCH)[3:5]) > ATOL)


def test_zero_pull_produces_two_open_hands():
    np.testing.assert_allclose(_targets(), DEX3_DUAL_OPEN, rtol=0, atol=ATOL)


def test_full_pull_produces_calibrated_thumb_index_targets():
    actual = _targets(left_value=0.0, right_value=0.0)
    expected = np.asarray(DEX3_LEFT_PINCH + DEX3_RIGHT_PINCH)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


def test_half_pull_is_exact_midpoint():
    actual = _targets(left_value=5.0, right_value=5.0)
    expected = 0.5 * np.asarray(DEX3_LEFT_PINCH + DEX3_RIGHT_PINCH)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


def test_left_and_right_triggers_are_independent():
    actual = _targets(left_value=0.0, right_value=10.0)
    expected = np.asarray(DEX3_LEFT_PINCH + DEX3_OPEN)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (15.0, 0.0),
        (10.0, 0.0),
        (6.51, 0.349),
        (6.50, 0.350),
        (6.49, 0.351),
        (5.0, 0.5),
        (0.0, 1.0),
        (-5.0, 1.0),
    ],
)
def test_finite_trigger_values_are_clamped_continuous_analog(value, expected):
    assert trigger_pull_ratio(False, value) == pytest.approx(expected, abs=ATOL)


def test_missing_helper_value_is_safe_open_even_when_boolean_is_true():
    assert trigger_pull_ratio(True, None) == 0.0


@pytest.mark.parametrize(
    ("pressed", "expected"),
    [(False, 0.0), (True, 1.0)],
)
def test_nonfinite_trigger_value_uses_boolean_without_nan(pressed, expected):
    ratio = trigger_pull_ratio(pressed, math.nan)
    assert ratio == expected
    assert math.isfinite(ratio)


@pytest.mark.parametrize(
    ("ee", "input_mode", "export", "expected"),
    [
        ("dex3", "controller", True, True),
        ("dex3", "controller", False, False),
        ("dex3", "hand", True, False),
        ("dex1", "controller", True, False),
    ],
)
def test_scope_gate_requires_dex3_controller_and_export(ee, input_mode, export, expected):
    assert (
        dex3_controller_pinch_enabled(
            ee=ee,
            input_mode=input_mode,
            gear_sonic_export=export,
        )
        is expected
    )
```

- [ ] **Step 3: Extend the raw WebXR pipeline fixture and tests**

In `tests/test_televuer_controller_payload.py`, add `numpy`, then replace the
existing post-stub `TeleVuer` import with this helper/wrapper import block:

```python
import numpy as np

from teleop.utils.dex3_controller_pinch import (
    DEX3_LEFT_PINCH,
    DEX3_OPEN,
    controller_gripper_targets,
)
from televuer.televuer import TeleVuer
from televuer.tv_wrapper import TeleVuerWrapper
```

Add a head matrix to `make_controller_televuer_stub()` and initialize all three poses as identity matrices:

```python
    tv.head_pose_shared = Array("d", 16, lock=True)
    tv.left_arm_pose_shared = Array("d", 16, lock=True)
    tv.right_arm_pose_shared = Array("d", 16, lock=True)
    identity = np.eye(4, dtype=float).reshape(-1, order="F").tolist()
    tv.head_pose_shared[:] = identity
    tv.left_arm_pose_shared[:] = identity
    tv.right_arm_pose_shared[:] = identity
```

Append the end-to-end helper and parameterized tests:

```python
def _raw_controller_payload_to_targets(left_state):
    tv = make_controller_televuer_stub()
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    asyncio.run(
        tv.on_controller_move(
            Event(
                {
                    "left": identity,
                    "right": identity,
                    "leftState": left_state,
                    "rightState": {"trigger": False, "triggerValue": 0.0},
                }
            ),
            None,
        )
    )
    wrapper = object.__new__(TeleVuerWrapper)
    wrapper.use_hand_tracking = False
    wrapper.return_hand_rot_data = False
    wrapper.tvuer = tv
    tele_data = wrapper.get_tele_data()
    return np.asarray(
        controller_gripper_targets(
            left_trigger=tele_data.left_ctrl_trigger,
            left_trigger_value=tele_data.left_ctrl_triggerValue,
            right_trigger=tele_data.right_ctrl_trigger,
            right_trigger_value=tele_data.right_ctrl_triggerValue,
        ),
        dtype=float,
    )


def test_raw_analog_trigger_reaches_wrapper_and_pinch_helper():
    actual = _raw_controller_payload_to_targets(
        {"trigger": True, "triggerValue": 0.25}
    )
    expected = np.asarray(DEX3_LEFT_PINCH + DEX3_OPEN, dtype=float)
    expected[:7] *= 0.25
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_missing_raw_trigger_value_is_safe_open_even_with_true_boolean():
    actual = _raw_controller_payload_to_targets({"trigger": True})
    np.testing.assert_allclose(actual, np.zeros(14), rtol=0, atol=1e-6)


def test_nonfinite_raw_trigger_value_uses_true_boolean_fallback():
    actual = _raw_controller_payload_to_targets(
        {"trigger": True, "triggerValue": float("nan")}
    )
    expected = np.asarray(DEX3_LEFT_PINCH + DEX3_OPEN, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
```

- [ ] **Step 4: Run the new tests and verify the red state**

Run the formatter, syntax check, linter, and final format check:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_televuer_controller_payload.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'teleop.utils.dex3_controller_pinch'`.

- [ ] **Step 5: Implement the minimal pure helper**

Create `teleop/utils/dex3_controller_pinch.py` with:

```python
"""Analog thumb-index pinch mapping for Dex3 XR controllers."""

from __future__ import annotations

import math
from collections.abc import Sequence


DEX3_OPEN = (0.0,) * 7
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
    1.273907,
    0.419395,
    0.0,
    0.0,
)
DEX3_DUAL_OPEN = DEX3_OPEN + DEX3_OPEN


def trigger_pull_ratio(trigger_pressed: bool, trigger_value: object) -> float:
    """Return 0=open and 1=full pull from the wrapper's 10-to-0 value."""

    try:
        value = float(trigger_value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 1.0 if bool(trigger_pressed) else 0.0
    return min(max((10.0 - value) / 10.0, 0.0), 1.0)


def _scaled_target(target: Sequence[float], ratio: float) -> list[float]:
    return [ratio * value for value in target]


def controller_gripper_targets(
    *,
    left_trigger: bool,
    left_trigger_value: object,
    right_trigger: bool,
    right_trigger_value: object,
) -> list[float]:
    """Build left-then-right Dex3 DDS targets for two controller triggers."""

    left_ratio = trigger_pull_ratio(left_trigger, left_trigger_value)
    right_ratio = trigger_pull_ratio(right_trigger, right_trigger_value)
    return _scaled_target(DEX3_LEFT_PINCH, left_ratio) + _scaled_target(
        DEX3_RIGHT_PINCH, right_ratio
    )


def dex3_controller_pinch_enabled(
    *, ee: str, input_mode: str, gear_sonic_export: bool
) -> bool:
    """Limit the primitive to Dex3 controller targets exported to GEAR-SONIC."""

    return ee == "dex3" and input_mode == "controller" and bool(gear_sonic_export)
```

- [ ] **Step 6: Run the pure and raw-pipeline tests and verify green**

Run the command from Step 4 again.

Expected: all tests in both files pass; no result contains NaN.

- [ ] **Step 7: Audit the external diff without staging or committing user work**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
git status --short
git diff --no-index /dev/null teleop/utils/dex3_controller_pinch.py || true
git diff -- tests/test_televuer_controller_payload.py
```

Expected: only the new helper and the intended test additions are attributable to this task. Do not run `git add` or `git commit` in this dirty checkout.

## Task 2: Wire the Helper into the Live Export Path

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/tests/test_teleop_keyboard_controls.py`
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`

- [ ] **Step 1: Add failing scope-aware selector tests to the existing main-module fixture**

At the top of `tests/test_teleop_keyboard_controls.py`, add:

```python
from types import SimpleNamespace

import numpy as np
import teleop.utils.dex3_controller_pinch as pinch
```

Inside `_load_teleop_module()`, immediately after `_stub_module("teleop.utils")`, retain the real pure helper beneath the stub package:

```python
    sys.modules["teleop.utils.dex3_controller_pinch"] = pinch
```

Append:

```python
def _controller_tele_data(left_value=10.0, right_value=10.0):
    return SimpleNamespace(
        left_ctrl_trigger=False,
        left_ctrl_triggerValue=left_value,
        right_ctrl_trigger=False,
        right_ctrl_triggerValue=right_value,
    )


def test_live_selector_returns_calibrated_pinch_only_inside_exact_scope():
    module = _load_teleop_module()
    tele_data = _controller_tele_data(left_value=0.0, right_value=10.0)

    enabled = module._controller_gripper_payload_if_enabled(
        ee="dex3",
        input_mode="controller",
        gear_sonic_export=True,
        tele_data=tele_data,
    )
    np.testing.assert_allclose(
        enabled,
        pinch.DEX3_LEFT_PINCH + pinch.DEX3_OPEN,
        rtol=0,
        atol=1e-6,
    )

    assert module._controller_gripper_payload_if_enabled(
        ee="dex3",
        input_mode="controller",
        gear_sonic_export=False,
        tele_data=tele_data,
    ) is None
    assert module._controller_gripper_payload_if_enabled(
        ee="dex3",
        input_mode="hand",
        gear_sonic_export=True,
        tele_data=tele_data,
    ) is None
    assert module._controller_gripper_payload_if_enabled(
        ee="dex1",
        input_mode="controller",
        gear_sonic_export=True,
        tele_data=tele_data,
    ) is None
```

- [ ] **Step 2: Run the selector test and verify it fails**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_teleop_keyboard_controls.py::test_live_selector_returns_calibrated_pinch_only_inside_exact_scope
```

Expected: FAIL with `AttributeError` because `_controller_gripper_payload_if_enabled` does not exist.

- [ ] **Step 3: Replace binary constants and functions with the pure helper**

In `teleop/teleop_hand_and_arm.py`, add:

```python
from teleop.utils.dex3_controller_pinch import (
    DEX3_DUAL_OPEN,
    controller_gripper_targets as dex3_controller_gripper_targets,
    dex3_controller_pinch_enabled,
)
```

Replace `XR_DEX3_OPEN`, `XR_DEX3_LEFT_CLOSED`, `XR_DEX3_RIGHT_CLOSED`, and the old dual-open expression with:

```python
XR_DEX3_DUAL_OPEN = list(DEX3_DUAL_OPEN)
```

Delete `_trigger_closed` and replace `_controller_gripper_targets` with:

```python
def _controller_gripper_targets(tele_data):
    return dex3_controller_gripper_targets(
        left_trigger=tele_data.left_ctrl_trigger,
        left_trigger_value=tele_data.left_ctrl_triggerValue,
        right_trigger=tele_data.right_ctrl_trigger,
        right_trigger_value=tele_data.right_ctrl_triggerValue,
    )


def _controller_gripper_payload_if_enabled(
    *, ee, input_mode, gear_sonic_export, tele_data
):
    if not dex3_controller_pinch_enabled(
        ee=ee,
        input_mode=input_mode,
        gear_sonic_export=gear_sonic_export,
    ):
        return None
    return _controller_gripper_targets(tele_data)
```

- [ ] **Step 4: Route the live export branch through the selector**

Replace the current controller/Dex3 hand-payload conditional with:

```python
                controller_gripper_payload = _controller_gripper_payload_if_enabled(
                    ee=args.ee,
                    input_mode=args.input_mode,
                    gear_sonic_export=args.gear_sonic_export,
                    tele_data=tele_data,
                )
                if controller_gripper_payload is not None:
                    dual_hand_payload = controller_gripper_payload
                elif args.ee == "dex3":
                    with dual_hand_data_lock:
                        dual_hand_payload = list(dual_hand_action_array[:])
```

Keep the existing `dual_hand_payload = []` initialization immediately before this block.

- [ ] **Step 5: Run the focused main wiring test and full external set**

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

Expected: every selected test passes, including the original keyboard regressions.

- [ ] **Step 6: Confirm the obsolete binary primitive is gone**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
rg -n "XR_DEX3_(LEFT|RIGHT)_CLOSED|_trigger_closed" teleop tests
```

Expected: no matches.

- [ ] **Step 7: Audit only the touched external paths**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
git diff --check -- teleop/teleop_hand_and_arm.py tests/test_teleop_keyboard_controls.py
git diff -- teleop/teleop_hand_and_arm.py tests/test_teleop_keyboard_controls.py
git status --short
```

Expected: the pinch hunks are limited to imports, constants/helper functions, live selection, and tests; all unrelated pre-existing changes remain present and unstaged.

## Task 3: Characterize the 14-Value JSON Bridge Split

**Files:**
- Modify: `gear_sonic/tests/test_xr_upperbody_bridge.py`

- [ ] **Step 1: Add the calibrated JSON-to-frame characterization test**

Add immediately after `test_split_xr_dex3_dual_hand_joints`:

```python
def test_calibrated_controller_pinch_json_splits_exact_dex3_sides() -> None:
    left = [-0.379616, 0.516712, 0.121406, 0.0, 0.0, -1.273903, -0.419393]
    right = [-0.379617, -0.516714, -0.121407, 1.273907, 0.419395, 0.0, 0.0]
    payload = json.loads(
        json.dumps(
            {
                "dual_arm_position": [0.0] * 14,
                "dual_hand_joints": left + right,
            }
        )
    )

    frame = normalize_live_source_payload(payload)

    np.testing.assert_allclose(frame.left_hand_joints, left, rtol=0, atol=1e-6)
    np.testing.assert_allclose(frame.right_hand_joints, right, rtol=0, atol=1e-6)
```

- [ ] **Step 2: Run the exact bridge characterization**

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py::test_calibrated_controller_pinch_json_splits_exact_dex3_sides
```

Expected: PASS because the bridge already preserves active-tracking hand order.

- [ ] **Step 3: Run the full bridge regression file**

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py
```

Expected: all prior 38 tests plus the new characterization pass.

- [ ] **Step 4: Commit only the workspace bridge test**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git add gear_sonic/tests/test_xr_upperbody_bridge.py
git diff --cached --check
git commit -m "test(teleop): cover calibrated Dex3 pinch split"
```

Expected: the commit contains only the single bridge test; `docs/option_b_vla_sonic_pipeline_spec.md` remains untracked.

## Task 4: Add the Reproducible MuJoCo Geometry Verifier

**Files:**
- Create: `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py`

- [ ] **Step 1: Run the specified verifier command before creation**

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate
```

Expected: FAIL with `can't open file` because the verifier does not exist.

- [ ] **Step 2: Implement the standalone verifier**

Create `gear_sonic/scripts/verify_xr_dex3_controller_pinch.py` with:

```python
#!/usr/bin/env python3
"""Verify Dex3 XR thumb-index pinch geometry against the G1 MuJoCo model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


SIDE_ORDER = {
    "left": (
        "thumb_0",
        "thumb_1",
        "thumb_2",
        "middle_0",
        "middle_1",
        "index_0",
        "index_1",
    ),
    "right": (
        "thumb_0",
        "thumb_1",
        "thumb_2",
        "index_0",
        "index_1",
        "middle_0",
        "middle_1",
    ),
}
DISTANCE_TOLERANCE_M = 1e-6
MAX_COMMAND_STEP_RAD = 0.25


def load_xr_targets(xr_root: Path) -> dict[str, np.ndarray]:
    module_path = xr_root / "teleop" / "utils" / "dex3_controller_pinch.py"
    spec = importlib.util.spec_from_file_location(
        "xr_dex3_controller_pinch_for_verification", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load XR pinch helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = {
        "left": np.asarray(module.DEX3_LEFT_PINCH, dtype=float),
        "right": np.asarray(module.DEX3_RIGHT_PINCH, dtype=float),
    }
    for side, target in targets.items():
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid {side} target: {target}")
    return targets


def _hand_ids(
    model: mujoco.MjModel, side: str
) -> tuple[np.ndarray, int, int]:
    qpos_addresses = []
    for suffix in SIDE_ORDER[side]:
        joint_name = f"{side}_hand_{suffix}_joint"
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"missing joint: {joint_name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    geom_ids = []
    for suffix in ("thumb_2", "index_1"):
        body_name = f"{side}_hand_{suffix}_link"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        candidates = np.flatnonzero(
            (model.geom_bodyid == body_id) & (model.geom_contype != 0)
        )
        if candidates.size != 1:
            raise ValueError(
                f"expected one collision geom on {body_name}, got {candidates.tolist()}"
            )
        geom_ids.append(int(candidates[0]))
    return np.asarray(qpos_addresses, dtype=int), geom_ids[0], geom_ids[1]


def _signed_distance_trace(
    model: mujoco.MjModel,
    side: str,
    states: list[np.ndarray],
) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos_addresses, thumb_geom, index_geom = _hand_ids(model, side)
    distances = []
    for state in states:
        data.qpos[:] = model.qpos0
        data.qpos[qpos_addresses] = state
        mujoco.mj_forward(model, data)
        distance = mujoco.mj_geomDistance(
            model,
            data,
            thumb_geom,
            index_geom,
            1.0,
            np.zeros(6, dtype=float),
        )
        distances.append(float(distance))
    return np.asarray(distances, dtype=float)


def idealized_slew_states(
    target: np.ndarray, max_step: float = MAX_COMMAND_STEP_RAD
) -> list[np.ndarray]:
    current = np.zeros_like(target)
    states = []
    for _ in range(100):
        if np.allclose(current, target, rtol=0, atol=1e-12):
            return states
        current = current + np.clip(target - current, -max_step, max_step)
        states.append(current.copy())
    raise AssertionError("idealized slew did not converge within 100 recurrence steps")


def _assert_trace(
    side: str,
    name: str,
    distances: np.ndarray,
    *,
    expected_steps: int,
) -> None:
    if distances.shape != (expected_steps,):
        raise AssertionError(
            f"{side} {name}: expected {expected_steps} samples, got {distances.size}"
        )
    if not np.all(distances[:-1] > 0.0):
        raise AssertionError(f"{side} {name}: pre-final contact {distances.tolist()}")
    if np.any(np.diff(distances) > DISTANCE_TOLERANCE_M):
        raise AssertionError(f"{side} {name}: non-monotonic {distances.tolist()}")
    if abs(distances[-1]) > DISTANCE_TOLERANCE_M:
        raise AssertionError(
            f"{side} {name}: final distance {distances[-1]} exceeds tolerance"
        )


def verify_pinch(model_path: Path, targets: dict[str, np.ndarray]) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    report = {"mujoco_version": mujoco.__version__, "sides": {}}
    alphas = np.linspace(0.0, 1.0, 11)
    for side, target in targets.items():
        scalar_states = [alpha * target for alpha in alphas]
        scalar_distances = _signed_distance_trace(model, side, scalar_states)
        _assert_trace(
            side,
            "scalar",
            scalar_distances,
            expected_steps=len(scalar_states),
        )

        slew_states = idealized_slew_states(target)
        if len(slew_states) != 6:
            raise AssertionError(
                f"{side} idealized slew: expected 6 recurrence steps, "
                f"got {len(slew_states)}"
            )
        slew_distances = _signed_distance_trace(model, side, slew_states)
        _assert_trace(
            side,
            "idealized_slew",
            slew_distances,
            expected_steps=6,
        )
        report["sides"][side] = {
            "scalar_distances_m": scalar_distances.tolist(),
            "idealized_slew_steps": len(slew_states),
            "idealized_slew_distances_m": slew_distances.tolist(),
        }
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_xr_targets(args.xr_root.resolve())
    report = verify_pinch(args.model.resolve(), targets)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verifier and inspect the exact output contract**

Run the command from Step 1 again.

Expected: exit code 0; JSON reports MuJoCo `3.8.1`, six idealized recurrence steps per side, and final distances near `-8.7e-8 m` left and `-5.3e-8 m` right. The verifier never uses `MjData.ncon`.

- [ ] **Step 4: Check formatting and syntax**

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
ruff format --no-cache gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
PYTHONDONTWRITEBYTECODE=1 .venv_teleop/bin/python -m py_compile \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
ruff check --no-cache gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
ruff format --check --no-cache gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
```

Expected: all three commands exit 0.

- [ ] **Step 5: Commit only the verifier**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git add gear_sonic/scripts/verify_xr_dex3_controller_pinch.py
git diff --cached --check
git commit -m "test(teleop): verify Dex3 pinch geometry"
```

Expected: the commit contains only the verifier; the unrelated untracked document remains untouched.

## Task 5: Full Software Verification and Guarded Hardware Acceptance

**Files:**
- Verify: all files named in Tasks 1-4
- Do not modify: `docs/option_b_vla_sonic_pipeline_spec.md`

- [ ] **Step 1: Run the complete external XR regression set**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPATH=teleop/televuer/src:. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_dex3_controller_pinch.py \
  tests/test_televuer_controller_payload.py \
  tests/test_teleop_keyboard_controls.py
```

Expected: all selected tests pass, including raw-payload, finite/non-finite, scope, and legacy keyboard cases.

- [ ] **Step 2: Run the complete bridge regression file**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_xr_upperbody_bridge.py
```

Expected: the prior 38 tests plus the calibrated split test pass.

- [ ] **Step 3: Run the reproducible geometry verifier**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python \
  gear_sonic/scripts/verify_xr_dex3_controller_pinch.py \
  --xr-root /home/jihun/work/unitree_official/xr_teleoperate
```

Expected: exit code 0 with positive pre-final distances, signed-distance tangency within `1e-6 m`, and six idealized recurrence steps for each hand.

- [ ] **Step 4: Audit repository state and scoped diffs**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git status --short
git log -3 --oneline

cd /home/jihun/work/unitree_official/xr_teleoperate
git status --short
git diff --check -- \
  teleop/teleop_hand_and_arm.py \
  tests/test_teleop_keyboard_controls.py \
  tests/test_televuer_controller_payload.py
git diff -- teleop/teleop_hand_and_arm.py
```

Expected: workspace commits contain only the bridge test and verifier; `docs/option_b_vla_sonic_pipeline_spec.md` remains untracked. The external checkout retains all prior dirty files plus the narrowly scoped pinch changes, with no unrelated reset or staging.

- [ ] **Step 5: Perform a no-robot ZMQ payload inspection before enabling control**

Launch the XR exporter with the existing command, but keep the GEAR-SONIC deploy process stopped or control disabled:

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

Expected while inspecting `xr_teleop` messages: open triggers emit fourteen zeros; a fully pulled left trigger emits `DEX3_LEFT_PINCH` followed by seven zeros; a fully pulled right trigger emits seven zeros followed by `DEX3_RIGHT_PINCH`; partial pulls scale continuously.

In a second terminal, print the live 14-value payload with:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python - <<'PY'
import json

import zmq

topic = b"xr_teleop"
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.setsockopt(zmq.SUBSCRIBE, topic)
subscriber.connect("tcp://127.0.0.1:5560")
try:
    while True:
        raw = subscriber.recv()
        payload = json.loads(raw[len(topic) :].decode("utf-8"))
        print(payload["dual_hand_joints"], flush=True)
finally:
    subscriber.close(linger=0)
    context.term()
PY
```

Expected: values update continuously as each trigger moves; press `Ctrl-C`
after open, half-pull, and full-pull observations for both hands.

- [ ] **Step 6: Perform guarded real-robot acceptance with a spotter and immediate stop access**

Prerequisites: robot supported, clear workspace, no object between fingers, E-stop/damping control held by a second person, GEAR-SONIC running with Dex3 enabled and `--max-close-ratio 1.0`, bridge connected to port `5560`, and XR tracking valid before pressing `r`. Use `--stop-hand-preset open` on the bridge if open fingers are required during exit.

Acceptance sequence:

1. Press `r` while both triggers are released; both hands must remain open.
2. Pull only the left trigger slowly; the left thumb and index must approach continuously while the left middle finger remains open.
3. Stop immediately if either pad passes the other, an unintended finger moves, or motion is discontinuous.
4. At full pull, confirm thumb-index pad contact without visible crossing.
5. Release the left trigger and confirm continuous reopening.
6. Repeat Steps 2-5 on the right hand.
7. Press `q`; verify the configured bridge exit preset, remembering that the default `tucked-thumb` preset may move the middle fingers.

Expected: both hands satisfy the thumb-index pinch contract independently during active tracking. Do not mark the goal complete until this guarded hardware observation is reported.
