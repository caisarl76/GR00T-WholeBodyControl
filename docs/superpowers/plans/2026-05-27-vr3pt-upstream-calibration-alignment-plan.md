# VR_3PT Upstream Calibration Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realign the controller-only VR_3PT teleop path to upstream's calibration contract by inserting a virtual-pelvis stage, reverting our local `ThreePointPose` deviations, adding a low-pass headset filter, and verifying the result via a parity test against NVlabs pinned commit `4f5e118ae82e0448f2b2f441024d59fe74e0a4d9`.

**Architecture:** A new `ControllerCalibState` class owns the virtual-pelvis estimate, two filters (orientation SLERP-EMA, Z position EMA), and the singular-yaw fallback. `_process_controller_3pt_pose` becomes a thin wrapper around it that applies the `openxr_unitree` basis and any legacy wrist bias. `ThreePointPose` is partially reverted to NVlabs upstream behavior: synchronous CALIB_FULL capture, neck-relative wrist residuals, kinematic-chain neck position. Manager-loop lifecycle wiring resets filters at the right moments (filter reset must precede the VR_3PT entry gate so the gate evaluates with the same state the first frame will use).

**Tech Stack:** Python 3.10, numpy, scipy.spatial.transform.Rotation (and Slerp), unittest (existing test pattern in repo), git.

**Spec reference:** `docs/superpowers/specs/2026-05-27-vr3pt-upstream-calibration-alignment-design.md`.

**Upstream pinned SHA for parity test:** `4f5e118ae82e0448f2b2f441024d59fe74e0a4d9`.

## Working-tree and commit conventions

This plan does **not** assume a clean working tree. The current workspace contains in-progress ramping/freeze edits and untracked safety-stack files. Do **not** stash or revert the user's work to satisfy a per-task TDD pattern.

- Each task's "Commit" step is the recommended checkpoint but **optional**. If you prefer to batch-commit, replace `git commit -m "..."` with `git add ...` and consolidate at the end of the plan.
- Do **not** add `Co-Authored-By:` trailers unless the user explicitly requests them.
- All git operations stay on the new feature branch created in Task 1. The branch carries the user's existing local changes from `pico4u-g1-real-teleop-safety` forward — those changes are intentional and remain part of the working tree.

---

## File Structure

**Modified (existing):**
- `gear_sonic/scripts/pico_manager_thread_server.py` — single Python file housing all helpers, `ControllerCalibState`, `_process_controller_3pt_pose`, `ThreePointPose` (partial revert), manager loop wiring, CLI parsing.
- `gear_sonic/tests/test_safety_smoke.py` — extended with three new operator-confirmation drill items.
- `docs/source/user_guide/real_robot_safety.md` — add standing-only-during-teleop constraint.

**New (tests + fixtures):**
- `gear_sonic/tests/test_headset_filters.py` — unit tests for both filters.
- `gear_sonic/tests/test_yaw_only.py` — unit tests for `yaw_only`.
- `gear_sonic/tests/test_controller_calib_state.py` — unit + lifecycle tests for `ControllerCalibState`.
- `gear_sonic/tests/test_controller_calib_pelvis_anchoring.py` — regression guard for the head-relative wrist subtraction bug.
- `gear_sonic/tests/test_three_point_pose_parity.py` — numerical parity vs NVlabs pinned commit `4f5e118ae82e0448f2b2f441024d59fe74e0a4d9`.
- `gear_sonic/tests/fixtures/threepoint_golden.json` — golden output captured from upstream at the pinned SHA.
- `gear_sonic/tests/fixtures/regen_threepoint_golden.py` — regenerates the golden in a temporary git worktree.

**New (hardware-validation reports — operator-generated):**
- `docs/superpowers/specs/2026-05-27-hardware-validation/G1_drop_z_distribution.md`.
- `docs/superpowers/specs/2026-05-27-hardware-validation/G2_forearm_twist_axis.md`.
- `docs/superpowers/specs/2026-05-27-hardware-validation/tau_sweep.md`.

**Branch:** create `vr3pt-upstream-calibration-alignment` from `pico4u-g1-real-teleop-safety`.

---

## Task 1: Create feature branch on top of current state

**Files:**
- No file changes (branch operation only).

**Important:** This task creates a new branch that carries the current dirty working tree forward. It does NOT stash, reset, or otherwise mutate the user's in-progress local changes (ramping/freeze edits + safety-stack untracked files).

- [ ] **Step 1: Record current status for reference**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git branch --show-current
git status --short > /tmp/vr3pt_alignment_pre_branch_status.txt
cat /tmp/vr3pt_alignment_pre_branch_status.txt
```

Expected: `git branch --show-current` prints `pico4u-g1-real-teleop-safety`. `/tmp/vr3pt_alignment_pre_branch_status.txt` captures the pre-branch state for later reference if needed.

- [ ] **Step 2: Create and switch to the feature branch — without resetting the tree**

```bash
git checkout -b vr3pt-upstream-calibration-alignment
git branch --show-current
git status --short
```

`git checkout -b` carries the working tree forward unchanged; the operation does not touch local edits. Confirm:

- `git branch --show-current` prints `vr3pt-upstream-calibration-alignment`.
- `git status --short` is identical to `/tmp/vr3pt_alignment_pre_branch_status.txt`.

- [ ] **Step 3: Do NOT stash, reset, clean, or revert**

If the engineer is tempted to "tidy up" the tree at this point — don't. The plan's TDD steps coexist with the existing edits. Task 8 in particular will modify code that the user has been actively iterating on; that's by design.

---

## Task 2: `HeadsetOrientationFilter`

**Files:**
- Create: `gear_sonic/tests/test_headset_filters.py`
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (insert after the `XRStalenessWatchdog` class)

- [ ] **Step 1: Write the failing test**

Create `gear_sonic/tests/test_headset_filters.py`:

```python
"""Unit tests for HeadsetOrientationFilter and HeadsetZFilter."""
import math
import unittest

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import (
    HeadsetOrientationFilter,
    HeadsetZFilter,
)


def quat_angle_deg(a, b):
    """Angle between two unit quaternions in degrees."""
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


class HeadsetOrientationFilterTest(unittest.TestCase):
    def test_first_call_initializes_from_input(self):
        f = HeadsetOrientationFilter(tau_s=1.0)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        out = f.update(q, dt=0.05)
        np.testing.assert_allclose(out, q)

    def test_first_call_after_reset_initializes_from_input(self):
        f = HeadsetOrientationFilter(tau_s=1.0)
        f.update(np.array([1.0, 0.0, 0.0, 0.0]), dt=0.05)
        f.reset()
        q2 = np.array([0.7071, 0.7071, 0.0, 0.0])
        out = f.update(q2, dt=0.05)
        np.testing.assert_allclose(out, q2, atol=1e-6)

    def test_zero_dt_returns_prior_state_no_snap(self):
        f = HeadsetOrientationFilter(tau_s=1.0)
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        f.update(q1, dt=0.05)
        q2 = np.array([0.7071, 0.7071, 0.0, 0.0])
        out = f.update(q2, dt=0.0)
        np.testing.assert_allclose(out, q1, atol=1e-6)

    def test_negative_dt_returns_prior_state(self):
        f = HeadsetOrientationFilter(tau_s=1.0)
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        f.update(q1, dt=0.05)
        q2 = np.array([0.7071, 0.7071, 0.0, 0.0])
        out = f.update(q2, dt=-0.01)
        np.testing.assert_allclose(out, q1, atol=1e-6)

    def test_dt_clamp_caps_alpha(self):
        f_clamp = HeadsetOrientationFilter(tau_s=1.0)
        f_clamp.update(np.array([1.0, 0.0, 0.0, 0.0]), dt=0.05)
        out_50ms = f_clamp.update(np.array([0.7071, 0.7071, 0.0, 0.0]), dt=0.05)
        f_clamp2 = HeadsetOrientationFilter(tau_s=1.0)
        f_clamp2.update(np.array([1.0, 0.0, 0.0, 0.0]), dt=0.05)
        out_500ms = f_clamp2.update(np.array([0.7071, 0.7071, 0.0, 0.0]), dt=0.500)
        # With clamp at 0.05 s, both calls should produce the same output.
        np.testing.assert_allclose(out_50ms, out_500ms, atol=1e-6)

    def test_long_sequence_converges(self):
        f = HeadsetOrientationFilter(tau_s=0.1)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        target = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90 deg about X
        f.update(identity, dt=0.05)
        for _ in range(100):  # 5 seconds = 50 tau, should converge
            out = f.update(target, dt=0.05)
        angle = quat_angle_deg(out, target)
        self.assertLess(angle, 2.0)


class HeadsetZFilterTest(unittest.TestCase):
    def test_first_call_initializes_from_input(self):
        f = HeadsetZFilter(tau_s=0.3)
        out = f.update(1.7, dt=0.05)
        self.assertAlmostEqual(out, 1.7)

    def test_zero_dt_returns_prior_state(self):
        f = HeadsetZFilter(tau_s=0.3)
        f.update(1.7, dt=0.05)
        out = f.update(2.0, dt=0.0)
        self.assertAlmostEqual(out, 1.7)

    def test_negative_dt_returns_prior_state(self):
        f = HeadsetZFilter(tau_s=0.3)
        f.update(1.7, dt=0.05)
        out = f.update(2.0, dt=-0.01)
        self.assertAlmostEqual(out, 1.7)

    def test_dt_clamp_caps_alpha(self):
        f_a = HeadsetZFilter(tau_s=0.3)
        f_a.update(0.0, dt=0.05)
        out_a = f_a.update(1.0, dt=0.05)
        f_b = HeadsetZFilter(tau_s=0.3)
        f_b.update(0.0, dt=0.05)
        out_b = f_b.update(1.0, dt=0.500)
        self.assertAlmostEqual(out_a, out_b, places=6)

    def test_long_sequence_converges(self):
        f = HeadsetZFilter(tau_s=0.1)
        f.update(0.0, dt=0.05)
        for _ in range(100):
            out = f.update(1.0, dt=0.05)
        self.assertLess(abs(out - 1.0), 0.01)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test — confirm failure**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_headset_filters.py
```

Expected: `ImportError: cannot import name 'HeadsetOrientationFilter'` (and `HeadsetZFilter`).

- [ ] **Step 3: Add `Slerp` import to `pico_manager_thread_server.py`**

In `gear_sonic/scripts/pico_manager_thread_server.py`, find the existing `from scipy.spatial.transform import Rotation as sRot` import near the top of the file. Add `Slerp` to the same line:

```python
from scipy.spatial.transform import Rotation as sRot, Slerp
```

- [ ] **Step 4: Implement the two filter classes**

In `gear_sonic/scripts/pico_manager_thread_server.py`, insert the following immediately after the `XRStalenessWatchdog` class definition (search for `class XRStalenessWatchdog:` to locate; the new classes go after its closing brace, before `### Parse 3 point pose from SMPL` comment):

```python
class HeadsetOrientationFilter:
    """SLERP-based exponential moving average on a quaternion (scalar-first wxyz).

    First call after construction or reset() initializes state from input
    and returns input unchanged (any dt). dt <= 0 on a primed filter returns
    the last state — does NOT snap to the new sample. dt is clamped to
    DT_CLAMP_MAX_S to bound single-step alpha during transient stalls.
    """

    DT_CLAMP_MAX_S = 0.05

    def __init__(self, tau_s: float = 1.0):
        self.tau_s = float(tau_s)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
        if self._state is None:
            self._state = np.asarray(quat_wxyz, dtype=np.float64).copy()
            return self._state.copy()
        if dt <= 0.0:
            return self._state.copy()
        dt_eff = min(float(dt), self.DT_CLAMP_MAX_S)
        alpha = 1.0 - float(np.exp(-dt_eff / self.tau_s))
        r_state = sRot.from_quat(self._state, scalar_first=True)
        r_new = sRot.from_quat(quat_wxyz, scalar_first=True)
        key_rots = sRot.concatenate([r_state, r_new])
        slerp = Slerp([0.0, 1.0], key_rots)
        self._state = slerp(alpha).as_quat(scalar_first=True)
        return self._state.copy()


class HeadsetZFilter:
    """1-D exponential moving average on a scalar (headset Z position).

    First-call and dt semantics match HeadsetOrientationFilter.
    """

    DT_CLAMP_MAX_S = 0.05

    def __init__(self, tau_s: float = 0.3):
        self.tau_s = float(tau_s)
        self._state: float | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, z: float, dt: float) -> float:
        if self._state is None:
            self._state = float(z)
            return self._state
        if dt <= 0.0:
            return self._state
        dt_eff = min(float(dt), self.DT_CLAMP_MAX_S)
        alpha = 1.0 - float(np.exp(-dt_eff / self.tau_s))
        self._state = (1.0 - alpha) * self._state + alpha * float(z)
        return self._state
```

- [ ] **Step 5: Run the test — confirm pass**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_headset_filters.py
```

Expected: `Ran 11 tests in ... OK`.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic/tests/test_headset_filters.py gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): add HeadsetOrientationFilter and HeadsetZFilter

SLERP-based EMA for headset orientation (default τ=1.0 s) and a scalar
EMA for headset Z position (default τ=0.3 s). Both have idempotent
first-call behavior and a 50 ms dt clamp to bound single-step alpha
during transient process stalls. Used by ControllerCalibState (next task)
to produce a stable virtual-pelvis estimate from headset-only tracking.
EOF
)"
```

---

## Task 3: `yaw_only` function

**Files:**
- Create: `gear_sonic/tests/test_yaw_only.py`
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (insert after `HeadsetZFilter`)

- [ ] **Step 1: Write the failing test**

Create `gear_sonic/tests/test_yaw_only.py`:

```python
"""Unit tests for yaw_only quaternion extraction (singularity-safe)."""
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.scripts.pico_manager_thread_server import yaw_only


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class YawOnlyTest(unittest.TestCase):
    def test_pure_yaw_input_returns_same_yaw(self):
        # 30 deg about Z (yaw)
        q = sRot.from_euler("z", 30, degrees=True).as_quat(scalar_first=True)
        out = yaw_only(q)
        # Extract yaw from output
        euler = sRot.from_quat(out, scalar_first=True).as_euler("zyx", degrees=True)
        self.assertAlmostEqual(euler[0], 30.0, places=4)
        self.assertAlmostEqual(euler[1], 0.0, places=4)
        self.assertAlmostEqual(euler[2], 0.0, places=4)

    def test_yaw_plus_pitch_returns_yaw_only(self):
        q = (sRot.from_euler("z", 45, degrees=True) *
             sRot.from_euler("y", 20, degrees=True)).as_quat(scalar_first=True)
        out = yaw_only(q)
        euler = sRot.from_quat(out, scalar_first=True).as_euler("zyx", degrees=True)
        # Yaw should be preserved up to projection error from the pitch component
        self.assertAlmostEqual(euler[1], 0.0, places=4)
        self.assertAlmostEqual(euler[2], 0.0, places=4)

    def test_straight_up_returns_fallback(self):
        # +X axis rotated to point at +Z = looking straight up
        q = sRot.from_euler("y", -90, degrees=True).as_quat(scalar_first=True)
        fallback = sRot.from_euler("z", 42, degrees=True).as_quat(scalar_first=True)
        out = yaw_only(q, fallback_quat=fallback)
        np.testing.assert_allclose(out, fallback)

    def test_straight_down_returns_fallback(self):
        q = sRot.from_euler("y", 90, degrees=True).as_quat(scalar_first=True)
        fallback = sRot.from_euler("z", -17, degrees=True).as_quat(scalar_first=True)
        out = yaw_only(q, fallback_quat=fallback)
        np.testing.assert_allclose(out, fallback)

    def test_straight_up_without_fallback_returns_identity(self):
        q = sRot.from_euler("y", -90, degrees=True).as_quat(scalar_first=True)
        out = yaw_only(q, fallback_quat=None)
        np.testing.assert_allclose(out, IDENTITY)

    def test_almost_singular_takes_compute_path_not_fallback(self):
        # Tilted 89.9 deg up — xy projection of fwd has norm ~ cos(89.9) ≈ 0.00175,
        # comfortably above the 1e-6 singularity threshold. yaw_only must take
        # the COMPUTE path and return identity (yaw = 0 for pure -Y rotation),
        # NOT the fallback. Use a clearly non-identity fallback so the assertion
        # can distinguish the two paths.
        q = sRot.from_euler("y", -89.9, degrees=True).as_quat(scalar_first=True)
        non_identity_fallback = sRot.from_euler("z", 30, degrees=True).as_quat(scalar_first=True)
        out = yaw_only(q, fallback_quat=non_identity_fallback)
        # Compute path returns identity for pure pitch; fallback would have
        # been the 30°-yaw quat. Assert we got identity.
        np.testing.assert_allclose(out, IDENTITY, atol=1e-6,
                                    err_msg="yaw_only took fallback path when it should have computed")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test — confirm failure**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_yaw_only.py
```

Expected: `ImportError: cannot import name 'yaw_only'`.

- [ ] **Step 3: Implement `yaw_only`**

In `gear_sonic/scripts/pico_manager_thread_server.py`, insert immediately after the `HeadsetZFilter` class (from Task 2):

```python
def yaw_only(quat_wxyz: np.ndarray,
             fallback_quat: np.ndarray | None = None) -> np.ndarray:
    """Return a yaw-only quaternion. Singularity-safe.

    When the rotated +X axis has near-zero horizontal projection (operator
    headset pointed straight up or straight down), `fallback_quat` is
    returned. If fallback is None, returns identity. This prevents a
    single-frame frame jump on extreme headset pitch.
    """
    r = sRot.from_quat(quat_wxyz, scalar_first=True)
    fwd = r.apply([1.0, 0.0, 0.0])
    xy_norm = float(np.linalg.norm(fwd[:2]))
    if xy_norm < 1e-6:
        if fallback_quat is not None:
            return np.asarray(fallback_quat, dtype=np.float64).copy()
        return np.array([1.0, 0.0, 0.0, 0.0])
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    return sRot.from_euler("z", yaw).as_quat(scalar_first=True)
```

- [ ] **Step 4: Run the test — confirm pass**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_yaw_only.py
```

Expected: `Ran 6 tests in ... OK`.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/tests/test_yaw_only.py gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): add yaw_only quaternion extraction

Singularity-safe yaw extraction from a quaternion. When the rotated +X
axis has near-zero horizontal projection (operator pointed straight up
or down), returns the supplied fallback (or identity). Used by
ControllerCalibState to derive virtual-pelvis yaw from the filtered
headset orientation without one-frame jumps on extreme pitch.
EOF
)"
```

---

## Task 4: `ControllerCalibState`

**Files:**
- Create: `gear_sonic/tests/test_controller_calib_state.py`
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` (insert after `yaw_only`)

- [ ] **Step 1: Write the failing test**

Create `gear_sonic/tests/test_controller_calib_state.py`:

```python
"""Unit + lifecycle tests for ControllerCalibState."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.scripts.pico_manager_thread_server import ControllerCalibState

IDENTITY_ROT = sRot.identity()
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _pose(x, y, z, rot=None):
    return np.array([x, y, z], dtype=np.float64), rot or IDENTITY_ROT


class ControllerCalibStateTest(unittest.TestCase):
    def test_construction_uses_default_drop_z(self):
        calib = ControllerCalibState(default_drop_z=0.7)
        self.assertEqual(calib.operator_drop_z, 0.7)
        np.testing.assert_allclose(calib._last_pelvis_yaw_quat, IDENTITY_QUAT)

    def test_capture_valid_sets_operator_drop_z(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        head_pos, _ = _pose(0.0, 0.0, 1.72)
        calib.capture(head_pos)
        self.assertAlmostEqual(calib.operator_drop_z, 1.72)

    def test_capture_out_of_range_falls_back_to_default(self):
        calib = ControllerCalibState(default_drop_z=0.65,
                                      drop_z_min=-0.5, drop_z_max=3.0)
        head_pos, _ = _pose(0.0, 0.0, 5.0)  # out of range
        calib.capture(head_pos)
        self.assertEqual(calib.operator_drop_z, 0.65)

    def test_capture_nan_falls_back_to_default(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        calib.capture(np.array([0.0, 0.0, float("nan")]))
        self.assertEqual(calib.operator_drop_z, 0.65)

    def test_capture_resets_last_pelvis_yaw_quat(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        calib._last_pelvis_yaw_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
        head_pos, _ = _pose(0.0, 0.0, 1.72)
        calib.capture(head_pos)
        np.testing.assert_allclose(calib._last_pelvis_yaw_quat, IDENTITY_QUAT)

    def test_reset_filters_preserve_yaw_true_keeps_last_yaw(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        stored = np.array([0.7071, 0.0, 0.0, 0.7071])
        calib._last_pelvis_yaw_quat = stored.copy()
        calib.reset_filters(preserve_yaw=True)
        np.testing.assert_allclose(calib._last_pelvis_yaw_quat, stored)

    def test_reset_filters_preserve_yaw_false_resets_to_identity(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        calib._last_pelvis_yaw_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
        calib.reset_filters(preserve_yaw=False)
        np.testing.assert_allclose(calib._last_pelvis_yaw_quat, IDENTITY_QUAT)

    def test_process_output_shape(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        l_pos, l_rot = _pose(0.2, 0.2, 1.0)
        r_pos, r_rot = _pose(0.2, -0.2, 1.0)
        h_pos, h_rot = _pose(0.0, 0.0, 1.7)
        row = calib.process(l_pos, l_rot, r_pos, r_rot, h_pos, h_rot, dt=0.0)
        self.assertEqual(row.shape, (3, 7))

    def test_process_rejects_negative_dt(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        l_pos, l_rot = _pose(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            calib.process(l_pos, l_rot, l_pos, l_rot, l_pos, l_rot, dt=-0.01)

    def test_process_rejects_nonfinite_pos(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        bad_pos = np.array([0.0, 0.0, float("inf")])
        with self.assertRaises(ValueError):
            calib.process(bad_pos, IDENTITY_ROT,
                          bad_pos, IDENTITY_ROT,
                          bad_pos, IDENTITY_ROT, dt=0.05)

    def test_process_before_capture_uses_default_drop_z(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        l_pos, l_rot = _pose(0.2, 0.2, 1.0)
        r_pos, r_rot = _pose(0.2, -0.2, 1.0)
        h_pos, h_rot = _pose(0.0, 0.0, 0.65)
        row = calib.process(l_pos, l_rot, r_pos, r_rot, h_pos, h_rot, dt=0.0)
        # With operator_drop_z = default = 0.65 and head_z = 0.65, pelvis is
        # at z=0; wrist row z component should be near controller's z (1.0).
        self.assertGreater(row[0, 2], 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test — confirm failure**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_state.py
```

Expected: `ImportError: cannot import name 'ControllerCalibState'`.

- [ ] **Step 3: Implement `ControllerCalibState`**

In `gear_sonic/scripts/pico_manager_thread_server.py`, insert immediately after `yaw_only`:

```python
class ControllerCalibState:
    """Owns the virtual-pelvis estimate, headset filters, and singular-yaw fallback
    for the controller-only VR_3PT path. ThreePointPose stays mode-agnostic;
    this class is the bridge between absolute-XR poses and pelvis-relative input.
    """

    IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

    def __init__(self,
                 orn_tau_s: float = 1.0,
                 z_tau_s: float = 0.3,
                 default_drop_z: float = 0.65,
                 drop_z_min: float = -0.5,
                 drop_z_max: float = 3.0):
        self.default_drop_z = float(default_drop_z)
        self.operator_drop_z = float(default_drop_z)
        self.drop_z_min = float(drop_z_min)
        self.drop_z_max = float(drop_z_max)
        self.orn_filter = HeadsetOrientationFilter(tau_s=orn_tau_s)
        self.z_filter = HeadsetZFilter(tau_s=z_tau_s)
        self._last_pelvis_yaw_quat = self.IDENTITY_QUAT.copy()

    def capture(self, head_pos_r: np.ndarray) -> None:
        """Capture session operator drop-Z from robot-frame head position.

        Assumes G1 pelvis reference Z = 0 in the robot/policy frame.
        If a future robot has nonzero pelvis_ref_z, change to:
            operator_drop_z = head_pos_r[2] - g1_pelvis_ref_z
        Called from the startup calibration flow (OFF -> PLANNER) only.
        """
        z = float(head_pos_r[2])
        if not np.isfinite(z) or z < self.drop_z_min or z > self.drop_z_max:
            print(f"[ControllerCalibState] WARNING: captured head_z={z:.2f} m "
                  f"outside [{self.drop_z_min}, {self.drop_z_max}]; falling back "
                  f"to default {self.default_drop_z:.2f} m.")
            self.operator_drop_z = self.default_drop_z
        else:
            self.operator_drop_z = z
            print(f"[ControllerCalibState] captured operator_drop_z = {z:.3f} m")
        self.reset_filters(preserve_yaw=False)

    def reset_filters(self, preserve_yaw: bool = True) -> None:
        """Reset filter state.

        preserve_yaw=True  (default): used by watchdog re-acquire and VR_3PT entry.
                                       Last good pelvis yaw is kept as the
                                       singular-case fallback for yaw_only.
        preserve_yaw=False:            used by capture() (startup calibration flow).
                                       Pelvis yaw resets to identity — explicit
                                       re-anchor of the session.
        """
        self.orn_filter.reset()
        self.z_filter.reset()
        if not preserve_yaw:
            self._last_pelvis_yaw_quat = self.IDENTITY_QUAT.copy()

    def process(self,
                l_ctrl_pos_r: np.ndarray, l_ctrl_rot_r: sRot,
                r_ctrl_pos_r: np.ndarray, r_ctrl_rot_r: sRot,
                head_pos_r: np.ndarray, head_rot_r: sRot,
                dt: float) -> np.ndarray:
        """All inputs already in robot frame (openxr_unitree applied upstream).
        Returns (3, 7) pelvis-relative row [x, y, z, qw, qx, qy, qz] per point.
        """
        for name, pos, rot in [
            ("l_ctrl", l_ctrl_pos_r, l_ctrl_rot_r),
            ("r_ctrl", r_ctrl_pos_r, r_ctrl_rot_r),
            ("head",   head_pos_r,   head_rot_r),
        ]:
            if pos.shape != (3,):
                raise ValueError(f"{name} pos malformed: shape={pos.shape}")
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"{name} pos contains non-finite values: {pos}")
            if not hasattr(rot, "as_quat"):
                raise ValueError(f"{name} rot must be a scipy.Rotation (or "
                                 f"compatible); got {type(rot)}")
            q = rot.as_quat(scalar_first=True)
            if not np.all(np.isfinite(q)):
                raise ValueError(f"{name} quaternion contains non-finite: {q}")
            q_norm = float(np.linalg.norm(q))
            if q_norm < 0.5 or q_norm > 2.0:
                raise ValueError(f"{name} quaternion norm = {q_norm} far from 1")
        if dt < 0.0:
            raise ValueError(f"negative dt={dt}")

        head_quat_wxyz = head_rot_r.as_quat(scalar_first=True)
        head_quat_filt = self.orn_filter.update(head_quat_wxyz, dt)
        head_z_filt = self.z_filter.update(float(head_pos_r[2]), dt)

        pelvis_pos = np.array([head_pos_r[0], head_pos_r[1],
                                head_z_filt - self.operator_drop_z],
                               dtype=np.float64)
        pelvis_quat = yaw_only(head_quat_filt,
                                fallback_quat=self._last_pelvis_yaw_quat)
        self._last_pelvis_yaw_quat = pelvis_quat

        row = np.zeros((3, 7), dtype=np.float32)
        r_pelvis_inv = sRot.from_quat(pelvis_quat, scalar_first=True).inv()
        l_quat = l_ctrl_rot_r.as_quat(scalar_first=True)
        r_quat = r_ctrl_rot_r.as_quat(scalar_first=True)
        for i, (p_pos, p_quat) in enumerate([
            (l_ctrl_pos_r, l_quat),
            (r_ctrl_pos_r, r_quat),
            (head_pos_r,   head_quat_filt),
        ]):
            row[i, :3] = r_pelvis_inv.apply(p_pos - pelvis_pos)
            row[i, 3:] = (r_pelvis_inv *
                          sRot.from_quat(p_quat, scalar_first=True)
                         ).as_quat(scalar_first=True)
        return row
```

- [ ] **Step 4: Run the test — confirm pass**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_state.py
```

Expected: `Ran 11 tests in ... OK`.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/tests/test_controller_calib_state.py gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): add ControllerCalibState (virtual-pelvis bridge)

Owns operator_drop_z + two filters + singular-yaw fallback for the
controller-only VR_3PT path. capture() sets operator_drop_z from a
robot-frame headset position (with sanity bounds + default fallback)
and resets filters with preserve_yaw=False. reset_filters() with
preserve_yaw=True is for watchdog re-acquire and VR_3PT entry.
process() validates input shape, finiteness, and quaternion norm
explicitly via raise ValueError (not assert), then produces
upstream-shape pelvis-relative (3, 7) rows.
EOF
)"
```

---

## Task 5: Head-relative regression guard test

**Files:**
- Create: `gear_sonic/tests/test_controller_calib_pelvis_anchoring.py`

This task adds a regression test that catches the head-relative wrist subtraction bug independently of `ThreePointPose`. The implementation in Task 4 already produces pelvis-relative output, so the test will pass immediately — its job is to prevent future regression.

- [ ] **Step 1: Write the regression test**

Create `gear_sonic/tests/test_controller_calib_pelvis_anchoring.py`:

```python
"""Regression guard: ControllerCalibState.process must produce pelvis-relative
rows, NOT head-relative rows.

Pelvis-relative output uses pelvis_z = head_z - operator_drop_z. When the
operator captures at head_z = 1.7 m, pelvis sits at z=0; a controller at
world z=1.0 m produces wrist_z = +1.0 in pelvis frame.

Head-relative output (the old bug) would subtract head_pos directly:
wrist_z = 1.0 - 1.7 = -0.7. Sign and magnitude both differ; the assertion
distinguishes them cleanly.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.scripts.pico_manager_thread_server import ControllerCalibState


IDENTITY_ROT = sRot.identity()


def _pos(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


class PelvisAnchoringTest(unittest.TestCase):
    def test_output_is_pelvis_relative_not_head_relative(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        head_pos = _pos(0.0, 0.0, 1.7)
        l_pos = _pos(0.2, +0.2, 1.0)
        r_pos = _pos(0.2, -0.2, 1.0)
        calib.capture(head_pos)  # operator_drop_z := 1.7
        row = calib.process(l_pos, IDENTITY_ROT,
                            r_pos, IDENTITY_ROT,
                            head_pos, IDENTITY_ROT, dt=0.0)
        # Pelvis-relative wrist Z is well above zero (controller above pelvis).
        # Head-relative bug would produce ~ -0.7 here.
        self.assertGreater(row[0, 2], 0.5,
                            f"left wrist z = {row[0,2]:.3f} suggests head-relative")
        self.assertGreater(row[1, 2], 0.5,
                            f"right wrist z = {row[1,2]:.3f} suggests head-relative")

    def test_head_lateral_motion_propagates_to_pelvis_frame(self):
        calib = ControllerCalibState(default_drop_z=0.65)
        calib.capture(_pos(0.0, 0.0, 1.7))
        l_pos = _pos(0.2, +0.2, 1.0)
        r_pos = _pos(0.2, -0.2, 1.0)
        # Frame A: head at origin xy
        head_a = _pos(0.0, 0.0, 1.7)
        row_a = calib.process(l_pos, IDENTITY_ROT,
                              r_pos, IDENTITY_ROT,
                              head_a, IDENTITY_ROT, dt=0.0)
        # Frame B: head shifts +0.1 m in X; wrists held fixed in world frame
        head_b = _pos(0.1, 0.0, 1.7)
        row_b = calib.process(l_pos, IDENTITY_ROT,
                              r_pos, IDENTITY_ROT,
                              head_b, IDENTITY_ROT, dt=0.05)
        # Pelvis tracks head xy, so wrists in pelvis frame should shift
        # opposite the head — wrist x in pelvis frame should decrease.
        self.assertLess(row_b[0, 0], row_a[0, 0] - 0.05,
                         "left wrist x in pelvis frame should track head motion")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test — should PASS immediately (Task 4 code is correct)**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_pelvis_anchoring.py
```

Expected: `Ran 2 tests in ... OK`.

If the test fails, the `ControllerCalibState.process` implementation from Task 4 has a frame error and must be re-examined before proceeding.

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/tests/test_controller_calib_pelvis_anchoring.py
git commit -m "$(cat <<'EOF'
test(pico_manager): regression guard for head-relative wrist subtraction

Two tests that catch the specific bug class regardless of ThreePointPose
behavior: (1) wrist Z must be positive when the controller is above the
virtual pelvis (pelvis-relative anchoring), not negative as would happen
under head-relative subtraction. (2) Lateral head motion must propagate
to the pelvis frame so wrists in pelvis frame shift opposite the head.
EOF
)"
```

---

## Task 6: Rewire wrapper + manager loop atomically (no broken intermediate)

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` — rewrite `_process_controller_3pt_pose`, update `ControllerPoseReader._run`, instantiate `ControllerCalibState` in `run_pico_manager`, thread it through `PlannerStreamer`, rewire startup calibration flow + VR_3PT entry + watchdog warn→ok reset.

**Why merged:** the new `_process_controller_3pt_pose` signature requires a `controller_calib` reference and a per-sample `dt`. The `sample["vr_3pt_pose_np"]` field disappears as a result, and every downstream consumer (`PlannerStreamer.run_once`, the OFF→PLANNER calibration block, the posture-marker logger, the entry gate) must be migrated in the same step. Splitting this across multiple tasks would commit a known-broken intermediate state.

**Frame consistency:** the controller-only path uses `openxr_unitree` as the **single** basis convention. The CLI flags for `controller_pose_convention` / `headset_pose_convention` / `headset_orientation_convention` remain accepted for backward compatibility and non-controller debug paths, but they are ignored by the controller-only VR_3PT processing path. `_process_controller_3pt_pose`, startup drop-Z capture, entry-gate target construction, and per-frame streaming all hardcode `openxr_unitree`. Only legacy `--*_controller_offset_rpy` remains active as a controller-specific bias.

**device_dt source:** `PlannerStreamer.run_once` computes `device_dt` from `sample["sample_monotonic_ns"]` (stamped by `ControllerPoseReader._run` per the existing XR-staleness safety stack), not from manager-loop wall time. Filter behavior tracks XR sample cadence; if the manager loop runs faster than XR samples arrive, `device_dt` is 0 and the filter idempotently returns the last state.

- [ ] **Step 1: Locate insertion points**

```bash
grep -n "^def _process_controller_3pt_pose\|class ControllerPoseReader\|^def run_pico_manager\|class PlannerStreamer\|^class XRStalenessWatchdog\|vr_3pt_pose_np" /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/scripts/pico_manager_thread_server.py
```

This produces the line numbers you'll need for every edit in this task. Keep the output handy.

- [ ] **Step 2: Replace `_process_controller_3pt_pose` body**

In `gear_sonic/scripts/pico_manager_thread_server.py`, replace the entire `_process_controller_3pt_pose` function (from `def _process_controller_3pt_pose(...)` through its closing `return ...` statement) with:

```python
def _process_controller_3pt_pose(
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    headset_pose: np.ndarray,
    dt: float,
    controller_calib: ControllerCalibState,
    left_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a pelvis-relative VR 3-point pose from headset + two controller XR poses.

    Single place that converts full controller samples from XR to robot frame.
    The startup calibration flow (manager loop) calls _pose_xr_to_robot
    separately on the headset for operator_drop_z capture — that is the only
    other place Q_openxr is applied to controller-mode data.

    Wrist orientation bias (--*_controller_offset_rpy) is applied here, BEFORE
    pelvis-relative transformation, to controller wrist quats only. Positions
    and the headset are untouched by the bias.
    """
    if controller_calib is None:
        raise ValueError("controller_calib is required for the controller-only "
                         "VR_3PT path; cannot be None")

    # Apply the single controller-only basis to each pose. Do not use the
    # CLI convention flags here: controller-only VR_3PT is defined in
    # openxr_unitree end-to-end so drop-Z capture, entry gate, and streaming
    # all operate in the same robot frame. _pose_xr_to_robot returns
    # (pos: ndarray(3,), rot: scipy.Rotation).
    l_pos, l_rot = _pose_xr_to_robot(
        left_controller_pose, scalar_first=False,
        convention="openxr_unitree",
    )
    r_pos, r_rot = _pose_xr_to_robot(
        right_controller_pose, scalar_first=False,
        convention="openxr_unitree",
    )
    head_pos, head_rot = _pose_xr_to_robot(
        headset_pose, scalar_first=False,
        convention="openxr_unitree",
    )

    # Legacy wrist orientation bias — orientation only, controllers only.
    if any(a != 0.0 for a in left_controller_offset_rpy):
        l_rot = l_rot * sRot.from_euler("xyz", left_controller_offset_rpy, degrees=True)
    if any(a != 0.0 for a in right_controller_offset_rpy):
        r_rot = r_rot * sRot.from_euler("xyz", right_controller_offset_rpy, degrees=True)

    return controller_calib.process(
        l_pos, l_rot,
        r_pos, r_rot,
        head_pos, head_rot,
        dt,
    )
```

- [ ] **Step 3: Update `ControllerPoseReader._run` to carry raw XR poses only**

In `ControllerPoseReader._run`, locate the existing call `vr_3pt_pose = _process_controller_3pt_pose(...)` and the subsequent `sample = {... "vr_3pt_pose_np": vr_3pt_pose ...}` build. Replace with a raw-pose-only sample dict. Preserve every existing non-derived key that downstream code still uses; at minimum the migrated sample must include:

- `left_controller_pose_xrt`, `right_controller_pose_xrt`, `headset_pose_xrt`
- `sample_monotonic_ns`
- any existing wall-clock timestamp key used by debug logging
- any existing SMPL/body-pose keys used when `controller_3pt` is false
- convention debug fields used by `Controller3PtCalibrationLogger` or hardware-log reports

Do not keep derived `vr_3pt_pose_np`; every consumer must build it from the raw poses in this same task.

```python
# The pelvis-relative (3, 7) row is produced downstream by
# PlannerStreamer.run_once, which owns the ControllerCalibState reference
# and computes device_dt from consecutive sample timestamps.
sample = {
    "left_controller_pose_xrt": left_pose,
    "right_controller_pose_xrt": right_pose,
    "headset_pose_xrt": head_pose,
    "sample_monotonic_ns": now_monotonic_ns,
    # Preserve existing non-derived keys such as timestamp_ns, body_poses_np,
    # body_joints_local, controller_pose_convention_debug, and any logger
    # metadata already present in the sample.
}
```

Remove the `vr_3pt_pose_np` key from `sample` entirely. Other consumers are rewired below in this same task.

- [ ] **Step 4: Add temporary local defaults + instantiate `ControllerCalibState` in `run_pico_manager`**

In `run_pico_manager`, just before the main `while True:` loop (and after the existing `xr_watchdog = XRStalenessWatchdog(...)` line), add:

```python
# Temporary local defaults for the new flags. Task 9 wires these to
# argparse; for this task we hardcode the spec defaults so the manager
# can run end-to-end immediately after this commit.
operator_drop_z = 0.65
operator_drop_z_min = -0.5
operator_drop_z_max = 3.0
headset_orn_lowpass_s = 1.0
headset_z_lowpass_s = 0.3

controller_calib = ControllerCalibState(
    orn_tau_s=headset_orn_lowpass_s,
    z_tau_s=headset_z_lowpass_s,
    default_drop_z=operator_drop_z,
    drop_z_min=operator_drop_z_min,
    drop_z_max=operator_drop_z_max,
)

# Convention compatibility notice: in controller-only mode, convention
# CLI flags are accepted but ignored. The controller-only VR_3PT path
# hardcodes openxr_unitree end-to-end for frame consistency.
if controller_3pt and (
    controller_pose_convention != "openxr_unitree"
    or headset_pose_convention != "openxr_unitree"
    or headset_orientation_convention != "openxr_unitree"
):
    print("[Manager] WARNING: controller-only VR_3PT ignores "
          "--controller_pose_convention, --headset_pose_convention, and "
          "--headset_orientation_convention; using openxr_unitree for "
          "controllers, headset position, and headset orientation.")
```

- [ ] **Step 5: Thread `controller_calib` into `PlannerStreamer`**

In `PlannerStreamer.__init__`, add the new kwarg (default None so non-controller_3pt paths still construct):

```python
def __init__(
    self,
    # ... existing args ...
    controller_calib: ControllerCalibState | None = None,
    # ... rest of existing args ...
):
    # ... existing init body ...
    self.controller_calib = controller_calib
    self._last_sample_ns: int | None = None
```

In `run_pico_manager`, where `PlannerStreamer(...)` is instantiated, add `controller_calib=controller_calib` to the kwargs.

- [ ] **Step 6: Rewire `PlannerStreamer.run_once` to use the new wrapper + sample-derived `device_dt`**

In `PlannerStreamer.run_once`, locate the block that reads `sample["vr_3pt_pose_np"]`. Replace with:

```python
if self.controller_3pt:
    # device_dt comes from consecutive XR sample timestamps, not manager
    # wall time. Filter behavior tracks XR sample cadence; if the
    # manager runs faster than samples arrive, device_dt is 0 and the
    # filter idempotently returns the last state.
    sample_ns = sample.get("sample_monotonic_ns")
    if self._last_sample_ns is None or sample_ns is None:
        device_dt = 0.0
    else:
        device_dt = max(0.0, (sample_ns - self._last_sample_ns) / 1e9)
    self._last_sample_ns = sample_ns

    raw_vr_3pt_pose = _process_controller_3pt_pose(
        sample["left_controller_pose_xrt"],
        sample["right_controller_pose_xrt"],
        sample["headset_pose_xrt"],
        dt=device_dt,
        controller_calib=self.controller_calib,
        left_controller_offset_rpy=self.left_controller_offset_rpy,
        right_controller_offset_rpy=self.right_controller_offset_rpy,
    )
    vr_3pt_pose = self.three_point.process_vr_3pt_pose(raw_vr_3pt_pose)
    # ... existing freeze-target / posture-marker / logger logic now
    # consumes raw_vr_3pt_pose and vr_3pt_pose (both local variables) ...
```

For `Controller3PtCalibrationLogger.log(...)` and `posture_marker.advance()` call sites that previously read `sample["vr_3pt_pose_np"]`, pass `raw_vr_3pt_pose` from the local variable instead.

- [ ] **Step 7: Rewire the startup calibration flow with hardcoded `openxr_unitree` for drop-Z**

Locate the `OFF → PLANNER` transition (search for `start_combo and not prev_start_combo`). Replace the existing controller-3pt branch with:

```python
elif current_mode == StreamMode.OFF:
    if start_combo and not prev_start_combo:
        new_mode = StreamMode.PLANNER
        # Startup calibration flow. NVlabs docs call this CALIB_FULL on
        # the first activation after a fresh ThreePointPose.
        sample = reader.get_latest()
        if sample is not None:
            if controller_3pt:
                # Hardcoded openxr_unitree: drop-Z must be captured in
                # the SAME frame that _process_controller_3pt_pose uses
                # per-frame. The CLI convention flags exist for backward
                # compatibility but the controller-only design assumes
                # openxr_unitree end-to-end.
                head_pos_r, _ = _pose_xr_to_robot(
                    sample["headset_pose_xrt"],
                    scalar_first=False,
                    convention="openxr_unitree",
                )
                controller_calib.capture(head_pos_r)

                pelvis_relative = _process_controller_3pt_pose(
                    sample["left_controller_pose_xrt"],
                    sample["right_controller_pose_xrt"],
                    sample["headset_pose_xrt"],
                    dt=0.0,
                    controller_calib=controller_calib,
                    left_controller_offset_rpy=left_controller_offset_rpy,
                    right_controller_offset_rpy=right_controller_offset_rpy,
                )
                three_point.calibrate_vr_3pt_now(pelvis_relative)
                if controller_3pt_logger is not None:
                    controller_3pt_logger.log(
                        "initial_calibration",
                        sample,
                        pelvis_relative,
                        three_point.process_vr_3pt_pose(pelvis_relative),
                        three_point,
                        None,
                    )
            else:
                three_point.calibrate_now(sample["body_poses_np"])
        else:
            print("[Manager] WARNING: No Pico data available for calibration")
```

- [ ] **Step 8: Rewire `check_vr3pt_entry_mismatch()` for raw XR samples**

Locate `PlannerStreamer.check_vr3pt_entry_mismatch()`. If it currently reads `sample["vr_3pt_pose_np"]`, replace that read with the same raw-pose wrapper path used by `run_once`.

Use the latest raw sample from the reader, compute `gate_dt` from `sample["sample_monotonic_ns"]` using the same `self._last_sample_ns` convention as `run_once` (0.0 if unavailable), then build the candidate target:

```python
raw_vr_3pt_pose = _process_controller_3pt_pose(
    sample["left_controller_pose_xrt"],
    sample["right_controller_pose_xrt"],
    sample["headset_pose_xrt"],
    dt=gate_dt,
    controller_calib=self.controller_calib,
    left_controller_offset_rpy=self.left_controller_offset_rpy,
    right_controller_offset_rpy=self.right_controller_offset_rpy,
)
candidate_vr_3pt_pose = self.three_point.process_vr_3pt_pose(raw_vr_3pt_pose)
```

The gate must compare `candidate_vr_3pt_pose` against FK using the exact same `ControllerCalibState` state and `openxr_unitree` wrapper that the first published VR_3PT frame will use. Do not use cached `sample["vr_3pt_pose_np"]`; that key is removed.

- [ ] **Step 9: VR_3PT entry — `reset_filters` BEFORE the gate**

Locate the mode-transition block where `new_mode == StreamMode.PLANNER_VR_3PT`. The existing code calls `planner_streamer.check_vr3pt_entry_mismatch()` first. Restructure:

```python
if new_mode == StreamMode.PLANNER_VR_3PT and current_mode != StreamMode.PLANNER_VR_3PT:
    # CRITICAL ORDERING: reset filters BEFORE the gate so the gate
    # evaluates with the same ControllerCalibState state that the
    # first VR_3PT frame will use. Accepted side effect: a refused
    # gate leaves filters reset — documented in the spec.
    if controller_calib is not None:
        controller_calib.reset_filters(preserve_yaw=True)
    if not planner_streamer.check_vr3pt_entry_mismatch():
        new_mode = current_mode  # gate refused; stay in current mode
```

The existing post-transition block that calls `recalibrate_for_vr3pt()` + `start_vr3pt_ramp()` runs only if the gate passed (the early `new_mode = current_mode` short-circuits the transition).

- [ ] **Step 10: Watchdog `warn → ok` filter reset**

Locate the `xr_watchdog.poll(...)` block. Add a previous-state tracker before the main `while True:` loop:

```python
prev_xr_state = "idle"
```

Inside the loop, after the `xr_state = xr_watchdog.poll(...)` call, add:

```python
if prev_xr_state == "warn" and xr_state == "ok":
    # XR re-acquired. Reset filters; do NOT trigger recalibration
    # (per NVlabs docs: wrist CALIB only on VR_3PT entry, not on
    # tracking blips).
    if controller_calib is not None:
        controller_calib.reset_filters(preserve_yaw=True)
prev_xr_state = xr_state
```

`estop` is terminal (manager exits), so there is no `estop → ok` transition within a session.

- [ ] **Step 11: Syntax check + run all tests**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python -m py_compile gear_sonic/scripts/pico_manager_thread_server.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_headset_filters.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_yaw_only.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_state.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_pelvis_anchoring.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_xr_staleness_watchdog.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_vr3pt_entry_gate.py
```

Expected: silent compile + all 6 test files PASS. If `test_xr_staleness_watchdog.py` or `test_vr3pt_entry_gate.py` fail with an import error caused by the function-signature change, that indicates a downstream consumer was missed; locate and fix before committing.

- [ ] **Step 12: Stage and commit (or stage only if you batch commits — see "Working-tree conventions")**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
refactor(pico_manager): atomically rewire wrapper + manager loop

Single atomic change so no broken intermediate state is committed:

- _process_controller_3pt_pose: thin wrapper applying hardcoded openxr_unitree +
  optional wrist orientation bias, delegating to
  ControllerCalibState.process. Head-relative wrist subtraction
  removed.

- ControllerPoseReader._run: emits raw XR poses only;
  sample["vr_3pt_pose_np"] removed.

- run_pico_manager: instantiates ControllerCalibState alongside
  xr_watchdog; warns that controller-only mode ignores convention CLI
  flags and uses openxr_unitree end-to-end for frame consistency.

- PlannerStreamer.run_once and check_vr3pt_entry_mismatch: compute dt from consecutive
  sample["sample_monotonic_ns"] values (XR sample cadence, not manager
  wall time); call _process_controller_3pt_pose directly; thread
  raw_vr_3pt_pose to logger / posture marker / entry-gate call sites.

- Startup calibration flow (OFF → PLANNER): applies _pose_xr_to_robot
  with convention="openxr_unitree" HARDCODED for drop-Z capture so
  operator_drop_z is in the same frame as the per-frame wrapper.

- VR_3PT entry: filter reset runs BEFORE the entry gate so the gate
  evaluates with the same state the first frame will use.

- Watchdog warn → ok transition: filter reset only (no automatic
  recalibration per NVlabs docs).
EOF
)"
```

---

## Task 7: Upstream parity test infrastructure (failing test first)

**Files:**
- Create: `gear_sonic/tests/fixtures/regen_threepoint_golden.py`
- Create: `gear_sonic/tests/fixtures/threepoint_golden.json` (generated)
- Create: `gear_sonic/tests/test_three_point_pose_parity.py`

- [ ] **Step 1: Write the fixture regenerator**

Create `gear_sonic/tests/fixtures/regen_threepoint_golden.py`:

```python
"""Regenerate threepoint_golden.json by running NVlabs upstream ThreePointPose
at the pinned commit in a temporary git worktree.

Usage:
    cd /home/jihun/work/GR00T-WholeBodyControl
    PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/fixtures/regen_threepoint_golden.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Pinned NVlabs upstream commit — see design spec.
UPSTREAM_REMOTE = "https://github.com/NVlabs/GR00T-WholeBodyControl"
UPSTREAM_COMMIT = "4f5e118ae82e0448f2b2f441024d59fe74e0a4d9"
UPSTREAM_FILE = "gear_sonic/scripts/pico_manager_thread_server.py"

OUTPUT_PATH = Path(__file__).parent / "threepoint_golden.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


# Fixed-seed pelvis-relative inputs. Each is a (3, 7) array with rows
# [L_wrist, R_wrist, Neck] and columns [x, y, z, qw, qx, qy, qz].
CALIB_INPUT = [
    [0.20, +0.20, -0.10,  1.0, 0.0, 0.0, 0.0],   # L wrist
    [0.20, -0.20, -0.10,  1.0, 0.0, 0.0, 0.0],   # R wrist
    [0.00,  0.00,  0.40,  1.0, 0.0, 0.0, 0.0],   # Neck (pelvis-relative)
]


def _quat_from_euler(roll_deg, pitch_deg, yaw_deg):
    from scipy.spatial.transform import Rotation as sRot
    r = sRot.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    return r.as_quat(scalar_first=True).tolist()


FRAMES = [
    ("neutral", [
        [0.20, +0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.20, -0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.00,  0.00,  0.40] + _quat_from_euler(0, 0, 0),
    ]),
    ("mild_lean_forward", [
        [0.25, +0.20, -0.05] + _quat_from_euler(0, 10, 0),
        [0.25, -0.20, -0.05] + _quat_from_euler(0, 10, 0),
        [0.00,  0.00,  0.40] + _quat_from_euler(0, 10, 0),
    ]),
    ("left_wrist_twist", [
        [0.20, +0.20, -0.10] + _quat_from_euler(45, 0, 0),
        [0.20, -0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.00,  0.00,  0.40] + _quat_from_euler(0, 0, 0),
    ]),
    ("right_wrist_lift", [
        [0.20, +0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.25, -0.20,  0.05] + _quat_from_euler(0, 30, 0),
        [0.00,  0.00,  0.40] + _quat_from_euler(0, 0, 0),
    ]),
    ("head_yaw_left", [
        [0.20, +0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.20, -0.20, -0.10] + _quat_from_euler(0, 0, 0),
        [0.00,  0.00,  0.40] + _quat_from_euler(0, 0, 20),
    ]),
    ("smpl_neutral_pelvis_rel", [
        [0.18, +0.22,  0.05] + _quat_from_euler(90, 0, 0),
        [0.18, -0.22,  0.05] + _quat_from_euler(-90, 0, 180),
        [0.00,  0.00,  0.35] + _quat_from_euler(0, 0, -90),
    ]),
    ("smpl_arms_raised_pelvis_rel", [
        [0.10, +0.20,  0.30] + _quat_from_euler(90, 30, 0),
        [0.10, -0.20,  0.30] + _quat_from_euler(-90, 30, 180),
        [0.00,  0.00,  0.35] + _quat_from_euler(0, 0, -90),
    ]),
]


def main():
    with tempfile.TemporaryDirectory(prefix="threepoint_golden_") as tmp:
        tmp_path = Path(tmp)
        worktree = tmp_path / "upstream"

        print(f"[regen] cloning upstream into {worktree}")
        subprocess.run([
            "git", "clone", UPSTREAM_REMOTE, str(worktree),
        ], check=True)
        subprocess.run([
            "git", "-C", str(worktree), "checkout", UPSTREAM_COMMIT,
        ], check=True)

        # Verify the upstream file exists at the pinned commit
        upstream_file_path = worktree / UPSTREAM_FILE
        if not upstream_file_path.exists():
            raise SystemExit(f"upstream file missing: {upstream_file_path}")

        # Import upstream ThreePointPose via spec.import_from_file pattern
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_upstream_pmts", str(upstream_file_path)
        )
        if spec is None or spec.loader is None:
            raise SystemExit("failed to build module spec for upstream file")

        # The upstream module imports gear_sonic submodules; we need to put
        # the upstream worktree on sys.path so those imports resolve.
        sys.path.insert(0, str(worktree))
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        ThreePointPose = module.ThreePointPose

        print("[regen] running upstream ThreePointPose on fixed inputs")
        tp = ThreePointPose(enable_vis_vr3pt=False, with_g1_robot=False)
        calib_arr = np.array(CALIB_INPUT, dtype=np.float32)
        tp.calibrate_vr_3pt_now(calib_arr)

        out_frames = []
        for label, raw in FRAMES:
            arr = np.array(raw, dtype=np.float32)
            out = tp.process_vr_3pt_pose(arr)
            out_frames.append({
                "label": label,
                "input": arr.tolist(),
                "expected_upstream": out.tolist(),
            })

        fixture = {
            "_metadata": {
                "source_repo": UPSTREAM_REMOTE,
                "source_commit": UPSTREAM_COMMIT,
                "source_file": UPSTREAM_FILE,
                "source_class": "ThreePointPose",
                "generated_at": "2026-05-27",
                "generator": "gear_sonic/tests/fixtures/regen_threepoint_golden.py",
            },
            "calib_input": calib_arr.tolist(),
            "frames": out_frames,
        }

        OUTPUT_PATH.write_text(json.dumps(fixture, indent=2))
        print(f"[regen] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the regenerator to produce the fixture**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/fixtures/regen_threepoint_golden.py
```

Expected output ends with `[regen] wrote .../threepoint_golden.json` and the file exists.

If the regenerator fails to import upstream `ThreePointPose` (e.g. missing dependencies in the cloned worktree), inspect the error: the upstream module may require `gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer` (for `get_g1_key_frame_poses`). In that case, set the local repo on `PYTHONPATH` and run from there so the helper module resolves locally:

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl .venv_teleop/bin/python \
    gear_sonic/tests/fixtures/regen_threepoint_golden.py
```

- [ ] **Step 3: Write the parity test**

Create `gear_sonic/tests/test_three_point_pose_parity.py`:

```python
"""Verifies our ThreePointPose._apply_calibration produces numerical parity
(np.allclose, atol=1e-7) with the NVlabs upstream version at the pinned SHA.

Failure indicates our local ThreePointPose has diverged from upstream — either
by accident or by needing the divergence to be intentional and documented.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import ThreePointPose


FIXTURE = Path(__file__).parent / "fixtures" / "threepoint_golden.json"


class ThreePointPoseParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(FIXTURE.read_text())
        cls.tp = ThreePointPose(enable_vis_vr3pt=False, with_g1_robot=False)
        cls.tp.calibrate_vr_3pt_now(
            np.array(cls.golden["calib_input"], dtype=np.float32)
        )

    def test_metadata_pinned_sha(self):
        meta = self.golden["_metadata"]
        self.assertEqual(meta["source_commit"],
                          "4f5e118ae82e0448f2b2f441024d59fe74e0a4d9")
        self.assertEqual(meta["source_class"], "ThreePointPose")

    def test_every_frame_matches_upstream(self):
        for case in self.golden["frames"]:
            with self.subTest(label=case["label"]):
                inp = np.array(case["input"], dtype=np.float32)
                out = self.tp.process_vr_3pt_pose(inp)
                expected = np.array(case["expected_upstream"], dtype=np.float32)
                np.testing.assert_allclose(out, expected, atol=1e-7,
                                            err_msg=case["label"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the parity test — expect FAILURE**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_three_point_pose_parity.py
```

Expected: at least the `mild_lean_forward` and `smpl_neutral_pelvis_rel` cases fail because our local `ThreePointPose._apply_calibration` still has the `_calibration_neck_pos_offset` consumption + the `_calibration_wrist_orientation_uses_neck` special case. These are reverted in Task 8.

If the test PASSES at this stage, our local `ThreePointPose` is already upstream-compatible and Task 8 may be a no-op. Verify the test really exercises the divergent code paths before declaring victory.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/tests/fixtures/regen_threepoint_golden.py \
        gear_sonic/tests/fixtures/threepoint_golden.json \
        gear_sonic/tests/test_three_point_pose_parity.py
git commit -m "$(cat <<'EOF'
test(pico_manager): add upstream parity test for ThreePointPose

Pinned to NVlabs commit 4f5e118ae82e0448f2b2f441024d59fe74e0a4d9. The
fixture is generated by a regenerator script that checks out the
upstream SHA in a temporary worktree and runs upstream ThreePointPose
on a fixed set of pelvis-relative inputs (controller-derived AND
SMPL-shaped). The test asserts numerical parity (np.allclose,
atol=1e-7); failure blocks merge.

This test is expected to FAIL until Task 8 reverts our local
ThreePointPose divergences (_calibration_neck_pos_offset and the
_calibration_wrist_orientation_uses_neck special case).
EOF
)"
```

---

## Task 8: Revert `ThreePointPose` to upstream behavior

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` — remove two fields, restore three methods to upstream behavior

This task makes the parity test from Task 7 pass.

- [ ] **Step 1: Locate the `ThreePointPose` class**

```bash
grep -n "^class ThreePointPose" /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/scripts/pico_manager_thread_server.py
```

- [ ] **Step 2: Remove `_calibration_neck_pos_offset` field**

In `ThreePointPose.__init__`, find the line:
```python
self._calibration_neck_pos_offset: np.ndarray | None = None
```
Delete it.

Find all other references:
```bash
grep -n "_calibration_neck_pos_offset" /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/scripts/pico_manager_thread_server.py
```
Each remaining reference is in `_capture_calibration` (capture), `_apply_calibration` (consume), `_clear_calibration` (reset), and a log statement in `_capture_calibration`. Delete all of them.

- [ ] **Step 3: Remove `_calibration_wrist_orientation_uses_neck` field and branches**

In `ThreePointPose.__init__`, delete:
```python
self._calibration_wrist_orientation_uses_neck = True
```

In `reset_with_measured_q`, delete:
```python
self._calibration_wrist_orientation_uses_neck = False
```

In `calibrate_vr_3pt_now`, delete:
```python
self._calibration_wrist_orientation_uses_neck = False
```

In `_capture_calibration`, find the branch:
```python
if self._calibration_wrist_orientation_uses_neck:
    lwrist_rot_corrected = calib_inv_rot * lwrist_rot_raw
    rwrist_rot_corrected = calib_inv_rot * rwrist_rot_raw
else:
    lwrist_rot_corrected = lwrist_rot_raw
    rwrist_rot_corrected = rwrist_rot_raw
```
Replace with the unconditional upstream form:
```python
lwrist_rot_corrected = calib_inv_rot * lwrist_rot_raw
rwrist_rot_corrected = calib_inv_rot * rwrist_rot_raw
```

In `_apply_calibration`, find similar branches and apply the same unconditional pattern.

In `_clear_calibration`, delete:
```python
self._calibration_wrist_orientation_uses_neck = True
```

Verify no remaining references:
```bash
grep -n "_calibration_wrist_orientation_uses_neck" /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/scripts/pico_manager_thread_server.py
```
Expected: no output.

- [ ] **Step 4: Restore `_apply_calibration`'s kinematic-chain neck position**

Locate the section in `_apply_calibration` that computes `calibrated[2, :3]`. In the current local code it reads:

```python
if self._calibration_neck_pos_offset is not None:
    calibrated[2, :3] = (
        calib_inv_rot.apply(vr_3pt_pose[2, :3]) - self._calibration_neck_pos_offset
    ).astype(np.float32)
```

Replace with the upstream kinematic chain (constants `TORSO_LINK_OFFSET_Z` and `NECK_LINK_LENGTH` are already defined on the class):

```python
# Upstream behavior: neck position is synthesized via the kinematic chain,
# NOT from VR-tracked headset translation. The third point's published
# position is purely a function of the calibrated neck orientation's Z axis.
neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
calibrated[2, :3] = (
    np.array([0, 0, self.TORSO_LINK_OFFSET_Z])
    + self.NECK_LINK_LENGTH * neck_z
).astype(np.float32)
```

- [ ] **Step 5: Remove neck-pos-offset capture from `_capture_calibration`**

Locate the line:
```python
self._calibration_neck_pos_offset = neck_pos_corrected - g1_neck_pos
```
Delete it.

Locate the lines that read `g1_poses["torso"]` for the neck position:
```python
g1_neck_pos = g1_poses["torso"]["position"]
```
and the variable `neck_pos_corrected`:
```python
neck_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[2, :3].copy())
```
Both are only used by the line we just deleted. Delete them too.

Also remove the log line that prints the neck pos offset (search for `Neck pos offset:`).

- [ ] **Step 6: Run the parity test — expect PASS**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_three_point_pose_parity.py
```

Expected: all subtests pass.

If any subtest still fails, the diff between our `ThreePointPose._apply_calibration` and upstream is incomplete. Compare the two side-by-side:

```bash
git show 4f5e118:gear_sonic/scripts/pico_manager_thread_server.py | \
    awk '/^class ThreePointPose/,/^class /' > /tmp/upstream_tpp.py
# (then inspect /tmp/upstream_tpp.py and diff against local)
```

- [ ] **Step 7: Verify other tests still pass**

```bash
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_headset_filters.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_yaw_only.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_state.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_pelvis_anchoring.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_xr_staleness_watchdog.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_vr3pt_entry_gate.py
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
fix(pico_manager): revert ThreePointPose to upstream calibration contract

Removes two local divergences from NVlabs upstream:

1. _calibration_neck_pos_offset (field + capture + consume + clear).
   Upstream synthesizes the third point's published position via the
   kinematic chain [0, 0, TORSO_LINK_OFFSET_Z] + NECK_LINK_LENGTH * neck_z.
   Our local code stored a per-operator neck position offset and consumed
   the raw VR-tracked headset translation; this injected headset bob and
   look-around translation into the policy's torso target.

2. _calibration_wrist_orientation_uses_neck (field + reset_with_measured_q
   + calibrate_vr_3pt_now + _capture_calibration branches + _apply_calibration
   branches + _clear_calibration). The flag bypassed the neck-relative
   wrist correction on reset_with_measured_q. Upstream applies the
   neck-relative correction unconditionally.

Upstream-parity test (test_three_point_pose_parity.py) now passes for
all seven fixture frames (controller-derived + SMPL-shaped pelvis-relative
inputs) against NVlabs commit 4f5e118ae82e0448f2b2f441024d59fe74e0a4d9.
EOF
)"
```


## Task 9: CLI flags + legacy bias warning

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py` — argparse section + the temporary variable definitions added in Task 6 Step 4

- [ ] **Step 1: Add new CLI flags**

In `gear_sonic/scripts/pico_manager_thread_server.py`, locate the argparse section (`grep -n "args = parser.parse_args" ...`). Add five new flags before `args = parser.parse_args()`:

```python
parser.add_argument(
    "--operator_drop_z",
    type=float,
    default=0.65,
    help=("Fallback operator drop-Z (meters) used until the startup "
          "calibration flow captures the session value. Default 0.65 m."),
)
parser.add_argument(
    "--operator_drop_z_min",
    type=float,
    default=-0.5,
    help=("Lower impossible-bound for captured operator_drop_z. Captured "
          "values outside [min, max] fall back to --operator_drop_z."),
)
parser.add_argument(
    "--operator_drop_z_max",
    type=float,
    default=3.0,
    help=("Upper impossible-bound for captured operator_drop_z."),
)
parser.add_argument(
    "--headset_orn_lowpass_s",
    type=float,
    default=1.0,
    help=("SLERP-EMA time constant on headset orientation (seconds). "
          "INITIAL default; sweep on hardware to set final value."),
)
parser.add_argument(
    "--headset_z_lowpass_s",
    type=float,
    default=0.3,
    help=("EMA time constant on headset Z position (seconds)."),
)
```

- [ ] **Step 2: Replace the temporary variable definitions with CLI reads**

Locate the temporary block from Task 6 Step 4:

```python
operator_drop_z = 0.65
operator_drop_z_min = -0.5
operator_drop_z_max = 3.0
headset_orn_lowpass_s = 1.0
headset_z_lowpass_s = 0.3
```

Replace with:

```python
operator_drop_z = args.operator_drop_z
operator_drop_z_min = args.operator_drop_z_min
operator_drop_z_max = args.operator_drop_z_max
headset_orn_lowpass_s = args.headset_orn_lowpass_s
headset_z_lowpass_s = args.headset_z_lowpass_s
```

- [ ] **Step 3: Add legacy bias warning at manager startup**

Locate the existing argparse section that defines `--left_controller_offset_rpy` and `--right_controller_offset_rpy`. After `args = parser.parse_args()` (and before `controller_calib` is instantiated), add:

```python
def _warn_if_legacy_offset_nonzero(name, value):
    if any(a != 0.0 for a in value):
        print(f"[Manager] Note: --{name}={tuple(value)} will be applied "
              f"as a pre-calibration bias this run. Calibration absorbs "
              f"the remainder of the controller-to-URDF wrist rotation "
              f"automatically. Set to (0, 0, 0) to rely on calibration alone.")

_warn_if_legacy_offset_nonzero("left_controller_offset_rpy",
                                tuple(args.left_controller_offset_rpy))
_warn_if_legacy_offset_nonzero("right_controller_offset_rpy",
                                tuple(args.right_controller_offset_rpy))
```

- [ ] **Step 4: Syntax check + run unit tests**

```bash
.venv_teleop/bin/python -m py_compile gear_sonic/scripts/pico_manager_thread_server.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_headset_filters.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_controller_calib_state.py
PYTHONPATH=. .venv_teleop/bin/python gear_sonic/tests/test_three_point_pose_parity.py
```

Expected: silent compile + all tests PASS.

- [ ] **Step 5: Verify the new CLI flags are discoverable**

```bash
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --help | \
    grep -E "operator_drop_z|headset_orn_lowpass|headset_z_lowpass"
```

Expected: all five new flags listed.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
feat(pico_manager): add CLI flags for virtual pelvis tuning + legacy warning

New flags:
  --operator_drop_z 0.65               (fallback until startup calibration flow captures)
  --operator_drop_z_min -0.5
  --operator_drop_z_max 3.0
  --headset_orn_lowpass_s 1.0          (SLERP-EMA τ; initial — set by sweep)
  --headset_z_lowpass_s 0.3            (Z EMA τ)

Legacy --left_controller_offset_rpy / --right_controller_offset_rpy
still supported; warns at startup if nonzero, explains that calibration
absorbs the remainder of the controller-to-URDF wrist rotation.
EOF
)"
```

---

## Task 10: Smoke-test extensions

**Files:**
- Modify: `gear_sonic/tests/test_safety_smoke.py` — append three new operator-confirmation drill items

- [ ] **Step 1: Locate the existing smoke test**

```bash
cat /home/jihun/work/GR00T-WholeBodyControl/gear_sonic/tests/test_safety_smoke.py
```

The existing file structure is: imports + a few `@pytest.mark.skipif`-gated tests that prompt the operator for `y/N` confirmation.

- [ ] **Step 2: Append three new drill items**

Append the following to `gear_sonic/tests/test_safety_smoke.py` (before the `if __name__ == "__main__":` block, if present, or at end of file):

```python
@pytest.mark.skipif(
    not _proc_running("pico_manager_thread_server"),
    reason="pico_manager must be running",
)
def test_startup_calibration_re_run_preserves_neck():
    """Trigger STOP, then START. operator_drop_z + wrist offsets re-anchor.
    Neck anchor preserved from first activation (upstream behavior).

    Operator verifies: after the second START, look LEFT. The robot's
    torso target should still reflect the original forward direction
    (anchored at first CALIB_FULL), NOT the new headset orientation.
    """
    print("\n[smoke] Press A+B+X+Y to STOP, then again to START. Look LEFT.")
    print("[smoke] Expected: robot torso reflects original calibration direction.")
    ans = input("[smoke] Did torso target stay at original direction? [y/N]: "
                ).strip().lower()
    assert ans == "y", "Startup-calibration neck-preservation drill failed"


@pytest.mark.skipif(
    not _proc_running("pico_manager_thread_server"),
    reason="pico_manager must be running",
)
def test_vr3pt_entry_filter_reset():
    """From PLANNER, click Left Stick. Console should log filter resets."""
    print("\n[smoke] In PLANNER mode, click Left Stick to enter VR_3PT.")
    print("[smoke] Expected console log: '[ControllerCalibState] orn_filter "
          "reset' or similar, AND [XRStalenessWatchdog] idle -> ok within "
          "the next ~200 ms.")
    ans = input("[smoke] Did you see filter reset + watchdog ok? [y/N]: "
                ).strip().lower()
    assert ans == "y", "VR_3PT entry filter reset drill failed"


@pytest.mark.skipif(
    not _proc_running("pico_manager_thread_server"),
    reason="pico_manager must be running",
)
def test_xr_re_acquire_filter_reset():
    """Cover the headset cameras for ~80 ms (warn band), then uncover.
    Console should log filter reset on warn → ok transition; next
    published target should not show a single-frame jump."""
    print("\n[smoke] In PLANNER_VR_3PT, cover PICO cameras ~80 ms, then "
          "uncover.")
    print("[smoke] Expected: [XRStalenessWatchdog] ok -> warn, then warn -> "
          "ok, AND filter reset log on the warn → ok transition. No visible "
          "VR_3PT target jump.")
    ans = input("[smoke] Did you see warn/ok transitions + no jump? "
                "[y/N]: ").strip().lower()
    assert ans == "y", "XR re-acquire filter reset drill failed"
```

- [ ] **Step 3: Verify the smoke test file syntax-compiles**

```bash
.venv_teleop/bin/python -m py_compile gear_sonic/tests/test_safety_smoke.py
```

Expected: silent success.

- [ ] **Step 4: Commit**

```bash
git add gear_sonic/tests/test_safety_smoke.py
git commit -m "$(cat <<'EOF'
test(safety): add three new VR_3PT calibration smoke-drill items

- Startup-calibration re-run preserves neck: operator triggers
  STOP → START and verifies the torso target keeps the original
  calibration's forward direction.
- VR_3PT entry filter reset: operator confirms filter reset + watchdog
  ok logs after clicking Left Stick.
- XR re-acquire filter reset: operator covers cameras ~80 ms, confirms
  warn → ok transitions + filter reset log + no target jump.

All three are manual-confirmation drill items per the existing pattern.
EOF
)"
```

---

## Task 11: Docs update — standing-only constraint

**Files:**
- Modify: `docs/source/user_guide/real_robot_safety.md`

- [ ] **Step 1: Append the standing-only constraint**

In `docs/source/user_guide/real_robot_safety.md`, locate the section "## Required Preflight" (or the closest equivalent). Add a new bullet under it:

```markdown
- **Operator must stand still during VR_3PT teleop.** The controller-only
  path estimates a virtual pelvis from the headset position; walking
  introduces vertical headset bob (±2 cm per step) that propagates into
  the policy's pelvis-relative wrist targets. A low-pass on headset Z
  reduces the effect (`--headset_z_lowpass_s 0.3` by default) but does
  not eliminate it. Real-robot teleop expects a standing operator.
```

- [ ] **Step 2: Verify Sphinx still builds**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/docs
make html 2>&1 | tail -10
```

Expected: no warnings about `real_robot_safety` rendering.

- [ ] **Step 3: Commit**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
git add docs/source/user_guide/real_robot_safety.md
git commit -m "$(cat <<'EOF'
docs(safety): note standing-only constraint for controller-only VR_3PT

The virtual-pelvis estimate is anchored to the headset; walking
introduces vertical bob into the policy's pelvis-relative targets.
Document this as an operator constraint alongside the existing
preflight items.
EOF
)"
```

---

## Task 12: Hardware-log gate G1 — `head_pos_r.z` distribution

**Files:**
- Create: `docs/superpowers/specs/2026-05-27-hardware-validation/G1_drop_z_distribution.md`

This is an operator-driven validation task: it requires running the manager + deploy + an actual PICO headset for 60 s, recording per-frame `head_pos_r.z`, and committing the resulting report. The implementation isn't considered complete until this report shows pass.

- [ ] **Step 1: Prepare the recording infrastructure**

The recording uses the existing `Controller3PtCalibrationLogger` (jsonl logger). Ensure it is enabled when running the manager:

```bash
mkdir -p outputs/hw_validation/G1
```

- [ ] **Step 2: Run deploy in sim**

In one terminal:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager sim
```

- [ ] **Step 3: Run the manager with logging for 60 s**

In another terminal:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --controller_3pt \
    --controller_pose_convention openxr_unitree \
    --headset_pose_convention openxr_unitree \
    --headset_orientation_convention openxr_unitree \
    --controller_3pt_log_dir outputs/hw_validation/G1 \
    --controller_3pt_log_interval 0.05
```

Operator: trigger the startup calibration flow by pressing A+B+X+Y from OFF (first activation in this manager session — this is what NVlabs docs call CALIB_FULL). Then stand normally facing the robot for 60 s. Press A+B+X+Y again to stop. Note the captured `operator_drop_z` printed by `[ControllerCalibState]`.

- [ ] **Step 4: Analyze the log**

Write a small one-shot analysis script (inline):

```bash
.venv_teleop/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

logs_dir = Path("outputs/hw_validation/G1")
files = sorted(logs_dir.glob("*.jsonl"))
latest = files[-1]
print(f"Reading {latest}")

zs = []
captured_drop_z = None
for line in latest.read_text().splitlines():
    ev = json.loads(line)
    if ev.get("event") != "vr3pt_stream":
        continue
    # head_pos_r.z lives under controller_pose_convention_debug → openxr_unitree
    # → headset_robot_position[2]. Adjust if your logger uses a different key.
    debug = ev.get("controller_pose_convention_debug", {})
    oxr = debug.get("openxr_unitree", {})
    if "headset_robot_position" in oxr:
        zs.append(float(oxr["headset_robot_position"][2]))
    elif "headset_robot_position_for_wrist_relative" in oxr:
        zs.append(float(oxr["headset_robot_position_for_wrist_relative"][2]))

if not zs:
    raise SystemExit("No vr3pt_stream events found with head_pos_r.z")

arr = np.array(zs)
print(f"samples: {len(arr)}")
print(f"min:     {arr.min():.4f} m")
print(f"max:     {arr.max():.4f} m")
print(f"median:  {np.median(arr):.4f} m")
print(f"p10/p90: {np.percentile(arr, 10):.4f} / {np.percentile(arr, 90):.4f} m")
print(f"std:     {arr.std():.4f} m")
print(f"band (p10 to p90): {np.percentile(arr, 90) - np.percentile(arr, 10):.4f} m")
PY
```

- [ ] **Step 5: Write the gate report**

Create `docs/superpowers/specs/2026-05-27-hardware-validation/G1_drop_z_distribution.md`:

```markdown
# G1 Hardware-Log Validation — `head_pos_r.z` Distribution

**Date:** YYYY-MM-DD (fill in)
**Operator:** Your Name
**Session length:** 60 s standing neutral
**Manager flags:** `--controller_3pt --controller_pose_convention openxr_unitree --headset_pose_convention openxr_unitree --headset_orientation_convention openxr_unitree`

## Results

| Metric | Value |
|---|---|
| Samples | (fill in) |
| Min | X.XXXX m |
| Max | X.XXXX m |
| Median | X.XXXX m |
| p10 | X.XXXX m |
| p90 | X.XXXX m |
| Band p10→p90 | X.XXXX m |
| Std | X.XXXX m |
| Captured `operator_drop_z` at first activation (CALIB_FULL) | X.XXXX m |

## Pass criteria

- [ ] All samples within `[--operator_drop_z_min, --operator_drop_z_max] = [-0.5, 3.0]` m.
- [ ] Band p10→p90 ≤ 0.10 m (operator was standing, not walking).
- [ ] Captured `operator_drop_z` within 0.10 m of session median.

## Recommendations

- Default `--operator_drop_z` (currently 0.65 m): change to (fill in) if the observed median is materially different.
- Default `--operator_drop_z_min` / `--operator_drop_z_max`: tighten from `[-0.5, 3.0]` to (fill in) if observed range is much narrower.

## Notes

(Any anomalies, dropouts, tracking quality observations, etc.)
```

- [ ] **Step 6: Commit (report + the analysis output if useful)**

```bash
git add docs/superpowers/specs/2026-05-27-hardware-validation/G1_drop_z_distribution.md
git commit -m "$(cat <<'EOF'
docs(hw-validation): G1 head_pos_r.z distribution report

60-second standing-neutral session. Confirms captured operator_drop_z
sits within session band, all samples are inside the impossible-bounds,
and the band is narrow enough for the virtual-pelvis assumption.
EOF
)"
```

---

## Task 13: Hardware-log gate G2 — forearm-twist axis check

**Files:**
- Create: `docs/superpowers/specs/2026-05-27-hardware-validation/G2_forearm_twist_axis.md`

- [ ] **Step 1: Prepare recording**

```bash
mkdir -p outputs/hw_validation/G2
```

Run deploy in sim (same as Task 12 Step 2) and the manager with logging (same flags as Task 12 Step 3, but pointed at the G2 log dir).

- [ ] **Step 2: Operator procedure**

1. Trigger the startup calibration flow (A+B+X+Y from OFF — first activation in this session, which is what NVlabs docs call CALIB_FULL) with operator in neutral pose, controllers held forward.
2. Enter VR_3PT mode (Left Stick Click).
3. Hold the right controller still in a neutral orientation for ~3 s — this is the "neutral reference" frame.
4. Twist the right controller about its longitudinal axis (forearm pronation/supination) through approximately ±60° back-and-forth, 5 cycles over ~20 s.
5. Stop. Note the time window of pure-twist motion.

- [ ] **Step 3: Analyze the log**

```bash
.venv_teleop/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as sRot

logs_dir = Path("outputs/hw_validation/G2")
latest = sorted(logs_dir.glob("*.jsonl"))[-1]
print(f"Reading {latest}")

# Right wrist orientation from each vr3pt_stream event. Adjust the key
# path if your logger uses different field names.
right_quats_wxyz = []
for line in latest.read_text().splitlines():
    ev = json.loads(line)
    if ev.get("event") != "vr3pt_stream":
        continue
    calib_target = ev.get("vr_3pt_calibrated_target", {})
    right_wrist = calib_target.get("right_wrist", {})
    orn = right_wrist.get("orientation")
    if isinstance(orn, dict) and "quat_wxyz" in orn:
        right_quats_wxyz.append(orn["quat_wxyz"])

if len(right_quats_wxyz) < 100:
    raise SystemExit(f"Too few samples: {len(right_quats_wxyz)}")

print(f"Loaded {len(right_quats_wxyz)} right-wrist quats")

# Pick a 'neutral' reference: average over the first ~3 s (≈60 samples at 20 Hz).
# Use the first 30 samples to be safe.
NEUTRAL_COUNT = 30
rots = [sRot.from_quat(q, scalar_first=True) for q in right_quats_wxyz]
# Mean rotation of neutral block via quat averaging
def mean_quat(quats):
    Q = np.array(quats)
    A = Q.T @ Q
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs[:, -1]  # leading eigenvector

neutral_quat = mean_quat(right_quats_wxyz[:NEUTRAL_COUNT])
r_neutral = sRot.from_quat(neutral_quat, scalar_first=True)

# For each subsequent sample, compute relative rotation R_after * R_neutral.inv()
# and extract its axis. Filter out tiny rotations (angle < 5°).
axes = []
for r_after in rots[NEUTRAL_COUNT:]:
    r_rel = r_after * r_neutral.inv()
    rotvec = r_rel.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle < np.deg2rad(5):
        continue
    axis = rotvec / angle
    axes.append(axis)

if not axes:
    raise SystemExit("No samples with >5° rotation from neutral")

axes_arr = np.array(axes)
print(f"Twisted samples: {len(axes_arr)}")

# Compare each axis to G1 right_wrist_yaw_link local X axis. In the
# pelvis-relative frame the policy sees, this axis is approximately
# [1, 0, 0] when the arm is in the URDF default pose with palm down. The
# exact link X direction depends on FK; for a static analysis we approximate
# the policy's expected longitudinal axis as the link's URDF X. Adjust this
# if your reference frame differs.
LINK_X = np.array([1.0, 0.0, 0.0])

# Use abs(dot) for sign ambiguity (axis vs -axis describe the same rotation).
dots = np.abs(axes_arr @ LINK_X)
print(f"abs(dot) median: {np.median(dots):.4f}")
print(f"abs(dot) p10:    {np.percentile(dots, 10):.4f}")
print(f"abs(dot) p90:    {np.percentile(dots, 90):.4f}")
print(f"required:        > {np.cos(np.deg2rad(10)):.4f}  (cos 10°)")
PY
```

- [ ] **Step 4: Write the gate report**

Create `docs/superpowers/specs/2026-05-27-hardware-validation/G2_forearm_twist_axis.md`:

```markdown
# G2 Hardware-Log Validation — Forearm-Twist Axis Check

**Date:** YYYY-MM-DD (fill in)
**Operator:** Your Name
**Manager flags:** `--controller_3pt --controller_pose_convention openxr_unitree --headset_pose_convention openxr_unitree --headset_orientation_convention openxr_unitree`

## Procedure

1. Startup calibration flow (A+B+X+Y from OFF; first activation = CALIB_FULL per NVlabs docs) with operator in neutral pose, controllers held forward.
2. Enter VR_3PT (Left Stick Click).
3. Hold right controller still ~3 s.
4. Twist right controller about longitudinal axis ±60° for 5 cycles over ~20 s.

## Method

For each post-neutral sample with rotation angle > 5°, compute relative rotation
`R_rel = R_after * R_neutral.inv()`, extract rotation axis, compare to G1
`right_wrist_yaw_link` local X (URDF longitudinal axis) using
`abs(np.dot(axis, link_x))` to allow sign ambiguity.

## Results

| Metric | Value |
|---|---|
| Twisted samples (angle > 5°) | (fill in) |
| abs(dot) median | (fill in) |
| abs(dot) p10 | (fill in) |
| abs(dot) p90 | (fill in) |
| Required (cos 10°) | 0.9848 |

## Pass criteria

- [ ] abs(dot) median > 0.9848 (alignment within 10°).
- [ ] abs(dot) p10 > 0.95 (only mild outliers).

## Conclusion

(`openxr_unitree` confirmed / not confirmed.)

If not confirmed: pause spec, re-derive basis convention. Consider sweeping
`controller_pose_convention` between `xrobotoolkit_unity` and `openxr_unitree`
and re-running this gate.

## Notes

(Any procedural anomalies, tracking quality, partial twists, etc.)
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-27-hardware-validation/G2_forearm_twist_axis.md
git commit -m "$(cat <<'EOF'
docs(hw-validation): G2 forearm-twist axis check

Empirical verification that openxr_unitree maps controller longitudinal
twist to G1 right_wrist_yaw_link's URDF X axis within 10°. Uses relative
rotation from a held-neutral reference and abs(dot) for sign-ambiguous
axis alignment.
EOF
)"
```

---

## Task 14: τ sweep — operator subjective tuning

**Files:**
- Create: `docs/superpowers/specs/2026-05-27-hardware-validation/tau_sweep.md`

- [ ] **Step 1: Run the manager with three τ values**

For each `tau ∈ {0.3, 0.5, 1.0}`, run the manager:

```bash
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --controller_3pt \
    --controller_pose_convention openxr_unitree \
    --headset_pose_convention openxr_unitree \
    --headset_orientation_convention openxr_unitree \
    --headset_orn_lowpass_s 0.3      # change between runs
```

Operator performs a fixed task per run: stand neutral → reach forward → small body lean forward (hold 3 s) → head nod down (head only, body still) → return neutral. Repeat 3 times per τ. Two operators rate on 0–5 scale.

- [ ] **Step 2: Write the sweep report**

Create `docs/superpowers/specs/2026-05-27-hardware-validation/tau_sweep.md`:

```markdown
# τ Sweep — `--headset_orn_lowpass_s` Tuning

**Date:** YYYY-MM-DD (fill in)
**Operators:** Operator A, Operator B
**Task:** stand neutral → reach forward → body lean forward (hold 3 s) → head nod down (head only) → return neutral

## Results

| τ (s) | Op A: responsiveness | Op A: quick-glance suppression | Op B: responsiveness | Op B: quick-glance suppression |
|---|---|---|---|---|
| 0.3 | / 5 | / 5 | / 5 | / 5 |
| 0.5 | / 5 | / 5 | / 5 | / 5 |
| 1.0 | / 5 | / 5 | / 5 | / 5 |

## Selected default

`--headset_orn_lowpass_s = X.X` (fill in).

## Notes

- Slow-nod ambiguity: any τ misreads slow chin-down as torso lean. Briefed
  to operators; not a code defect.
- (Other observations.)
```

- [ ] **Step 3: Update the CLI default in the source if a different value wins**

If the sweep selects something other than 1.0 s, edit `gear_sonic/scripts/pico_manager_thread_server.py`:

```python
parser.add_argument(
    "--headset_orn_lowpass_s",
    type=float,
    default=0.5,                  # CHANGED from 1.0 per τ sweep
    help=("SLERP-EMA time constant on headset orientation (seconds). "
          "Tuned to 0.5 s per τ sweep (2026-05-27)."),
)
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-27-hardware-validation/tau_sweep.md \
        gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "$(cat <<'EOF'
docs(hw-validation): τ sweep report + updated CLI default

Two operators × three τ values × subjective 0-5 ratings on
responsiveness and quick-glance suppression. Default
--headset_orn_lowpass_s updated to (winning value) based on aggregated
scores.
EOF
)"
```

---

## Self-review

**1. Spec coverage check:**

| Spec section | Plan task |
|---|---|
| New helpers (filters, yaw_only) | Tasks 2-3 |
| `ControllerCalibState` | Task 4 |
| Pelvis-anchoring regression test | Task 5 |
| `_process_controller_3pt_pose` rewrite | Task 6 |
| `ThreePointPose` partial revert | Task 8 (after parity-test setup in Task 7) |
| Manager loop wiring (instantiate, startup, gate-order, watchdog) | Task 6 (merged with wrapper rewrite) |
| CLI flags + legacy warning | Task 9 |
| Smoke test extensions | Task 10 |
| `real_robot_safety.md` standing-only constraint | Task 11 |
| Hardware-log gate G1 (`head_pos_r.z`) | Task 12 |
| Hardware-log gate G2 (forearm twist) | Task 13 |
| τ sweep | Task 14 |
| Upstream parity test (numerical) | Task 7 (setup) + Task 8 (passes after revert) |

All spec sections covered.

**2. Placeholder scan:** Pass. No "TBD"/"implement later" in step bodies. The hardware-log report templates contain (fill in) placeholders by design — those are filled in by the operator running the gate, not by the plan author. The fixture JSON's `[...]` are populated by `regen_threepoint_golden.py` at the task's runtime.

**3. Type-consistency check:**

| Symbol | Defined | Used in |
|---|---|---|
| `HeadsetOrientationFilter.update(quat_wxyz, dt) -> np.ndarray` | Task 2 | Task 4 (`ControllerCalibState.process`) |
| `HeadsetZFilter.update(z, dt) -> float` | Task 2 | Task 4 |
| `yaw_only(quat_wxyz, fallback_quat=None) -> np.ndarray` | Task 3 | Task 4 |
| `ControllerCalibState(orn_tau_s, z_tau_s, default_drop_z, drop_z_min, drop_z_max)` | Task 4 | Tasks 6 + 9 (manager wiring + CLI) |
| `ControllerCalibState.capture(head_pos_r: np.ndarray) -> None` | Task 4 | Task 6 (startup flow) |
| `ControllerCalibState.reset_filters(preserve_yaw: bool = True) -> None` | Task 4 | Task 6 (VR_3PT entry, watchdog warn → ok) |
| `ControllerCalibState.process(l_pos, l_rot, r_pos, r_rot, h_pos, h_rot, dt) -> ndarray(3,7)` | Task 4 | Task 6 (`_process_controller_3pt_pose`) |
| `_process_controller_3pt_pose(...)` new signature with `controller_calib` and `dt` | Task 6 | Task 6 (`PlannerStreamer.run_once`, startup flow — same atomic task) |

All signatures consistent across tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-vr3pt-upstream-calibration-alignment-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Pairs well with the TDD-heavy task structure: each task has its own failing-test step that bounds what the subagent must do.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints for review.

Which approach?
