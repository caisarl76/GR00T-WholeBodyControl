# Inspire FTP POSE Stability and Hand Slew Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent zero-angle PICO elbow frames from terminating POSE streaming and limit Inspire FTP hand motion to 1 rad/s per active physical joint.

**Architecture:** Make the shared axis-angle decomposition finite at its mathematical zero and twist-degenerate limits. Keep raw PICO intent unchanged, but make `InspireCommandState` slew its applied simulator target toward fresh commands or fail-open using one symmetric per-motor rate vector.

**Tech Stack:** Python 3.10, NumPy, SciPy Rotation, PyTorch-backed PICO pose utilities, pytest, MuJoCo, YAML, Ruff.

---

## File map

- Create `gear_sonic/tests/test_rotation_conversion.py`: focused numerical regression coverage for zero and degenerate axis-angle decomposition.
- Modify `gear_sonic/trl/utils/rotation_conversion.py`: stable axis-angle and twist normalization.
- Modify `gear_sonic/tests/test_inspire_ftp_zmq.py`: fresh-command slew, reversal, stale-open, and compatibility tests.
- Modify `gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py`: symmetric command slew and legacy keyword aliases.
- Modify `gear_sonic/utils/mujoco_sim/base_sim.py`: prefer the symmetric slew configuration key while accepting the old key.
- Modify `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml`: name the configured 1 rad/s rates as slew rates.
- Modify `docs/inspire-ftp-mujoco-runbook.md`: record expected travel times and operator validation signals.

### Task 1: Make swing/twist decomposition finite

**Files:**
- Create: `gear_sonic/tests/test_rotation_conversion.py`
- Modify: `gear_sonic/trl/utils/rotation_conversion.py:605-618`

- [ ] **Step 1: Write the failing zero-rotation tests**

Create `gear_sonic/tests/test_rotation_conversion.py` with exact-zero, mixed-batch, and 180-degree twist-degenerate cases:

```python
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa


def _xyzw(quaternions_wxyz: np.ndarray) -> np.ndarray:
    return quaternions_wxyz[:, [1, 2, 3, 0]]


def _assert_unit_finite(quaternions: np.ndarray) -> None:
    assert np.all(np.isfinite(quaternions))
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12)


def test_decompose_rotation_aa_maps_zero_to_identity_twist_and_swing():
    twist, swing = decompose_rotation_aa(
        np.zeros((1, 3), dtype=np.float64), np.array([0.0, 1.0, 0.0])
    )

    np.testing.assert_array_equal(twist, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(swing, [[1.0, 0.0, 0.0, 0.0]])


def test_decompose_rotation_aa_keeps_mixed_batch_finite_and_reconstructable():
    rotations = np.array([[0.0, 0.0, 0.0], [0.2, -0.3, 0.1]], dtype=np.float64)
    twist, swing = decompose_rotation_aa(rotations, np.array([0.0, 1.0, 0.0]))

    _assert_unit_finite(twist)
    _assert_unit_finite(swing)
    reconstructed = Rotation.from_quat(_xyzw(twist)) * Rotation.from_quat(_xyzw(swing))
    np.testing.assert_allclose(reconstructed.as_matrix(), Rotation.from_rotvec(rotations).as_matrix(), atol=1e-12)


def test_decompose_rotation_aa_uses_identity_twist_for_degenerate_projection():
    rotations = np.array([[np.pi, 0.0, 0.0]], dtype=np.float64)
    twist, swing = decompose_rotation_aa(rotations, np.array([0.0, 1.0, 0.0]))

    _assert_unit_finite(twist)
    _assert_unit_finite(swing)
    np.testing.assert_allclose(twist, [[1.0, 0.0, 0.0, 0.0]], atol=1e-12)
    reconstructed = Rotation.from_quat(_xyzw(twist)) * Rotation.from_quat(_xyzw(swing))
    np.testing.assert_allclose(reconstructed.as_matrix(), Rotation.from_rotvec(rotations).as_matrix(), atol=1e-12)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider gear_sonic/tests/test_rotation_conversion.py
```

Expected: the zero and mixed tests fail because the current function returns NaN quaternions and emits a divide warning.

- [ ] **Step 3: Implement stable zero limits**

Replace `decompose_rotation_aa()` with a float64-safe calculation that avoids evaluating a zero denominator and falls back to identity twist for a degenerate projection:

```python
def decompose_rotation_aa(rotation_aa, v2):
    rotation_aa = np.asarray(rotation_aa, dtype=np.float64)
    angle = np.linalg.norm(rotation_aa, axis=1)[:, None]
    half_angle_scale = np.full_like(angle, 0.5)
    np.divide(
        np.sin(angle / 2.0),
        angle,
        out=half_angle_scale,
        where=angle > np.finfo(np.float64).eps,
    )
    w = np.cos(angle / 2.0)
    v = half_angle_scale * rotation_aa
    q = np.concatenate([w, v], axis=1)

    v_twist = np.dot(v, v2)[:, None] * v2
    q_twist_raw = np.concatenate([w, v_twist], axis=1)
    twist_norm = np.linalg.norm(q_twist_raw, axis=1)[:, None]
    q_twist = np.zeros_like(q_twist_raw)
    np.divide(
        q_twist_raw,
        twist_norm,
        out=q_twist,
        where=twist_norm > np.finfo(np.float64).eps,
    )
    q_twist[twist_norm[:, 0] <= np.finfo(np.float64).eps, 0] = 1.0

    q_twist_inv = q_twist * np.array([1.0, -1.0, -1.0, -1.0])
    q_swing = quaternion_multiply_np(q_twist_inv, q)
    return q_twist, q_swing
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command again.

Expected: `3 passed`, with no runtime warning.

- [ ] **Step 5: Commit the numerical fix**

```bash
git add gear_sonic/tests/test_rotation_conversion.py gear_sonic/trl/utils/rotation_conversion.py
git commit -m "fix: handle zero PICO axis-angle rotations"
```

### Task 2: Slew fresh and stale Inspire commands symmetrically

**Files:**
- Modify: `gear_sonic/tests/test_inspire_ftp_zmq.py:87-140`
- Modify: `gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py:20-228`
- Modify: `gear_sonic/utils/mujoco_sim/base_sim.py:24-29,169-177`
- Modify: `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml:57-68`

- [ ] **Step 1: Write failing fresh-command slew tests**

Update the fresh-command test and add reversal/alias tests:

```python
def test_command_state_starts_open_and_slews_toward_fresh_command():
    speed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    state = InspireCommandState(stale_after_s=0.25, max_slew_speed=speed)

    startup_left, startup_right = state.advance(now=10.0, dt=0.01)
    np.testing.assert_array_equal(startup_left, OPEN)
    np.testing.assert_array_equal(startup_right, OPEN)

    state.accept(np.zeros(6), np.zeros(6), now=10.0)
    fresh_left, fresh_right = state.advance(now=10.1, dt=0.5)
    expected = OPEN - speed * 0.5
    np.testing.assert_allclose(fresh_left, expected)
    np.testing.assert_allclose(fresh_right, expected)


def test_command_state_slew_reverses_without_overshoot():
    state = InspireCommandState(max_slew_speed=np.ones(6))
    state.accept(np.zeros(6), np.zeros(6), now=0.0)
    closed_left, _ = state.advance(now=0.1, dt=0.4)
    np.testing.assert_allclose(closed_left, 0.6)

    state.accept(OPEN, OPEN, now=0.1)
    reopened_left, reopened_right = state.advance(now=0.2, dt=0.25)
    np.testing.assert_allclose(reopened_left, 0.85)
    np.testing.assert_allclose(reopened_right, 0.85)


def test_command_state_dt_zero_keeps_applied_output_unchanged():
    state = InspireCommandState(max_slew_speed=np.ones(6))
    state.accept(np.zeros(6), np.zeros(6), now=0.0)
    left, right = state.advance(now=0.1, dt=0.0)
    np.testing.assert_array_equal(left, OPEN)
    np.testing.assert_array_equal(right, OPEN)


def test_max_open_speed_remains_a_legacy_alias():
    speed = np.full(6, 0.25)
    state = InspireCommandState(max_open_speed=speed)
    np.testing.assert_array_equal(state.max_slew_speed, speed)

    with pytest.raises(ValueError, match="only one"):
        InspireCommandState(max_slew_speed=speed, max_open_speed=speed)
```

Keep atomic-validation/subscriber tests focused by constructing their state
with `max_slew_speed=np.full(6, 100.0)`. In the stale test, first advance with
a sufficiently large `dt` while the command is fresh so the output reaches
closed, then assert that stale steps reopen by exactly `speed * dt`.

- [ ] **Step 2: Run the ZMQ state tests and verify RED**

Run:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_zmq.py
```

Expected: failures report that `max_slew_speed` is not accepted and fresh commands still jump directly to their target.

- [ ] **Step 3: Implement the symmetric slew and aliases**

In `inspire_ftp_hand.py`, rename the canonical default and accept exactly one keyword:

```python
DEFAULT_MAX_SLEW_SPEED = 1.0 / CLOSED_RADIANS
DEFAULT_MAX_OPEN_SPEED = DEFAULT_MAX_SLEW_SPEED


class InspireCommandState:
    def __init__(
        self,
        *,
        stale_after_s: float = 0.25,
        max_slew_speed: Sequence[float] | NDArray[np.floating] | None = None,
        max_open_speed: Sequence[float] | NDArray[np.floating] | None = None,
    ) -> None:
        if max_slew_speed is not None and max_open_speed is not None:
            raise ValueError("specify only one of max_slew_speed and max_open_speed")
        selected_speed = max_slew_speed if max_slew_speed is not None else max_open_speed
        if selected_speed is None:
            selected_speed = DEFAULT_MAX_SLEW_SPEED
        speed = np.asarray(selected_speed, dtype=np.float64)
        if speed.shape != (6,) or not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
            raise ValueError("max_slew_speed must contain six finite positive values")
        self.max_slew_speed = speed.copy()
        self.max_open_speed = self.max_slew_speed
```

Retain the existing timestamp validation and replace the branch that directly
assigns fresh output with bounded movement toward a selected target:

```python
        fresh = (
            self.last_valid is not None
            and self.last_receive_monotonic is not None
            and timestamp - self.last_receive_monotonic <= self.stale_after_s
        )
        target = self.last_valid if fresh else (OPEN, OPEN)
        maximum_delta = self.max_slew_speed * step
        self.output = tuple(
            current + np.clip(goal - current, -maximum_delta, maximum_delta)
            for current, goal in zip(self.output, target)
        )
```

In `base_sim.py`, import `DEFAULT_MAX_SLEW_SPEED` and instantiate with:

```python
max_slew_speed=self.config.get(
    "INSPIRE_HAND_MAX_SLEW_SPEED",
    self.config.get("INSPIRE_HAND_MAX_OPEN_SPEED", DEFAULT_MAX_SLEW_SPEED),
),
```

Rename the YAML key to `INSPIRE_HAND_MAX_SLEW_SPEED` and retain the six current
normalized values; they already equal `1.0 / CLOSED_RADIANS`.

- [ ] **Step 4: Run the ZMQ and MuJoCo tests and verify GREEN**

Run:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/tests/test_inspire_ftp_mujoco.py
```

Expected: all tests pass. If an existing test was intended to test decoding or
atomic validation rather than speed, inject the explicit high slew rate rather
than weakening the new rate assertions.

- [ ] **Step 5: Commit the slew implementation**

```bash
git add \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/utils/mujoco_sim/base_sim.py \
  gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml
git commit -m "feat: slew Inspire FTP hand targets"
```

### Task 3: Document and verify the complete change

**Files:**
- Modify: `docs/inspire-ftp-mujoco-runbook.md:18-24,125-139`

- [ ] **Step 1: Add operator-visible speed and stability expectations**

Add this text to the command-contract and PICO validation sections:

```markdown
MuJoCo applies a per-motor slew corresponding to 1 rad/s in each active
physical joint. Full travel is approximately 1.44 s for the four fingers,
0.59 s for thumb bend, and 1.16 s for thumb rotation. The ZMQ logger displays
raw PICO intent, while the viewer displays this slew-limited applied motion.

An exact neutral elbow rotation is valid input and must not terminate POSE
streaming. Keep POSE active through a neutral-arm hold for at least 15 seconds
before considering the live validation passed.
```

- [ ] **Step 2: Run formatting and focused tests**

```bash
ruff format --check --no-cache \
  gear_sonic/trl/utils/rotation_conversion.py \
  gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/utils/mujoco_sim/base_sim.py \
  gear_sonic/tests/test_rotation_conversion.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py

ruff check --no-cache \
  gear_sonic/trl/utils/rotation_conversion.py \
  gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/utils/mujoco_sim/base_sim.py \
  gear_sonic/tests/test_rotation_conversion.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py

PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_rotation_conversion.py \
  gear_sonic/tests/test_inspire_ftp_contract.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/tests/test_inspire_ftp_mujoco.py \
  gear_sonic/tests/test_inspire_ftp_bridge.py \
  gear_sonic/tests/test_inspire_ftp_launch.py
```

Expected: Ruff exits zero and the complete Inspire subset passes.

- [ ] **Step 3: Run headless MuJoCo structural and dynamics checks**

```bash
PYTHONPATH=. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py --motor-sweep --duration 2

PYTHONPATH=. \
  /home/jihun/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py \
  --contact-cycle --duration 15 --open-hold 5
```

Expected: both JSON reports contain `"passed": true`; the contact report is
finite and records nonzero contacts.

- [ ] **Step 4: Inspect the final diff and commit documentation**

```bash
git diff --check
git status --short
git diff --stat HEAD~2
git add docs/inspire-ftp-mujoco-runbook.md
git commit -m "docs: add Inspire hand slew validation"
```

- [ ] **Step 5: Hand off the live operator validation**

Do not launch PICO, SONIC, or the MuJoCo viewer automatically. Give the operator
the existing four-terminal commands and require these signals:

1. manager remains in `POSE` through a 15-second neutral-arm hold;
2. no divide warning, zero-quaternion error, or manager shutdown occurs;
3. raw ZMQ values still span `[0, 1]`;
4. visible finger travel matches the documented slew durations;
5. stale/stop behavior returns both hands gradually to open.
