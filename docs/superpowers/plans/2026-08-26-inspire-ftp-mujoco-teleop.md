# Inspire FTP MuJoCo Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive both six-motor Inspire FTP hands from PICO 4 Ultra controls in GEAR-SONIC MuJoCo simulation without changing SONIC's 29-DoF body policy or commanding real hardware.

**Architecture:** Add a hardware-compatible normalized six-motor contract and PICO profile, then route the existing ZMQ hand fields to a Python MuJoCo adapter. The adapter controls six named active joints per hand in a pinned Unitree FTP model and uses equality constraints for the remaining joints; the C++ Dex3 path is disabled whenever this profile is selected.

**Tech Stack:** Python 3.10, NumPy, pyzmq, MuJoCo, pytest, tyro, Bash, existing GEAR-SONIC C++ deploy binary

---

## File Map

| File | Responsibility |
|---|---|
| `gear_sonic/utils/teleop/inspire_ftp.py` | Six-motor order, normalized/physical conversions, PICO trigger/grip mapping, validation |
| `gear_sonic/utils/teleop/zmq/zmq_message_decoder.py` | Shared decoder for `pose` and `planner` binary messages |
| `gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py` | Latest-value subscriber, stale-command state machine, named MuJoCo actuator layout |
| `gear_sonic/scripts/build_inspire_ftp_mujoco_model.py` | Deterministically transplant the pinned official FTP hand model onto the existing SONIC G1 body |
| `gear_sonic/data/robots/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` | Vendored official Unitree source at commit `4ddbf6d` |
| `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_ftp.xml` | Generated 29-body + 24-hand-joint model |
| `gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml` | Floor/light scene including the generated model |
| `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml` | Inspire-specific MuJoCo counts, scene, and joint-name lists |
| `gear_sonic/scripts/pico_manager_thread_server.py` | Select Inspire controller mapper while preserving Dex3 default |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | Install and step the named Inspire plant in Inspire mode |
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` | Make Dex3 DDS hand endpoints optional |
| `gear_sonic/utils/mujoco_sim/configs.py` | Add `hand_profile` and choose the matching sim YAML |
| `gear_sonic/scripts/launch_data_collection.py` | Propagate one hand profile to sim, PICO, and C++ deploy |
| `gear_sonic_deploy/deploy.sh` | Parse and forward `--disable-dex3-hands` |
| `gear_sonic/scripts/verify_inspire_ftp_mujoco.py` | Headless model/count/coupling/per-motor sweep report |
| `gear_sonic/tests/test_inspire_ftp_contract.py` | Contract and controller mapping tests |
| `gear_sonic/tests/test_inspire_ftp_zmq.py` | Decoder, latest-message, and watchdog tests |
| `gear_sonic/tests/test_inspire_ftp_mujoco.py` | Model layout, conversion, and sweep tests |
| `gear_sonic/tests/test_inspire_ftp_launch.py` | Launcher ownership and CLI propagation tests |

### Task 1: Six-motor contract and PICO mapping

**Files:**
- Create: `gear_sonic/utils/teleop/inspire_ftp.py`
- Create: `gear_sonic/tests/test_inspire_ftp_contract.py`

- [ ] **Step 1: Write failing order and endpoint tests**

```python
def test_normalized_endpoints_follow_ftp_convention():
    np.testing.assert_allclose(normalized_to_radians(np.ones(6)), np.zeros(6))
    np.testing.assert_allclose(
        normalized_to_radians(np.zeros(6)),
        [1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641],
    )

def test_pico_trigger_closes_five_bend_motors_and_grip_rotates_thumb():
    np.testing.assert_allclose(map_pico_controls(1.0, 0.0), [0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(map_pico_controls(0.0, 1.0), [1, 1, 1, 1, 1, 0])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_teleop/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_contract.py
```

Expected: collection fails because `gear_sonic.utils.teleop.inspire_ftp` does not exist.

- [ ] **Step 3: Implement the immutable contract**

Implement these public definitions:

```python
MOTOR_NAMES = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
CLOSED_RADIANS = np.array([1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641])
OPEN = np.ones(6, dtype=np.float64)

def validate_normalized(values: ArrayLike) -> np.ndarray: ...
def normalized_to_radians(values: ArrayLike) -> np.ndarray: ...
def radians_to_normalized(values: ArrayLike) -> np.ndarray: ...
def map_pico_controls(trigger_close: float, grip_close: float, *, deadzone: float = 0.05) -> np.ndarray: ...
```

`validate_normalized` requires shape `(6,)`, finite values, and range `[0, 1]` with no silent clipping. `map_pico_controls` validates finite scalar inputs, applies a symmetric endpoint deadzone, and returns `[1-trigger] * 5 + [1-grip]`.

- [ ] **Step 4: Add rejection and round-trip cases, then verify GREEN**

Cover wrong lengths, NaN/Inf, out-of-range values, open/closed/midpoint round trips, and deadzone endpoints. Run the Task 1 command and require all cases to pass.

- [ ] **Step 5: Commit the contract**

```bash
git add gear_sonic/utils/teleop/inspire_ftp.py gear_sonic/tests/test_inspire_ftp_contract.py
git commit -m "feat: define Inspire FTP hand contract"
```

### Task 2: PICO hand profile

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py`
- Test: `gear_sonic/tests/test_inspire_ftp_contract.py`

- [ ] **Step 1: Write a failing profile-selection test**

Exercise `compute_hand_joints_from_inputs(..., hand_profile="inspire_ftp")` with controller trigger/grip values and assert two arrays of shape `(1, 6)`. Also assert the omitted/default profile still returns the current Dex3 `(1, 7)` representation.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: `compute_hand_joints_from_inputs` rejects the new keyword.

- [ ] **Step 3: Add `--hand-profile {dex3,inspire_ftp}` and route both loops**

Make `dex3` the CLI default. Pass the selected profile through the pose streamer and `PlannerStreamer` to the existing shared `compute_hand_joints_from_inputs` function. In Inspire mode, read each controller's `triggerValue` and `gripValue` and call `map_pico_controls`; do not invoke the Dex3 hand solver.

- [ ] **Step 4: Verify pose and planner paths produce the same six values**

Run the Task 1 tests plus the existing PICO navigation tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_contract.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py
```

- [ ] **Step 5: Commit the PICO profile**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py gear_sonic/tests/test_inspire_ftp_contract.py
git commit -m "feat: map PICO controls to Inspire FTP motors"
```

### Task 3: Pinned Unitree model and deterministic generator

**Files:**
- Create: `gear_sonic/data/robots/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`
- Create: `gear_sonic/scripts/build_inspire_ftp_mujoco_model.py`
- Create: `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_ftp.xml`
- Create: `gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml`
- Create: `gear_sonic/tests/test_inspire_ftp_mujoco.py`

- [ ] **Step 1: Vendor and verify the official source**

Fetch the URDF from Unitree commit `4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48`, add it through `apply_patch`, and record its SHA-256 in the generator. The generator exits nonzero when the source hash changes.

- [ ] **Step 2: Write a failing generated-model count test**

```python
model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
assert model.njnt == 54  # floating base + 29 body + 24 hand
assert model.nu == 41    # 29 body + 6 left + 6 right
assert model.neq == 12   # six mimic constraints per hand
```

Also assert all twelve active actuator names exist and no actuator name ends in `_2_joint`, `_3_joint`, or `_4_joint` for a dependent finger joint.

- [ ] **Step 3: Run the model test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_sim/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_mujoco.py
```

Expected: the Inspire scene is absent.

- [ ] **Step 4: Implement deterministic hand transplantation**

The generator must:

1. load the existing SONIC `g1_29dof_with_hand.xml` body;
2. remove only Dex3 hand geoms, bodies, and 14 hand actuators;
3. convert the vendored Unitree URDF with MuJoCo and extract the `left_base_link` and `right_base_link` subtrees;
4. attach them below the existing wrist-yaw bodies with the official fixed transforms;
5. add position actuators only for `little_1`, `ring_1`, `middle_1`, `index_1`, `thumb_2`, and `thumb_1` on each side; and
6. add equality constraints with dependent joint as `joint1` and its driver as `joint2`, preserving the official ratios.

The script writes canonical indented XML and immediately reloads it to enforce the counts from Step 2.

- [ ] **Step 5: Generate, rerun, and verify GREEN**

```bash
.venv_sim/bin/python gear_sonic/scripts/build_inspire_ftp_mujoco_model.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_sim/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_mujoco.py
```

- [ ] **Step 6: Commit source, generator, generated model, and test**

```bash
git add gear_sonic/data/robots/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf \
  gear_sonic/scripts/build_inspire_ftp_mujoco_model.py \
  gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_ftp.xml \
  gear_sonic/data/robot_model/model_data/g1/scene_41dof_inspire_ftp.xml \
  gear_sonic/tests/test_inspire_ftp_mujoco.py
git commit -m "feat: add coupled G1 Inspire FTP MuJoCo model"
```

### Task 4: Shared ZMQ decoder and safe command state

**Files:**
- Create: `gear_sonic/utils/teleop/zmq/zmq_message_decoder.py`
- Modify: `gear_sonic/scripts/run_data_exporter.py`
- Create: `gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py`
- Create: `gear_sonic/tests/test_inspire_ftp_zmq.py`

- [ ] **Step 1: Write failing decoder and watchdog tests**

Build real `pose` and `planner` messages with six hand fields. Assert the decoder returns shape `(6,)`, an invalid shape is rejected, startup returns `OPEN`, a command is active before 250 ms, and a stale command monotonically ramps toward `OPEN` at the configured per-motor speed.

- [ ] **Step 2: Verify RED**

Run the new test file and require missing-module failures.

- [ ] **Step 3: Extract the existing binary decoder**

Move `unpack_pose_message` from `run_data_exporter.py` into the shared module without changing its wire behavior. Import it back into the exporter. Add explicit header-size, dtype, shape, payload-length, and duplicate-field validation.

- [ ] **Step 4: Implement pure `InspireCommandState`**

The class owns `last_valid`, `last_receive_monotonic`, and `output`. `accept(left, right, now)` validates both commands atomically. `advance(now, dt)` returns open before the first command, follows the latest valid command while fresh, and uses `np.minimum(output + max_speed * dt, OPEN)` after 250 ms.

- [ ] **Step 5: Implement latest-value `InspireFtpZmqSubscriber`**

Subscribe to both `pose` and `planner` on port 5556 with `RCVHWM=1`, zero linger, and nonblocking drain-to-latest behavior. Decode only messages containing both hand fields. The subscriber never creates a publisher and never imports FTP SDK types.

- [ ] **Step 6: Verify GREEN and exporter regression**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/tests/test_xr_upperbody_bridge.py
```

- [ ] **Step 7: Commit the receiver boundary**

```bash
git add gear_sonic/utils/teleop/zmq/zmq_message_decoder.py \
  gear_sonic/scripts/run_data_exporter.py \
  gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py
git commit -m "feat: receive safe Inspire hand commands in simulation"
```

### Task 5: Named MuJoCo plant and body-only DDS bridge

**Files:**
- Modify: `gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py`
- Modify: `gear_sonic/utils/mujoco_sim/base_sim.py`
- Modify: `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`
- Modify: `gear_sonic/utils/mujoco_sim/robot.py`
- Modify: `gear_sonic/tests/test_inspire_ftp_mujoco.py`

- [ ] **Step 1: Write failing named-layout tests**

Resolve all active joint qpos/qvel addresses and actuator IDs by exact name. Assert normalized open/closed targets, measured-state round trips, and that each hand actuator ID is distinct from all body actuator IDs.

- [ ] **Step 2: Verify RED**

Expected: the plant does not yet expose `resolve`, `write_targets`, or `read_normalized_state`.

- [ ] **Step 3: Implement `InspireFtpMujocoPlant`**

Use `mj_name2id` for every configured joint and actuator. `write_targets(left, right)` writes radian position targets directly into the twelve hand actuator slots. `read_normalized_state()` reads the six active qpos addresses per side and converts them back to normalized values.

- [ ] **Step 4: Install the plant only for `HAND_TYPE: inspire_ftp`**

In `DefaultEnv`, retain current Dex3 code for the default profile. In Inspire mode:

- classify hand joints from explicit config lists rather than substrings;
- size kinematic state from `NUM_HAND_JOINTS=12` and actuator state from `NUM_HAND_MOTORS=6`;
- leave the 29 body torque loop unchanged;
- advance the command state and write hand position targets once per sim step; and
- report six normalized measured hand values separately from the 24 physical hand joints.

- [ ] **Step 5: Disable Dex3 DDS objects in Inspire mode**

Add `ENABLE_DEX3_DDS_HANDS` defaulting to true. When false, do not construct `rt/dex3/*` publishers/subscribers, do not wait for Dex3 commands, and do not publish Dex3 hand state. Low-state, odometry, IMU, and wireless-controller behavior must remain unchanged.

- [ ] **Step 6: Run model, bridge, and existing simulator tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_sim/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_mujoco.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py
```

- [ ] **Step 7: Commit simulator integration**

```bash
git add gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/utils/mujoco_sim/base_sim.py \
  gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py \
  gear_sonic/utils/mujoco_sim/robot.py \
  gear_sonic/tests/test_inspire_ftp_mujoco.py
git commit -m "feat: drive Inspire FTP hands in MuJoCo"
```

### Task 6: Profile configuration and single-owner launcher

**Files:**
- Create: `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml`
- Modify: `gear_sonic/utils/mujoco_sim/configs.py`
- Modify: `gear_sonic/scripts/launch_data_collection.py`
- Modify: `gear_sonic_deploy/deploy.sh`
- Create: `gear_sonic/tests/test_inspire_ftp_launch.py`

- [ ] **Step 1: Write failing configuration/command tests**

Assert `SimLoopConfig(hand_profile="inspire_ftp").load_wbc_yaml()` selects the Inspire scene, six motors, twelve joints, and disabled Dex3 DDS. Assert launcher command builders include:

```text
run_sim_loop.py --hand-profile inspire_ftp
pico_manager_thread_server.py --hand-profile inspire_ftp
deploy.sh --disable-dex3-hands sim
```

- [ ] **Step 2: Verify RED**

Run the launch test and require failures for the missing field and shell option.

- [ ] **Step 3: Add the Inspire YAML and typed profile option**

Copy the current SONIC body gains and limits unchanged. Change only scene, hand type/counts/names, hand actuator names, equality expectations, `ENABLE_DEX3_DDS_HANDS: false`, and the ZMQ timeout/speed settings.

- [ ] **Step 4: Refactor launcher string construction into tested helpers**

Add `hand_profile: Literal["dex3", "inspire_ftp"] = "dex3"`. The Inspire branch passes all three ownership flags atomically; the Dex3 branch produces the current commands byte-for-byte.

- [ ] **Step 5: Teach `deploy.sh` to forward the existing C++ flag**

Parse `--disable-dex3-hands` as a boolean before interface resolution and append it to the final binary command. Do not reinterpret it as an interface. Add a shell `--dry-run` or factored argument-print path so the pytest can verify forwarding without launching C++.

- [ ] **Step 6: Verify GREEN and shell syntax**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_teleop/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_launch.py
bash -n gear_sonic_deploy/deploy.sh
```

- [ ] **Step 7: Commit profile wiring**

```bash
git add gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire_ftp.yaml \
  gear_sonic/utils/mujoco_sim/configs.py \
  gear_sonic/scripts/launch_data_collection.py \
  gear_sonic_deploy/deploy.sh \
  gear_sonic/tests/test_inspire_ftp_launch.py
git commit -m "feat: launch SONIC sim with Inspire hand ownership"
```

### Task 7: Headless motor sweep and SONIC regression

**Files:**
- Create: `gear_sonic/scripts/verify_inspire_ftp_mujoco.py`
- Modify: `gear_sonic/tests/test_inspire_ftp_mujoco.py`

- [ ] **Step 1: Write a failing per-motor sweep test**

For each side and each motor, settle open, close only that motor, step 400 simulation frames, and assert:

- every state and control value is finite;
- the intended active joint moves toward its closed target;
- unrelated active joints stay within `1e-3` rad;
- dependent joint error stays below `2e-3` rad; and
- all joint limits remain satisfied.

- [ ] **Step 2: Verify RED before adding the script**

The test must fail because the sweep API is missing.

- [ ] **Step 3: Implement the headless verifier**

Support `--motor-sweep`, `--duration`, and `--json-output`. Default to no viewer and no external ZMQ. Emit model counts, maximum constraint error, maximum joint-limit error, and a result for each of twelve motors.

- [ ] **Step 4: Run the focused sweep**

```bash
.venv_sim/bin/python gear_sonic/scripts/verify_inspire_ftp_mujoco.py --motor-sweep --json-output /tmp/inspire_ftp_mujoco_report.json
```

Expected: exit 0 and twelve passing motor results.

- [ ] **Step 5: Run a bounded full SONIC headless regression**

Start the simulator with `hand_profile=inspire_ftp`, C++ deploy with Dex3 disabled, and a scripted local ZMQ hand publisher. Run open hands for 5 seconds and a slow close/open cycle for 10 seconds. Record pelvis height, fall checks, non-finite values, and maximum hand coupling error. Stop all three processes explicitly after the bounded run.

- [ ] **Step 6: Commit verification tooling**

```bash
git add gear_sonic/scripts/verify_inspire_ftp_mujoco.py gear_sonic/tests/test_inspire_ftp_mujoco.py
git commit -m "test: verify Inspire FTP MuJoCo motor sweeps"
```

### Task 8: Final verification and operator handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-inspire-ftp-mujoco-teleop-design.md` only if verified behavior differs
- Create: `docs/inspire-ftp-mujoco-runbook.md`

- [ ] **Step 1: Run the complete focused suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_inspire_ftp_contract.py \
  gear_sonic/tests/test_inspire_ftp_zmq.py \
  gear_sonic/tests/test_inspire_ftp_launch.py \
  gear_sonic/tests/test_pico_planner_joystick.py \
  gear_sonic/tests/test_pico_streamer_navigation.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_sim/bin/python -m pytest -q -p no:cacheprovider gear_sonic/tests/test_inspire_ftp_mujoco.py
bash -n gear_sonic_deploy/deploy.sh
ruff check --no-cache \
  gear_sonic/utils/teleop/inspire_ftp.py \
  gear_sonic/utils/teleop/zmq/zmq_message_decoder.py \
  gear_sonic/utils/mujoco_sim/inspire_ftp_hand.py \
  gear_sonic/scripts/build_inspire_ftp_mujoco_model.py \
  gear_sonic/scripts/verify_inspire_ftp_mujoco.py
```

- [ ] **Step 2: Write the runbook from observed commands**

Document the exact headless model test, motor sweep, full simulated launch with `--hand-profile inspire_ftp`, expected six-value PICO fields, shutdown behavior, and the explicit statement that this phase never publishes FTP DDS commands.

- [ ] **Step 3: Check the diff and repository cleanliness**

```bash
git diff --check
git status --short
```

- [ ] **Step 4: Commit the verified runbook**

```bash
git add docs/inspire-ftp-mujoco-runbook.md docs/superpowers/specs/2026-08-26-inspire-ftp-mujoco-teleop-design.md
git commit -m "docs: add Inspire FTP MuJoCo runbook"
```

- [ ] **Step 5: Stop before live PICO or PC2 hardware**

Report the automated and headless regression evidence. Request separate user approval before opening a MuJoCo viewer, depending on live XRoboToolkit state, SSHing to PC2, or starting any real FTP DDS publisher.
