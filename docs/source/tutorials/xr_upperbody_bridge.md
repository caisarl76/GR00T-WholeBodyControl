# XR Upper-Body Bridge

Use this bridge when you have XR/HOMIE-style **upper-body trajectory data only** and want GEAR-SONIC to keep controlling the Unitree G1 lower body for balance and safe posture.

```{admonition} Safety Boundary
:class: danger
The bridge only publishes ZMQ `command` and `planner` messages to GEAR-SONIC. It must not run together with another process that writes Unitree body `LowCmd` commands. GEAR-SONIC must remain the only body motor command writer.
```

## Input Format

The bridge accepts `.jsonl` or `.npz` trajectories. Each frame must contain:

| Field | Shape | Required | Notes |
|---|---:|---:|---|
| `upper_body_position` | 17 | Yes, unless `dual_arm_position` is present | GEAR-SONIC upper-body joint order, radians |
| `dual_arm_position` | 14 | Yes, unless `upper_body_position` is present | Unitree `xr_teleoperate` G1_29 arm order |
| `upper_body_velocity` | 17 | No | Defaults to zeros |
| `dual_arm_velocity` | 14 | No | Mapped like `dual_arm_position` when present |
| `left_hand_joints` | 7 | No | Dex3 left hand target |
| `right_hand_joints` | 7 | No | Dex3 right hand target |
| `dual_hand_joints` | 14 | No | Unitree XR Dex3 action order: left 7 followed by right 7 |
| `timestamp` | 1 | No | Used for traceability; playback rate comes from `--hz` |

For Unitree `xr_teleoperate` G1_29 arm trajectories, `dual_arm_position` is:

```text
[
  left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
  left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
  right_wrist_roll, right_wrist_pitch, right_wrist_yaw
]
```

The bridge maps that into GEAR-SONIC's 17-DoF upper-body order by prepending neutral waist joints and interleaving left/right arm joints.

For Unitree `xr_teleoperate` Dex3 trajectories, `dual_hand_joints` is already in command order: left hand 7 motor targets followed by right hand 7 motor targets. The bridge splits this into `left_hand_joints` and `right_hand_joints` before publishing.

Optional lower-body planner fields are also accepted, but the default is a stable idle lower-body command:

```json
{"mode": 0, "movement": [0, 0, 0], "facing": [1, 0, 0], "speed": -1, "height": -1}
```

Example JSONL frame:

```json
{"timestamp": 0.0, "upper_body_position": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "left_hand_joints": [0, 0, 0, 0, 0, 0, 0], "right_hand_joints": [0, 0, 0, 0, 0, 0, 0]}
```

## Dry Run

From the repository root:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/xr_upperbody_bridge.py \
  --input path/to/upperbody.jsonl \
  --dry-run \
  --once
```

This validates frame shapes and builds GEAR-SONIC planner messages without opening a ZMQ socket.

Before sim or real hardware, inspect the trajectory:

```bash
python gear_sonic/scripts/xr_upperbody_bridge.py \
  --input path/to/upperbody.jsonl \
  --inspect \
  --synthesize-velocities
```

`--inspect` reports frame count, joint ranges, the first mapped 17-DoF GEAR-SONIC upper-body frame, and warnings for values that look too large for radians. `--synthesize-velocities` fills missing upper-body velocities from adjacent frames using timestamps when available, or `--hz` otherwise.

## Sim2Sim Validation

Run three terminals.

**Terminal 1 — MuJoCo sim:**

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 — GEAR-SONIC deployment:**

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
```

**Terminal 3 — XR upper-body bridge:**

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/xr_upperbody_bridge.py \
  --input path/to/upperbody.jsonl \
  --hz 50 \
  --synthesize-velocities \
  --loop \
  --start-control
```

The bridge publishes `planner=true` continuously. GEAR-SONIC receives idle lower-body commands plus upper-body targets, then its policy remains responsible for balance and posture.

## Live GenONAI / xr_teleoperate Source

For live Quest 3 or PICO 4 Ultra testing, use Unitree `xr_teleoperate` in GEAR-SONIC export mode to publish the solved arm and hand action arrays over a small JSON ZMQ PUB socket. The bridge subscribes to that stream and republishes GEAR-SONIC planner messages.

Expected JSON payload:

```json
{
  "timestamp": 1780000000.0,
  "dual_arm_position": [14 values from sol_q],
  "dual_hand_joints": [14 values from dual_hand_action_array],
  "mode": 1,
  "movement": [0.0, 0.0, 0.0],
  "facing": [1.0, 0.0, 0.0],
  "speed": 0.4,
  "height": -1.0
}
```

The local `xr_teleoperate` checkout used with this bridge has a `--gear-sonic-export` flag. In that mode it computes `sol_q` and Dex3 actions, publishes them, and skips direct arm/hand command writes.

Run `xr_teleoperate` in export mode:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
python teleop/teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode hand \
  --sim \
  --gear-sonic-export \
  --gear-sonic-export-port 5560 \
  --gear-sonic-export-topic xr_teleop
```

For Quest 3 or PICO 4 Ultra controller teleoperation, use controller mode:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
python teleop/teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode controller \
  --sim \
  --img-server-ip 127.0.0.1 \
  --gear-sonic-export \
  --gear-sonic-export-port 5560 \
  --gear-sonic-export-topic xr_teleop \
  --gear-sonic-ramp-in-s 2.0 \
  --gear-sonic-ramp-out-s 2.0 \
  --gear-sonic-exit-hold-s 2.0 \
  --gear-sonic-debug
```

Controller mapping:

| Input | Action |
|---|---|
| Left index trigger | Left Dex3 gripper close/open |
| Right index trigger | Right Dex3 gripper close/open |
| Left stick | Walk / strafe |
| Right stick X | Turn / yaw |
| Right stick Y | Adjust squat/kneel height |
| A + X | Pause/resume export and toggle collection state |
| B + Y | Pause/resume export and toggle collection state |
| A + B | Next locomotion mode |
| X + Y | Previous locomotion mode |
| A + B + X + Y | Publish final idle command and quit XR export; GEAR-SONIC stays on |

Locomotion modes match the GEAR-SONIC planner convention:

| Mode | Meaning |
|---:|---|
| 0 | Idle |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5 | Kneel two legs |
| 6 | Kneel |

Modes 4, 5, and 6 publish `height` in the planner message. Height is clamped to `0.2..0.8`; walking modes publish `height=-1`.

Internally, the export payload is equivalent to:

```python
import json
import time
import zmq

ctx = zmq.Context.instance()
xr_pub = ctx.socket(zmq.PUB)
xr_pub.bind("tcp://*:5560")

# Inside the teleop loop, after sol_q and dual_hand_action_array are updated:
payload = {
    "timestamp": time.time(),
    "dual_arm_position": sol_q.astype(float).tolist(),
    "dual_hand_joints": list(dual_hand_action_array[:]),
    "mode": mode,
    "movement": movement,
    "facing": facing,
    "speed": speed,
    "height": height,
}
xr_pub.send(b"xr_teleop" + json.dumps(payload).encode("utf-8"))
```

Then run the bridge in live-source mode:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/xr_upperbody_bridge.py \
  --source zmq-json \
  --source-host localhost \
  --source-port 5560 \
  --source-topic xr_teleop \
  --hz 50 \
  --start-control
```

This keeps Unitree/XR as the upper-body source only. Do not let `xr_teleoperate` write body arm commands while this bridge is controlling GEAR-SONIC; publish `sol_q`/hand actions and let GEAR-SONIC be the only body command writer.

## Full Simulation XR Teleop Setup

Use this four-terminal flow to test the Unitree `xr_teleoperate` export path against GEAR-SONIC in MuJoCo. Do not run `pico_manager_thread_server.py` at the same time; both it and this bridge publish to the GEAR-SONIC ZMQ manager port.

**Terminal 1 -- MuJoCo sim:**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 -- GEAR-SONIC deployment:**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
```

**Terminal 3 -- Unitree-to-GEAR-SONIC bridge:**

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/xr_upperbody_bridge.py \
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

Before enabling XR tracking, wait for one of these bridge messages:

```text
seeded upper-body ramp from feedback
seeded upper-body ramp from late feedback
```

The bridge rejects unseeded live arm control by default. Do not use
`--allow-unseeded-start-control` on the robot.

When Unitree XR enters exit ramp-out, exit hold, or sends `stop=true`, the
bridge owns the exit ramp instead of forwarding raw Unitree exit frames. It
ramps the upper body to `--stop-upper-body-preset`, commands a partial Dex3
thumb tuck, then publishes the exact final frame for `--stop-final-hold-s`
seconds so the controller can settle at the final pose. `straight` and
`deploy-standing` both use the normal final standing target. Use
`--stop-upper-body-preset calib-full` only when you explicitly want the all-zero
calibration pose. Use `--stop-hand-preset none` to restore the old hand
behavior, `source` to hold the last XR hand command, or `open` for the all-zero
open hand target.

By default the live bridge also anchors Unitree controller locomotion heading to
the robot's measured startup yaw from `g1_debug`. This prevents the initial
Unitree planner default `facing=[1, 0, 0]` from rotating the robot toward world
+X when teleop starts. Use `--no-anchor-planner-heading` only for sources that
already publish absolute world-frame `movement` and `facing` vectors.

**Terminal 4 -- Unitree XR export in simulation mode:**

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
conda activate tv
python teleop/teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode controller \
  --sim \
  --img-server-ip 127.0.0.1 \
  --gear-sonic-export \
  --gear-sonic-export-port 5560 \
  --gear-sonic-export-topic xr_teleop \
  --gear-sonic-ramp-in-s 2.0 \
  --gear-sonic-ramp-out-s 2.0 \
  --gear-sonic-exit-hold-s 2.0 \
  --gear-sonic-debug
```

Data flow:

```text
Quest/Pico browser -> Unitree xr_teleoperate -> tcp://*:5560 topic xr_teleop
  -> xr_upperbody_bridge -> tcp://*:5556 command/planner/manager_state
  -> gear_sonic_deploy --input-type zmq_manager sim -> MuJoCo
```

Optional read-only dashboard:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/xr_teleop_dashboard.py \
  --host 127.0.0.1 \
  --port 8088 \
  --no-camera
```

Then open `http://127.0.0.1:8088`.

## Data Collection

Use the existing data exporter flow while the bridge is publishing. The exporter subscribes to C++ robot state output and the same planner stream, so upper-body and hand targets are recorded through the existing SONIC VLA dataset fields.

The bridge also publishes `manager_state` with `stream_mode=5` by default. This matches the planner/VR teleop recording path in `run_data_exporter.py`, so planner and hand fields are treated as active teleop data. Change this only if you also update the exporter logic.

## Real Robot

Do not start real robot control from this tutorial alone. Before real hardware use:

1. Validate the same trajectory in sim.
2. Confirm no HOMIE, Unitree XR, or other process is writing body `LowCmd`.
3. Follow `docs/source/user_guide/real_robot_safety.md`.
4. Keep a safety operator ready at the keyboard and emergency stop.
