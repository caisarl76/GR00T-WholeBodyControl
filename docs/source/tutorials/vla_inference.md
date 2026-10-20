# VLA Inference

This guide covers running a trained Isaac-GR00T VLA policy on the Unitree G1 robot
using the Sonic whole-body control stack.

## Overview

The inference pipeline consists of:

1. **Isaac-GR00T PolicyServer** — loads the VLA model and serves actions over ZMQ
2. **VLA inference client** (`run_vla_inference.py`) — reads camera + robot state,
   queries the PolicyServer, and publishes actions to the C++ control loop
3. **C++ deploy** (`gear_sonic_deploy`) — executes whole-body control on the robot
4. **Camera server** — provides camera images over ZMQ (runs as a systemd service)
5. **Data exporter** (optional) — records episodes during inference

```
┌──────────────────────┐
│  Isaac-GR00T         │
│  PolicyServer        │
│  (GPU machine)       │
└──────┬───────────────┘
       │ ZMQ REQ/REP
       ▼
┌─────────────────────┐    ZMQ TCP    ┌──────────────────────┐
│  VLA Inference      │ ◄─────────── │  Camera Server       │
│  (run_vla_inference)│              │  (on robot)          │
└────┬───────────┬────┘              └──────────────────────┘
     │           │
     │ ZMQ PUB   │ ZMQ SUB
     │ (actions) │ (state)
     ▼           ▼
┌─────────────────────┐
│  C++ Deploy         │
│  (gear_sonic_deploy)│
└─────────────────────┘
```

## Prerequisites

### 1. Isaac-GR00T PolicyServer

The PolicyServer runs on a machine with a GPU. It loads your finetuned VLA model
and serves inference over ZMQ.

Install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and start the server:

```bash
# On the GPU machine (from the Isaac-GR00T repo)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### 2. Inference Environment

On the inference machine (can be the same as the PolicyServer or a separate PC):

```bash
bash install_scripts/install_inference.sh
```

This creates `.venv_inference` with the Isaac-GR00T PolicyClient and all
inference dependencies.

### 3. Camera Server

The camera server should be running as a systemd service on the robot.
See [Data Collection](data_collection.md) for camera server setup.

### 4. C++ Deploy

The `gear_sonic_deploy` binary must be built. See the main README.

### SONIC v1.1 Checkpoint

Use the `sonic_v1_1/` checkpoint when the VLA policy was trained against
the robot-heading-normalized SONIC controller. It uses a 10-frame SMPL/wrist
reference horizon and was trained with wrist-pose augmentation. It is not the
low-latency checkpoint.

```bash
python download_from_hf.py --sonic-v1-1
```

Launch the matching C++ controller:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    --motor-kp-scale 4,10=1.5 \
    --motor-kd-scale 4,10=1.5 \
    real
```

This v1.1 tuning scales the left and right ankle-pitch motors (hardware
indices `4` and `10`). The resulting whole-body stability improves observed
wrist tracking; it does not directly scale the wrist motors.

Or pass the same model pair to the Python launcher:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/sonic_v1_1/model \
    --deploy-obs-config policy/sonic_v1_1/observation_config.yaml \
    --deploy-motor-kp-scale 4,10=1.5 \
    --deploy-motor-kd-scale 4,10=1.5 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

### Low-Latency Teleoperation Checkpoint

The `low_latency/` checkpoint is configured for responsive whole-body
teleoperation. Its SMPL encoder uses 4 future reference frames at 50 Hz
(approximately 80 ms of reference lookahead), compared with 10 frames
(approximately 200 ms) in the default release. This is reference lookahead,
not total end-to-end system latency.

Download the deployment files from Hugging Face:

```bash
python download_from_hf.py --low-latency
```

Then launch `gear_sonic_deploy` with the low-latency model prefix and matching
observation config:

**C++ deploy:**

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

For simulation, replace `real` with `sim`. The `--cp` value is a model prefix:
`deploy.sh` appends `_encoder.onnx` and `_decoder.onnx` internally.

**Python launcher:**

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

The Python launcher starts the same C++ deploy command in a tmux pane, then runs
the Python VLA inference client, keyboard publisher, and optional data exporter.

## Action Space

The Sonic embodiment (`unitree_g1_sonic`) uses a 78-dimensional action
space: 64-dim motion token + 7-dim left hand joints + 7-dim right hand joints.

## Quick Start — tmux Launcher

The easiest way to run inference is with the all-in-one tmux launcher:

```bash
# Real robot
python gear_sonic/scripts/launch_inference.py \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164

# Simulation
python gear_sonic/scripts/launch_inference.py --sim \
    --prompt "pick up the apple"

# Without data recording
python gear_sonic/scripts/launch_inference.py \
    --no-data-exporter \
    --prompt "pick up the apple"
```

The launcher defaults to `--initial-pose calib_full`. In that mode, `i` ramps
to the configured SONIC initial pose and holds PLANNER mode. Pressing `p`
resumes inference; the first policy action switches to POSE mode. The
feedback-paced straight-standing workflow below applies when you explicitly
launch with `--initial-pose planner_standing` (the PnP evaluation presets use
that mode).

The launcher creates a tmux session with four panes:

| Pane | Component | Description |
|------|-----------|-------------|
| 0 (top-left) | C++ Deploy | Whole-body controller |
| 1 (bottom-left) | Keyboard Publisher | Type keyboard commands here |
| 2 (top-right) | VLA Inference | Policy client + action loop |
| 3 (bottom-right) | Data Exporter | Records episodes (optional) |

### Keyboard Controls

Type these keys in the **Keyboard Publisher** pane (pane 1):

| Key | Action |
|-----|--------|
| `k` | Start / stop the C++ control loop |
| `i` | Reset according to `--initial-pose`: ramp to CALIB_FULL by default, or return to planner standing in `planner_standing` mode |
| `m` | Enter manual planner repositioning / stop movement and hold PLANNER mode |
| `p` | Pause / resume policy inference |
| `[` | Toggle left hand open/closed (initial pose) |
| `]` | Toggle right hand open/closed (initial pose) |
| `t <text>` | Change the inference prompt (e.g., `t pick up the cup`) |
| `c` | Start recording; press again to finish and save (data exporter) |
| `x` | Discard an active recording (data exporter) |

### Typical Workflow

1. Wait for all panes to initialize
2. Click on **pane 0** (C++ Deploy) and press Enter to confirm deployment
3. Switch to **pane 1** (Keyboard Publisher)
4. Press `k` to start the C++ control loop (starts in PLANNER mode)
5. Press `i` to ramp to the configured initial pose and hold PLANNER mode.
   The default CALIB_FULL ramp takes 2 seconds. If your
   task starts from a different pose than the default, see
   [Customizing the Initial Pose](#customizing-the-initial-pose) below.
6. Press `p` to unpause the inference loop; the first policy action enters POSE mode
7. The robot will begin executing VLA-predicted actions
8. Press `p` to pause, `k` to stop the control loop when done

For PnP evaluation, launch with `--initial-pose planner_standing` (the
evaluation presets select this explicitly), then use this reset sequence:

1. Press `i` to return to straight standing in PLANNER mode. Wait until the
   robot has settled before continuing. This uses the existing `straight`
   planner preset: upright waist, shoulder pitch 0.2 rad, shoulder roll
   ±0.2 rad, and elbows 0.6 rad. SONIC controls the legs and balance.
   The return starts from measured joints, preserves heading, and opens both
   hands unless `[` / `]` selected closed hands. Fresh robot state is required.
2. Press `p` to resume inference using fresh observations. The first policy
   action switches to POSE mode.

Between attempts in `planner_standing` mode, use `p` (pause) → `i` (return to
standing) → wait for the robot to settle → `p` (evaluate). Every `i` pauses
the policy and discards cached and in-flight policy results. The standing target advances at up to
0.5 rad/s per joint and stays within 0.15 rad ahead of measured feedback;
stale feedback holds the last target. `k` remains available during the return.
Restart the inference client to load changes to this keyboard behavior.

The planner-standing reset preserves the measured world heading at `i`. This
heading and feedback behavior applies when `--initial-pose planner_standing`
is selected. The client uses deploy's `reference_heading_quat` feedback to convert that heading into
the current planner reference frame on every publication, including after a
POSE → PLANNER reinitialization. This also keeps manual movement directions
consistent after turning. Rebuild/restart deploy and restart the inference
client together for this heading fix; `deploy.sh` builds before launching.
Older deploy binaries without this telemetry are rejected by the reset with
an explicit rebuild message.
In `planner_standing` mode, if `i` arrives before the POSE switch is reflected
in feedback, the client remains paused and asks you to retry after the switch settles; it never uses
the previous planner session's heading reference for the new reset.

### Keyboard repositioning between trials

Keep the original `--input-type zmq_manager` deployment. The inference client
translates keyboard commands into ZMQ planner messages. Once deploy supplies
the heading telemetry described above, no input-handler change is needed.
Enter each key followed by **Enter** in the
inference keyboard pane:

| Key | Action in manual planner mode |
|-----|-------------------------------|
| `m` | Enter manual mode / stop movement and hold PLANNER mode |
| `w` / `s` | Toggle forward / backward |
| `a` / `d` | Toggle left / right |
| `q` / `e` | Toggle turning left / right |
| `z` | Stop movement |

Sequence: `k → i → wait → p → p → i → wait → m → reposition → m → i → wait → p`.
Evaluation stays paused in manual mode. Press a movement key once to start;
press the same key again to stop. A different movement key switches direction;
`z` stops all movement. **Movement continues until cancelled; there is no jog
timeout.** Translation commands 0.2 m/s relative to the measured heading when
selected. Turning continuously keeps a small ±5° target ahead of measured yaw.
Missing/stale robot feedback cancels movement without replay on recovery.

`m` sends idle before holding paused PLANNER mode. Press `i` and then `p` for
the next trial. `p` is blocked during
manual mode; `i` cancels movement and resets standing; `k` stops the controller.
If `i` cannot reset because feedback is unavailable, the cancelled manual hold
remains active until reset succeeds or you exit with `m`. The recorder still
uses `c` to start or save a recording and `x` to discard; these are not movement keys.

Restart the workstation inference client and keyboard publisher to load these
controls. No H100 server change is required.

## Manual Setup (Without tmux)

If you prefer to run each component in separate terminals:

### Terminal 1 — Isaac-GR00T PolicyServer (GPU machine)

```bash
# From the Isaac-GR00T repo
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Terminal 2 — C++ Deploy

```bash
cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

Low-latency variant:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the apple"
```

Manual C++ deploy equivalent:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

### Terminal 3 — VLA Inference

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/run_vla_inference.py \
    --host <policy_server_ip> \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

### Terminal 4 — Data Exporter (optional)

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

## Configuration Reference

### VLA Inference (`run_vla_inference.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | PolicyServer host |
| `--port` | `5550` | PolicyServer port |
| `--embodiment-tag` | `unitree_g1_sonic` | Embodiment tag |
| `--prompt` | `demo` | Language prompt |
| `--action-publish-rate` | `50` | Action publish rate (Hz) |
| `--action-horizon` | `40` | Actions per inference chunk |
| `--rate` | `2.5` | Inference rate (Hz) |
| `--camera-host` | `localhost` | Camera server host |
| `--camera-port` | `5555` | Camera server port |
| `--initial-pose-blend-duration` | `1.0` | Seconds to blend to initial pose (0 = instant snap) |
| `--verbose-timing` | `false` | Always print loop timing |

### tmux Launcher (`launch_inference.py`)

The launcher exposes all the above flags plus deploy and data exporter options.
Use `--deploy-motor-kp-scale` and `--deploy-motor-kd-scale` to forward hardware
gain specifications to the C++ controller.
Run `python gear_sonic/scripts/launch_inference.py --help` for the full list.

## Remote PolicyServer

When running the PolicyServer on a separate GPU machine:

```bash
# On the inference machine, point to the remote server
python gear_sonic/scripts/launch_inference.py \
    --policy-host <gpu_machine_ip> \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the apple"
```

Make sure port 5550 (or your chosen port) is accessible between the two machines.

## Left-only pnp_trash evaluation with SONIC on PC2

The left-only models each trained on 44 complete episodes for 20,000 steps.
Run this preset **on the workstation**, from the repository root:

```bash
# Full-episode prompt; H100 GPU 7, checkpoint-20000
python gear_sonic/scripts/launch_pnp_trash_left_eval.py full --object "pill bottle"

# Four operator-controlled subtask prompts; H100 GPU 6, checkpoint-20000
python gear_sonic/scripts/launch_pnp_trash_left_eval.py subtasks --object "pill bottle"
```

Choose one model at a time. Both use the `sonic_inference` tmux session and the
same action/keyboard ports. An existing session is preserved and the launcher
prints its reattach command. Add `--dry-run` to either command to preview the
configuration without opening tmux, connecting to servers, or publishing actions.
Supported objects are `apple`, `cup`, `pill bottle`, `pill box`, and `red bottle`.

| Connection | Address |
|------------|---------|
| Full-prompt left-only PolicyServer | `192.168.75.173:15552` |
| Subtask left-only PolicyServer | `192.168.75.173:15553` |
| PC2 camera | `192.168.0.223:5555` |
| PC2 SONIC state and robot configuration | `192.168.0.223:5557` |
| Workstation action publisher, subscribed to by PC2 | `192.168.0.62:5556` |
| Workstation keyboard publisher | `localhost:5580` |

The preset accepts `--policy-host`, `--robot-host`, and `--action-host` if these
addresses change. `--action-host` must be an address on the workstation.
It uses live PC2 camera/state, a 40-frame action horizon, and 50 Hz publication.

The existing PC2 deployment must use `--input-type zmq_manager`,
`--zmq-host 192.168.0.62`, `--output-type zmq`, and these SONIC v1.1 assets:

```text
--cp policy/sonic_v1_1/model
--obs-config policy/sonic_v1_1/observation_config.yaml
--planner planner/target_vel/V2/planner_sonic.onnx
```

Keep its deploy console open. The workstation preset leaves that process under
the operator's control; it does not SSH to PC2 or start another C++ deployment.

| Pane | Content |
|------|---------|
| 0, top-left | PC2 deployment notes and an available shell |
| 1, bottom-left | Keyboard publisher: enter controls and `t <prompt>` |
| 2, top-right | GR00T inference client, initially paused |
| 3, bottom-right | Data exporter using PC2 camera/state |

Use the existing startup procedure in [Typical Workflow](#typical-workflow),
with deployment confirmation in the **PC2 console**. `i` uses the straight
standing planner preset described above. No `k`, `i`, or `p`
commands are sent automatically. Keep PC2's controller state consistent with
the client's initial stopped state before starting a new client.

The full model gets its complete prompt at startup. For the subtask model,
the initial prompt is `approach brown table`. Enter these lines in pane 1
as each real subtask completes (replace the object to match `--object`):

```text
t pick the pill bottle
t turn left and approach the trash bin
t put it in to the trash bin
```

Use task completion to choose prompt transitions on the real robot. The dataset's
recorded transition timestamps are for recorded-video evaluation.

Recording starts with `c`; press `c` again to finish and save, or `x` to discard. Each
launch records under a fresh `outputs/pnp_trash_left_eval_<variant>_<object>_<UTC>/`
directory, labeled with the full task prompt. The recorder does not add the live
subtask prompt transitions as per-frame labels.

To switch models, pause with `p` if running, stop the controller with `k` if
running, and confirm the stop in the PC2 console. Then close the workstation
session with `tmux kill-session -t sonic_inference` and run the other preset.
Detaching with `Ctrl+b`, then `d`, leaves the session running.

## pnp_table_260908 evaluation

The completed table subtask model uses a separate H100 Docker PolicyServer at
`192.168.75.173:15554`. From the workstation repository root:

```bash
python3 gear_sonic/scripts/launch_pnp_table_eval.py

# Full-task checkpoint on H100 GPU 7, Docker server port 15555
python3 gear_sonic/scripts/launch_pnp_table_eval.py full
```

Add `--dry-run` to preview the command. This uses the same four-pane PC2 layout
described above, with the inference client initially paused. For the default
subtask variant, the initial prompt is `approach the table`; use `t grasp the bottle` and then `t pick up the bottle`
in the keyboard pane as the real subtasks complete. Recordings use a fresh
`outputs/pnp_table_260908_subtask_eval_<UTC>/` directory and the full task label
`approach the table and pick the bottle`.

The `full` variant connects to `192.168.75.173:15555`, keeps the full task prompt
throughout the episode, and records under `outputs/pnp_table_260908_full_task_eval_<UTC>/`.
Use one workstation client at a time; both variants share the `sonic_inference`
tmux session and action/keyboard ports.

See [the table evaluation report](../../pnp_table_260908_evaluation.md) for the
checkpoint, recorded-data results, Docker commands, and operator instructions.

## Latency Compensation

The inference loop automatically compensates for network and compute latency.
When a new action chunk arrives, the system calculates how many actions in the
chunk are already "stale" based on the time elapsed since inference started,
and skips to the appropriate action index. This is controlled by `--action-publish-rate`
and `--action-horizon`.

## Customizing the Initial Pose

When you press `i`, the inference client blends the robot smoothly from its
current configuration to a predefined **initial pose** encoded as a 64-dim
latent motion token. This pose should match the starting configuration your
demonstrations typically begin from.

### When to Change the Initial Pose

You should update the initial motion token if:

- Your collected demonstrations start from a pose far from the default
  (e.g., arms raised, holding an object, or a different standing stance)
- You switch to a different SONIC checkpoint (each checkpoint has its own
  latent space — the same token produces different poses across checkpoints)
- The robot is snapping to a dangerous or unstable configuration on `i` press

### Where to Change It

Edit `gear_sonic/utils/inference/initial_poses.py`:

```python
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
        # Replace with your 64-dim token
        ...
    ],
    dtype=np.float32,
)
```

### How to Find a Good Token

1. **From data collection:** Look at the first action frame of a good demonstration
   episode. The `action.motion_token` column in the parquet file at `frame_index=0`
   gives you the latent token for that pose.

2. **From the C++ deploy:** Put the robot in the desired starting pose via teleop,
   then read the most recent latent token published on the ZMQ action channel.

### Blend Duration

The blend duration controls how quickly the robot transitions to the initial pose:

```bash
# Default: 1 second smooth blend
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 1.0

# Faster blend (0.5 seconds)
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 0.5

# Instant snap (no interpolation, legacy behavior)
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 0
```

```{warning}
Setting `--initial-pose-blend-duration` too low (or to 0) can cause jerky motion,
especially if the robot's current pose is far from the initial pose. The default
1-second blend is safe for most configurations.
```
