# pnp_table_260908 evaluation

The subtask `checkpoint-20000` is ready for operator evaluation. It completed
training with exit code `0` and final logged loss `0.0246`. The Docker
PolicyServer is running on H100 GPU **6** at **192.168.75.173:15554**; a workstation
PolicyClient ping succeeded. No workstation tmux session or robot control was
started during this setup.

The full-task `checkpoint-20000` also completed with exit code `0` and now runs
in Docker on H100 GPU **7** at **192.168.75.173:15555**. Its prompt is fixed to
`approach the table and pick the bottle`. Both model servers can remain running;
run one workstation client at a time.
The full-task server passed a workstation ping and an inference request using
episode 0, frame 0 from its dataset. It returned finite motion-token `[1,40,64]`
and left/right-hand `[1,40,7]` arrays with the exact full-task prompt. This launch
check did not run the full 34-episode evaluation for that model.

## Run the four-pane workstation launcher

From `/home/jihun/work/GR00T-WholeBodyControl`:

```bash
# Preview without launching anything
python3 gear_sonic/scripts/launch_pnp_table_eval.py --dry-run

# Run when ready for the real-robot evaluation
python3 gear_sonic/scripts/launch_pnp_table_eval.py

# Full-task model, with its complete prompt throughout the episode
python3 gear_sonic/scripts/launch_pnp_table_eval.py full
```

| Pane | Component |
|------|-----------|
| 0, top-left | PC2 deployment notes and an available shell |
| 1, bottom-left | Keyboard publisher |
| 2, top-right | Selected table model's inference client, initially paused |
| 3, bottom-right | Data exporter |

| Connection | Address |
|------------|---------|
| Table subtask PolicyServer | `192.168.75.173:15554` |
| Table full-task PolicyServer | `192.168.75.173:15555` |
| PC2 camera | `192.168.0.223:5555` |
| PC2 SONIC state/configuration | `192.168.0.223:5557` |
| Workstation action publisher | `192.168.0.62:5556` |
| Workstation keyboard publisher | `localhost:5580` |

The preset uses a 40-frame action horizon and 50 Hz action publication. Endpoint
overrides are `--policy-host`, `--policy-port`, `--robot-host`, and `--action-host`.
The action address must belong to the workstation.

Keep the existing SONIC v1.1 deployment and its console on PC2. It must subscribe
to `192.168.0.62:5556` with `--input-type zmq_manager --zmq-host 192.168.0.62`
and publish state with `--output-type zmq`, using these existing assets:

```text
--cp policy/sonic_v1_1/model
--obs-config policy/sonic_v1_1/observation_config.yaml
--planner planner/target_vel/V2/planner_sonic.onnx
```

Follow the established [operator startup procedure](source/tutorials/vla_inference.md#typical-workflow),
with deployment confirmation in the PC2 console. `i` pauses evaluation and
returns through SONIC's straight standing planner preset, paced by measured
joint feedback while preserving heading. Wait for standing before pressing
`p` to switch to POSE mode with fresh policy output. The preset sends no startup controls
automatically. The keyboard controls remain `k` for controller start/stop,
`i` for initial pose, and `p` for policy pause/resume.

For the subtask model, the initial prompt is exactly **approach the table**. As each subtask completes,
enter the next prompt in pane 1:

```text
t grasp the bottle
t pick up the bottle
```

Use actual task completion for these transitions. Recorded dataset timestamps
are not a schedule for the real robot.

For the full-task model, retain **approach the table and pick the bottle** for
the entire episode; the preset does not prompt for language transitions.

Use `c` to start recording, press `c` again to finish and save, or `x` to discard.
Each launch uses a fresh `outputs/pnp_table_260908_subtask_eval_<UTC>/` directory
or `outputs/pnp_table_260908_full_task_eval_<UTC>/` for the full-task model,
with full task label `approach the table and pick the bottle`. The existing
recorder does not store live prompt changes as per-frame subtask annotations.

The tmux session is `sonic_inference`. An existing session is preserved and
causes the launcher to stop, preventing two clients from sharing action ports.
When switching from an older model, stop its policy/controller using the
established PC2 procedure before closing its workstation session.

## Subtask recorded-data evaluation results

All **34 episodes / 22,057 frames** were evaluated with exit code `0` using the
pinned Isaac-GR00T episode loader, recorded camera frames and states, and the
dataset's corrected per-frame language labels. The evaluator used 1,132 queries,
consuming at most 20 actions per query from the model's 40-action prediction.
It truncates the previous prediction at a prompt change and queries again on
the first frame carrying the new prompt. All 34 prompt boundaries were checked
against the saved query records, including boundaries off the 20-frame grid.

| Action group | Dimensions | MAE | MSE |
|--------------|-----------:|----:|----:|
| All actions | 78 | 0.00933793 | 0.00112695 |
| Motion token | 64 | 0.01044392 | 0.000527743 |
| Left hand | 7 | 0 | 0 |
| Right hand | 7 | 0.00856402 | 0.00773236 |

| Prompt | Frames | MAE | MSE |
|--------|-------:|----:|----:|
| approach the table | 12,662 | 0.00934716 | 0.000572067 |
| grasp the bottle | 1,912 | 0.00811877 | 0.00122806 |
| pick up the bottle | 7,483 | 0.00963383 | 0.00204003 |

These are unnormalized errors on **training recordings**, weighted by frame and
dimension. The combined score mixes motion-token and hand units. The left-hand
target and predicted channels are all zero, so their zero error provides no
evidence about active left-hand control. These results measure recorded action
reconstruction; they do not establish held-out generalization or real-robot
task success.

Median policy request time was 85.6 ms; p95 was 137.8 ms. These measurements are
from the Docker evaluation client on H100 and exclude the workstation/robot
camera and action-publication pipeline.

All predictions were finite. The saved arrays independently reproduced the
aggregate metrics, frame count, prompt counts, and complete boundary coverage.
Machine-readable copies: [summary](artifacts/pnp_table_260908_subtask_eval/summary.json)
and [audit](artifacts/pnp_table_260908_subtask_eval/audit.json).

## Containers and artifacts

All H100 training, inference, evaluation, and Python diagnostics run inside
Docker. Use the H100 host only for Docker management through SSH. No packages
were installed on the host.

- Subtask server container: `jihun_gr00t_n17_pnp_table_subtask_eval_gpu6_20260908`.
- Full-task server container: `jihun_gr00t_n17_pnp_table_full_task_eval_gpu7_20260908`.
- Evaluation client: ran inside `jihun_gr00t_n17_pnp_table_gpu6_20260908` and exited successfully.
- Image: `sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`.
- Each server exposes only its physical GPU as `cuda:0`; container port 5550 maps
  to host port 15554 for subtask or 15555 for full-task.
- Both servers mount the run root and existing Hugging Face cache read-only.
- Its `serve_pnp_table_offline.py` entry point reuses the training launcher's
  existing Cosmos identity shim and runs the official GR00T server with offline
  Hugging Face loading. Docker restart policy is `unless-stopped`.

H100 run root:
`/mnt/data01/jhkim/gr00t_runs/pnp_table_260908_n17_gpu6_20260908`.

Paths relative to that root:

| Artifact | Path |
|----------|------|
| Checkpoint | `subtask/train/pnp-table-260908-subtask-gpu6/checkpoint-20000/` |
| Full-task checkpoint | `full_task/train/pnp-table-260908-full-task-gpu7/checkpoint-20000/` |
| Evaluation log and exit code | `subtask/evaluation/open_loop_20260908/` |
| Summary and 34 per-episode JSON/NPZ pairs | `subtask/evaluation/open_loop_20260908/results/` |
| Server entry point | `evidence/serve_pnp_table_offline.py` |
| Executed evaluator | `evidence/evaluate_gr00t_subtasks_open_loop.py` |

The NPZ files contain target/predicted actions, frame prompts, query starts, and
action-group dimensions. A local copy is under
`/tmp/pnp-table-260908-subtask-eval/results/`.

Inspect the server through Docker:

```bash
ssh h100 docker logs --tail 50 jihun_gr00t_n17_pnp_table_subtask_eval_gpu6_20260908
ssh h100 docker logs --tail 50 jihun_gr00t_n17_pnp_table_full_task_eval_gpu7_20260908
```

To repeat recorded-data evaluation, choose a **new** output directory:

```bash
ssh h100 docker exec -w /workspace -e PYTHONPATH=/workspace \
  jihun_gr00t_n17_pnp_table_gpu6_20260908 \
  python /outputs/evidence/evaluate_gr00t_subtasks_open_loop.py \
  --dataset-path /outputs/subtask/dataset_view \
  --output-dir /outputs/subtask/evaluation/manual_repeat/results \
  --host 192.168.75.173 --port 15554 --stride 20
```

Implementation checks: five targeted tests passed, covering prompt-boundary
chunking, the new preset, and the existing launcher/left-only presets. Ruff and
the launcher's dry-run passed. No hardware test was performed.
