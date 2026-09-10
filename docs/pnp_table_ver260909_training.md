# pnp_table_ver260909 full-task training

This run trains GR00T N1.7 on the retained table-picking demonstrations from
September 8–9. The user reported that the earlier full-task model's real-robot
success rate was the same as or higher than the subtask model, and requested
full-task training for the new object-picking dataset.

Initialization is from the **official GR00T N1.7 3B base**, as explicitly chosen
by the user. It uses the same 20,000-step recipe as the previous table models.
The two physical stages, approaching the table and picking the object, share
one full-task language prompt throughout each episode.

## Dataset

Local source: `outputs/pnp_table_ver260909`, resolving to
`/mnt/data/jihun/datasets/G1_WBT_GR00T/pnp_table_ver260909`.

| Exact full-task prompt | Episodes | Frames |
|------------------------|---------:|-------:|
| approach the table and pick the bottle | 40 | 25,652 |
| approach the table and pick the water bottle | 1 | 633 |
| approach the table and pick the apple | 15 | 8,944 |
| approach the table and pick the brown bottle | 6 | 3,376 |
| Total | 62 | 38,605 |

All four verified prompt strings are preserved. The export has 62 retained
episodes after the user's deletion of 25 source episodes, contiguous episode
indices, and no interval exclusions in the retained episodes. All 62 episodes
belong to the supplied training split; no held-out split or class reweighting
was introduced. Per-object real-robot accuracy remains to be measured after
training, especially for the one-demonstration water-bottle label.

Validation completed before launch:

- All 62 videos fully decoded and matched their episode frame counts at
  640×480 and 50 FPS.
- All 1,922 numeric episode/column combinations matched the declared array
  shapes and Arrow dtypes and contained finite values.
- Frame, episode, global indices, 50 Hz timestamps, and per-frame task mappings
  passed validation. Features and modality mapping match the previous dataset.
- All 136 transferred files matched SHA-256 hashes: 317,367,621 bytes total.
- The pinned native GR00T loader checked every frame's language and decoded
  a sample for every prompt, with 46 state dimensions, 78 action dimensions,
  and action horizon 40.

H100 dataset path: `/mnt/data01/jhkim/datasets/pnp_table_ver260909`.
The container mounts it read-only at `/dataset`. A writable metadata copy at
`/outputs/dataset_view/meta` holds generated statistics; its `data` and `videos`
directories link to the read-only dataset.

## Training run

- Container: `jihun_gr00t_n17_pnp_table_ver260909_full_task_gpu7_20260910`.
- Physical GPU: **7**, exposed as CUDA device 0 in the container.
- Image: `sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`.
- Official base revision: `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Cosmos revision: `9ce19a195e423419c349abfc86fd07178b230561`.
- 20,000 steps; batch size 32; accumulation 1; BF16; no DeepSpeed.
- AdamW, learning rate `1e-4`, cosine schedule, warmup ratio 0.05,
  weight decay `1e-5`, seed 42.
- Same projector/diffusion fine-tuning, frozen LLM/visual backbone, state
  dropout 0.2, color jitter, and data sampling settings as the previous run.
- Save every 1,000 steps; retain five resumable checkpoints.
- Existing Hugging Face cache is read-only and loaded offline. W&B uses the
  existing credential mounted read-only and read inside the container.

The effective training-arguments audit confirmed batch size 32, BF16,
accumulation 1, learning rate `1e-4`, and no active DeepSpeed. The fresh-start
audit found no previous checkpoint in the new experiment directory.

Launch verification on September 10: training reached **step 410 / 20,000**.
The latest logged loss was **0.6058** (previous: 0.6119), with finite gradient
norms and no traceback or out-of-memory error.

Completion subsequently verified: training exited with code `0`, and
`checkpoint-20000/trainer_state.json` records step `20000/20000`. The final
checkpoint contains all three model weight shards.

W&B run: <https://wandb.ai/jihun-kim/gr00t-n1.7-pnp-table/runs/76n586bw>.

H100 run root:
`/mnt/data01/jhkim/gr00t_runs/pnp_table_ver260909_n17_full_task_gpu7_20260910`.

| Artifact | Path relative to the run root |
|----------|-------------------------------|
| Training log | `train/train.log` |
| Training exit code, when finished | `train/exit` |
| Trainer PID | `train/child.pid` |
| Exact training command | `train/command.txt` |
| Checkpoint directory | `train/pnp-table-ver260909-full-task-gpu7/checkpoint-<step>/` |
| Dataset audit | `evidence/pnp-table-ver260909-validation.json` |
| Native loader and transfer check | `evidence/native-loader-probe.json` |
| Source SHA-256 manifest | `evidence/pnp-table-ver260909.sha256` |
| Training launcher | `evidence/run_training.sh` |
| Pinned offline training entry point | `evidence/launch_gr00t_n17_pinned_finetune.py` |

Monitor through Docker:

```bash
ssh h100 docker exec jihun_gr00t_n17_pnp_table_ver260909_full_task_gpu7_20260910 \
  tail -f /outputs/train/train.log
```

All H100 training and diagnostic programs run inside Docker; the host is used
only for Docker management through SSH. Existing inference services were
preserved. No host software was installed.

## Inference server

The completed model is served at **192.168.75.173:15556** on H100 GPU **7**.

- Container: `jihun_gr00t_n17_pnp_table_ver260909_eval_gpu7_20260910`.
- Checkpoint inside the container:
  `/outputs/train/pnp-table-ver260909-full-task-gpu7/checkpoint-20000`.
- Same pinned Docker image and offline Cosmos loading used by the earlier servers.
- Read-only mounts for the run root and Hugging Face cache.
- Container port `5550` is published as `192.168.75.173:15556`.
- Restart policy: `unless-stopped`.

The workstation ping and a recorded episode-0/frame-0 inference request passed.
The response contained finite motion-token `[1,40,64]` and left/right-hand
`[1,40,7]` arrays. No robot control was launched.

Use the exact full-task prompt for the chosen object, as listed in the dataset
table above. The existing four-pane preset can connect with:

```bash
python3 gear_sonic/scripts/launch_pnp_table_eval.py full --policy-port 15556
```

That preset starts with `approach the table and pick the bottle` and retains its
existing recording-name convention. Other object prompts can be sent through
the keyboard pane as complete `t approach the table and pick the <object>`
commands; the recorder's task label remains the startup label.

The keyboard sequence is `k` → `i` → wait for straight standing → `p`.
Between attempts, use `p` → `i` → wait → `p`. `i` pauses evaluation, discards
old policy results, and returns through SONIC's straight standing planner
preset using measured joint feedback and the current heading. Restart the
workstation inference client to load this reset behavior; the H100 model
server does not need a restart.

Keep `zmq_manager` for keyboard repositioning. Use `m` to enter manual
planner mode, `w/s` forward/back, `a/d` sideways, `q/e` to turn, and `z` to
stop. Enter after each key. Movement stays on until the same key is pressed
again or `z` stops it; another movement key switches direction.
`m` again returns to paused POSE mode; use
`i → wait → p` for the next trial. See
[keyboard repositioning](source/tutorials/vla_inference.md#keyboard-repositioning-between-trials).

For the posture-only heading fix, rebuild and restart SONIC deploy as well as
the inference client. `i` holds the measured world heading while the posture
returns to standing, using the new deploy heading-reference telemetry.

Inspect the server log through Docker:

```bash
ssh h100 docker logs --tail 50 jihun_gr00t_n17_pnp_table_ver260909_eval_gpu7_20260910
```
