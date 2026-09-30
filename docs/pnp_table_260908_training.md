# pnp_table_260908 GR00T N1.7 training

The two dataset variants completed parallel training on H100, each in a Docker
container with one physical GPU. Both completed 20,000 fine-tuning steps after
initialization from the official GR00T N1.7 3B base checkpoint.

| Variant | GPU | Container |
|---------|----:|-----------|
| Subtask | 6 | `jihun_gr00t_n17_pnp_table_gpu6_20260908` |
| Full-task | 7 | `jihun_gr00t_n17_pnp_table_full_task_gpu7_20260908` |

Parallel launch verification on September 8: subtask training reached step 5,110
and full-task training reached step 139. Latest logged losses were `0.0822` and
`1.1025`, respectively, with no traceback or out-of-memory error in either log.
The subtask `checkpoint-1000` was already checked for the expected model/trainer
files and finite loss history. The resolved recipe uses batch size 32,
accumulation 1, BF16, learning rate `1e-4`, checkpoint interval 1,000, and no
active DeepSpeed.
The subtask process continued without restarting when the full-task job was
moved from the serial queue to GPU 7 at the user's request.

Subtask completion verified on September 8: training exited with code `0`,
`trainer_state.json` records step `20000/20000`, and the final logged loss was
`0.0246`. Its checkpoint is now served for evaluation on H100 GPU 6 at
`192.168.75.173:15554`. See [evaluation results and the operator command](pnp_table_260908_evaluation.md).
Full-task completion was subsequently verified: its per-run exit code is `0`
and `checkpoint-20000/trainer_state.json` records `20000/20000`. Its Docker
PolicyServer now runs on GPU 7 at `192.168.75.173:15555`; the workstation preset
selects it with `python3 gear_sonic/scripts/launch_pnp_table_eval.py full`.
The recorded-data metrics in the evaluation report are for the subtask model.

H100 run root:
`/mnt/data01/jhkim/gr00t_runs/pnp_table_260908_n17_gpu6_20260908`.

## Data acceptance

Both datasets contain 34 episodes, 22,057 frames, and 34 ego-view videos at
640×480 / 50 FPS. Every video decoded and matched its parquet frame count.
Numeric values, dimensions, statistics, and episode/frame indices passed
validation. The variants have identical non-language data and video hashes.

| Dataset | Prompt | Frames |
|---------|--------|-------:|
| `pnp_table_260908_subtask` | approach the table | 12,662 |
| `pnp_table_260908_subtask` | grasp the bottle | 1,912 |
| `pnp_table_260908_subtask` | pick up the bottle | 7,483 |
| `pnp_table_260908_full_task` | approach the table and pick the bottle | 22,057 |

Subtask labels match `reviewed_annotations_v3` and the human review snapshot
SHA-256 `66fed28839015354aaf4471c87d0d1ff58e9e8a17229036e1e4a0f5167c6672f`.
The actual pinned GR00T loader checked every frame's language and decoded a
training observation for each variant: 46 state dimensions and 78 action
dimensions. All 155 transferred files matched their local SHA-256 manifests.

H100 dataset parent: `/mnt/data01/jhkim/datasets/pnp_table_260908`.
The original dataset directories beneath it are mounted read-only at `/datasets`.
Each run has a separate writable metadata view under
`/outputs/<variant>/dataset_view`, with data and videos linked to the read-only
source. The source datasets are not rewritten by GR00T statistics generation.

## Training recipe

- Image: `sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`.
- Official GR00T base revision: `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Cosmos revision: `9ce19a195e423419c349abfc86fd07178b230561`.
- PyTorch 2.7.1 / CUDA 12.8; each container exposes its assigned physical GPU
  as CUDA device 0.
- 20,000 steps per variant, batch size 32, gradient accumulation 1, learning
  rate `1e-4`, BF16, action horizon 40, standard SONIC embodiment configuration.
- Checkpoint every 1,000 steps; retain five full resumable checkpoints per run.
- Same tested offline launcher and optimizer/augmentation recipe as the previous
  N1.7 training. The task adapter changes only its W&B project to
  `gr00t-n1.7-pnp-table`.
- Hugging Face cache and W&B credential mounted from the existing H100 paths.
  The credential is read inside the container and never stored in Docker's
  configured environment. No host software was installed.

Each GPU runs one fine-tuning process. GPU 6's existing services used approximately
20 GiB and GPU 7's used approximately 13 GiB before adding their training jobs;
these services were preserved.

## Outputs and monitoring

Paths below are relative to the H100 run root:

| Variant | Experiment directory |
|---------|----------------------|
| Subtask | `subtask/train/pnp-table-260908-subtask-gpu6/` |
| Full-task | `full_task/train/pnp-table-260908-full-task-gpu7/` |

Each `<variant>/train/` contains `train.log`, `command.txt`, `child.pid`,
the resolved training arguments, the fresh-start record, and an `exit` file
when finished. Each experiment directory contains its `checkpoint-*` folders.

`execution-layout.json` records the parallel arrangement. Use each
`<variant>/train/exit` for its eventual training exit code.

The original serial queue waited for the subtask child to record its exit.
It then stopped at the reserved `full_task/train` directory, preventing a duplicate
GPU 6 full-task launch. Its observed root-level `queue.exit=1` is expected at that guard;
it is a retired scheduler result, not either model's training result. The queue
files no longer describe the overall two-GPU training status.

```bash
ssh h100 docker exec jihun_gr00t_n17_pnp_table_gpu6_20260908 tail -f /outputs/subtask/train/train.log
ssh h100 docker exec jihun_gr00t_n17_pnp_table_full_task_gpu7_20260908 tail -f /outputs/full_task/train/train.log
```

Run H100 workloads and diagnostic programs inside Docker. The host is used only
for Docker management through SSH; do not install or run Python/evaluation jobs
directly on the host.

Subtask W&B run: <https://wandb.ai/jihun-kim/gr00t-n1.7-pnp-table/runs/c7ul6vw0>.
Full-task W&B run: <https://wandb.ai/jihun-kim/gr00t-n1.7-pnp-table/runs/oh1oarmu>.

The run's `evidence/` directory retains dataset validation, SHA-256 manifests,
container configurations, native loader results, the pinned launcher,
`run_queue.sh`, and `run_full_task_gpu7.sh`.
