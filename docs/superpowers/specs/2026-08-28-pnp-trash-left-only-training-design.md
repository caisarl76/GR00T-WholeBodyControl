# PnP Trash Left-Only Dataset and GR00T N1.7 Training Design

## Goal

Publish two reproducible LeRobot v2.1 datasets containing only the 44
turn-left PnP-trash episodes, then start two fresh 20,000-step GR00T N1.7
fine-tunes on the H100 server:

- one model with a single full-episode prompt; and
- one model with four timestamped subtask prompts.

Neither run may resume or load a completed mixed-direction checkpoint.

## Verified Input

The immutable source dataset is `outputs/pnp_trash`, and its annotation
workbook is `outputs/pnp_trash/pnp_trash.xlsx`. The existing full-prompt and
subtask variants have identical source mappings. Their 72 valid episodes
divide into 44 turn-left episodes containing 93,312 frames and 28 turn-right
episodes.

The selected source episode indices are:

```text
4, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 26, 27,
62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79,
80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91
```

These indices are acceptance evidence, not a second manually maintained
selection input. The exporter derives the selection from the workbook.

## Dataset Architecture

### Direction selection

Extend the existing XLSX exporter with an optional direction filter. The
default value is `all`, preserving the current unfiltered behavior; the new
value is `left`.

Direction classification uses only subtask 3. After case folding and
whitespace normalization, a prompt must contain exactly one of `turn left`
or `turn right`. A left-filtered publication fails before staging outputs if
any valid episode contains both phrases or neither phrase, or if the complete
workbook does not classify to exactly 44 left and 28 right episodes.

Filtering occurs after workbook validation but before task-map construction,
episode numbering, Parquet rewriting, statistics, video copying, and
provenance generation. No frame interval is removed. Every selected episode
retains its complete trajectory, video, and four subtasks.

### Outputs and consumable publication

The two final dataset paths are:

- `outputs/pnp_trash_full_prompt_left_only`
- `outputs/pnp_trash_subtasks_left_only`

Both paths must be nonexistent before publication. An existing path is an
error even when it is an empty directory.

Output episodes are ordered by source episode index and renumbered from 0
through 43. The exporter regenerates global indices, task indices, episode
statistics, task and split metadata, Parquet files, and video paths. Both
variants contain the same ordered source mapping and the same 93,312 frames.

The existing exporter stages and validates both datasets, then renames their
directories sequentially. Because a process or host crash can occur between
those renames, directory presence alone does not make this pair consumable.
After both final paths exist and pass validation, the exporter atomically
writes this shared marker in the same parent filesystem:

```text
outputs/pnp_trash_left_only.release.json
```

The marker contains a format version, `state: complete`, direction, selected
and excluded counts, source-manifest and workbook hashes, and the final
manifest hash of each variant. It is written to a temporary regular file,
flushed and fsynced, renamed into place, and followed by a parent-directory
fsync. A consumer must validate this marker and both referenced manifests.
One visible dataset without the valid marker is an incomplete release and
must not be loaded, copied, or trained. Recovery from an incomplete release
is a separate explicit operation; the exporter does not silently remove a
visible directory.

### Provenance compatibility

Legacy and default unfiltered datasets keep provenance `schema_version: 1`
and do not require a `selection` object. Validation with `direction=all`
continues to accept that exact contract.

A left-filtered dataset uses `schema_version: 2` and requires:

```json
{
  "selection": {
    "direction": "left",
    "candidate_episodes": 72,
    "selected_episodes": 44,
    "excluded_episodes": 28
  }
}
```

Schema 2 retains the existing source-manifest and workbook SHA-256 values.
Each episode row continues to record source/output indices, length,
boundaries, snapped boundary frames and timestamps, and emitted prompts.
Direction-filtered validation rejects schema 1; unfiltered validation rejects
a schema-2 selection that claims to be left-filtered. The immutable raw
dataset and workbook, not either previous derived dataset, remain the source.

## Dataset Validation and Tests

The `validate_only` path accepts the same direction filter and reconstructs
the expected annotations from the immutable source and workbook. A left-only
release passes only if:

- each variant has 44 episodes and 93,312 frames;
- output episode indices are contiguous from 0 through 43;
- all frame/global/task indices, statistics, paths, and videos are valid;
- both variants have identical ordered source mappings and episode lengths;
- every selected subtask-3 prompt is left and no task contains `turn right`;
- schema-2 provenance reports the exact selection and source/workbook hashes;
- both final manifests match the shared complete release marker; and
- source and workbook hashes are unchanged across publication.

Tests cover direction normalization and rejection, the exact 44/28 split,
preservation of schema-1 default exports, schema-2 filtered exports,
provenance validation, cross-variant mapping, preexisting-path rejection,
failure before publication, and the rule that an absent or mismatched release
marker makes a sequentially exposed pair non-consumable.

## H100 Transfer Contract

The remote final paths are:

- `/mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_full_prompt_left_only`
- `/mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks_left_only`
- `/mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_left_only.release.json`

All three must be nonexistent before transfer. Copy into a uniquely named
staging directory, verify both manifests against the local release marker,
then rename the dataset directories and atomically write the remote marker.
Training gates require the valid remote marker; neither a staging path nor an
unmarked final directory is a dataset input.

## Immutable Training Inputs

Use H100 source commit
`626af89d3e914ec92eab5323e23b9ed44a7b26c8` and image ID
`sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`.
The tag `jihun/gr00t-n1.7:626af89` is descriptive only. Every container's
effective `.Image` must equal the immutable ID.

Resolve the base model locally and offline to this mounted-cache snapshot:

```text
/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/
snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
```

The launcher receives that snapshot path, not the mutable Hub name. The cache
must contain only the verified revision expected by this design. Do not
install packages or modify software on the server host.

## Containers and Writable Dataset Views

Create isolated containers and output roots:

| Variant | GPU | Container | Output root |
| --- | ---: | --- | --- |
| Full prompt | 7 | `jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828` | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828` |
| Subtasks | 6 | `jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828` | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828` |

Each container receives only its assigned physical GPU and mounts its
left-only dataset read-only at `/dataset`, its output root read-write at
`/outputs`, `/mnt/data01/huggingface` read-write at
`/root/.cache/huggingface`, and the existing W&B key as a read-only file. The
runtime uses `--init`, host IPC, unlimited memlock, a 67,108,864-byte stack
limit, no restart policy, and `sleep infinity`.

Direct GR00T loading from `/dataset:ro` is forbidden because the pinned
statistics path unconditionally rewrites `meta/relative_stats.json`. Before
any loader probe, smoke, gate, or production run, each container creates:

```text
/outputs/dataset_view/
  meta/    writable copy of /dataset/meta
  data     absolute symlink to /dataset/data
  videos   absolute symlink to /dataset/videos
```

Every GR00T command uses `/outputs/dataset_view`. A sorted SHA-256 manifest of
all regular files under `/dataset/meta` is recorded before view construction,
after statistics generation, after each smoke/gate, and at production
handoff. Every source metadata manifest must be identical. The initial copied
metadata manifest must match the source before GR00T writes only the view's
`relative_stats.json`. Source dataset manifests and the remote release marker
are also revalidated before production launch.

## Exact Fresh Training Recipe

Both variants use the official pinned `examples/finetune.sh` launcher and the
same recipe except for dataset, GPU, output, and experiment name:

- embodiment `UNITREE_G1_SONIC`;
- modality configuration `gr00t/configs/data/embodiment_configs.py`;
- maximum steps 20,000 and logging interval 10;
- global batch size 32, gradient accumulation 1, and four loader workers;
- AdamW Torch, learning rate `1e-4`, cosine schedule, warmup ratio `0.05`,
  weight decay `1e-5`, and maximum gradient norm `1.0`;
- BF16 and TF32 enabled, FP16 disabled, DeepSpeed stage 2, and gradient
  checkpointing disabled;
- seed 42, shuffling enabled, episode sampling rate `0.1`, shard size 1,024,
  and 100,000 shards per epoch;
- color jitter brightness `0.3`, contrast `0.4`, saturation `0.5`, and hue
  `0.08`;
- state dropout `0.2`;
- projector, diffusion model, and VLLN trainable; LLM and visual backbone
  frozen, with zero top LLM layers tuned;
- checkpoint interval 1,000 and retention limit five;
- `save_only_model=false`, producing complete resumable trainer state; and
- W&B online under project `gr00t-n1.7-pnp-trash`.

The distinct production experiment names are:

- `pnp-trash-full-prompt-left-only-gpu7-20260828`
- `pnp-trash-subtasks-left-only-gpu6-20260828`

Before execution, persist the complete launcher command, launcher and Python
entrypoint hashes, redacted environment, image/container inspection, base
snapshot revision, and expected recipe. Each run must preserve the resolved
`config.yaml`, `conf.yaml`, dataset statistics, processor/model configuration,
and training arguments emitted by the trainer. A post-start comparison must
show that the resolved configuration equals this contract.

## Resource Authority and Baseline Gate

No existing process or container is stopped, signaled, removed, or restarted
automatically. Existing allocations count against capacity even when they are
unrelated. An allocation that prevents a gate from passing blocks the launch.

The only potential stop candidates in this workflow are the two evaluation
servers below, and even these require a separate explicit user approval at
execution time for the exact current container IDs:

- `jihun_gr00t_n17_pnp_trash_full_eval_gpu7_20260828` (observed ID
  `815619a5a57c6a6e06d19514d8a05ffe359b0ae95c0979a0065184c579a98f7e`);
- `jihun_gr00t_n17_pnp_trash_subtasks_eval_gpu6_20260828` (observed ID
  `e90f21341c0fa533ae9673638faf7a3e371b5db2ba96bb447448dfac79f99877`).

IDs must be resolved again immediately before requesting approval; a changed
ID invalidates earlier approval. Completed mixed-direction trainer containers
and unrelated host processes are not stop candidates. Before an approved stop,
record Docker inspection, running state, restart policy, ports, image ID, and
the exact `docker start <container-id>` restoration command. If production is
not launched, restore the container immediately. If production needs the
capacity, restore it after the corresponding training job reaches terminal
exit, unless the user gives a different explicit instruction.

Before creating new containers, sample physical GPUs 6 and 7 once per second
for 30 seconds. Record UTC timestamp, index, UUID, used and total MiB, memory
fraction, compute utilization, and the process table. Baseline passes only if:

- every sample uses the expected H100 with 81,559 MiB total memory;
- maximum used memory on each GPU is below 25% (20,389.75 MiB);
- the same PID, process name, and GPU UUID set appears throughout, each
  process's allocation range is at most 256 MiB, and aggregate used-memory
  range is at most 512 MiB during the window; and
- ECC, retired-page, and row-remap health do not worsen.

Any failure blocks container creation or training and produces a baseline
result artifact; it does not grant authority to reclaim capacity.

## Smoke and Concurrent VRAM Gates

Run independent one-step W&B smokes through each writable dataset view. Each
must exit zero, emit a finite step-1 loss, save a complete checkpoint-1, and
persist its resolved configuration.

Then run both five-step jobs concurrently. Begin the monitor 30 seconds before
launch, sample once per second until 30 seconds after both jobs exit, and impose
a 30-minute wall-clock timeout. At least one sample must show both training
processes alive on their assigned GPUs. The gate passes only if:

- both jobs exit zero with finite losses and complete checkpoint-5 directories;
- every aggregate memory sample is strictly below 95% of 81,559 MiB
  (77,481.05 MiB), leaving more than 4,077.95 MiB free;
- neither log contains CUDA OOM, traceback, NaN, or infinite loss;
- GPU health does not worsen; and
- no nonbaseline, non-gate process appears on either GPU.

The monitor writes CSV rows with the baseline fields and JSONL process
snapshots. A result JSON records variant PIDs, container IDs, start/end times,
overlap evidence, per-GPU baseline/peak/free MiB and fraction, exit codes,
checkpoint verdicts, error scans, health deltas, threshold, and final
`pass`/`fail`. A timeout or monitor failure is a gate failure.

Re-run the 30-second baseline immediately before production. Production may
launch only if the process set still matches the accepted post-gate state and
both baselines remain below 25%.

## Complete Checkpoint Definition

A checkpoint named `checkpoint-N` is complete only when all conditions hold:

- no `.part`, `.tmp`, `.incomplete`, or zero-length file exists;
- `model.safetensors.index.json` parses, every referenced shard exists and is
  nonempty, the referenced shard set exactly equals the present
  `model-*-of-*.safetensors` set, and all tensor keys map to present shards;
- nonempty `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` load without
  error; optimizer state and parameter groups are present, scheduler
  `last_epoch` matches N, and RNG state contains the expected Python, NumPy,
  CPU, and CUDA state;
- `trainer_state.json` parses and `global_step` equals N;
- nonempty `training_args.bin`, `config.json`, `processor_config.json`,
  `statistics.json`, and `embodiment_id.json` exist; and
- `experiment_cfg` contains nonempty `config.yaml`, `conf.yaml`,
  `dataset_statistics.json`, `final_model_config.json`, and
  `final_processor_config.json`.

The checkpoint verifier writes a hash-and-size manifest and a structured
verdict. Directory existence alone is never checkpoint evidence.

## Production Status Model

Both 20,000-step production wrappers run detached and record shell PID, child
training PID, log, W&B identity, and an atomic eventual exit-status file.
Status is reported in distinct states:

- `process_started`: within 15 minutes, the wrapper and Python child are alive,
  the child is on the assigned GPU, W&B has a distinct run ID, dataset/model
  initialization appears in the log, and no exit-status file or fatal error
  exists;
- `first_step_finite`: within 30 minutes, a finite optimizer-step loss is
  recorded with no fatal error; and
- `training_healthy`: within 120 minutes, `first_step_finite` holds and a
  complete checkpoint-1000 passes the checkpoint verifier.

Missing a deadline fails that acceptance state and blocks any claim that the
run is healthy. It does not authorize killing or restarting the process; the
job is left intact pending explicit user direction. Initial handoff may report
`process_started`, but setup acceptance requires both jobs to reach
`training_healthy`. Completion remains a separate state requiring exit zero
and a complete checkpoint-20000.

## Artifact Retention and Cleanup

Smoke and concurrent-gate checkpoints are preserved by default. Cleanup is a
separate destructive action and may be proposed only after both production
runs have complete checkpoint-6000 directories and remain healthy. At that
time, resolve and report every candidate checkpoint directory under the two
new `smoke` and concurrent-gate roots, its size, and the fact that it is not a
production checkpoint. Request explicit user approval for those exact paths.
No approval is inferred from this design or from prior cleanup. Logs,
resolved configurations, W&B identities, monitor data, result JSON, and
checkpoint-verifier manifests are never cleanup targets.

## Failure Handling and Handoff

Each stage writes its PID where applicable, full log, atomic exit status, and
structured verdict beneath its assigned output root. A failed publication,
transfer, metadata-view check, loader probe, smoke, baseline, or concurrent
gate prevents production launch. A failed production run preserves complete
checkpoints and trainer state and is not silently restarted or resumed.

The secret-free handoff reports:

- local and remote release markers and both dataset manifest hashes;
- source commit, immutable image and base-model revisions, entrypoint hashes,
  container IDs, mounts, and GPU assignment;
- source metadata hashes before and after every GR00T stage;
- W&B run IDs and URLs;
- smoke, concurrent-gate, and checkpoint-verifier verdicts;
- production status and deadline for each state;
- current PIDs, latest finite loss, and latest complete checkpoint; and
- all preexisting container/process state, including any explicitly approved
  stop and its restoration status.

Credentials and secret contents never appear in commands, Docker environment
metadata, logs, evidence, or the repository.
