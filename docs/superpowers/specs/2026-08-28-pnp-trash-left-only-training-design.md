# PnP Trash Left-Only Dataset and GR00T N1.7 Training Design

## Goal

Publish two reproducible LeRobot v2.1 datasets containing only the 44
turn-left PnP-trash episodes, then start two fresh 20,000-step GR00T N1.7
fine-tunes on the H100 server:

- one model with a single full-episode prompt; and
- one model with four timestamped subtask prompts.

The new runs must not resume or otherwise load either completed
mixed-direction checkpoint.

## Verified Input

The immutable source dataset is `outputs/pnp_trash`, and its annotation
workbook is `outputs/pnp_trash/pnp_trash.xlsx`. The already-published full
and subtask variants have identical source-episode mappings. Their 72 valid
episodes divide into:

- 44 turn-left episodes containing 93,312 frames; and
- 28 turn-right episodes.

The selected source episode indices are:

```text
4, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 26, 27,
62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79,
80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91
```

These indices are acceptance evidence, not a second manually maintained
selection input. The exporter must derive the selection from the workbook.

## Dataset Architecture

### Direction selection

Extend the existing XLSX annotation exporter with an optional direction
filter whose default is `all`, preserving existing behavior. The new
supported value is `left`.

Direction classification uses only subtask 3. After case folding and
whitespace normalization, a prompt must contain exactly one of `turn left`
or `turn right`. A left-filtered publication fails before writing outputs if
any valid episode contains both phrases or neither phrase, or if the complete
workbook does not classify to exactly 44 left and 28 right episodes.

The filter is applied to the validated `EpisodeAnnotation` sequence before
the task map, episode numbering, Parquet rewriting, statistics, video copy,
or provenance generation. No frame interval is removed. Every selected
episode retains all four subtasks and its complete video and state/action
trajectory.

### Outputs

Publish the two self-contained datasets together:

- `outputs/pnp_trash_full_prompt_left_only`
- `outputs/pnp_trash_subtasks_left_only`

Output episodes are ordered by source episode index and renumbered from 0
through 43. The exporter regenerates global indices, task indices, episode
statistics, task metadata, split metadata, Parquet files, and video paths.
Both variants must contain the same ordered source-to-output mapping and the
same 93,312 frames.

Publication retains the existing empty-destination and atomic pair behavior:
an existing nonempty destination is an error, and a failure cannot leave one
variant published without the other.

### Provenance

Keep the existing provenance schema compatible and add a `selection` object
to each `meta/annotation_provenance.json`:

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

Existing source-manifest and workbook SHA-256 values remain mandatory. Each
episode provenance row continues to record source/output indices, length,
boundaries, snapped boundary frames and timestamps, and emitted prompts.
The exporter must not rely on the previous derived datasets as sources.

## Dataset Validation

The normal `validate_only` path accepts the same direction filter and
reconstructs the expected filtered annotations from the immutable source and
workbook. A left-only dataset passes only if:

- each variant has 44 episodes and 93,312 frames;
- output episode indices are contiguous from 0 through 43;
- all frame/global/task indices, statistics, paths, and videos are valid;
- both variants have identical ordered source mappings and episode lengths;
- every selected subtask-3 prompt is classified as left and no task contains
  `turn right`;
- provenance reports the exact selection counts and current source/workbook
  hashes; and
- a fresh manifest hash can be recorded for transfer verification.

Tests cover direction normalization and rejection, the exact 44/28 split,
preservation of the default unfiltered export, filtered full/subtask
publication, provenance, cross-variant mapping, validation failures, and
atomic failure behavior.

## H100 Publication

After local validation, copy and hash-verify the datasets at:

- `/mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_full_prompt_left_only`
- `/mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks_left_only`

Transfer uses a staging directory followed by finalization only after the
remote manifest matches the local manifest. No training process may read a
staging or partial path.

## Training Environment

Use the existing H100 checkout at commit
`626af89d3e914ec92eab5323e23b9ed44a7b26c8` and the existing image
`jihun/gr00t-n1.7:626af89`. Do not install packages or modify software on the
server host.

Create isolated containers and output roots:

| Variant | GPU | Container | Output root |
| --- | ---: | --- | --- |
| Full prompt | 7 | `jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828` | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828` |
| Subtasks | 6 | `jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828` | `/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828` |

Each container receives only its assigned GPU and mounts:

- its left-only dataset read-only at `/dataset`;
- its output root read-write at `/outputs`;
- `/mnt/data01/huggingface` at `/root/.cache/huggingface`; and
- the existing W&B key as a read-only Docker secret file.

The containers use host IPC and the same shared-memory and resource settings
as the completed mixed-direction runs. Secret contents must never appear in
commands, logs, inspect artifacts, or the handoff report.

## Fresh Training Contract

Both jobs initialize from `nvidia/GR00T-N1.7-3B`. They do not pass a resume
path and do not mount the previous mixed-direction checkpoints as training
inputs.

Reuse the proven recipe independently for both variants:

- embodiment: `UNITREE_G1_SONIC`;
- modality configuration:
  `gr00t/configs/data/embodiment_configs.py`;
- maximum steps: 20,000;
- global batch size: 32;
- data-loader workers: 4;
- state dropout probability: 0.2;
- episode sampling rate: 0.1;
- shard size: 1,024;
- shards per epoch: 100,000;
- save interval: 1,000 steps; and
- retain five complete resumable checkpoints.

Use W&B online in project `gr00t-n1.7-pnp-trash` with distinct experiment
identities:

- `pnp-trash-full-prompt-left-only-gpu7-20260828`
- `pnp-trash-subtasks-left-only-gpu6-20260828`

## Launch Gates

The execution order is fail-closed:

1. Verify the exact checkout/image, remote dataset hashes, dataset loader,
   output nonexistence, W&B-key metadata, container-name availability, and
   current GPU 6/7 process and memory state.
2. Run an independent one-step smoke for each dataset. Each smoke must exit
   zero, report a finite loss, and save a complete checkpoint and experiment
   configuration.
3. Run both variants concurrently for five steps. Require finite losses,
   complete checkpoints, no CUDA out-of-memory event, no data-loader error,
   and adequate steady-state GPU headroom.
4. If existing evaluation-server containers prevent safe headroom, stop only
   those evaluation containers and repeat the concurrent gate. Do not remove
   them or their checkpoints.
5. Launch both detached 20,000-step production jobs only after all gates pass.

Smoke and concurrent-gate weights may be removed after their logs,
configuration, exit status, and validation evidence are retained. Production
outputs are never written into or over the previous mixed-direction runs.

## Failure Handling and Handoff

Every smoke, gate, and production script writes its PID, full log, and atomic
exit-status file under its assigned output root. A failed gate preserves its
diagnostic artifacts and prevents production launch. A failed production run
preserves all complete checkpoints and trainer state for diagnosis; it is not
silently restarted or resumed as part of this workflow.

The launch handoff reports, without secrets:

- local and remote dataset manifest hashes;
- exact source commit, image ID, container names, mounts, and GPU assignment;
- W&B run IDs/URLs;
- smoke and concurrent-gate results;
- production PIDs and active process state;
- latest finite loss and checkpoint for each run; and
- whether the previous evaluation servers remained running or were stopped.

Training is considered launched, not completed, when both detached production
processes are alive, bound to their designated GPUs, have initialized their
datasets and W&B runs, and have either emitted a finite first-step metric or
are demonstrably progressing through model/data initialization without an
exit-status file.
