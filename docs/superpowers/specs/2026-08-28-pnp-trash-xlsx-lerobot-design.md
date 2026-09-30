# PnP Trash XLSX-to-LeRobot Dataset Design

**Date:** 2026-08-28
**Status:** Approved by the user's instruction to proceed

## Objective

Convert the annotations in `outputs/pnp_trash/pnp_trash.xlsx` into two
separate, training-ready LeRobot v2.1 datasets without modifying
`outputs/pnp_trash`:

- `outputs/pnp_trash_subtasks`: four timestamped task labels per valid
  episode; and
- `outputs/pnp_trash_full_prompt`: one episode-wide task label per valid
  episode.

Both datasets must retain only spreadsheet rows whose `Valid` value is `1`,
remain self-contained for transfer to the H100, and load through the GR00T
N1.7 LeRobot data path.

## Source Facts

The source contains 92 episodes and 198,846 frames at 50 Hz. The workbook has
one meaningful row for every source episode. Seventy-two rows are valid,
covering 154,625 frames. Every valid row has nonempty `subtask1` through
`subtask4`, `time1` through `time3`, and `Full Prompt` values. `time4` is empty
for all valid rows because subtask 4 continues to the episode end. The three
boundaries in every valid row are strictly increasing and fall inside the
source episode duration.

## Considered Approaches

### Materialized independent exports (selected)

Rewrite the required parquet identifiers and language indices, copy retained
videos and metadata into staging directories, validate them, and atomically
publish the two outputs. This consumes roughly two copies of the retained
video bytes but produces portable datasets with no dependency on the source.

### Hardlinked or symlinked exports

This saves local disk space. It is rejected because symlinks are not portable
to the H100 and hardlinks make ownership and later mutation surprising.

### In-place source rewrite

This uses the least disk space. It is rejected because it destroys the raw
recording dataset, cannot represent both annotation variants at once, and
makes annotation mistakes difficult to recover from.

## Converter Interface

Add a repository script with explicit paths and safe defaults:

```text
python gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --subtasks-output-path outputs/pnp_trash_subtasks \
  --full-prompt-output-path outputs/pnp_trash_full_prompt
```

The command refuses to overwrite either final output. It builds both outputs
in unique sibling staging directories. A failure removes only its own staging
directories and leaves the source and any pre-existing final output intact.
The final directories are renamed into place only after both variants pass
structural validation.

The XLSX reader uses Python's standard ZIP/XML support for the small scalar
schema needed here. This avoids adding a runtime package solely for workbook
ingestion while supporting the shared-string, inline-string, numeric, and
boolean cell encodings needed by ordinary XLSX files.

## Annotation Contract

Column names are matched case-insensitively after trimming whitespace.
Required columns are `episode`, `valid`, `subtask1`, `time1`, `subtask2`,
`time2`, `subtask3`, `time3`, `subtask4`, and `full prompt`. Duplicate episode
rows, duplicate headers, missing source episodes, non-integral episode/valid
values, missing required annotations, non-finite times, unordered boundaries,
and out-of-range boundaries are fatal.

Only `valid == 1` rows are exported. Other rows are omitted. Retained episodes
are ordered by source episode index and assigned output episode indices
`0..N-1`.

For the subtask variant, every source episode is divided into exhaustive,
non-overlapping half-open intervals:

1. `[episode start, time1)` uses `subtask1`;
2. `[time1, time2)` uses `subtask2`;
3. `[time2, time3)` uses `subtask3`; and
4. `[time3, episode end]` uses `subtask4`.

Each boundary snaps to the nearest source timestamp; equal-distance ties use
the earlier frame. The allowed snap error is at most half a nominal frame plus
a small float tolerance. The snapped frames must remain strictly increasing
and leave every subtask with at least one frame. A boundary frame belongs to
the later subtask.

For the full-prompt variant, every frame in an episode uses that row's exact
trimmed `Full Prompt` value. Prompt wording is not normalized or corrected;
the workbook is authoritative.

## Deterministic LeRobot Export

Each output is a LeRobot v2.1 directory with `data/`, `videos/`, and `meta/`.
For every retained episode:

- preserve frame count, frame order, timestamps, sensor values, actions,
  teleoperation data, Arrow field types, and schema metadata;
- replace `episode_index` with the contiguous output episode index;
- preserve the already-contiguous per-episode `frame_index`;
- replace global `index` with a contiguous dataset-wide index; and
- replace `task_index` according to the selected annotation variant.

Prompt strings are deduplicated exactly. Subtask tasks are ordered by subtask
number and then UTF-8 prompt bytes. Full prompts are ordered by UTF-8 bytes.
Indices are assigned contiguously from zero. This makes `tasks.jsonl`
independent of workbook row order.

`episodes.jsonl` records the new episode index, source length, and that
episode's prompt strings in temporal order. `info.json` updates episode,
frame, task, video, chunk, and train-split totals. `modality.json` is copied
unchanged because it already maps `annotation.human.task_description` to
`task_index`.

Retained videos are copied byte-for-byte to paths using the new episode
indices. They are independent regular files rather than links.

`episodes_stats.jsonl` retains unchanged feature statistics and recalculates
the statistics for `episode_index`, `index`, and `task_index` from the emitted
parquet. Each output also contains `meta/annotation_provenance.json`, including
source/workbook SHA-256 values, source-to-output episode mapping, original and
snapped boundary values, task mapping, and output variant.

## Validation

The converter validates both staged outputs before publication:

- required LeRobot metadata exists and parses;
- totals, split, task count, video count, and episode lengths agree;
- output episode, frame, and global indices are contiguous;
- all parquet task indices resolve through `tasks.jsonl`;
- every subtask episode has exactly four nonempty runs in workbook order;
- every full-prompt episode has exactly one run matching `Full Prompt`;
- timestamps and all non-identifier/non-task columns equal their source
  Arrow arrays;
- every copied video exists, is a regular non-symlink file, has a different
  inode from the source, and has the same SHA-256;
- source files have not changed during the export; and
- provenance covers exactly the 72 valid source episodes.

Automated tests first cover workbook parsing and rejection cases, timestamp
snapping (including float32 boundary artifacts), deterministic task maps, and
a small end-to-end synthetic LeRobot export. After generating the real
datasets, run the repository validator plus the available Isaac-GR00T N1.7
statistics/loader acceptance path. If the real GR00T checkout or environment
is unavailable locally, report that gate separately rather than representing
structural validation as loader validation.

## Non-Goals

- Training or starting a training job on the H100.
- Changing spreadsheet wording or validity decisions.
- Trimming frames within valid episodes.
- Re-encoding videos.
- Modifying GR00T N1.7 or the source dataset.
