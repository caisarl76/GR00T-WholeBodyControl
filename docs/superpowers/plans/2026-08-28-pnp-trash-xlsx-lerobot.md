# PnP Trash XLSX-to-LeRobot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, generate, and validate separate four-subtask and full-prompt LeRobot v2.1 datasets from the valid rows in `outputs/pnp_trash/pnp_trash.xlsx`.

**Architecture:** A focused library parses the workbook, validates annotations, constructs deterministic task maps, rewrites retained parquet identifiers/language indices, copies videos, emits metadata/provenance, and validates staged outputs. A thin CLI supplies safe paths and publishes both variants only after both pass. The source dataset is read-only throughout.

**Tech Stack:** Python 3.10, standard-library ZIP/XML/JSON/hash/copy utilities, PyArrow, pytest, Tyro, LeRobot v2.1 metadata, Isaac-GR00T N1.7 loader validation.

---

## File Map

- Create `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`: workbook parsing, annotation models, boundary snapping, deterministic task maps, export, provenance, and structural validation.
- Create `gear_sonic/scripts/annotate_pnp_trash_dataset.py`: Tyro CLI and two-output staging/publication orchestration.
- Create `gear_sonic/tests/test_lerobot_xlsx_annotations.py`: unit and synthetic end-to-end coverage.
- Modify `docs/source/tutorials/data_collection.md`: record the reproducible conversion and validation commands.
- Generate `outputs/pnp_trash_subtasks/`: self-contained timestamped-subtask dataset.
- Generate `outputs/pnp_trash_full_prompt/`: self-contained full-prompt dataset.

### Task 1: Parse and validate XLSX annotations

**Files:**
- Create: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Create: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing parser tests**

Add a minimal XLSX fixture writer in the test module that produces workbook,
relationship, shared-string, and worksheet XML in a ZIP. Test mixed-case
headers, blank formatted rows, shared and inline strings, numeric cells,
boolean `valid`, valid-row selection, complete expected-episode coverage, and
exact prompt preservation:

```python
def test_load_annotations_selects_valid_rows_and_preserves_prompts(tmp_path):
    xlsx = write_xlsx(tmp_path / "annotations.xlsx", [
        ["episode", "Valid", "subtask1", "time1", "subtask2", "time2",
         "subtask3", "time3", "subtask4", "Full Prompt"],
        [0, 0, "", "", "", "", "", "", "", ""],
        [1, 1, "approach brown table", 1.0, "pick the cup", 2.0,
         "turn left and approach the trash bin", 3.0,
         "put it in to the trash bin", "  exact full prompt  "],
    ])
    rows = load_annotations(xlsx, expected_episodes={0, 1})
    assert [row.episode for row in rows] == [1]
    assert rows[0].subtasks == (
        "approach brown table", "pick the cup",
        "turn left and approach the trash bin", "put it in to the trash bin",
    )
    assert rows[0].full_prompt == "exact full prompt"
```

Add parameterized rejection tests for duplicate/missing headers, duplicate
episodes, fractional identifiers, `valid` outside `{0,1}`, missing valid-row
cells, non-finite times, and unordered boundaries.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
```

Expected: collection fails because `lerobot_xlsx_annotations` does not exist.

- [ ] **Step 3: Implement the annotation model and XLSX parser**

Define the public model and parser:

```python
@dataclass(frozen=True)
class EpisodeAnnotation:
    episode: int
    subtasks: tuple[str, str, str, str]
    boundaries_s: tuple[float, float, float]
    full_prompt: str

def load_annotations(
    path: Path,
    expected_episodes: set[int] | None = None,
) -> list[EpisodeAnnotation]:
    with ZipFile(path) as archive:
        rows = _read_first_worksheet(archive)
    return _validate_and_select_rows(rows, expected_episodes)
```

Resolve the first worksheet through `xl/workbook.xml` and its relationships,
decode shared/inline strings plus numeric/boolean cells, ignore rows whose
episode cell is empty, normalize only header spelling and outer prompt
whitespace, validate every meaningful row, and return valid annotations sorted
by episode.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the Task 1 command. Expected: parser tests pass with no warnings.

- [ ] **Step 5: Commit parser behavior**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: parse pnp trash xlsx annotations"
```

### Task 2: Map timestamp boundaries and prompts deterministically

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Modify: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing snapping and task-map tests**

Cover float32 artifacts and exact run semantics:

```python
def test_snap_boundaries_assigns_boundary_frame_to_later_subtask():
    timestamps = pa.array(np.arange(0, 4.02, 0.02), type=pa.float32())
    frames = snap_boundary_frames(timestamps, (1.0, 2.0, 3.0), fps=50)
    assert frames == (50, 100, 150)
    assert build_run_steps(len(timestamps), frames).tolist() == (
        [0] * 50 + [1] * 50 + [2] * 50 + [3] * 51
    )
```

Test a farther-than-half-frame boundary, zero-length snapped interval, and
stable maps under shuffled annotation input. The subtask map sorts by
`(step_number, prompt UTF-8 bytes)`; the full map sorts by prompt UTF-8 bytes.

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 pytest command. Expected: failures name the missing snapping
and task-map functions.

- [ ] **Step 3: Implement boundary and task-map helpers**

Define:

```python
def snap_boundary_frames(
    timestamps: pa.Array | pa.ChunkedArray,
    boundaries_s: tuple[float, float, float],
    fps: int,
) -> tuple[int, int, int]:
    values = np.asarray(timestamps.to_numpy(), dtype=np.float64)
    starts = tuple(_nearest_frame(values, boundary, fps) for boundary in boundaries_s)
    _validate_nonempty_runs(len(values), starts)
    return starts

def build_run_steps(length: int, starts: tuple[int, int, int]) -> np.ndarray:
    return np.repeat(np.arange(4), np.diff((0, *starts, length)))

def build_task_map(
    annotations: Sequence[EpisodeAnnotation], variant: Literal["subtasks", "full_prompt"]
) -> dict[str, int]:
    ordered_prompts = _ordered_unique_prompts(annotations, variant)
    return {prompt: index for index, prompt in enumerate(ordered_prompts)}
```

Compare the insertion point and preceding timestamp, prefer the preceding
frame on an exact distance tie, enforce `0 < start1 < start2 < start3 < length`,
and reject snap error above `0.5 / fps + 1e-6`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 pytest command. Expected: all current tests pass.

- [ ] **Step 5: Commit task assignment behavior**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: map xlsx times to lerobot tasks"
```

### Task 3: Export a synthetic self-contained LeRobot dataset

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Modify: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write a failing end-to-end export test**

Build a two-episode source fixture with fixed-size-list sensor columns,
float32 timestamps, all five identifier columns, schema metadata, metadata
JSON/JSONL, and dummy MP4 bytes. Mark one episode valid. Export both variants
and assert:

```python
assert subtask_info["total_episodes"] == 1
assert subtask_info["total_tasks"] == 4
assert full_info["total_tasks"] == 1
assert subtask_table["episode_index"].to_pylist() == [0] * 201
assert full_table["task_index"].to_pylist() == [0] * 201
assert task_runs(subtask_table["task_index"]) == [50, 50, 50, 51]
assert source_sensor.equals(subtask_table["observation.state"])
assert output_video.read_bytes() == source_video.read_bytes()
assert output_video.stat().st_ino != source_video.stat().st_ino
```

Also assert exact `tasks.jsonl`, `episodes.jsonl`, `info.json`,
`episodes_stats.jsonl`, and source-to-output provenance contents.

- [ ] **Step 2: Run the end-to-end test and verify RED**

Run the Task 1 pytest command. Expected: failure because export functions are
missing.

- [ ] **Step 3: Implement export construction**

Define:

```python
@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    variant: Literal["subtasks", "full_prompt"]
    episodes: int
    frames: int
    tasks: int

def export_variant(
    source: Path,
    destination: Path,
    annotations: Sequence[EpisodeAnnotation],
    variant: Literal["subtasks", "full_prompt"],
    source_sha256: str,
    workbook_sha256: str,
) -> ExportResult:
    context = _prepare_export(source, destination, annotations, variant)
    _write_episodes(context)
    _write_metadata_and_provenance(context, source_sha256, workbook_sha256)
    return context.result()
```

Read source metadata and each retained parquet with PyArrow. Replace only
`episode_index`, global `index`, and `task_index` using arrays of the original
Arrow field types; preserve `frame_index`, timestamp, all payload columns,
field metadata, and schema metadata. Write parquet to the source path pattern,
copy videos with `shutil.copy2`, transform per-episode statistics for changed
columns, and emit deterministic compact JSONL plus indented JSON metadata and
provenance.

- [ ] **Step 4: Run the end-to-end test and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit dataset export**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: export annotated lerobot variants"
```

### Task 4: Add structural validation and safe two-output CLI publication

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Create: `gear_sonic/scripts/annotate_pnp_trash_dataset.py`
- Modify: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing validator and CLI tests**

Corrupt, one at a time, metadata totals, episode indices, global indices,
task resolution, expected run count, sensor arrays, source/video hashes, video
link type, and provenance mapping; assert a targeted `DatasetValidationError`.
Test that the CLI refuses either existing output and that a forced second
variant failure publishes neither final path.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 1 pytest command. Expected: validator/CLI tests fail because the
interfaces are absent.

- [ ] **Step 3: Implement validation and orchestration**

Define:

```python
def validate_variant(
    source: Path,
    output: Path,
    annotations: Sequence[EpisodeAnnotation],
    variant: Literal["subtasks", "full_prompt"],
) -> dict[str, object]:
    report = _validate_metadata_and_parquet(source, output, annotations, variant)
    _validate_videos_and_provenance(source, output, annotations, report)
    return report

def export_both(
    dataset_path: Path,
    annotations_path: Path,
    subtasks_output_path: Path,
    full_prompt_output_path: Path,
) -> tuple[ExportResult, ExportResult]:
    return _stage_validate_and_publish_both(
        dataset_path,
        annotations_path,
        subtasks_output_path,
        full_prompt_output_path,
    )
```

Read the source episode indices and pass them as `expected_episodes` while
loading the workbook, then hash the source manifest before building. Create
unique staging siblings with
`tempfile.mkdtemp(dir=output.parent)`, export and validate both, compare the
source manifest again, then rename both staging directories. If the second
rename fails, rename the first published output back to staging before cleanup
so publication remains all-or-nothing under ordinary same-filesystem errors.
The CLI uses a Tyro dataclass with the four paths from the design and prints
the validation summaries.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Run static checks**

```bash
ruff check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
ruff format --check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit CLI and validator**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: validate and publish pnp trash datasets"
```

### Task 5: Document and generate the real datasets

**Files:**
- Modify: `docs/source/tutorials/data_collection.md`
- Generate: `outputs/pnp_trash_subtasks/`
- Generate: `outputs/pnp_trash_full_prompt/`

- [ ] **Step 1: Add the conversion runbook**

Document the exact CLI command, immutable-source rule, output meanings,
72-episode expectation, and the standalone validation command.

- [ ] **Step 2: Commit the runbook**

```bash
git add docs/source/tutorials/data_collection.md
git commit -m "docs: add xlsx annotation export workflow"
```

- [ ] **Step 3: Record the source manifest and confirm outputs are absent**

Run a read-only SHA-256 manifest over `outputs/pnp_trash` excluding the two
future sibling outputs. Confirm neither final output path exists.

- [ ] **Step 4: Generate both real datasets**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv_data_collection/bin/python \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --subtasks-output-path outputs/pnp_trash_subtasks \
  --full-prompt-output-path outputs/pnp_trash_full_prompt
```

Expected: two validation summaries, each reporting 72 episodes, 154,625
frames, 72 videos, and 9 tasks; subtask episodes have four runs and full-prompt
episodes have one run.

- [ ] **Step 5: Confirm source immutability**

Regenerate the source manifest and require byte-for-byte equality with the
pre-export manifest.

### Task 6: Run final repository and GR00T N1.7 acceptance gates

**Files:**
- Verify: `outputs/pnp_trash_subtasks/`
- Verify: `outputs/pnp_trash_full_prompt/`

- [ ] **Step 1: Run complete unit and static verification**

Run the Task 1 pytest command and both Task 4 Ruff commands. Expected: zero
failures and zero lint/format findings.

- [ ] **Step 2: Run a fresh standalone structural validation**

Invoke the CLI/library validation-only path for both final outputs, rather
than relying on the generation-time report. Expected: 72 episodes, 154,625
frames, 72 videos, 9 resolved tasks, and correct run counts in both outputs.

- [ ] **Step 3: Run Isaac-GR00T statistics generation**

For each output, from the available Isaac-GR00T checkout, run:

```bash
python gr00t/data/stats.py \
  --dataset-path /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_subtasks \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py
```

Repeat with
`/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_full_prompt`.
Expected: each command exits zero and generates aggregate statistics for all
72 episodes.

- [ ] **Step 4: Run the real GR00T N1.7 episode loader acceptance probe**

Instantiate `LeRobotEpisodeLoader` with
`MODALITY_CONFIGS["unitree_g1_sonic"]` for each dataset, traverse all 72
episodes, and resolve `language.annotation.human.task_description` frame by
frame. Assert four ordered prompt runs per subtask episode and one exact
full-prompt run per full-prompt episode.

- [ ] **Step 5: Audit final deliverables**

Compare every requirement in the approved design against fresh command output,
dataset metadata, parquet/task runs, provenance, video hashes/inodes, and the
source manifest. Report any unavailable external GR00T gate explicitly; do not
mark the goal complete until both requested datasets exist and all locally
available acceptance evidence has been read.
