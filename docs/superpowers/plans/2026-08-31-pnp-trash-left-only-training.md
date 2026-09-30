# PnP Trash Left-Only Dataset and GR00T N1.7 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish two validated 44-episode left-only LeRobot datasets and launch two fresh, W&B-enabled, pinned GR00T N1.7 training jobs on H100 GPUs 7 and 6.

**Architecture:** Extend the existing XLSX exporter with a fail-closed direction-selection and release-marker contract, keeping legacy schema-v1 exports unchanged. Add three focused operational scripts: an audited pinned-model launch shim, a payload-level checkpoint verifier, and a quantitative GPU-gate monitor. Generate and validate locally, transfer through an unconsumable staging path, then pass loader, smoke, concurrency, freshness, and health gates before production launch.

**Tech Stack:** Python 3.10/3.12, pytest, PyArrow, NumPy, Safetensors, PyTorch, LeRobot v2.1, Docker, NVIDIA H100, NVIDIA Isaac-GR00T commit `626af89d3e914ec92eab5323e23b9ed44a7b26c8`, W&B, SSH.

---

## Pinned Upstream References

- NVIDIA launcher mapping: `https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/experiment/launch_finetune.py`
- NVIDIA runtime and effective `TrainingArguments`: `https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/experiment/experiment.py`
- NVIDIA trainer resume behavior: `https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/experiment/trainer.py`
- NVIDIA Cosmos selector: `https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L476-L483`
- Pinned Cosmos revision: `https://huggingface.co/api/models/nvidia/Cosmos-Reason2-2B/revision/9ce19a195e423419c349abfc86fd07178b230561`

## File Map

- Modify `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`: direction classification, filtered selection, schema-v2 provenance, release-marker write/validation, and pair publication.
- Modify `gear_sonic/scripts/annotate_pnp_trash_dataset.py`: left-only CLI fields and release validation.
- Modify `gear_sonic/tests/test_lerobot_xlsx_annotations.py`: unit and end-to-end coverage for selection, provenance, marker preconditions, and release validation.
- Modify `docs/source/tutorials/data_collection.md`: documented left-only generation and validation commands.
- Create `gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py`: exact offline GR00T/Cosmos configuration shim using pinned upstream runtime code.
- Create `gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py`: shim configuration, selector, snapshot, and freshness tests without importing GR00T locally.
- Create `gear_sonic/scripts/verify_gr00t_n17_checkpoint.py`: structural, tensor-payload, trainer-state, and offline load verification.
- Create `gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py`: synthetic sharded-checkpoint and corruption tests.
- Create `gear_sonic/scripts/verify_gr00t_training_attempt.py`: real-log, Trainer-argument, W&B-identity, and immutable verdict verification for bounded attempts.
- Create `gear_sonic/tests/test_verify_gr00t_training_attempt.py`: fixture-driven terminal-metric, anchor-order, W&B, loss, and no-clobber tests.
- Create `gear_sonic/scripts/monitor_gr00t_gpu_gate.py`: 1 Hz baseline/gate sampling and structured pass/fail artifacts.
- Create `gear_sonic/tests/test_monitor_gr00t_gpu_gate.py`: threshold, stability, overlap, process-set, timeout, and artifact tests.
- Create after launch `docs/superpowers/progress/2026-08-31-pnp-trash-left-only-training.md`: secret-free execution evidence and current health state.

### Task 1: Add direction classification and selection

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Test: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing direction-selection tests**

Add imports for `AnnotationSelection` and `select_annotations_by_direction`, then add:

```python
def test_select_annotations_by_direction_keeps_complete_left_episodes() -> None:
    left = _annotation(4, ("approach", "pick", "  TURN   LEFT toward bin ", "put"), "full left")
    right = _annotation(9, ("approach", "pick", "turn right toward bin", "put"), "full right")

    selected, selection = select_annotations_by_direction(
        [right, left],
        "left",
        expected_counts=(1, 1),
    )

    assert [row.episode for row in selected] == [4]
    assert selection == AnnotationSelection(
        direction="left",
        candidate_episodes=2,
        selected_episodes=1,
        excluded_episodes=1,
    )
    assert selected[0].subtasks[0:4] == left.subtasks


@pytest.mark.parametrize(
    "prompt",
    ["approach the trash bin", "turn left then turn right"],
)
def test_select_annotations_by_direction_rejects_ambiguous_turn_prompt(prompt: str) -> None:
    row = _annotation(1, ("approach", "pick", prompt, "put"), "full")

    with pytest.raises(AnnotationError, match="exactly one turn direction"):
        select_annotations_by_direction([row], "left", expected_counts=(1, 0))


def test_select_annotations_by_direction_requires_exact_counts() -> None:
    left = _annotation(1, ("approach", "pick", "turn left", "put"), "full")

    with pytest.raises(AnnotationError, match="expected left=44 right=28"):
        select_annotations_by_direction([left], "left", expected_counts=(44, 28))


def test_select_annotations_by_direction_all_preserves_legacy_order() -> None:
    rows = [
        _annotation(2, ("a", "b", "unclassified", "d"), "two"),
        _annotation(1, ("a", "b", "unclassified", "d"), "one"),
    ]

    selected, selection = select_annotations_by_direction(rows, "all")

    assert selected == rows
    assert selection is None
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  -k 'select_annotations_by_direction'
```

Expected: collection fails because the two new symbols do not exist.

- [ ] **Step 3: Implement the minimal selector**

Add `Literal` to the module's `typing` imports and add near
`EpisodeAnnotation`:

```python
DirectionFilter = Literal["all", "left"]


@dataclass(frozen=True)
class AnnotationSelection:
    direction: Literal["left"]
    candidate_episodes: int
    selected_episodes: int
    excluded_episodes: int


def _turn_direction(prompt: str) -> Literal["left", "right"]:
    normalized = " ".join(prompt.casefold().split())
    has_left = re.search(r"\bturn left\b", normalized) is not None
    has_right = re.search(r"\bturn right\b", normalized) is not None
    if has_left == has_right:
        raise AnnotationError(
            f"subtask3 must contain exactly one turn direction, got {prompt!r}"
        )
    return "left" if has_left else "right"


def select_annotations_by_direction(
    annotations: Sequence[EpisodeAnnotation],
    direction_filter: DirectionFilter,
    *,
    expected_counts: tuple[int, int] | None = None,
) -> tuple[list[EpisodeAnnotation], AnnotationSelection | None]:
    if direction_filter == "all":
        return list(annotations), None
    if direction_filter != "left":
        raise AnnotationError(f"unknown direction filter: {direction_filter!r}")
    classified = [(row, _turn_direction(row.subtasks[2])) for row in annotations]
    left = sorted((row for row, direction in classified if direction == "left"), key=lambda row: row.episode)
    right_count = sum(direction == "right" for _row, direction in classified)
    if expected_counts is None:
        raise AnnotationError("left direction filter requires expected direction counts")
    if (len(left), right_count) != expected_counts:
        raise AnnotationError(
            f"direction counts differ: expected left={expected_counts[0]} right={expected_counts[1]}, "
            f"actual left={len(left)} right={right_count}"
        )
    return left, AnnotationSelection(
        direction="left",
        candidate_episodes=len(classified),
        selected_episodes=len(left),
        excluded_episodes=right_count,
    )
```

- [ ] **Step 4: Run the focused tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  -k 'select_annotations_by_direction'
```

Expected: four selected tests pass.

- [ ] **Step 5: Commit the selector**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: select left-only pnp trash episodes"
```

### Task 2: Add schema-v2 filtered provenance

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Test: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing provenance tests**

Add these tests (and import `AnnotationSelection`):

```python
def test_filtered_provenance_uses_schema_two_selection(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    output = tmp_path / "subtasks"
    annotation = _annotation(
        0,
        ("approach", "pick", "turn left and approach", "put"),
        "approach, pick, turn left and approach, put",
    )
    selection = AnnotationSelection("left", 2, 1, 1)
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
        selection=selection,
    )
    provenance = json.loads(
        (output / "meta/annotation_provenance.json").read_text(encoding="utf-8")
    )

    validate_variant(
        source,
        output,
        [annotation],
        "subtasks",
        workbook_sha256="b" * 64,
        source_manifest_sha256="a" * 64,
        selection=selection,
    )

    assert provenance["schema_version"] == 2
    assert provenance["selection"] == {
        "direction": "left",
        "candidate_episodes": 2,
        "selected_episodes": 1,
        "excluded_episodes": 1,
    }


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("direction", "right"),
        ("candidate_episodes", 3),
        ("selected_episodes", 2),
        ("excluded_episodes", 0),
    ],
)
def test_filtered_provenance_rejects_corrupt_selection(
    tmp_path: Path,
    field: str,
    corrupt_value: object,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    output = tmp_path / "subtasks"
    annotation = _annotation(
        0,
        ("approach", "pick", "turn left and approach", "put"),
        "approach, pick, turn left and approach, put",
    )
    selection = AnnotationSelection("left", 2, 1, 1)
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
        selection=selection,
    )
    provenance_path = output / "meta/annotation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["selection"][field] = corrupt_value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="selection"):
        validate_variant(
            source,
            output,
            [annotation],
            "subtasks",
            workbook_sha256="b" * 64,
            source_manifest_sha256="a" * 64,
            selection=selection,
        )


def test_complete_export_provenance_remains_schema_one(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    output = tmp_path / "subtasks"
    annotation = _annotation(
        0,
        ("approach", "pick", "turn left and approach", "put"),
        "approach, pick, turn left and approach, put",
    )
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
    )
    provenance = json.loads(
        (output / "meta/annotation_provenance.json").read_text(encoding="utf-8")
    )

    assert provenance["schema_version"] == 1
    assert "selection" not in provenance
```

The filtered test's exact selection assertion is:

```python
assert provenance["schema_version"] == 2
assert provenance["selection"] == {
    "direction": "left",
    "candidate_episodes": 2,
    "selected_episodes": 1,
    "excluded_episodes": 1,
}
```

- [ ] **Step 2: Run the provenance tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  -k 'provenance and (selection or complete_export)'
```

Expected: filtered-schema assertions fail because `export_variant` always emits schema 1.

- [ ] **Step 3: Extend export and validation signatures**

Change both functions to accept the same optional selection:

```python
def export_variant(
    source: Path,
    destination: Path,
    annotations: Sequence[EpisodeAnnotation],
    variant: AnnotationVariant,
    source_manifest_sha256: str,
    workbook_sha256: str,
    *,
    selection: AnnotationSelection | None = None,
) -> ExportResult:
    """Materialize one self-contained LeRobot annotation variant."""


def validate_variant(
    source: Path,
    output: Path,
    annotations: Sequence[EpisodeAnnotation],
    variant: AnnotationVariant,
    workbook_sha256: str,
    source_manifest_sha256: str | None = None,
    *,
    selection: AnnotationSelection | None = None,
) -> dict[str, object]:
    """Validate an output against its source and annotation contract."""
```

Build provenance with:

```python
task_rows = [
    {"task_index": task_index, "task": prompt}
    for prompt, task_index in sorted(task_map.items(), key=lambda item: item[1])
]
provenance = {
    "schema_version": 2 if selection is not None else 1,
    "variant": variant,
    "source": {
        "dataset_path": str(source.resolve()),
        "manifest_sha256": source_manifest_sha256,
    },
    "annotations": {"workbook_sha256": workbook_sha256},
    "tasks": task_rows,
    "episodes": provenance_episodes,
}
if selection is not None:
    provenance["selection"] = {
        "direction": selection.direction,
        "candidate_episodes": selection.candidate_episodes,
        "selected_episodes": selection.selected_episodes,
        "excluded_episodes": selection.excluded_episodes,
    }
```

Validation requires schema 1 and no selection when `selection is None`; otherwise it requires schema 2 and exact dataclass field equality.

- [ ] **Step 4: Run provenance and legacy tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  -k 'provenance or complete_export'
```

Expected: all selected tests pass, including legacy schema 1.

- [ ] **Step 5: Commit provenance support**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: record left-only annotation provenance"
```

### Task 3: Add consumable release markers

**Files:**
- Modify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Test: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write failing marker tests**

Cover these exact cases:

```python
def _write_two_direction_fixture_annotations(source: Path) -> Path:
    left = valid_row(episode=0)
    left[-1] = InlineText("approach, pick, turn left, put")
    right = valid_row(episode=1)
    right[6] = "turn right and approach the trash bin"
    right[-1] = InlineText("approach, pick, turn right, put")
    return write_xlsx(source / "pnp_trash.xlsx", [HEADERS, left, right])


def _make_existing_path(path: Path, kind: str) -> None:
    if kind == "file":
        path.write_text("occupied", encoding="utf-8")
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = path.parent / "marker-target"
        target.write_text("target", encoding="utf-8")
        path.symlink_to(target)
    elif kind == "broken_symlink":
        path.symlink_to(path.parent / "missing-marker-target")
    else:
        raise AssertionError(f"unknown existing path kind: {kind}")


def _publish_filtered_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    export_both(
        source,
        workbook,
        subtasks,
        full_prompt,
        direction_filter="left",
        expected_direction_counts=(1, 1),
        release_marker_path=marker,
    )
    return source, subtasks, full_prompt, marker


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_export_both_rejects_preexisting_release_marker(tmp_path: Path, kind: str) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    marker = tmp_path / "pnp_trash_left_only.release.json"
    _make_existing_path(marker, kind)

    with pytest.raises(AnnotationError, match="release marker already exists"):
        export_both(
            source,
            workbook,
            tmp_path / "subtasks",
            tmp_path / "full_prompt",
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )


def test_filtered_export_publishes_valid_release_marker(tmp_path: Path) -> None:
    _source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    reports = validate_release_marker(marker, subtasks, full_prompt)
    assert reports["state"] == "complete"
    assert reports["selected_episodes"] == 1


def test_release_without_marker_is_not_consumable(tmp_path: Path) -> None:
    _source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    marker.unlink()
    with pytest.raises(DatasetValidationError, match="release marker"):
        validate_release_marker(marker, subtasks, full_prompt)


def test_release_marker_rejects_mismatched_manifest_and_mapping(tmp_path: Path) -> None:
    _source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    sub_provenance = json.loads(
        (subtasks / "meta/annotation_provenance.json").read_text(encoding="utf-8")
    )
    full_provenance = json.loads(
        (full_prompt / "meta/annotation_provenance.json").read_text(encoding="utf-8")
    )
    mapping = lambda value: [
        (row["source_episode_index"], row["output_episode_index"], row["length"])
        for row in value["episodes"]
    ]
    assert mapping(sub_provenance) == mapping(full_provenance)

    value = json.loads(marker.read_text(encoding="utf-8"))
    value["datasets"]["subtasks"]["manifest_sha256"] = "0" * 64
    marker.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="manifest"):
        validate_release_marker(marker, subtasks, full_prompt)
```

The fixture writer must create one valid left and one valid right source episode so selection counts are `(1, 1)`.

- [ ] **Step 2: Run marker tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py -k 'release_marker'
```

Expected: missing API/signature failures.

- [ ] **Step 3: Implement atomic marker helpers**

Add `import os` and:

```python
def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if _path_lexists(temporary_path):
            temporary_path.unlink()
```

Add `validate_release_marker(marker, subtasks, full_prompt)` that rejects
non-regular/symlink markers, parses format version 1 and `state == "complete"`,
requires the exact two dataset basenames, and recomputes both dataset
manifests. It must also read both provenance files and require their
source-manifest hash, workbook hash, and schema-v2 selection object to agree
exactly with each other and with the marker before returning the parsed
selection summary.

- [ ] **Step 4: Extend `export_both` without changing legacy calls**

Use this signature:

```python
def export_both(
    dataset_path: Path,
    annotations_path: Path,
    subtasks_output_path: Path,
    full_prompt_output_path: Path,
    *,
    direction_filter: DirectionFilter = "all",
    expected_direction_counts: tuple[int, int] | None = None,
    release_marker_path: Path | None = None,
) -> tuple[ExportResult, ExportResult]:
```

Before resolving paths, reject any raw output or marker for which `_path_lexists` is true. For `left`, require a marker path, call `select_annotations_by_direction`, pass selection to both export/validate calls, publish both directories, calculate their manifests, then atomically write:

```python
{
    "format_version": 1,
    "state": "complete",
    "direction": "left",
    "candidate_episodes": selection.candidate_episodes,
    "selected_episodes": selection.selected_episodes,
    "excluded_episodes": selection.excluded_episodes,
    "source_manifest_sha256": before_manifest,
    "workbook_sha256": workbook_sha256,
    "datasets": {
        "subtasks": {"path": outputs[0].name, "manifest_sha256": dataset_manifest_sha256(outputs[0])},
        "full_prompt": {"path": outputs[1].name, "manifest_sha256": dataset_manifest_sha256(outputs[1])},
    },
}
```

Legacy `direction_filter="all"` requires `release_marker_path is None` and publishes exactly as before.

- [ ] **Step 5: Run marker, rollback, and legacy tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  -k 'release_marker or export_both or complete_export'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit release publication**

```bash
git add gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
git commit -m "feat: publish consumable pnp trash releases"
```

### Task 4: Expose and document the left-only CLI

**Files:**
- Modify: `gear_sonic/scripts/annotate_pnp_trash_dataset.py`
- Modify: `docs/source/tutorials/data_collection.md`
- Test: `gear_sonic/tests/test_lerobot_xlsx_annotations.py`

- [ ] **Step 1: Write the failing CLI validation test**

Add:

```python
def test_cli_validates_complete_left_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    workbook = source / "pnp_trash.xlsx"
    reports = annotation_cli_main(
        AnnotatePnpTrashConfig(
            dataset_path=source,
            annotations_path=workbook,
            subtasks_output_path=subtasks,
            full_prompt_output_path=full_prompt,
            direction_filter="left",
            expected_left_episodes=1,
            expected_right_episodes=1,
            release_marker_path=marker,
            validate_only=True,
        )
    )

    assert reports[0]["episodes"] == 1
    assert reports[1]["episodes"] == 1
    printed = capsys.readouterr().out
    assert '"direction": "left"' in printed
    assert '"state": "complete"' in printed


def test_cli_left_export_rejects_broken_marker_symlink(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    marker.symlink_to(tmp_path / "missing-release-marker")

    with pytest.raises(AnnotationError, match="release marker already exists"):
        annotation_cli_main(
            AnnotatePnpTrashConfig(
                dataset_path=source,
                annotations_path=workbook,
                subtasks_output_path=subtasks,
                full_prompt_output_path=full_prompt,
                direction_filter="left",
                expected_left_episodes=1,
                expected_right_episodes=1,
                release_marker_path=marker,
                validate_only=False,
            )
        )

    assert not subtasks.exists()
    assert not full_prompt.exists()
```

- [ ] **Step 2: Run the CLI test and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py -k 'cli and left'
```

Expected: config constructor rejects the new fields.

- [ ] **Step 3: Implement CLI fields and validation flow**

Add fields:

```python
direction_filter: Literal["all", "left"] = "all"
expected_left_episodes: int = 44
expected_right_episodes: int = 28
release_marker_path: Path = Path("outputs/pnp_trash_left_only.release.json")
```

Use this flow in `main` (with the existing `_source_episode_indices` helper):

```python
source = config.dataset_path.resolve()
workbook = config.annotations_path.resolve()
raw_subtasks = config.subtasks_output_path
raw_full_prompt = config.full_prompt_output_path
raw_marker = config.release_marker_path
expected_counts = (
    (config.expected_left_episodes, config.expected_right_episodes)
    if config.direction_filter == "left"
    else None
)
marker_path = raw_marker if config.direction_filter == "left" else None

if not config.validate_only:
    export_both(
        source,
        workbook,
        raw_subtasks,
        raw_full_prompt,
        direction_filter=config.direction_filter,
        expected_direction_counts=expected_counts,
        release_marker_path=marker_path,
    )

workbook_bytes = workbook.read_bytes()
all_annotations = load_annotations_bytes(
    workbook_bytes,
    expected_episodes=_source_episode_indices(source),
)
annotations, selection = select_annotations_by_direction(
    all_annotations,
    config.direction_filter,
    expected_counts=expected_counts,
)
source_manifest = dataset_manifest_sha256(source)
workbook_sha256 = hashlib.sha256(workbook_bytes).hexdigest()
subtasks = raw_subtasks.absolute()
full_prompt = raw_full_prompt.absolute()
reports = (
    validate_variant(
        source,
        subtasks,
        annotations,
        "subtasks",
        workbook_sha256=workbook_sha256,
        source_manifest_sha256=source_manifest,
        selection=selection,
    ),
    validate_variant(
        source,
        full_prompt,
        annotations,
        "full_prompt",
        workbook_sha256=workbook_sha256,
        source_manifest_sha256=source_manifest,
        selection=selection,
    ),
)
payload: dict[str, object] = {"outputs": reports}
if marker_path is not None:
    payload["release"] = validate_release_marker(
        marker_path.absolute(), subtasks, full_prompt
    )
print(json.dumps(payload, indent=2))
return reports
```

Do not call `Path.resolve()` on either configured output or the release marker
before `export_both`; doing so would erase the identity of a broken-symlink
target. Resolve only the immutable source/workbook. Pass the raw configured
publication paths into the exporter's `lexists` precondition, then use their
absolute non-symlink locations for validation.

- [ ] **Step 4: Document exact generation and validation commands**

Add to `docs/source/tutorials/data_collection.md`:

```bash
python gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --direction-filter left \
  --expected-left-episodes 44 \
  --expected-right-episodes 28 \
  --subtasks-output-path outputs/pnp_trash_subtasks_left_only \
  --full-prompt-output-path outputs/pnp_trash_full_prompt_left_only \
  --release-marker-path outputs/pnp_trash_left_only.release.json

python gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --direction-filter left \
  --expected-left-episodes 44 \
  --expected-right-episodes 28 \
  --subtasks-output-path outputs/pnp_trash_subtasks_left_only \
  --full-prompt-output-path outputs/pnp_trash_full_prompt_left_only \
  --release-marker-path outputs/pnp_trash_left_only.release.json \
  --validate-only
```

- [ ] **Step 5: Run complete exporter tests and lint**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
ruff check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
ruff format --check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
```

Expected: all tests and both Ruff commands exit zero.

- [ ] **Step 6: Commit CLI and documentation**

```bash
git add gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  docs/source/tutorials/data_collection.md
git commit -m "docs: add left-only pnp trash export workflow"
```

### Task 5: Generate and validate the real local release

**Files:**
- Create ignored: `outputs/pnp_trash_full_prompt_left_only/`
- Create ignored: `outputs/pnp_trash_subtasks_left_only/`
- Create ignored: `outputs/pnp_trash_left_only.release.json`

- [ ] **Step 1: Prove all three targets are absent without deleting anything**

```bash
python3 -c 'import os; paths=["outputs/pnp_trash_full_prompt_left_only","outputs/pnp_trash_subtasks_left_only","outputs/pnp_trash_left_only.release.json"]; assert all(not os.path.lexists(p) for p in paths), paths'
```

Expected: exit zero. Any failure blocks generation and is reported; do not remove the path.

- [ ] **Step 2: Generate the release**

```bash
.venv_data_collection/bin/python \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --direction-filter left \
  --expected-left-episodes 44 \
  --expected-right-episodes 28 \
  --subtasks-output-path outputs/pnp_trash_subtasks_left_only \
  --full-prompt-output-path outputs/pnp_trash_full_prompt_left_only \
  --release-marker-path outputs/pnp_trash_left_only.release.json
```

Expected: both variants report 44 episodes and 93,312 frames; the marker reports left=44 and excluded=28.

- [ ] **Step 3: Validate independently**

```bash
.venv_data_collection/bin/python \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  --dataset-path outputs/pnp_trash \
  --annotations-path outputs/pnp_trash/pnp_trash.xlsx \
  --direction-filter left \
  --expected-left-episodes 44 \
  --expected-right-episodes 28 \
  --subtasks-output-path outputs/pnp_trash_subtasks_left_only \
  --full-prompt-output-path outputs/pnp_trash_full_prompt_left_only \
  --release-marker-path outputs/pnp_trash_left_only.release.json \
  --validate-only
```

Expected: exit zero and identical counts.

- [ ] **Step 4: Record local hashes and cross-variant mapping evidence**

```bash
.venv_data_collection/bin/python -c '
import json
from pathlib import Path
from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import dataset_manifest_sha256
roots=[Path("outputs/pnp_trash_subtasks_left_only"),Path("outputs/pnp_trash_full_prompt_left_only")]
provenance=[json.loads((root/"meta/annotation_provenance.json").read_text()) for root in roots]
mapping=lambda value:[(row["source_episode_index"],row["output_episode_index"],row["length"]) for row in value["episodes"]]
expected_sources=[4,5,6,7,11,13,14,15,16,17,18,19,20,23,24,26,27,62,63,64,66,67,68,69,70,71,72,73,74,76,77,78,79,80,81,82,83,85,86,87,88,89,90,91]
assert mapping(provenance[0]) == mapping(provenance[1])
assert len(mapping(provenance[0])) == 44
assert [row[0] for row in mapping(provenance[0])] == expected_sources
assert [row[1] for row in mapping(provenance[0])] == list(range(44))
assert sum(row[2] for row in mapping(provenance[0])) == 93312
print(*(f"{root.name} {dataset_manifest_sha256(root)}" for root in roots),sep="\n")
'
```

Capture the same evidence for transfer:

```bash
.venv_data_collection/bin/python -c '
from pathlib import Path
from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import dataset_manifest_sha256
for root in (Path("outputs/pnp_trash_subtasks_left_only"), Path("outputs/pnp_trash_full_prompt_left_only")):
    print(root.name, dataset_manifest_sha256(root))
' | tee /tmp/pnp-trash-left-local-hashes.txt
```

Expected: two manifest lines and no assertion failure. Save the hashes for the transfer gate.

### Task 6: Implement the pinned offline launch shim

**Files:**
- Create: `gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py`
- Create: `gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py`

- [ ] **Step 1: Write failing pure-helper tests**

Import `json`, `Path`, `SimpleNamespace` from `types`, and `pytest`; then test
constants and helpers against fake configs:

```python
def test_apply_runtime_pins_preserves_selector_name_and_sets_offline_fields(tmp_path: Path) -> None:
    config = fake_config()
    apply_runtime_pins(config, cache_root=tmp_path)
    assert config.model.model_name == "nvidia/Cosmos-Reason2-2B"
    assert config.model.model_revision == "9ce19a195e423419c349abfc86fd07178b230561"
    assert config.training.transformers_local_files_only is True
    assert config.training.transformers_cache_dir == str(tmp_path / "hub")


def test_resolve_snapshot_matches_realistic_hub_cache_layout(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots"
        / COSMOS_REVISION
    )
    expected.mkdir(parents=True)

    def realistic_snapshot_download(
        *, repo_id: str, revision: str, cache_dir: str, local_files_only: bool
    ) -> str:
        assert local_files_only is True
        repo_folder_name = f"models--{repo_id.replace('/', '--')}"
        snapshot = Path(cache_dir) / repo_folder_name / "snapshots" / revision
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        return str(snapshot)

    assert (
        resolve_snapshot(tmp_path, snapshot_download=realistic_snapshot_download)
        == expected.resolve()
    )


def test_assert_fresh_experiment_rejects_checkpoint(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    (experiment / "checkpoint-1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="fresh experiment"):
        assert_fresh_experiment(
            experiment,
            get_last_checkpoint=lambda _path: str(experiment / "checkpoint-1"),
            audit_path=tmp_path / "freshness-runtime.json",
        )


def test_assert_fresh_experiment_creates_empty_dir_and_records_none(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    audit_path = tmp_path / "freshness-runtime.json"

    assert_fresh_experiment(
        experiment,
        get_last_checkpoint=lambda path: None if Path(path).is_dir() else "missing",
        audit_path=audit_path,
    )

    assert experiment.is_dir()
    assert list(experiment.iterdir()) == []
    assert json.loads(audit_path.read_text()) == {
        "experiment": str(experiment),
        "get_last_checkpoint": None,
    }


def test_verify_backbone_selector_requires_qwen3_backbone() -> None:
    config = fake_config()
    verify_backbone_selector(config, lambda _model: FakeQwen3Backbone)


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_parse_preflight_only_accepts_closed_boolean(
    value: str | None,
    expected: bool,
) -> None:
    assert parse_preflight_only(value) is expected


def test_parse_preflight_only_rejects_other_values() -> None:
    with pytest.raises(RuntimeError, match="GR00T_PINNED_PREFLIGHT_ONLY"):
        parse_preflight_only("true")


def test_assert_recipe_contract_rejects_skip_weight_loading() -> None:
    config = fake_config()
    config.training.skip_weight_loading = True
    with pytest.raises(RuntimeError, match="training.skip_weight_loading"):
        assert_recipe_contract(config)


def _training_arguments_payload(deepspeed: object) -> dict[str, object]:
    return {
        "deepspeed": deepspeed,
        "report_to": ["wandb"],
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 1000,
        "save_total_limit": 5,
        "save_only_model": False,
        "fp16": False,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 4,
        "seed": 42,
    }


def test_training_arguments_audit_records_effective_deepspeed_none(tmp_path: Path) -> None:
    payload = _training_arguments_payload(None)
    arguments = SimpleNamespace(
        deepspeed=None,
        to_dict=lambda: payload,
    )
    module = SimpleNamespace(TrainingArguments=lambda **_kwargs: arguments)
    destination = tmp_path / "training-arguments.json"
    install_training_arguments_audit(module, destination)

    returned = module.TrainingArguments(deepspeed=None)

    assert returned is arguments
    assert json.loads(destination.read_text())["deepspeed"] is None


def test_training_arguments_audit_rejects_active_deepspeed(tmp_path: Path) -> None:
    deepspeed = {"zero_optimization": {"stage": 2}}
    arguments = SimpleNamespace(
        deepspeed=deepspeed,
        to_dict=lambda: _training_arguments_payload(deepspeed),
    )
    module = SimpleNamespace(TrainingArguments=lambda **_kwargs: arguments)
    install_training_arguments_audit(module, tmp_path / "training-arguments.json")

    with pytest.raises(RuntimeError, match="deepspeed"):
        module.TrainingArguments(deepspeed=arguments.deepspeed)
```

- [ ] **Step 2: Run the tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py
```

Expected: module import failure.

- [ ] **Step 3: Implement import-safe helpers and the GR00T main path**

The module must not import GR00T at module import time. Define constants,
`apply_runtime_pins`, `assert_offline_environment`, `resolve_snapshot`,
`assert_fresh_experiment`, `assert_recipe_contract`,
`run_offline_preflight`, `install_training_arguments_audit`, and
`verify_backbone_selector`. After Tyro has
created `ft_config`, build the config with the pinned launcher's complete
mapping below; these assignments are the audited contract and none may be
omitted:

`cache_root` means the mounted Hugging Face home root. Both
`apply_runtime_pins` and `resolve_snapshot` must derive
`hub_cache = cache_root / "hub"`. Set the Transformers `cache_dir` to that
Hub cache and pass the same Hub cache to `snapshot_download`; a Hub lookup
constructs `hub_cache / repo_folder_name(...) / "snapshots" / revision`.
For the production mount this distinguishes `/root/.cache/huggingface` from
the effective `/root/.cache/huggingface/hub` cache without changing the mount.

```python
dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
config = get_default_config().load_dict(
    {
        "data": {
            "download_cache": False,
            "datasets": [
                {
                    "dataset_paths": dataset_paths,
                    "mix_ratio": 1.0,
                    "embodiment_tag": embodiment_tag,
                }
            ],
        }
    }
)
config.load_config_path = None

config.model.tune_llm = ft_config.tune_llm
config.model.tune_visual = ft_config.tune_visual
config.model.tune_projector = ft_config.tune_projector
config.model.tune_diffusion_model = ft_config.tune_diffusion_model
config.model.state_dropout_prob = ft_config.state_dropout_prob
config.model.random_rotation_angle = ft_config.random_rotation_angle
config.model.color_jitter_params = ft_config.color_jitter_params
config.model.extra_augmentation_config = (
    json.loads(ft_config.extra_augmentation_config)
    if ft_config.extra_augmentation_config
    else None
)
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.model_name = COSMOS_MODEL_ID
config.model.backbone_trainable_params_fp32 = True
config.model.use_relative_action = True

config.training.experiment_name = ft_config.experiment_name
config.training.start_from_checkpoint = ft_config.base_model_path
config.training.optim = "adamw_torch"
config.training.global_batch_size = ft_config.global_batch_size
config.training.dataloader_num_workers = ft_config.dataloader_num_workers
config.training.learning_rate = ft_config.learning_rate
config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
config.training.output_dir = ft_config.output_dir
config.training.save_steps = ft_config.save_steps
config.training.save_total_limit = ft_config.save_total_limit
config.training.num_gpus = ft_config.num_gpus
config.training.use_wandb = ft_config.use_wandb
config.training.max_steps = ft_config.max_steps
config.training.weight_decay = ft_config.weight_decay
config.training.warmup_ratio = ft_config.warmup_ratio
config.training.wandb_project = ft_config.wandb_project
config.training.save_only_model = ft_config.save_only_model
config.training.skip_weight_loading = ft_config.skip_weight_loading

config.data.shard_size = ft_config.shard_size
config.data.episode_sampling_rate = ft_config.episode_sampling_rate
config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
apply_runtime_pins(config, cache_root=Path("/root/.cache/huggingface"))
```

`assert_recipe_contract(config)` compares the resolved values below and raises
with a field-by-field diff. Fields mapped from `ft_config` must still be checked
here so a missing CLI argument cannot silently inherit an unsuitable default:

```python
expected = {
    "training.optim": "adamw_torch",
    "training.global_batch_size": 32,
    "training.batch_size": None,
    "training.dataloader_num_workers": 4,
    "training.learning_rate": 1e-4,
    "training.gradient_accumulation_steps": 1,
    "training.lr_scheduler_type": "cosine",
    "training.weight_decay": 1e-5,
    "training.warmup_ratio": 0.05,
    "training.warmup_steps": 0,
    "training.max_grad_norm": 1.0,
    "training.logging_steps": 10,
    "training.num_gpus": 1,
    "training.use_ddp": False,
    "training.tf32": True,
    "training.fp16": False,
    "training.bf16": True,
    "training.gradient_checkpointing": False,
    "training.save_only_model": False,
    "training.skip_weight_loading": False,
    "training.save_total_limit": 5,
    "training.wandb_project": "gr00t-n1.7-pnp-trash",
    "data.shard_size": 1024,
    "data.episode_sampling_rate": 0.1,
    "data.num_shards_per_epoch": 100000,
    "data.shuffle": True,
    "data.seed": 42,
    "model.tune_llm": False,
    "model.tune_visual": False,
    "model.tune_projector": True,
    "model.tune_diffusion_model": True,
    "model.tune_vlln": True,
    "model.tune_top_llm_layers": 0,
    "model.state_dropout_prob": 0.2,
    "model.random_rotation_angle": None,
    "model.extra_augmentation_config": None,
    "model.color_jitter_params": {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    },
}
```

`main` imports pinned `FinetuneConfig`, `get_default_config`, modality loader,
`EmbodimentTag`, `get_backbone_cls`, `get_last_checkpoint`, and `run` only after
the offline environment has passed validation. Immediately before any such
import, it sets `NO_ALBUMENTATIONS_UPDATE=1` in the real process environment so
the pinned Albumentations import chain cannot request `pypi.org`. Accept an
existing exact value of `1`, but fail closed on any other preexisting value
rather than overwriting it. When a separate environment mapping is injected
for pure tests, restore both that mapping and the real process environment to
their prior state after dependency loading; normal production use mutates
`os.environ` and retains `1`. `REQUIRED_OFFLINE_ENV` remains the strict three
Hugging Face offline flags below. Main then resolves the embodiment, loads the
modality module, applies the mapping above, asserts the Cosmos snapshot, calls
`verify_backbone_selector(config.model, get_backbone_cls)`, and calls
`assert_fresh_experiment` on
`Path(config.training.output_dir) / config.training.experiment_name` before
`run(config)`. It also calls `assert_recipe_contract(config)`. The single-GPU
post-start gate separately reads the effective Hugging Face
`TrainingArguments` and requires `deepspeed is None`; the unused serialized
GR00T default `training.deepspeed_stage == 2` is not treated as active
DeepSpeed.

`assert_fresh_experiment` requires the exact experiment path to be absent by
`lexists`, creates that directory once, immediately calls pinned
`get_last_checkpoint(str(experiment))`, requires `None`, requires the directory
still be empty, and atomically writes the result to the absolute nonexistent
path in `GR00T_FRESHNESS_AUDIT_PATH`. It never accepts or reuses a preexisting
empty directory. This ordering proves both required facts: the directory was
absent immediately before this launch, and pinned checkpoint discovery
returned `None` after the shim created the empty directory required by that
API.

`GR00T_PINNED_PREFLIGHT_ONLY` is the sole operational branch. Reject any value
other than `0` or `1`. With value `1`, require `config.training.use_wandb is
False`, instantiate pinned
`gr00t.model.gr00t_n1d7.setup.Gr00tN1d7Pipeline(config, save_cfg_dir)`, call
`pipeline.setup()` so the official local-only model, processor, and real
dataset paths all load, record the model/processor class names, release the
objects, run `gc.collect()` and `torch.cuda.empty_cache()`, then exit without
calling `run(config)`. With value `0` or unset, require
`config.training.use_wandb is True` and call `run(config)`.

Preflight cleanup attempts both dataset closes even when either fails, clears
all pipeline-held model, processor, dataset, and collator references, and calls
`gc.collect()` and `torch.cuda.empty_cache()` exactly once. A setup exception
remains primary and chains any cleanup failures; without a setup exception,
the first cleanup failure is primary and chains later cleanup failures.

Before the normal `run(config)` branch, require an absolute nonexistent path
from `GR00T_TRAINING_ARGS_AUDIT_PATH`. `install_training_arguments_audit`
wraps the pinned `gr00t.experiment.experiment.TrainingArguments` constructor,
calls the original constructor unchanged, asserts the returned object's
`deepspeed is None`, and atomically writes a secret-free JSON projection of
`deepspeed,report_to,per_device_train_batch_size,gradient_accumulation_steps,
learning_rate,lr_scheduler_type,weight_decay,warmup_ratio,max_grad_norm,
logging_steps,save_steps,save_total_limit,save_only_model,fp16,bf16,tf32,
gradient_checkpointing,optim,dataloader_num_workers,seed`. It then returns the
original `TrainingArguments` object. Add pure tests with a fake constructor
that prove `deepspeed=None` is recorded and a non-null value raises before the
trainer is constructed.

Use these immutable constants:

```python
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT_RELATIVE = (
    "models--nvidia--Cosmos-Reason2-2B/"
    "snapshots/9ce19a195e423419c349abfc86fd07178b230561"
)
ALBUMENTATIONS_UPDATE_ENV = "NO_ALBUMENTATIONS_UPDATE"
ALBUMENTATIONS_UPDATE_VALUE = "1"
REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
```

The script exits before `run(config)` if the experiment directory preexists,
a checkpoint resolves, the selector differs, or the local revision resolves
elsewhere.

All smoke, concurrent-gate, and production wrappers invoke this exact command
shape, substituting only `MAX_STEPS`, `SAVE_STEPS`, `OUTPUT_DIR`, and
`EXPERIMENT_NAME` with the attempt-specific values:

```bash
cd /workspace
uv run python /outputs/evidence/launch_gr00t_n17_pinned_finetune.py \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset-path /outputs/dataset_view \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --save-total-limit 5 \
  --save-steps "$SAVE_STEPS" \
  --max-steps "$MAX_STEPS" \
  --use-wandb \
  --wandb-project gr00t-n1.7-pnp-trash \
  --global-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --learning-rate 0.0001 \
  --weight-decay 0.00001 \
  --warmup-ratio 0.05 \
  --state-dropout-prob 0.2 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  --episode-sampling-rate 0.1 \
  --shard-size 1024 \
  --num-shards-per-epoch 100000
```

- [ ] **Step 4: Run tests and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py
ruff check --no-cache gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py
ruff format --check --no-cache gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the shim**

```bash
git add gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py
git commit -m "feat: pin offline gr00t finetune models"
```

### Task 7: Implement checkpoint payload verification

**Files:**
- Create: `gear_sonic/scripts/verify_gr00t_n17_checkpoint.py`
- Create: `gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py`

- [ ] **Step 1: Write synthetic checkpoint tests**

Import `json`, `Path`, `pytest`, `torch`, and
`safetensors.torch.save_file`, then build the exact synthetic fixture and
corruption table below:

```python
def _write_checkpoint_fixture(root: Path, step: int = 5) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    shards = {
        "model-00001-of-00002.safetensors": {"action.weight": torch.ones(1)},
        "model-00002-of-00002.safetensors": {"projector.weight": torch.ones(1)},
    }
    for filename, tensors in shards.items():
        save_file(tensors, checkpoint / filename)
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {
            "action.weight": "model-00001-of-00002.safetensors",
            "projector.weight": "model-00002-of-00002.safetensors",
        },
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    torch.save(
        {"state": {0: {"step": torch.tensor(5)}}, "param_groups": [{"params": [0]}]},
        checkpoint / "optimizer.pt",
    )
    torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
    torch.save(
        {"python": (3, (1,), None), "numpy": ("MT19937",), "cpu": b"cpu", "cuda": [b"cuda"]},
        checkpoint / "rng_state.pth",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    for filename in (
        "training_args.bin",
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ):
        (checkpoint / filename).write_bytes(b"fixture")
    experiment_cfg = checkpoint / "experiment_cfg"
    experiment_cfg.mkdir()
    for filename in (
        "config.yaml",
        "conf.yaml",
        "dataset_statistics.json",
        "final_model_config.json",
        "final_processor_config.json",
    ):
        (experiment_cfg / filename).write_bytes(b"fixture")
    return checkpoint


def test_valid_sharded_checkpoint_passes_payload_verification(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    verdict = verify_checkpoint_structure(checkpoint, expected_step=5)
    assert verdict["status"] == "pass"
    assert verdict["tensor_keys"] == ["action.weight", "projector.weight"]


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("corrupt_shard", "Safetensors|safetensors"),
        ("index_mismatch", "tensor key map"),
        ("unreferenced_shard", "unreferenced"),
        ("unsharded", "sharded-only"),
        ("pytorch_bin", "PyTorch|bin"),
        ("trainer_step", "global_step"),
        ("scheduler_step", "last_epoch"),
        ("rng_key", "RNG"),
        ("partial", "partial"),
        ("temporary", "temporary|tmp"),
        ("incomplete", "incomplete"),
        ("zero_length", "zero-length"),
    ],
)
def test_checkpoint_corruption_fails_closed(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    if corruption == "corrupt_shard":
        (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"corrupt")
    elif corruption == "index_mismatch":
        path = checkpoint / "model.safetensors.index.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["weight_map"]["action.weight"] = "model-00002-of-00002.safetensors"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif corruption == "unreferenced_shard":
        save_file({"extra": torch.ones(1)}, checkpoint / "model-00003-of-00003.safetensors")
    elif corruption == "unsharded":
        save_file({"unexpected": torch.ones(1)}, checkpoint / "model.safetensors")
    elif corruption == "pytorch_bin":
        (checkpoint / "pytorch_model.bin").write_bytes(b"forbidden")
    elif corruption == "trainer_step":
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 4}), encoding="utf-8"
        )
    elif corruption == "scheduler_step":
        torch.save({"last_epoch": 4}, checkpoint / "scheduler.pt")
    elif corruption == "rng_key":
        torch.save(
            {"python": (), "numpy": (), "cpu": b"cpu"}, checkpoint / "rng_state.pth"
        )
    elif corruption == "partial":
        (checkpoint / "leftover.part").write_bytes(b"partial")
    elif corruption == "temporary":
        (checkpoint / "leftover.tmp").write_bytes(b"temporary")
    elif corruption == "incomplete":
        (checkpoint / "leftover.incomplete").write_bytes(b"incomplete")
    elif corruption == "zero_length":
        (checkpoint / "zero.txt").touch()
    else:
        raise AssertionError(corruption)

    with pytest.raises(CheckpointError, match=match):
        verify_checkpoint_structure(checkpoint, expected_step=5)
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_inference/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
```

Expected: module import failure.

- [ ] **Step 3: Implement structural and tensor verification**

Implement `verify_checkpoint_structure(checkpoint: Path, expected_step: int)`. Parse `model.safetensors.index.json`, require the exact sharded file set, then:

```python
actual_weight_map: dict[str, str] = {}
total_tensor_bytes = 0
for shard in sorted(referenced_shards):
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            if key in actual_weight_map:
                raise CheckpointError(f"duplicate tensor key: {key}")
            tensor = handle.get_tensor(key)
            total_tensor_bytes += tensor.numel() * tensor.element_size()
            actual_weight_map[key] = shard
if actual_weight_map != weight_map:
    raise CheckpointError("actual tensor key map differs from index")
if index["metadata"].get("total_size") != total_tensor_bytes:
    raise CheckpointError("indexed total_size differs from tensor payload")
```

Load `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` individually with
`torch.load(path, map_location="cpu", weights_only=False)`, validate exact
fields, validate `trainer_state.json`, required configs, forbidden suffixes,
and write SHA-256/size and tensor-key manifests.

- [ ] **Step 4: Add fail-closed offline model/processor load**

Add `verify_offline_load(checkpoint, cache_root, cosmos_revision)`. It sets CUDA invisible and all offline variables before importing Transformers/GR00T, verifies `get_backbone_cls(model_config)` returns `Qwen3Backbone`, then loads the checkpoint model and processor with canonical Cosmos ID, exact revision, `local_files_only=True`, and read-only cache. Reject missing or unexpected weights and capture a log/result without printing credentials.

Expose one CLI with positional `checkpoint` and required `--expected-step`,
`--cache-root`, `--cosmos-revision`, `--output-dir`, plus boolean
`--offline-load`. It writes `files.json`, `tensor-keys.json`,
`offline-load.log`, and `verdict.json` atomically beneath a nonexistent output
directory and exits nonzero unless both structural and requested offline-load
checks pass.

- [ ] **Step 5: Run tests and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_inference/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
ruff check --no-cache gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
ruff format --check --no-cache gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit checkpoint verification**

```bash
git add gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
git commit -m "feat: verify complete gr00t checkpoints"
```

### Task 8: Implement quantitative GPU-gate monitoring

**Files:**
- Create: `gear_sonic/scripts/monitor_gr00t_gpu_gate.py`
- Create: `gear_sonic/tests/test_monitor_gr00t_gpu_gate.py`

- [ ] **Step 1: Write failing evaluator tests**

Import `csv`, `json`, `Path`, and `pytest`. Use one explicit in-memory sample
schema and assert the thresholds and process contracts directly:

```python
def _sample(
    used_mib: int,
    processes: list[tuple[int, str, int]],
    *,
    second: int = 0,
    gpu_index: int = 7,
) -> dict[str, object]:
    return {
        "timestamp_utc": f"2026-08-31T00:00:{second:02d}Z",
        "gpu_index": gpu_index,
        "gpu_uuid": f"GPU-fixture-{gpu_index}",
        "memory_used_mib": used_mib,
        "memory_total_mib": 81559,
        "utilization_gpu_percent": 50,
        "processes": [
            {"pid": pid, "name": name, "used_memory_mib": memory}
            for pid, name, memory in processes
        ],
    }


def test_evaluate_baseline_accepts_stable_low_memory() -> None:
    samples = [_sample(13000, [(101, "eval", 6500)], second=i) for i in range(30)]
    result = evaluate_baseline(samples=samples, total_mib=81559)
    assert result["status"] == "pass"


def test_evaluate_baseline_rejects_wrong_h100_total() -> None:
    result = evaluate_baseline(
        samples=[_sample(13000, [(101, "eval", 6500)])], total_mib=80000
    )
    assert result["status"] == "fail"
    assert any("81559" in reason for reason in result["reasons"])


def test_evaluate_baseline_rejects_health_regression() -> None:
    result = evaluate_baseline(
        samples=[_sample(13000, [(101, "eval", 6500)])],
        total_mib=81559,
        health_before={"ecc_uncorrected": 0, "retired_pages": 0, "row_remap_pending": 0},
        health_after={"ecc_uncorrected": 1, "retired_pages": 0, "row_remap_pending": 0},
    )
    assert result["status"] == "fail"
    assert any("health" in reason for reason in result["reasons"])


def test_evaluate_concurrent_gate_rejects_95_percent_sample() -> None:
    samples = []
    for second in range(3):
        samples.extend(
            [
                _sample(77482, [(101, "full", 76000)], second=second, gpu_index=7),
                _sample(30000, [(202, "subtasks", 29000)], second=second, gpu_index=6),
            ]
        )
    result = evaluate_concurrent_gate(
        samples=samples,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("memory ceiling" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    ("samples", "reason"),
    [
        (
            [_sample(13000, [(101, "eval", 6500)]), _sample(13000, [(102, "eval", 6500)], second=1)],
            "process set",
        ),
        (
            [_sample(13000, [(101, "eval", 1000)]), _sample(13000, [(101, "eval", 1257)], second=1)],
            "process memory range",
        ),
        (
            [_sample(13000, [(101, "eval", 6500)]), _sample(13513, [(101, "eval", 6500)], second=1)],
            "aggregate memory range",
        ),
    ],
)
def test_evaluate_baseline_rejects_instability(
    samples: list[dict[str, object]],
    reason: str,
) -> None:
    result = evaluate_baseline(samples=samples, total_mib=81559)
    assert result["status"] == "fail"
    assert any(reason in item for item in result["reasons"])


def test_concurrent_gate_requires_overlap_and_exact_processes() -> None:
    no_overlap = [_sample(20000, [(101, "full", 10000)])]
    result = evaluate_concurrent_gate(
        samples=no_overlap,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("overlap" in item for item in result["reasons"])

    unexpected = [
        _sample(30000, [(101, "full", 10000), (303, "other", 1000)], gpu_index=7),
        _sample(30000, [(202, "subtasks", 10000)], gpu_index=6),
    ]
    result = evaluate_concurrent_gate(
        samples=unexpected,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("unexpected process" in item for item in result["reasons"])


def test_concurrent_gate_rejects_timeout() -> None:
    result = evaluate_concurrent_gate(
        samples=[
            _sample(30000, [(101, "full", 10000)], gpu_index=7),
            _sample(30000, [(202, "subtasks", 10000)], gpu_index=6),
        ],
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
        timed_out=True,
    )
    assert result["status"] == "fail"
    assert any("timeout" in item for item in result["reasons"])
```

Use this artifact-contract test:

```python
def test_write_artifacts_uses_exact_field_sets(tmp_path: Path) -> None:
    samples = [_sample(13000, [(101, "eval", 6500)])]
    result = {
        "status": "pass",
        "reasons": [],
        "mode": "baseline",
        "thresholds": {},
        "summary": {},
        "health_before": {},
        "health_after": {},
    }
    csv_path = tmp_path / "samples.csv"
    jsonl_path = tmp_path / "processes.jsonl"
    result_path = tmp_path / "result.json"
    write_artifacts(samples, result, csv_path, jsonl_path, result_path)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    process = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    stored_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(row) == {
        "timestamp_utc",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
    }
    assert set(process) == {
        "timestamp_utc",
        "gpu_index",
        "gpu_uuid",
        "pid",
        "name",
        "used_memory_mib",
    }
    assert set(stored_result) == {
        "status",
        "reasons",
        "mode",
        "thresholds",
        "summary",
        "health_before",
        "health_after",
    }
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
```

Expected: module import failure.

- [ ] **Step 3: Implement sampler and pure evaluators**

The CLI accepts `--mode baseline|concurrent`, `--gpu-indices`,
`--duration-seconds`, `--timeout-seconds`, `--expected-pid-file`,
`--exit-file`, `--prelaunch-seconds`, `--post-exit-seconds`,
`--csv-path`, `--process-jsonl-path`, `--result-json-path`,
`--health-before-path`, and `--health-after-path`. Sample `nvidia-smi` once per
second. Use exact limits:

```python
EXPECTED_TOTAL_MIB = 81559
BASELINE_MAX_FRACTION = 0.25
GATE_MAX_FRACTION = 0.95
PROCESS_RANGE_MAX_MIB = 256
AGGREGATE_RANGE_MAX_MIB = 512
```

For concurrent mode, each expected-PID JSON file has exact keys
`label,gpu_index,pid`; each exit-file argument is `label=/absolute/path`.
Capture the per-GPU baseline process sets during `--prelaunch-seconds`, wait
for both PID files, require each PID on its declared GPU, allow only baseline
plus declared gate PIDs, latch at least one sample where both declared PIDs
are simultaneously live, and continue for `--post-exit-seconds` after both
exit files appear. A missing PID/exit file, GPU mismatch, or 30-minute timeout
is a failed result, not authority to signal a process.
The result `summary` contains normalized
`baseline_processes_by_gpu` and `final_processes_by_gpu` mappings so the
production prelaunch baseline can prove it starts from the accepted post-gate
process state.

Write result JSON atomically and return nonzero for any failed condition.

- [ ] **Step 4: Run tests and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
ruff check --no-cache gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
ruff format --check --no-cache gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit monitoring**

```bash
git add gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
git commit -m "feat: gate concurrent gr00t gpu usage"
```

### Task 9: Run the local implementation gate

**Files:**
- Verify: `gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py`
- Verify: `gear_sonic/scripts/annotate_pnp_trash_dataset.py`
- Verify: `gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py`
- Verify: `gear_sonic/scripts/verify_gr00t_n17_checkpoint.py`
- Verify: `gear_sonic/scripts/verify_gr00t_training_attempt.py`
- Verify: `gear_sonic/scripts/monitor_gr00t_gpu_gate.py`
- Verify: the five corresponding test modules listed in the file map.

- [ ] **Step 1: Run the complete focused suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_data_collection/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_verify_gr00t_training_attempt.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv_inference/bin/python -m pytest -q -p no:cacheprovider \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py
```

Expected: all tests pass.

- [ ] **Step 2: Run Ruff and compile checks**

```bash
ruff check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/scripts/verify_gr00t_training_attempt.py \
  gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py \
  gear_sonic/tests/test_verify_gr00t_training_attempt.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
ruff format --check --no-cache \
  gear_sonic/utils/data_collection/lerobot_xlsx_annotations.py \
  gear_sonic/scripts/annotate_pnp_trash_dataset.py \
  gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/scripts/verify_gr00t_training_attempt.py \
  gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  gear_sonic/tests/test_lerobot_xlsx_annotations.py \
  gear_sonic/tests/test_launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/tests/test_verify_gr00t_n17_checkpoint.py \
  gear_sonic/tests/test_verify_gr00t_training_attempt.py \
  gear_sonic/tests/test_monitor_gr00t_gpu_gate.py
PYTHONDONTWRITEBYTECODE=1 .venv_teleop/bin/python -m py_compile \
  gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
  gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
  gear_sonic/scripts/verify_gr00t_training_attempt.py \
  gear_sonic/scripts/monitor_gr00t_gpu_gate.py
```

Expected: all commands exit zero.

- [ ] **Step 3: Confirm clean patch scope**

```bash
git diff --check
git status --short
```

Expected: only explicitly retained user changes and generated ignored datasets remain; implementation files are committed.

### Task 10: Perform a non-destructive H100 preflight

**Files:**
- Read on H100: source, image, datasets parent, cache, secrets, containers, GPUs.
- Create on H100: one user-owned `mktemp -d` evidence directory for the baseline only.

- [ ] **Step 1: Verify immutable source/image/model facts**

Run:

```bash
ssh h100 'set -euo pipefail
GR00T_REPO=/home/kube/jihun/Isaac-GR00T
IMAGE_TAG=jihun/gr00t-n1.7:626af89
IMAGE_ID=sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db
GR00T_SNAPSHOT=/mnt/data01/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
COSMOS_SNAPSHOT=/mnt/data01/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561
WANDB_KEY=/mnt/data01/jhkim/secrets/wandb_api_key
test "$(git -C "$GR00T_REPO" rev-parse HEAD)" = 626af89d3e914ec92eab5323e23b9ed44a7b26c8
test -z "$(git -C "$GR00T_REPO" status --porcelain)"
test "$(docker image inspect "$IMAGE_TAG" --format "{{.Id}}")" = "$IMAGE_ID"
test -d "$GR00T_SNAPSHOT"
test -d "$COSMOS_SNAPSHOT"
test -f "$WANDB_KEY"
test ! -L "$WANDB_KEY"
test -s "$WANDB_KEY"
test "$(stat -c %a "$WANDB_KEY")" = 600
stat -c "wandb_key_type=%F mode=%a bytes=%s" "$WANDB_KEY"'
```

This requires:

- `/home/kube/jihun/Isaac-GR00T` HEAD equals `626af89d3e914ec92eab5323e23b9ed44a7b26c8` and is clean;
- image tag resolves to `sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db`;
- GR00T snapshot `2fc962b973bccdd5d8ce4f67cc63b264d6886495` exists;
- Cosmos ref/revision resolves to `9ce19a195e423419c349abfc86fd07178b230561`;
- W&B key is a nonempty regular mode-600 file, reporting metadata only.

Expected: every assertion passes. Any failure stops the workflow without writes.

- [ ] **Step 2: Enforce fresh namespaces before initialization**

```bash
ssh h100 'set -euo pipefail
for path in \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828 \
  /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_full_prompt_left_only \
  /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks_left_only \
  /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_left_only.release.json
do
  test ! -e "$path"
  test ! -L "$path"
done
for name in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  test -z "$(docker ps -a --filter "name=^/${name}$" --format "{{.ID}}")"
done'
```

Expected: all absent. Do not delete or rename any conflicting object.

- [ ] **Step 3: Run the 30-second GPU baseline**

```bash
PREFLIGHT_DIR=$(ssh h100 'mktemp -d /mnt/data01/jhkim/gr00t_runs/.pnp-left-preflight-XXXXXXXX')
test -n "$PREFLIGHT_DIR"
rsync -a --protect-args \
  gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  "h100:$PREFLIGHT_DIR/monitor_gr00t_gpu_gate.py"
LOCAL_MONITOR_SHA=$(sha256sum gear_sonic/scripts/monitor_gr00t_gpu_gate.py | cut -d " " -f 1)
REMOTE_MONITOR_SHA=$(ssh h100 "sha256sum '$PREFLIGHT_DIR/monitor_gr00t_gpu_gate.py' | cut -d ' ' -f 1")
test "$LOCAL_MONITOR_SHA" = "$REMOTE_MONITOR_SHA"
ssh h100 "python3 '$PREFLIGHT_DIR/monitor_gr00t_gpu_gate.py' \
  --mode baseline \
  --gpu-indices 6 7 \
  --duration-seconds 30 \
  --timeout-seconds 60 \
  --csv-path '$PREFLIGHT_DIR/baseline.csv' \
  --process-jsonl-path '$PREFLIGHT_DIR/processes.jsonl' \
  --result-json-path '$PREFLIGHT_DIR/result.json' \
  --health-before-path '$PREFLIGHT_DIR/health-before.json' \
  --health-after-path '$PREFLIGHT_DIR/health-after.json'"
ssh h100 "python3 -c 'import json; value=json.load(open(\"$PREFLIGHT_DIR/result.json\")); assert value[\"status\"] == \"pass\", value'"
printf 'preflight_evidence=%s\n' "$PREFLIGHT_DIR"
```

Expected: each GPU stays below 20,389.75 MiB, PID sets are stable, and health is unchanged. Any failure blocks further mutation.

### Task 11: Transfer and finalize the remote dataset release

**Files:**
- Create on H100: the two remote dataset directories and release marker from the spec.

- [ ] **Step 1: Create a unique staging directory**

```bash
DATASET_PARENT=/mnt/data01/jhkim/datasets/pnp_trash
TRANSFER_STAGE=$(ssh h100 "mktemp -d '$DATASET_PARENT/.left-only-transfer-XXXXXXXX'")
test -n "$TRANSFER_STAGE"
ssh h100 "python3 -c 'from pathlib import Path; parent=Path(\"$DATASET_PARENT\").resolve(); stage=Path(\"$TRANSFER_STAGE\").resolve(); assert stage.parent == parent; print(stage)'"
printf '%s\n' "$TRANSFER_STAGE" > /tmp/pnp-trash-left-transfer-stage.txt
```

- [ ] **Step 2: Transfer both datasets and the marker into staging**

```bash
TRANSFER_STAGE=$(cat /tmp/pnp-trash-left-transfer-stage.txt)
rsync -a --protect-args outputs/pnp_trash_full_prompt_left_only/ \
  "h100:$TRANSFER_STAGE/pnp_trash_full_prompt_left_only/"
rsync -a --protect-args outputs/pnp_trash_subtasks_left_only/ \
  "h100:$TRANSFER_STAGE/pnp_trash_subtasks_left_only/"
rsync -a --protect-args outputs/pnp_trash_left_only.release.json \
  "h100:$TRANSFER_STAGE/pnp_trash_left_only.release.json"
```

- [ ] **Step 3: Verify staged manifests**

Run the exact manifest algorithm used by
`dataset_manifest_sha256` and compare it with the staged marker:

```bash
TRANSFER_STAGE=$(cat /tmp/pnp-trash-left-transfer-stage.txt)
ssh h100 "tee '$TRANSFER_STAGE/verify_release.py' >/dev/null" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(root: Path) -> str:
    digest = sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in dataset: {path}")
        if not path.is_file():
            continue
        record = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        count += 1
    if count == 0:
        raise RuntimeError(f"empty dataset: {root}")
    return digest.hexdigest()


stage = Path(sys.argv[1]).resolve()
marker = json.loads((stage / "pnp_trash_left_only.release.json").read_text())
assert marker["format_version"] == 1
assert marker["state"] == "complete"
assert marker["direction"] == "left"
assert marker["candidate_episodes"] == 72
assert marker["selected_episodes"] == 44
assert marker["excluded_episodes"] == 28
for variant, basename in {
    "full_prompt": "pnp_trash_full_prompt_left_only",
    "subtasks": "pnp_trash_subtasks_left_only",
}.items():
    assert marker["datasets"][variant]["path"] == basename
    actual = dataset_manifest(stage / basename)
    assert marker["datasets"][variant]["manifest_sha256"] == actual
    print(variant, actual)
PY
ssh h100 "python3 '$TRANSFER_STAGE/verify_release.py' '$TRANSFER_STAGE'"
```

- [ ] **Step 4: Finalize fail-closed**

```bash
TRANSFER_STAGE=$(cat /tmp/pnp-trash-left-transfer-stage.txt)
ssh h100 "python3 - '$TRANSFER_STAGE'" <<'PY'
import json
import os
from pathlib import Path
import sys

stage = Path(sys.argv[1]).resolve()
parent = Path("/mnt/data01/jhkim/datasets/pnp_trash").resolve()
assert stage.parent == parent
full = parent / "pnp_trash_full_prompt_left_only"
subtasks = parent / "pnp_trash_subtasks_left_only"
marker = parent / "pnp_trash_left_only.release.json"
for path in (full, subtasks, marker):
    assert not os.path.lexists(path), f"final target exists: {path}"
value = json.loads((stage / marker.name).read_text(encoding="utf-8"))
os.rename(stage / full.name, full)
os.rename(stage / subtasks.name, subtasks)
temporary = parent / f".{marker.name}.{os.getpid()}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, marker)
directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(marker)
PY
```

```bash
TRANSFER_STAGE=$(cat /tmp/pnp-trash-left-transfer-stage.txt)
ssh h100 "python3 '$TRANSFER_STAGE/verify_release.py' /mnt/data01/jhkim/datasets/pnp_trash"
```

If a failure exposes a directory without the marker, report an incomplete
release and stop; do not remove it automatically.

Expected: the remote marker validates and both datasets report 44 episodes and 93,312 frames.

### Task 12: Create isolated run roots and containers

**Files:**
- Create on H100: the two exact output roots and containers from the spec.

- [ ] **Step 1: Recheck absence and baseline**

```bash
ssh h100 'set -euo pipefail
for path in \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
do
  test ! -e "$path"
  test ! -L "$path"
done
for name in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  test -z "$(docker ps -a --filter "name=^/${name}$" --format "{{.ID}}")"
done'
PRECREATE_DIR=$(ssh h100 'mktemp -d /mnt/data01/jhkim/gr00t_runs/.pnp-left-precreate-XXXXXXXX')
rsync -a --protect-args gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
  "h100:$PRECREATE_DIR/monitor_gr00t_gpu_gate.py"
ssh h100 "python3 '$PRECREATE_DIR/monitor_gr00t_gpu_gate.py' \
  --mode baseline --gpu-indices 6 7 --duration-seconds 30 --timeout-seconds 60 \
  --csv-path '$PRECREATE_DIR/baseline.csv' \
  --process-jsonl-path '$PRECREATE_DIR/processes.jsonl' \
  --result-json-path '$PRECREATE_DIR/result.json' \
  --health-before-path '$PRECREATE_DIR/health-before.json' \
  --health-after-path '$PRECREATE_DIR/health-after.json'"
ssh h100 "python3 -c 'import json; value=json.load(open(\"$PRECREATE_DIR/result.json\")); assert value[\"status\"] == \"pass\", value'"
```

- [ ] **Step 2: Create output roots and copy audited scripts**

```bash
ssh h100 'set -euo pipefail
install -d -m 0777 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828
install -d -m 0777 /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
install -d -m 0777 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence
install -d -m 0777 /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/evidence'
for RUN_ROOT in \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
do
  rsync -a --protect-args \
    gear_sonic/scripts/launch_gr00t_n17_pinned_finetune.py \
    gear_sonic/scripts/verify_gr00t_n17_checkpoint.py \
    gear_sonic/scripts/verify_gr00t_training_attempt.py \
    gear_sonic/scripts/monitor_gr00t_gpu_gate.py \
    outputs/pnp_trash_left_only.release.json \
    /tmp/pnp-trash-left-local-hashes.txt \
    "h100:$RUN_ROOT/evidence/"
done
TRANSFER_STAGE=$(cat /tmp/pnp-trash-left-transfer-stage.txt)
for RUN_ROOT in \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
do
  ssh h100 "cp '$TRANSFER_STAGE/verify_release.py' '$RUN_ROOT/evidence/verify_release.py'"
done
for SCRIPT in \
  launch_gr00t_n17_pinned_finetune.py \
  verify_gr00t_n17_checkpoint.py \
  verify_gr00t_training_attempt.py \
  monitor_gr00t_gpu_gate.py
do
  LOCAL_SHA=$(sha256sum "gear_sonic/scripts/$SCRIPT" | cut -d " " -f 1)
  for RUN_ROOT in \
    /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
    /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
  do
    REMOTE_SHA=$(ssh h100 "sha256sum '$RUN_ROOT/evidence/$SCRIPT' | cut -d ' ' -f 1")
    test "$LOCAL_SHA" = "$REMOTE_SHA"
  done
done
```

- [ ] **Step 3: Create persistent containers by immutable image ID**

```bash
ssh h100 'docker run -d \
  --name jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  --gpus device=7 \
  --init --ipc=host --restart=no \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --label gr00t.source_commit=626af89d3e914ec92eab5323e23b9ed44a7b26c8 \
  --label gr00t.dataset_variant=full_prompt_left_only \
  -v /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_full_prompt_left_only:/dataset:ro \
  -v /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828:/outputs:rw \
  -v /mnt/data01/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/data01/jhkim/secrets/wandb_api_key:/run/secrets/wandb_api_key:ro \
  sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db \
  sleep infinity' | tee /tmp/pnp-trash-full-left-container-id.txt
ssh h100 'docker run -d \
  --name jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  --gpus device=6 \
  --init --ipc=host --restart=no \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --label gr00t.source_commit=626af89d3e914ec92eab5323e23b9ed44a7b26c8 \
  --label gr00t.dataset_variant=subtasks_left_only \
  -v /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks_left_only:/dataset:ro \
  -v /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828:/outputs:rw \
  -v /mnt/data01/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/data01/jhkim/secrets/wandb_api_key:/run/secrets/wandb_api_key:ro \
  sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db \
  sleep infinity' | tee /tmp/pnp-trash-subtasks-left-container-id.txt
```

- [ ] **Step 4: Verify effective runtime**

```bash
ssh h100 'set -euo pipefail
for item in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828
do
  name=${item%%:*}
  root=${item#*:}
  docker inspect "$name" > "$root/evidence/container-inspect.json"
  test "$(docker inspect "$name" --format "{{.Image}}")" = sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db
  test "$(docker inspect "$name" --format "{{.State.Running}}")" = true
  test "$(docker inspect "$name" --format "{{.HostConfig.IpcMode}}")" = host
  test "$(docker inspect "$name" --format "{{.HostConfig.RestartPolicy.Name}}")" = no
  docker exec "$name" python -c "import gr00t,torch; assert torch.__version__.startswith(\"2.7.1\"); assert torch.version.cuda == \"12.8\"; assert torch.cuda.device_count() == 1; assert \"H100\" in torch.cuda.get_device_name(0); assert str(gr00t.__file__).startswith(\"/workspace/\"); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),gr00t.__file__)"
  docker exec "$name" sha256sum /workspace/gr00t/experiment/launch_finetune.py /workspace/gr00t/experiment/experiment.py /workspace/gr00t/experiment/trainer.py > "$root/evidence/upstream-entrypoints.sha256"
done'
```

Then validate both inspection payloads:

```bash
ssh h100 'python3 - \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/container-inspect.json \
  7 /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_full_prompt_left_only' <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())[0]
assert value["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == [sys.argv[2]]
mounts = {row["Destination"]: row for row in value["Mounts"]}
assert mounts["/dataset"]["Source"] == sys.argv[3] and not mounts["/dataset"]["RW"]
assert mounts["/outputs"]["RW"]
assert not mounts["/root/.cache/huggingface"]["RW"]
assert not mounts["/run/secrets/wandb_api_key"]["RW"]
PY
ssh h100 'python3 - \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/evidence/container-inspect.json \
  6 /mnt/data01/jhkim/datasets/pnp_trash/pnp_trash_subtasks_left_only' <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())[0]
assert value["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == [sys.argv[2]]
mounts = {row["Destination"]: row for row in value["Mounts"]}
assert mounts["/dataset"]["Source"] == sys.argv[3] and not mounts["/dataset"]["RW"]
assert mounts["/outputs"]["RW"]
assert not mounts["/root/.cache/huggingface"]["RW"]
assert not mounts["/run/secrets/wandb_api_key"]["RW"]
PY
```

Any mismatch blocks the workflow; containers remain present for inspection.

### Task 13: Build writable views and pass loader/model probes

**Files:**
- Create per run: `/outputs/dataset_view`, metadata manifests, model manifests, probe logs/results.

- [ ] **Step 1: Hash immutable source metadata**

For each container below, write a sorted relative-path/size/SHA-256 manifest
of `/dataset/meta` before copying:

```bash
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec -i '$CONTAINER' python -" <<'PY'
from hashlib import sha256
import json
from pathlib import Path

root = Path("/dataset/meta")
records = []
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
    if path.is_symlink():
        raise RuntimeError(f"source metadata symlink: {path}")
    if not path.is_file():
        continue
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
assert records
Path("/outputs/evidence/source-meta-before.json").write_text(
    json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
done
```

- [ ] **Step 2: Build the writable view**

```bash
ssh h100 'set -euo pipefail
for name in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  docker exec "$name" bash -lc '\''
    set -euo pipefail
    test ! -e /outputs/dataset_view
    test ! -L /outputs/dataset_view
    mkdir /outputs/dataset_view
    cp -a /dataset/meta /outputs/dataset_view/meta
    ln -s /dataset/data /outputs/dataset_view/data
    ln -s /dataset/videos /outputs/dataset_view/videos
    test "$(readlink /outputs/dataset_view/data)" = /dataset/data
    test "$(readlink /outputs/dataset_view/videos)" = /dataset/videos
    test -w /outputs/dataset_view/meta
  '\''
done'
```

- [ ] **Step 3: Manifest selected model snapshots**

```bash
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec -i '$CONTAINER' python -" <<'PY'
from hashlib import sha256
import json
from pathlib import Path

pairs = {
    "gr00t": (
        Path("/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B"),
        Path("/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    ),
    "cosmos": (
        Path("/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"),
        Path("/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"),
    ),
}
for label, (model_root, snapshot) in pairs.items():
    resolved_root = model_root.resolve()
    records = []
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
        if not path.is_file():
            continue
        target = path.resolve()
        if resolved_root not in target.parents:
            raise RuntimeError(f"{label} file resolves outside model cache: {path} -> {target}")
        digest = sha256()
        with target.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        records.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "resolved_path": target.relative_to(resolved_root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    assert records
    Path(f"/outputs/evidence/{label}-snapshot.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
PY
done
```

Install and immediately run a fail-closed revalidator in each evidence
directory; later gates invoke the same immutable script by hash:

```bash
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec -i '$CONTAINER' tee /outputs/evidence/verify_input_manifests.py >/dev/null" <<'PY'
from hashlib import sha256
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_plain(root: Path, manifest: Path) -> None:
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert actual_paths == {row["path"] for row in expected}
    for row in expected:
        path = root / row["path"]
        assert not path.is_symlink()
        assert path.stat().st_size == row["bytes"]
        assert digest(path) == row["sha256"]


def verify_snapshot(snapshot: Path, model_root: Path, manifest: Path) -> None:
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    assert actual_paths == {row["path"] for row in expected}
    for row in expected:
        path = (model_root / row["resolved_path"]).resolve()
        assert model_root.resolve() in path.parents
        assert path.stat().st_size == row["bytes"]
        assert digest(path) == row["sha256"]


verify_plain(
    Path("/dataset/meta"), Path("/outputs/evidence/source-meta-before.json")
)
verify_snapshot(
    Path("/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    Path("/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B"),
    Path("/outputs/evidence/gr00t-snapshot.json"),
)
verify_snapshot(
    Path("/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"),
    Path("/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"),
    Path("/outputs/evidence/cosmos-snapshot.json"),
)
print("immutable-input-manifests-ok")
PY
  ssh h100 "docker exec '$CONTAINER' bash -lc 'sha256sum /outputs/evidence/verify_input_manifests.py > /outputs/evidence/verify_input_manifests.sha256'"
  ssh h100 "docker exec '$CONTAINER' python /outputs/evidence/verify_input_manifests.py"
done
```

- [ ] **Step 4: Run selector and offline model/processor probe**

Run each probe sequentially and preserve its unique output:

```bash
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:full-prompt-left-probe \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:subtasks-left-probe
do
  CONTAINER=${ITEM%%:*}
  EXPERIMENT_NAME=${ITEM#*:}
  ssh h100 "docker exec -i '$CONTAINER' bash -s -- '$EXPERIMENT_NAME'" <<'BASH'
set -euo pipefail
EXPERIMENT_NAME=$1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export GR00T_PINNED_PREFLIGHT_ONLY=1 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
export GR00T_FRESHNESS_AUDIT_PATH="/outputs/evidence/$EXPERIMENT_NAME-freshness.json"
test ! -e "/outputs/probe/$EXPERIMENT_NAME"
cd /workspace
uv run python /outputs/evidence/launch_gr00t_n17_pinned_finetune.py \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset-path /outputs/dataset_view \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py \
  --num-gpus 1 \
  --output-dir /outputs/probe \
  --experiment-name "$EXPERIMENT_NAME" \
  --save-total-limit 5 --save-steps 1000 --max-steps 20000 \
  --wandb-project gr00t-n1.7-pnp-trash \
  --global-batch-size 32 --gradient-accumulation-steps 1 \
  --learning-rate 0.0001 --weight-decay 0.00001 --warmup-ratio 0.05 \
  --state-dropout-prob 0.2 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 --episode-sampling-rate 0.1 \
  --shard-size 1024 --num-shards-per-epoch 100000 \
  > "/outputs/evidence/$EXPERIMENT_NAME.log" 2>&1
BASH
done
```

Expected: both logs contain the canonical Cosmos ID, exact revision/snapshot,
`Qwen3Backbone`, and successful official model/processor/dataset class names;
neither log contains a network request or W&B run.

- [ ] **Step 5: Run the real GR00T loader over each writable view**

```bash
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:1 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:4
do
  CONTAINER=${ITEM%%:*}
  EXPECTED_RUNS=${ITEM#*:}
  ssh h100 "docker exec -i '$CONTAINER' python - '$EXPECTED_RUNS'" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

expected_runs = int(sys.argv[1])
loader = LeRobotEpisodeLoader(
    "/outputs/dataset_view", MODALITY_CONFIGS["unitree_g1_sonic"]
)
provenance = json.loads(
    Path("/dataset/meta/annotation_provenance.json").read_text(encoding="utf-8")
)
expected_prompts = {
    row["output_episode_index"]: row["prompts"] for row in provenance["episodes"]
}
expected_sources = [
    4, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 26, 27,
    62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80,
    81, 82, 83, 85, 86, 87, 88, 89, 90, 91,
]
assert [row["source_episode_index"] for row in provenance["episodes"]] == expected_sources
assert [row["output_episode_index"] for row in provenance["episodes"]] == list(range(44))
assert len(loader) == 44
column = "language.annotation.human.task_description"
total_frames = 0
for episode_index in range(len(loader)):
    frame = loader._load_parquet_data(episode_index)
    values = frame[column].tolist()
    runs = [values[0]] + [right for left, right in zip(values, values[1:]) if left != right]
    assert len(runs) == expected_runs
    assert runs == expected_prompts[episode_index]
    assert all("turn right" not in value.casefold() for value in runs)
    total_frames += len(frame)
assert total_frames == 93312
assert loader.get_dataset_statistics()

root = Path("/dataset/meta")
records = []
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
    if path.is_symlink():
        raise RuntimeError(f"source metadata symlink: {path}")
    if not path.is_file():
        continue
    digest = sha256(path.read_bytes()).hexdigest()
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    )
before = json.loads(Path("/outputs/evidence/source-meta-before.json").read_text())
assert records == before
print("loader-contract-ok", len(loader), total_frames, expected_runs)
PY
done
```

### Task 14: Run independent one-step smokes

**Files:**
- Create per run: unique `smoke/YYYYmmddTHHMMSSZ-randomhex/` directories, logs, exit files, W&B identities, checkpoint verdicts.

The exact production recipe keeps `logging_steps=10`, so one-step and five-step
attempts are expected to have no per-step loss row in `trainer_state.json`.
Validate their loss from the single official terminal Trainer metrics dictionary
instead. The observed pinned runtime emits exactly the four required keys
`train_runtime`, `train_samples_per_second`, `train_steps_per_second`, and
`train_loss`; `epoch` is optional, loss is finite without a sign restriction,
runtime is positive, and throughput rates are nonnegative. Its buffered output order is exactly one
`Model saved` anchor, exactly one `Training completed` anchor, then the terminal
dictionary with no duplicate or unexpected keys. The audited source verifier
also correlates the W&B setup ID, every rendered local path, and the exact unique
`https://wandb.ai/<entity>/gr00t-n1.7-pnp-trash/runs/<id>` URL. It captures every
input through a no-follow file descriptor, records identity and SHA-256 manifests,
revalidates them immediately before publishing, and writes a no-clobber final
marker. Do not modify, reuse, or delete the completed failed smoke; allocate new
attempt IDs after the persistence fix is installed.

- [ ] **Step 1: Allocate unique attempt names**

Install one audited wrapper in both containers, compile-check the already
hash-verified source verifier, and allocate two unique IDs:

```bash
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec -i '$CONTAINER' tee /outputs/evidence/run_training_attempt.sh >/dev/null" <<'BASH'
#!/usr/bin/env bash
set -uo pipefail
if [ "$#" -ne 4 ]; then
  printf 'usage: %s ATTEMPT_ROOT EXPERIMENT_NAME MAX_STEPS SAVE_STEPS\n' "$0" >&2
  exit 64
fi
ATTEMPT_ROOT=$1
EXPERIMENT_NAME=$2
MAX_STEPS=$3
SAVE_STEPS=$4
python - "$ATTEMPT_ROOT" <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
if os.path.lexists(path):
    raise RuntimeError(f"attempt root already exists: {path}")
path.mkdir(parents=True)
PY
atomic_value() {
  local target=$1
  local value=$2
  local temporary="${target}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$target"
}
atomic_value "$ATTEMPT_ROOT/wrapper.pid" "$$"
OUTPUT_DIR="$ATTEMPT_ROOT"
cd /workspace
uv run python - "$OUTPUT_DIR" "$EXPERIMENT_NAME" > "$ATTEMPT_ROOT/freshness.json" <<'PY'
import json
import os
from pathlib import Path
import sys

experiment = Path(sys.argv[1]) / sys.argv[2]
assert not os.path.lexists(experiment)
assert not list(experiment.parent.glob("checkpoint-*"))
print(json.dumps({"experiment": str(experiment), "prelaunch_absent": True}))
PY
FRESH_STATUS=$?
if [ "$FRESH_STATUS" -ne 0 ]; then
  atomic_value "$ATTEMPT_ROOT/exit" "$FRESH_STATUS"
  exit "$FRESH_STATUS"
fi
WANDB_API_KEY=$(python -c 'from pathlib import Path; value=Path("/run/secrets/wandb_api_key").read_text().strip(); print(value.split("=",1)[1] if value.startswith("WANDB_API_KEY=") else value)')
if [ -z "$WANDB_API_KEY" ]; then
  atomic_value "$ATTEMPT_ROOT/exit" 65
  exit 65
fi
export WANDB_API_KEY WANDB_MODE=online CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export GR00T_PINNED_PREFLIGHT_ONLY=0
export GR00T_TRAINING_ARGS_AUDIT_PATH="$ATTEMPT_ROOT/training-arguments.json"
export GR00T_FRESHNESS_AUDIT_PATH="$ATTEMPT_ROOT/freshness-runtime.json"
COMMAND=(
  uv run python /outputs/evidence/launch_gr00t_n17_pinned_finetune.py
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
  --dataset-path /outputs/dataset_view
  --embodiment-tag UNITREE_G1_SONIC
  --modality-config-path gr00t/configs/data/embodiment_configs.py
  --num-gpus 1
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$EXPERIMENT_NAME"
  --save-total-limit 5
  --save-steps "$SAVE_STEPS"
  --max-steps "$MAX_STEPS"
  --use-wandb
  --wandb-project gr00t-n1.7-pnp-trash
  --global-batch-size 32
  --gradient-accumulation-steps 1
  --learning-rate 0.0001
  --weight-decay 0.00001
  --warmup-ratio 0.05
  --state-dropout-prob 0.2
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
  --dataloader-num-workers 4
  --episode-sampling-rate 0.1
  --shard-size 1024
  --num-shards-per-epoch 100000
)
printf '%q ' "${COMMAND[@]}" > "$ATTEMPT_ROOT/command.txt"
printf '\n' >> "$ATTEMPT_ROOT/command.txt"
"${COMMAND[@]}" > "$ATTEMPT_ROOT/train.log" 2>&1 &
CHILD_PID=$!
atomic_value "$ATTEMPT_ROOT/child.pid" "$CHILD_PID"
wait "$CHILD_PID"
STATUS=$?
atomic_value "$ATTEMPT_ROOT/exit" "$STATUS"
exit "$STATUS"
BASH
  ssh h100 "docker exec '$CONTAINER' chmod 700 /outputs/evidence/run_training_attempt.sh"
  ssh h100 "docker exec '$CONTAINER' bash -lc 'sha256sum /outputs/evidence/run_training_attempt.sh > /outputs/evidence/run_training_attempt.sha256'"
  ssh h100 "docker exec '$CONTAINER' test -f /outputs/evidence/verify_gr00t_training_attempt.py"
  ssh h100 "docker exec '$CONTAINER' python -m py_compile /outputs/evidence/verify_gr00t_training_attempt.py"
done
FULL_SMOKE_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')
SUBTASK_SMOKE_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')
printf '%s\n' "$FULL_SMOKE_ID" > /tmp/pnp-trash-full-left-smoke-id.txt
printf '%s\n' "$SUBTASK_SMOKE_ID" > /tmp/pnp-trash-subtasks-left-smoke-id.txt
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 python -c 'import os; assert not os.path.lexists(\"/outputs/smoke/$FULL_SMOKE_ID\")'"
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 python -c 'import os; assert not os.path.lexists(\"/outputs/smoke/$SUBTASK_SMOKE_ID\")'"
```

- [ ] **Step 2: Launch the full-prompt smoke**

```bash
FULL_SMOKE_ID=$(cat /tmp/pnp-trash-full-left-smoke-id.txt)
FULL_SMOKE_EXPERIMENT="pnp-trash-full-prompt-left-smoke-$FULL_SMOKE_ID"
ssh h100 "docker exec -d \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  '/outputs/smoke/$FULL_SMOKE_ID' '$FULL_SMOKE_EXPERIMENT' 1 1"
```

- [ ] **Step 3: Verify the full-prompt smoke**

Poll the following command at intervals no longer than 30 seconds. Stop after
30 minutes without killing or restarting the attempt; a missing exit file at
that deadline blocks the subtask smoke and requires user direction.

```bash
FULL_SMOKE_ID=$(cat /tmp/pnp-trash-full-left-smoke-id.txt)
ssh h100 "test -f '/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/smoke/$FULL_SMOKE_ID/exit' && cat '/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/smoke/$FULL_SMOKE_ID/exit' || echo running"
```

After terminal exit, run:

```bash
FULL_SMOKE_ID=$(cat /tmp/pnp-trash-full-left-smoke-id.txt)
FULL_SMOKE_EXPERIMENT="pnp-trash-full-prompt-left-smoke-$FULL_SMOKE_ID"
FULL_SMOKE_ROOT="/outputs/smoke/$FULL_SMOKE_ID"
FULL_SMOKE_CHECKPOINT="$FULL_SMOKE_ROOT/$FULL_SMOKE_EXPERIMENT/checkpoint-1"
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  python /outputs/evidence/verify_gr00t_training_attempt.py \
  '$FULL_SMOKE_ROOT' '$FULL_SMOKE_CHECKPOINT' 1 1"
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  python /outputs/evidence/verify_input_manifests.py"
ssh h100 "docker exec -e CUDA_VISIBLE_DEVICES= \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  python /outputs/evidence/verify_gr00t_n17_checkpoint.py \
  '$FULL_SMOKE_CHECKPOINT' --expected-step 1 \
  --cache-root /root/.cache/huggingface \
  --cosmos-revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --output-dir '$FULL_SMOKE_ROOT/checkpoint-verdict' --offline-load"
```

- [ ] **Step 4: Launch the subtask smoke only after Step 3 passes**

```bash
SUBTASK_SMOKE_ID=$(cat /tmp/pnp-trash-subtasks-left-smoke-id.txt)
SUBTASK_SMOKE_EXPERIMENT="pnp-trash-subtasks-left-smoke-$SUBTASK_SMOKE_ID"
ssh h100 "docker exec -d \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  '/outputs/smoke/$SUBTASK_SMOKE_ID' '$SUBTASK_SMOKE_EXPERIMENT' 1 1"
```

- [ ] **Step 5: Verify the subtask smoke**

Poll at intervals no longer than 30 seconds for at most 30 minutes:

```bash
SUBTASK_SMOKE_ID=$(cat /tmp/pnp-trash-subtasks-left-smoke-id.txt)
ssh h100 "test -f '/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/smoke/$SUBTASK_SMOKE_ID/exit' && cat '/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/smoke/$SUBTASK_SMOKE_ID/exit' || echo running"
```

After terminal exit, run:

```bash
SUBTASK_SMOKE_ID=$(cat /tmp/pnp-trash-subtasks-left-smoke-id.txt)
SUBTASK_SMOKE_EXPERIMENT="pnp-trash-subtasks-left-smoke-$SUBTASK_SMOKE_ID"
SUBTASK_SMOKE_ROOT="/outputs/smoke/$SUBTASK_SMOKE_ID"
SUBTASK_SMOKE_CHECKPOINT="$SUBTASK_SMOKE_ROOT/$SUBTASK_SMOKE_EXPERIMENT/checkpoint-1"
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  python /outputs/evidence/verify_gr00t_training_attempt.py \
  '$SUBTASK_SMOKE_ROOT' '$SUBTASK_SMOKE_CHECKPOINT' 1 1"
ssh h100 "docker exec jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  python /outputs/evidence/verify_input_manifests.py"
ssh h100 "docker exec -e CUDA_VISIBLE_DEVICES= \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  python /outputs/evidence/verify_gr00t_n17_checkpoint.py \
  '$SUBTASK_SMOKE_CHECKPOINT' --expected-step 1 \
  --cache-root /root/.cache/huggingface \
  --cosmos-revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --output-dir '$SUBTASK_SMOKE_ROOT/checkpoint-verdict' --offline-load"
FULL_SMOKE_ID=$(cat /tmp/pnp-trash-full-left-smoke-id.txt)
FULL_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/smoke/$FULL_SMOKE_ID/training-attempt-verdict.json"
SUBTASK_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/smoke/$SUBTASK_SMOKE_ID/training-attempt-verdict.json"
ssh h100 "python3 - '$FULL_VERDICT' '$SUBTASK_VERDICT'" <<'PY'
import json
from pathlib import Path
import sys

ids = [json.loads(Path(path).read_text())["wandb"]["run_id"] for path in sys.argv[1:]]
assert len(ids) == len(set(ids)) == 2, ids
PY
```

Expected: both independent smokes pass. Preserve all artifacts.

### Task 15: Run the concurrent five-step VRAM gate

**Files:**
- Create per run: unique concurrent attempt paths, logs, exits, checkpoint-5 verdicts; create shared CSV/JSONL/result evidence.

- [ ] **Step 1: Re-run the accepted 30-second baseline**

```bash
GATE_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')
printf '%s\n' "$GATE_ID" > /tmp/pnp-trash-left-concurrent-gate-id.txt
GATE_EVIDENCE="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/concurrent-$GATE_ID"
ssh h100 "mkdir '$GATE_EVIDENCE'"
ssh h100 "python3 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/monitor_gr00t_gpu_gate.py \
  --mode baseline --gpu-indices 6 7 --duration-seconds 30 --timeout-seconds 60 \
  --csv-path '$GATE_EVIDENCE/baseline.csv' \
  --process-jsonl-path '$GATE_EVIDENCE/baseline-processes.jsonl' \
  --result-json-path '$GATE_EVIDENCE/baseline-result.json' \
  --health-before-path '$GATE_EVIDENCE/baseline-health-before.json' \
  --health-after-path '$GATE_EVIDENCE/baseline-health-after.json'"
ssh h100 "python3 -c 'import json; value=json.load(open(\"$GATE_EVIDENCE/baseline-result.json\")); assert value[\"status\"] == \"pass\", value'"
```

- [ ] **Step 2: Start the monitor and both unique five-step attempts**

```bash
GATE_ID=$(cat /tmp/pnp-trash-left-concurrent-gate-id.txt)
GATE_EVIDENCE="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/concurrent-$GATE_ID"
FULL_GATE_EXPERIMENT="pnp-trash-full-prompt-left-gate-$GATE_ID"
SUBTASK_GATE_EXPERIMENT="pnp-trash-subtasks-left-gate-$GATE_ID"
ssh h100 "nohup python3 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/monitor_gr00t_gpu_gate.py \
  --mode concurrent --gpu-indices 6 7 \
  --prelaunch-seconds 30 --post-exit-seconds 30 --timeout-seconds 1800 \
  --expected-pid-file '$GATE_EVIDENCE/full.pid.json' \
  --expected-pid-file '$GATE_EVIDENCE/subtasks.pid.json' \
  --exit-file 'full=/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/gate/$GATE_ID/exit' \
  --exit-file 'subtasks=/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/gate/$GATE_ID/exit' \
  --csv-path '$GATE_EVIDENCE/samples.csv' \
  --process-jsonl-path '$GATE_EVIDENCE/processes.jsonl' \
  --result-json-path '$GATE_EVIDENCE/result.json' \
  --health-before-path '$GATE_EVIDENCE/health-before.json' \
  --health-after-path '$GATE_EVIDENCE/health-after.json' \
  > '$GATE_EVIDENCE/monitor.log' 2>&1 & echo \$! > '$GATE_EVIDENCE/monitor.pid'"
sleep 30
ssh h100 "docker exec -d jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  '/outputs/gate/$GATE_ID' '$FULL_GATE_EXPERIMENT' 5 5"
ssh h100 "docker exec -d jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  '/outputs/gate/$GATE_ID' '$SUBTASK_GATE_EXPERIMENT' 5 5"
ssh h100 "set -euo pipefail
resolve_pid() {
  name=\$1
  experiment=\$2
  destination=\$3
  gpu=\$4
  label=\$5
  pid=
  for _attempt in \$(seq 1 180); do
    pid=\$(docker top \"\$name\" -eo pid,args | awk -v value=\"\$experiment\" '\$0 ~ value && /launch_gr00t_n17_pinned_finetune.py/ {print \$1}')
    if test \"\$(printf '%s\\n' \"\$pid\" | sed '/^\$/d' | wc -l)\" = 1; then
      break
    fi
    sleep 5
  done
  test \"\$(printf '%s\\n' \"\$pid\" | sed '/^\$/d' | wc -l)\" = 1
  python3 -c 'import json,sys; print(json.dumps({\"label\":sys.argv[1],\"gpu_index\":int(sys.argv[2]),\"pid\":int(sys.argv[3])}))' \"\$label\" \"\$gpu\" \"\$pid\" > \"\$destination.tmp\"
  mv \"\$destination.tmp\" \"\$destination\"
}
resolve_pid jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 '$FULL_GATE_EXPERIMENT' '$GATE_EVIDENCE/full.pid.json' 7 full
resolve_pid jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 '$SUBTASK_GATE_EXPERIMENT' '$GATE_EVIDENCE/subtasks.pid.json' 6 subtasks"
```

- [ ] **Step 3: Enforce the 30-minute gate**

Poll this command at intervals no longer than 30 seconds. If `result.json` is
still absent at 30 minutes, do not signal any process:

```bash
GATE_ID=$(cat /tmp/pnp-trash-left-concurrent-gate-id.txt)
GATE_EVIDENCE="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/concurrent-$GATE_ID"
ssh h100 "test -f '$GATE_EVIDENCE/result.json' && cat '$GATE_EVIDENCE/result.json' || echo running"
```

After the monitor exits, run:

```bash
GATE_ID=$(cat /tmp/pnp-trash-left-concurrent-gate-id.txt)
GATE_EVIDENCE="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/concurrent-$GATE_ID"
FULL_GATE_EXPERIMENT="pnp-trash-full-prompt-left-gate-$GATE_ID"
SUBTASK_GATE_EXPERIMENT="pnp-trash-subtasks-left-gate-$GATE_ID"
ssh h100 "python3 -c 'import json; value=json.load(open(\"$GATE_EVIDENCE/result.json\")); assert value[\"status\"] == \"pass\", value'"
for ITEM in \
  "jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:/outputs/gate/$GATE_ID:$FULL_GATE_EXPERIMENT" \
  "jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:/outputs/gate/$GATE_ID:$SUBTASK_GATE_EXPERIMENT"
do
  CONTAINER=${ITEM%%:*}
  REMAINDER=${ITEM#*:}
  ATTEMPT_ROOT=${REMAINDER%%:*}
  EXPERIMENT=${REMAINDER#*:}
  CHECKPOINT="$ATTEMPT_ROOT/$EXPERIMENT/checkpoint-5"
  ssh h100 "docker exec '$CONTAINER' \
    python /outputs/evidence/verify_gr00t_training_attempt.py \
    '$ATTEMPT_ROOT' '$CHECKPOINT' 5 5"
  ssh h100 "docker exec '$CONTAINER' python /outputs/evidence/verify_input_manifests.py"
  ssh h100 "docker exec -e CUDA_VISIBLE_DEVICES= '$CONTAINER' \
    python /outputs/evidence/verify_gr00t_n17_checkpoint.py '$CHECKPOINT' \
    --expected-step 5 --cache-root /root/.cache/huggingface \
    --cosmos-revision 9ce19a195e423419c349abfc86fd07178b230561 \
    --output-dir '$ATTEMPT_ROOT/checkpoint-verdict' --offline-load"
done
FULL_SMOKE_ID=$(cat /tmp/pnp-trash-full-left-smoke-id.txt)
SUBTASK_SMOKE_ID=$(cat /tmp/pnp-trash-subtasks-left-smoke-id.txt)
FULL_SMOKE_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/smoke/$FULL_SMOKE_ID/training-attempt-verdict.json"
SUBTASK_SMOKE_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/smoke/$SUBTASK_SMOKE_ID/training-attempt-verdict.json"
FULL_GATE_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/gate/$GATE_ID/training-attempt-verdict.json"
SUBTASK_GATE_VERDICT="/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/gate/$GATE_ID/training-attempt-verdict.json"
ssh h100 "python3 - '$FULL_SMOKE_VERDICT' '$SUBTASK_SMOKE_VERDICT' '$FULL_GATE_VERDICT' '$SUBTASK_GATE_VERDICT'" <<'PY'
import json
from pathlib import Path
import sys

ids = [json.loads(Path(path).read_text())["wandb"]["run_id"] for path in sys.argv[1:]]
assert len(ids) == len(set(ids)) == 4, ids
PY
```

- [ ] **Step 4: Handle capacity failure without reclaiming resources**

On any non-pass result, collect only these read-only diagnostics and stop the
workflow:

```bash
ssh h100 'docker ps --no-trunc; docker top jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 -eo pid,ppid,etime,args; docker top jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 -eo pid,ppid,etime,args; nvidia-smi -i 6,7'
```

Do not stop an evaluation server unless the user separately approves its
freshly resolved full container ID after inspect/restart evidence is shown.

### Task 16: Launch both fresh 20,000-step production runs

**Files:**
- Create per run: fixed production experiment directory, `train.sh`, redacted environment, command, PID files, log, W&B identity, exit file on termination.

- [ ] **Step 1: Re-run the baseline and freshness gates**

```bash
GATE_ID=$(cat /tmp/pnp-trash-left-concurrent-gate-id.txt)
PRODUCTION_GATE=/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/production-prelaunch
ssh h100 "test ! -e '$PRODUCTION_GATE' && test ! -L '$PRODUCTION_GATE' && mkdir '$PRODUCTION_GATE'"
ssh h100 "python3 /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/monitor_gr00t_gpu_gate.py \
  --mode baseline --gpu-indices 6 7 --duration-seconds 30 --timeout-seconds 60 \
  --csv-path '$PRODUCTION_GATE/baseline.csv' \
  --process-jsonl-path '$PRODUCTION_GATE/processes.jsonl' \
  --result-json-path '$PRODUCTION_GATE/result.json' \
  --health-before-path '$PRODUCTION_GATE/health-before.json' \
  --health-after-path '$PRODUCTION_GATE/health-after.json'"
ssh h100 "python3 - '$GATE_ID' '$PRODUCTION_GATE/result.json'" <<'PY'
import json
from pathlib import Path
import sys

gate_id, production_result = sys.argv[1:]
gate = json.loads(
    Path(
        "/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828"
        f"/evidence/concurrent-{gate_id}/result.json"
    ).read_text()
)
production = json.loads(Path(production_result).read_text())
assert gate["status"] == "pass"
assert production["status"] == "pass"
assert gate["summary"]["final_processes_by_gpu"] == production["summary"]["baseline_processes_by_gpu"]
PY
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:pnp-trash-full-prompt-left-only-gpu7-20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:pnp-trash-subtasks-left-only-gpu6-20260828
do
  CONTAINER=${ITEM%%:*}
  EXPERIMENT=${ITEM#*:}
  ssh h100 "docker exec -i '$CONTAINER' python - '$EXPERIMENT'" <<'PY'
import os
from pathlib import Path
import sys

train_root = Path("/outputs/train")
experiment = train_root / sys.argv[1]
assert not os.path.lexists(train_root)
assert not os.path.lexists(experiment)
assert not list(experiment.glob("checkpoint-*"))
PY
done
```

- [ ] **Step 2: Persist exact launch inputs**

```bash
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:pnp-trash-full-prompt-left-only-gpu7-20260828:7 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:pnp-trash-subtasks-left-only-gpu6-20260828:6
do
  CONTAINER=${ITEM%%:*}
  REST=${ITEM#*:}
  EXPERIMENT=${REST%%:*}
  GPU=${REST#*:}
  ssh h100 "docker exec '$CONTAINER' python /outputs/evidence/verify_input_manifests.py"
  ssh h100 "docker exec '$CONTAINER' bash -lc 'sha256sum \
    /outputs/evidence/launch_gr00t_n17_pinned_finetune.py \
    /outputs/evidence/run_training_attempt.sh \
    /workspace/gr00t/experiment/launch_finetune.py \
    /workspace/gr00t/experiment/experiment.py \
    /workspace/gr00t/experiment/trainer.py \
    > /outputs/evidence/production-entrypoints.sha256'"
  ssh h100 "docker exec '$CONTAINER' bash -lc 'diff -u /workspace/gr00t/experiment/launch_finetune.py /outputs/evidence/launch_gr00t_n17_pinned_finetune.py > /outputs/evidence/production-launcher.diff; test \$? -eq 1'"
  ssh h100 "docker exec -i '$CONTAINER' python - '$EXPERIMENT' '$GPU'" <<'PY'
import json
from pathlib import Path
import sys

experiment, gpu = sys.argv[1:]
value = {
    "experiment": experiment,
    "gpu": int(gpu),
    "max_steps": 20000,
    "save_steps": 1000,
    "image_id": "sha256:917a790c576f3f00e1a3007e4594b350f75dfd33f77765da093afc3d2593d1db",
    "source_commit": "626af89d3e914ec92eab5323e23b9ed44a7b26c8",
    "gr00t_revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    "cosmos_model_id": "nvidia/Cosmos-Reason2-2B",
    "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
    "environment": {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "online",
        "WANDB_API_KEY_SOURCE": "/run/secrets/wandb_api_key",
        "WANDB_API_KEY_VALUE_RECORDED": False,
    },
}
Path(f"/outputs/evidence/production-{gpu}-expected.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
done
```

- [ ] **Step 3: Launch full prompt on GPU 7**

```bash
ssh h100 'docker exec -d \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  /outputs/train pnp-trash-full-prompt-left-only-gpu7-20260828 20000 1000'
```

- [ ] **Step 4: Launch subtasks on GPU 6**

```bash
ssh h100 'docker exec -d \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828 \
  bash /outputs/evidence/run_training_attempt.sh \
  /outputs/train pnp-trash-subtasks-left-only-gpu6-20260828 20000 1000'
```

- [ ] **Step 5: Confirm `process_started` within 15 minutes**

Poll this exact check at intervals no longer than 30 seconds for at most 15
minutes:

```bash
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:7 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:6
do
  CONTAINER=${ITEM%%:*}
  GPU=${ITEM#*:}
  ssh h100 "docker exec -i '$CONTAINER' python - '$GPU'" <<'PY'
import json
import os
from pathlib import Path
import re
import sys

root = Path("/outputs/train")
assert root.joinpath("wrapper.pid").is_file()
assert root.joinpath("child.pid").is_file()
for filename in ("wrapper.pid", "child.pid"):
    os.kill(int(root.joinpath(filename).read_text().strip()), 0)
assert not os.path.lexists(root / "exit")
log = root.joinpath("train.log").read_text(errors="replace")
lowered = log.casefold()
assert "resuming from checkpoint" not in lowered
assert "traceback" not in lowered and "out of memory" not in lowered
audit = json.loads(root.joinpath("training-arguments.json").read_text())
assert audit["deepspeed"] is None
freshness = json.loads(root.joinpath("freshness-runtime.json").read_text())
assert freshness["get_last_checkpoint"] is None
match = re.search(r"https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9_-]+)", log)
assert match, "W&B run URL not initialized"
root.joinpath("wandb-run.json").write_text(
    json.dumps({"gpu": int(sys.argv[1]), "run_id": match.group(1), "url": match.group(0)}) + "\n"
)
PY
  ssh h100 "docker top '$CONTAINER' -eo pid,ppid,etime,args"
done
ssh h100 'python3 - <<'"'"'PY'"'"'
import json
from pathlib import Path
full=json.loads(Path("/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/train/wandb-run.json").read_text())
sub=json.loads(Path("/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/train/wandb-run.json").read_text())
assert full["run_id"] != sub["run_id"]
PY'
```

The already-audited container device request proves the physical GPU. If this
check has not passed at 15 minutes, report current logs/processes and request
direction without signaling either run.

- [ ] **Step 6: Confirm `first_step_finite` within 30 minutes**

Poll the logs at intervals no longer than 30 seconds with this parser:

```bash
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec -i '$CONTAINER' python -" <<'PY'
import math
from pathlib import Path
import re

text = Path("/outputs/train/train.log").read_text(errors="replace")
matches = re.findall(r"['\"]loss['\"]\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
assert matches, "no emitted loss yet"
first_loss = float(matches[0])
assert math.isfinite(first_loss)
assert re.search(r"(?:global_)?step['\"]?\s*[:=]\s*1\b|\b1/20000\b", text)
print("first_step_finite", first_loss)
PY
done
```

If this has not passed at 30 minutes, do not kill or restart; report and
request direction.

### Task 17: Verify healthy training and record the handoff

**Files:**
- Create: `docs/superpowers/progress/2026-08-31-pnp-trash-left-only-training.md`

- [ ] **Step 1: Wait for complete checkpoint-1000 within 120 minutes**

Poll these paths at intervals no longer than 60 seconds:

```bash
ssh h100 'for path in \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/train/pnp-trash-full-prompt-left-only-gpu7-20260828/checkpoint-1000 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/train/pnp-trash-subtasks-left-only-gpu6-20260828/checkpoint-1000
do
  if test -d "$path"; then echo "present $path"; else echo "waiting $path"; fi
done'
```

When both are present, require the relative file/size/mtime snapshot to remain
identical across two reads 60 seconds apart:

```bash
for LABEL in full subtasks
do
  if [ "$LABEL" = full ]; then
    CHECKPOINT=/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/train/pnp-trash-full-prompt-left-only-gpu7-20260828/checkpoint-1000
  else
    CHECKPOINT=/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/train/pnp-trash-subtasks-left-only-gpu6-20260828/checkpoint-1000
  fi
  ssh h100 "find '$CHECKPOINT' -type f -printf '%P %s %T@\\n' | LC_ALL=C sort" > "/tmp/pnp-left-$LABEL-checkpoint-1000.first"
done
sleep 60
for LABEL in full subtasks
do
  if [ "$LABEL" = full ]; then
    CHECKPOINT=/mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/train/pnp-trash-full-prompt-left-only-gpu7-20260828/checkpoint-1000
  else
    CHECKPOINT=/mnt/data01/jhkim/gr00t_runs/pnp_trash_subtasks_left_only_n17_20260828/train/pnp-trash-subtasks-left-only-gpu6-20260828/checkpoint-1000
  fi
  ssh h100 "find '$CHECKPOINT' -type f -printf '%P %s %T@\\n' | LC_ALL=C sort" > "/tmp/pnp-left-$LABEL-checkpoint-1000.second"
  cmp "/tmp/pnp-left-$LABEL-checkpoint-1000.first" "/tmp/pnp-left-$LABEL-checkpoint-1000.second"
done
```

Then verify them sequentially:

```bash
for ITEM in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828:pnp-trash-full-prompt-left-only-gpu7-20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828:pnp-trash-subtasks-left-only-gpu6-20260828
do
  CONTAINER=${ITEM%%:*}
  EXPERIMENT=${ITEM#*:}
  CHECKPOINT="/outputs/train/$EXPERIMENT/checkpoint-1000"
  VERDICT_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')
  VERDICT="/outputs/train/${EXPERIMENT}-checkpoint-1000-verdict-$VERDICT_ID"
  ssh h100 "docker exec -e CUDA_VISIBLE_DEVICES= '$CONTAINER' \
    python /outputs/evidence/verify_gr00t_n17_checkpoint.py '$CHECKPOINT' \
    --expected-step 1000 --cache-root /root/.cache/huggingface \
    --cosmos-revision 9ce19a195e423419c349abfc86fd07178b230561 \
    --output-dir '$VERDICT' --offline-load"
  ssh h100 "docker exec -i '$CONTAINER' python - '$CHECKPOINT'" <<'PY'
import json
import math
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
state = json.loads((checkpoint / "trainer_state.json").read_text())
assert state["global_step"] == 1000
losses = [float(row["loss"]) for row in state["log_history"] if "loss" in row]
assert losses and all(math.isfinite(value) for value in losses)
print("latest_finite_loss", losses[-1])
PY
done
```

If both complete verdicts are not available within 120 minutes, leave both
runs unchanged, capture logs/processes, and request direction.

- [ ] **Step 2: Revalidate immutable inputs**

```bash
ssh h100 'python3 \
  /mnt/data01/jhkim/gr00t_runs/pnp_trash_full_prompt_left_only_n17_20260828/evidence/verify_release.py \
  /mnt/data01/jhkim/datasets/pnp_trash'
for CONTAINER in \
  jihun_gr00t_n17_pnp_trash_full_prompt_left_gpu7_20260828 \
  jihun_gr00t_n17_pnp_trash_subtasks_left_gpu6_20260828
do
  ssh h100 "docker exec '$CONTAINER' python /outputs/evidence/verify_input_manifests.py"
done
```

- [ ] **Step 3: Write the secret-free progress report**

Create
`docs/superpowers/progress/2026-08-31-pnp-trash-left-only-training.md`
with these exact sections and only observed evidence values (never copy an
environment dump or credential):

```markdown
# PnP Trash Left-Only GR00T N1.7 Training Progress

## Immutable Inputs

## Dataset Release

## Containers and GPU Assignment

## Independent Smoke Verdicts

## Concurrent VRAM Gate

## Production Process State

## W&B Runs

## Finite-Loss and Checkpoint-1000 Evidence

## Safety, Authority, and Cleanup State
```

Under those sections record the exact commits/images/revisions, dataset
counts/hashes, container IDs/mounts/GPUs, baseline/concurrent evidence paths,
W&B URLs, PIDs, finite losses, checkpoint verdict paths, deadline timestamps,
preexisting process state, and any approved stop/restoration. State verbatim:
`Smoke and gate checkpoints are preserved; no cleanup is authorized.`

- [ ] **Step 4: Verify and commit only the progress report**

```bash
git diff --check -- docs/superpowers/progress/2026-08-31-pnp-trash-left-only-training.md
git add docs/superpowers/progress/2026-08-31-pnp-trash-left-only-training.md
git commit -m "docs: record left-only gr00t training launch"
```

Expected: the commit contains exactly the progress report. Both runs are `training_healthy`; completion remains pending until exit zero and complete checkpoint-20000.

## Explicit Non-Actions

- Do not install anything on the H100 host.
- Do not overwrite or remove any local/remote dataset, release marker, run root, experiment directory, checkpoint, container, or process.
- Do not stop evaluation servers or unrelated jobs without separate approval for exact current IDs.
- Do not clean smoke/gate weights before both production runs have complete checkpoint-6000 and the user approves exact resolved paths.
- Do not claim completion at launch or checkpoint-1000; report continuing training until both exit zero with complete checkpoint-20000.
