from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tyro

import gear_sonic.scripts.annotate_pnp_trash_dataset as annotation_cli_module
from gear_sonic.scripts.annotate_pnp_trash_dataset import (
    AnnotatePnpTrashConfig,
    main as annotation_cli_main,
)
import gear_sonic.utils.data_collection.lerobot_xlsx_annotations as annotations_module
from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import (
    AnnotationError,
    AnnotationSelection,
    DatasetValidationError,
    EpisodeAnnotation,
    build_run_steps,
    build_task_map,
    dataset_manifest_sha256,
    export_both,
    export_variant,
    load_annotations,
    select_annotations_by_direction,
    snap_boundary_frames,
    validate_release_marker,
    validate_variant,
)

HEADERS = [
    "episode",
    "Valid",
    "subtask1",
    "time1",
    "subtask2",
    "time2",
    "subtask3",
    "time3",
    "subtask4",
    "Full Prompt",
]


@dataclass(frozen=True)
class InlineText:
    value: str


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def write_xlsx(path: Path, rows: list[list[object]]) -> Path:
    shared: list[str] = []
    shared_indices: dict[str, int] = {}

    def shared_index(value: str) -> int:
        if value not in shared_indices:
            shared_indices[value] = len(shared)
            shared.append(value)
        return shared_indices[value]

    row_xml: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_number, value in enumerate(row, start=1):
            if value is None or value == "":
                continue
            reference = f"{_column_name(column_number)}{row_number}"
            if isinstance(value, InlineText):
                cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{escape(value.value)}</t></is></c>')
            elif isinstance(value, str):
                cells.append(f'<c r="{reference}" t="s"><v>{shared_index(value)}</v></c>')
            elif isinstance(value, bool):
                cells.append(f'<c r="{reference}" t="b"><v>{int(value)}</v></c>')
            else:
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData></worksheet>"
    )
    shared_strings = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">'
        + "".join(f"<si><t>{escape(value)}</t></si>" for value in shared)
        + "</sst>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Annotations" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return path


def valid_row(*, episode: object = 1, valid: object = 1) -> list[object]:
    return [
        episode,
        valid,
        "approach brown table",
        1.0,
        "pick the cup",
        2.0,
        "turn left and approach the trash bin",
        3.0,
        "put it in to the trash bin",
        InlineText("  exact full prompt  "),
    ]


def test_load_annotations_selects_valid_rows_and_preserves_prompts(tmp_path: Path) -> None:
    xlsx = write_xlsx(
        tmp_path / "annotations.xlsx",
        [HEADERS, [0, 0], valid_row(valid=True), [None] * 10],
    )

    rows = load_annotations(xlsx, expected_episodes={0, 1})

    assert [row.episode for row in rows] == [1]
    assert rows[0].subtasks == (
        "approach brown table",
        "pick the cup",
        "turn left and approach the trash bin",
        "put it in to the trash bin",
    )
    assert rows[0].boundaries_s == (1.0, 2.0, 3.0)
    assert rows[0].full_prompt == "exact full prompt"


@pytest.mark.parametrize(
    ("rows", "expected_episodes", "match"),
    [
        ([HEADERS[:-1], valid_row()[:-1]], {1}, "missing required columns"),
        (
            [[*HEADERS[:-1], "episode"], valid_row()],
            {1},
            "duplicate header",
        ),
        ([HEADERS, valid_row(), valid_row()], {1}, "duplicate episode"),
        ([HEADERS, valid_row(episode=1.5)], {1}, "episode must be an integer"),
        ([HEADERS, valid_row(valid=2)], {1}, "valid must be 0 or 1"),
        ([HEADERS, [*valid_row()[:2], None, *valid_row()[3:]]], {1}, "subtask1"),
        (
            [HEADERS, [*valid_row()[:3], "nan", *valid_row()[4:]]],
            {1},
            "time1 must be finite",
        ),
        (
            [HEADERS, [*valid_row()[:5], 0.5, *valid_row()[6:]]],
            {1},
            "strictly increasing",
        ),
        ([HEADERS, valid_row()], {0, 1}, "missing episode rows"),
    ],
)
def test_load_annotations_rejects_invalid_workbooks(
    tmp_path: Path,
    rows: list[list[object]],
    expected_episodes: set[int],
    match: str,
) -> None:
    xlsx = write_xlsx(tmp_path / "invalid.xlsx", rows)

    with pytest.raises(AnnotationError, match=match):
        load_annotations(xlsx, expected_episodes=expected_episodes)


def test_snap_boundaries_assigns_boundary_frame_to_later_subtask() -> None:
    timestamps = pa.array(np.arange(0, 4.02, 0.02), type=pa.float32())

    frames = snap_boundary_frames(timestamps, (1.0, 2.0, 3.0), fps=50)

    assert frames == (50, 100, 150)
    assert build_run_steps(len(timestamps), frames).tolist() == ([0] * 50 + [1] * 50 + [2] * 50 + [3] * 51)


def test_snap_boundaries_rejects_excessive_snap_error() -> None:
    timestamps = pa.array([0.0, 0.02, 0.10, 0.12, 0.14], type=pa.float32())

    with pytest.raises(AnnotationError, match="farther than half a frame"):
        snap_boundary_frames(timestamps, (0.06, 0.10, 0.12), fps=50)


def test_snap_boundaries_rejects_empty_runs() -> None:
    timestamps = pa.array(np.arange(0, 1.02, 0.02), type=pa.float32())

    with pytest.raises(AnnotationError, match="nonempty subtask runs"):
        snap_boundary_frames(timestamps, (0.001, 0.002, 0.003), fps=50)


def _annotation(
    episode: int,
    subtasks: tuple[str, str, str, str],
    full_prompt: str,
) -> EpisodeAnnotation:
    return EpisodeAnnotation(
        episode=episode,
        subtasks=subtasks,
        boundaries_s=(1.0, 2.0, 3.0),
        full_prompt=full_prompt,
    )


def test_select_annotations_by_direction_keeps_complete_left_episodes() -> None:
    first_left = _annotation(
        4,
        ("approach", "pick", "  TURN   LEFT toward bin ", "put"),
        "full left first",
    )
    second_left = _annotation(
        7,
        ("approach other", "pick other", "turn left toward other bin", "put other"),
        "full left second",
    )
    right = _annotation(9, ("approach", "pick", "turn right toward bin", "put"), "full right")

    selected, selection = select_annotations_by_direction(
        [second_left, right, first_left],
        "left",
        expected_counts=(2, 1),
    )

    assert selected == [first_left, second_left]
    assert selection == AnnotationSelection(
        direction="left",
        candidate_episodes=3,
        selected_episodes=2,
        excluded_episodes=1,
    )


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


def test_select_annotations_by_direction_requires_expected_counts() -> None:
    left = _annotation(1, ("approach", "pick", "turn left", "put"), "full")

    with pytest.raises(AnnotationError, match="requires expected direction counts"):
        select_annotations_by_direction([left], "left")


def test_select_annotations_by_direction_all_preserves_legacy_order() -> None:
    rows = [
        _annotation(2, ("a", "b", "unclassified", "d"), "two"),
        _annotation(1, ("a", "b", "unclassified", "d"), "one"),
    ]

    selected, selection = select_annotations_by_direction(rows, "all")

    assert selected == rows
    assert selection is None


def test_build_task_map_is_stable_under_annotation_order() -> None:
    first = _annotation(1, ("approach a", "pick", "turn left", "drop"), "full z")
    second = _annotation(2, ("approach z", "pick", "turn right", "drop"), "full a")

    expected_subtasks = {
        "approach a": 0,
        "approach z": 1,
        "pick": 2,
        "turn left": 3,
        "turn right": 4,
        "drop": 5,
    }
    assert build_task_map([first, second], "subtasks") == expected_subtasks
    assert build_task_map([second, first], "subtasks") == expected_subtasks
    assert build_task_map([first, second], "full_prompt") == {"full a": 0, "full z": 1}
    assert build_task_map([second, first], "full_prompt") == {"full a": 0, "full z": 1}


def test_build_task_map_rejects_unknown_variant() -> None:
    annotation = _annotation(1, ("a", "b", "c", "d"), "full")

    with pytest.raises(AnnotationError, match="unknown annotation variant"):
        build_task_map([annotation], "invalid")  # type: ignore[arg-type]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _make_source_dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video_dir = root / "videos/chunk-000/observation.images.ego_view"
    video_dir.mkdir(parents=True)

    episode_length = 201
    for episode in range(2):
        state = pa.array(
            [[float(episode), float(frame)] for frame in range(episode_length)],
            type=pa.list_(pa.float64(), 2),
        )
        table = pa.table(
            {
                "observation.state": state,
                "timestamp": pa.array(np.arange(episode_length, dtype=np.float32) * np.float32(0.02)),
                "frame_index": pa.array(np.arange(episode_length), type=pa.int64()),
                "episode_index": pa.array([episode] * episode_length, type=pa.int64()),
                "index": pa.array(
                    np.arange(episode * episode_length, (episode + 1) * episode_length),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * episode_length, type=pa.int64()),
            }
        ).replace_schema_metadata({b"huggingface": b'{"fixture":true}'})
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
        (video_dir / f"episode_{episode:06d}.mp4").write_bytes(f"video-{episode}".encode())

    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 402,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 50,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
        "features": {
            "observation.images.ego_view": {"dtype": "video", "shape": [2, 2, 3]},
            "observation.state": {"dtype": "float64", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/modality.json").write_text(
        json.dumps({"annotation": {"human.task_description": {"original_key": "task_index"}}}),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": episode, "tasks": ["source task"], "length": episode_length} for episode in range(2)],
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "source task"}])
    _write_jsonl(
        root / "meta/episodes_stats.jsonl",
        [
            {
                "episode_index": episode,
                "stats": {
                    "observation.state": {"count": [episode_length]},
                    "timestamp": {
                        "min": [0.0],
                        "max": [4.0],
                        "mean": [2.0],
                        "std": [1.1604596790352808],
                        "count": [episode_length],
                    },
                    "frame_index": {
                        "min": [0],
                        "max": [200],
                        "mean": [100.0],
                        "std": [58.02298395176403],
                        "count": [episode_length],
                    },
                    "episode_index": {
                        "min": [episode],
                        "max": [episode],
                        "mean": [float(episode)],
                        "std": [0.0],
                        "count": [episode_length],
                    },
                    "index": {
                        "min": [episode * episode_length],
                        "max": [(episode + 1) * episode_length - 1],
                        "mean": [episode * episode_length + 100.0],
                        "std": [58.02298395176403],
                        "count": [episode_length],
                    },
                    "task_index": {
                        "min": [0],
                        "max": [0],
                        "mean": [0.0],
                        "std": [0.0],
                        "count": [episode_length],
                    },
                },
            }
            for episode in range(2)
        ],
    )
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_export_variant_builds_self_contained_lerobot_datasets(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    annotation = _annotation(
        1,
        ("approach", "pick", "turn and approach", "drop"),
        "approach, pick, turn and approach, drop",
    )
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"

    subtask_result = export_variant(
        source,
        subtasks,
        [annotation],
        "subtasks",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
    )
    full_result = export_variant(
        source,
        full_prompt,
        [annotation],
        "full_prompt",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
    )

    assert (subtask_result.episodes, subtask_result.frames, subtask_result.tasks) == (1, 201, 4)
    assert (full_result.episodes, full_result.frames, full_result.tasks) == (1, 201, 1)
    subtask_info = json.loads((subtasks / "meta/info.json").read_text(encoding="utf-8"))
    full_info = json.loads((full_prompt / "meta/info.json").read_text(encoding="utf-8"))
    assert subtask_info["total_episodes"] == full_info["total_episodes"] == 1
    assert subtask_info["total_frames"] == full_info["total_frames"] == 201
    assert subtask_info["total_tasks"] == 4
    assert full_info["total_tasks"] == 1
    assert subtask_info["splits"] == full_info["splits"] == {"train": "0:1"}

    source_table = pq.read_table(source / "data/chunk-000/episode_000001.parquet")
    subtask_table = pq.read_table(subtasks / "data/chunk-000/episode_000000.parquet")
    full_table = pq.read_table(full_prompt / "data/chunk-000/episode_000000.parquet")
    assert subtask_table["episode_index"].to_pylist() == [0] * 201
    assert subtask_table["index"].to_pylist() == list(range(201))
    assert subtask_table["frame_index"].to_pylist() == list(range(201))
    assert full_table["task_index"].to_pylist() == [0] * 201
    assert np.bincount(subtask_table["task_index"].to_numpy()).tolist() == [50, 50, 50, 51]
    assert source_table["observation.state"].equals(subtask_table["observation.state"])
    assert source_table["timestamp"].equals(subtask_table["timestamp"])
    assert source_table.schema.metadata == subtask_table.schema.metadata

    assert _read_jsonl(subtasks / "meta/tasks.jsonl") == [
        {"task": "approach", "task_index": 0},
        {"task": "pick", "task_index": 1},
        {"task": "turn and approach", "task_index": 2},
        {"task": "drop", "task_index": 3},
    ]
    assert _read_jsonl(full_prompt / "meta/tasks.jsonl") == [
        {"task": "approach, pick, turn and approach, drop", "task_index": 0}
    ]
    assert _read_jsonl(subtasks / "meta/episodes.jsonl") == [
        {
            "episode_index": 0,
            "length": 201,
            "tasks": ["approach", "pick", "turn and approach", "drop"],
        }
    ]

    stats = _read_jsonl(subtasks / "meta/episodes_stats.jsonl")
    assert stats[0]["episode_index"] == 0
    assert stats[0]["stats"]["episode_index"]["min"] == [0]
    assert stats[0]["stats"]["index"]["max"] == [200]
    assert stats[0]["stats"]["task_index"]["max"] == [3]

    source_video = source / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4"
    output_video = subtasks / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    assert output_video.read_bytes() == source_video.read_bytes()
    assert output_video.stat().st_ino != source_video.stat().st_ino

    provenance = json.loads((subtasks / "meta/annotation_provenance.json").read_text(encoding="utf-8"))
    assert provenance["variant"] == "subtasks"
    assert provenance["source"]["manifest_sha256"] == "a" * 64
    assert provenance["annotations"]["workbook_sha256"] == "b" * 64
    assert provenance["episodes"] == [
        {
            "source_episode_index": 1,
            "output_episode_index": 0,
            "length": 201,
            "boundaries_s": [1.0, 2.0, 3.0],
            "boundary_frames": [50, 100, 150],
            "boundary_timestamps_s": pytest.approx([1.0, 2.0, 3.0]),
            "prompts": ["approach", "pick", "turn and approach", "drop"],
        }
    ]


@pytest.mark.parametrize("path_style", ["absolute", "traversal"])
def test_export_variant_rejects_unsafe_source_controlled_paths_before_writing(
    tmp_path: Path,
    path_style: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if path_style == "absolute":
        info["data_path"] = str(
            source.resolve() / "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
    else:
        info["data_path"] = "../source/data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    source_parquet = source / "data/chunk-000/episode_000001.parquet"
    source_bytes = source_parquet.read_bytes()
    output = tmp_path / "output"

    with pytest.raises(AnnotationError, match="safe relative"):
        export_variant(
            source,
            output,
            [
                _annotation(
                    1,
                    ("approach", "pick", "turn and approach", "drop"),
                    "approach, pick, turn and approach, drop",
                )
            ],
            "subtasks",
            source_manifest_sha256="a" * 64,
            workbook_sha256="b" * 64,
        )

    assert source_parquet.read_bytes() == source_bytes
    assert not output.exists()


def _rewrite_column(path: Path, name: str, values: list[object]) -> None:
    table = pq.read_table(path)
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    replacement = pa.array(values, type=field.type)
    pq.write_table(table.set_column(index, field, replacement), path)


@pytest.mark.parametrize("variant", ["subtasks", "full_prompt"])
def test_validate_variant_accepts_complete_export(tmp_path: Path, variant: str) -> None:
    source = _make_source_dataset(tmp_path / "source")
    annotation = _annotation(
        1,
        ("approach", "pick", "turn and approach", "drop"),
        "approach, pick, turn and approach, drop",
    )
    manifest_sha256 = dataset_manifest_sha256(source)
    output = tmp_path / variant
    export_variant(
        source,
        output,
        [annotation],
        variant,  # type: ignore[arg-type]
        source_manifest_sha256=manifest_sha256,
        workbook_sha256="b" * 64,
    )

    report = validate_variant(
        source,
        output,
        [annotation],
        variant,  # type: ignore[arg-type]
        workbook_sha256="b" * 64,
    )

    assert report == {
        "variant": variant,
        "episodes": 1,
        "frames": 201,
        "tasks": 4 if variant == "subtasks" else 1,
        "videos": 1,
        "expected_runs_per_episode": 4 if variant == "subtasks" else 1,
    }


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
    provenance = json.loads((output / "meta/annotation_provenance.json").read_text(encoding="utf-8"))

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
    ("selection", "turn_prompt"),
    [
        (AnnotationSelection("left", 99, 99, 0), "turn left and approach"),
        (AnnotationSelection("left", 2, 1, 0), "turn left and approach"),
        (AnnotationSelection("left", 0, 1, -1), "turn left and approach"),
        (
            AnnotationSelection("right", 1, 1, 0),  # type: ignore[arg-type]
            "turn left and approach",
        ),
        (AnnotationSelection("left", 1, 1, 0), "turn right and approach"),
    ],
    ids=[
        "false-selected-count",
        "inconsistent-count-algebra",
        "negative-count",
        "invalid-direction",
        "retained-right-annotation",
    ],
)
def test_filtered_export_rejects_selection_inconsistent_with_annotations(
    tmp_path: Path,
    selection: AnnotationSelection,
    turn_prompt: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    output = tmp_path / "subtasks"
    annotation = _annotation(
        0,
        ("approach", "pick", turn_prompt, "put"),
        "approach, pick, turn and approach, put",
    )

    with pytest.raises(AnnotationError, match="selection"):
        export_variant(
            source,
            output,
            [annotation],
            "subtasks",
            source_manifest_sha256="a" * 64,
            workbook_sha256="b" * 64,
            selection=selection,
        )

    assert not output.exists()


def test_filtered_validation_rejects_self_consistent_forged_selection(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    output = tmp_path / "subtasks"
    annotation = _annotation(
        0,
        ("approach", "pick", "turn left and approach", "put"),
        "approach, pick, turn left and approach, put",
    )
    valid_selection = AnnotationSelection("left", 1, 1, 0)
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256="a" * 64,
        workbook_sha256="b" * 64,
        selection=valid_selection,
    )
    forged_selection = AnnotationSelection("left", 99, 99, 0)
    provenance_path = output / "meta/annotation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["selection"] = {
        "direction": forged_selection.direction,
        "candidate_episodes": forged_selection.candidate_episodes,
        "selected_episodes": forged_selection.selected_episodes,
        "excluded_episodes": forged_selection.excluded_episodes,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="selection"):
        validate_variant(
            source,
            output,
            [annotation],
            "subtasks",
            workbook_sha256="b" * 64,
            source_manifest_sha256="a" * 64,
            selection=forged_selection,
        )


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("direction", "right"),
        ("candidate_episodes", 3),
        ("candidate_episodes", 2.0),
        ("selected_episodes", 2),
        ("selected_episodes", True),
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


@pytest.mark.parametrize(
    ("corrupt_value", "selection"),
    [
        (2.0, AnnotationSelection("left", 2, 1, 1)),
        (True, None),
    ],
)
def test_provenance_rejects_type_confused_schema_version(
    tmp_path: Path,
    corrupt_value: object,
    selection: AnnotationSelection | None,
) -> None:
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
        selection=selection,
    )
    provenance_path = output / "meta/annotation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["schema_version"] = corrupt_value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="schema_version"):
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
    provenance = json.loads((output / "meta/annotation_provenance.json").read_text(encoding="utf-8"))

    assert provenance["schema_version"] == 1
    assert "selection" not in provenance


def test_validate_variant_requires_exact_workbook_digest(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    annotation = _annotation(
        1,
        ("approach", "pick", "turn and approach", "drop"),
        "approach, pick, turn and approach, drop",
    )
    output = tmp_path / "subtasks"
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256=dataset_manifest_sha256(source),
        workbook_sha256="b" * 64,
    )

    with pytest.raises(TypeError):
        validate_variant(source, output, [annotation], "subtasks")


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("metadata", "total_frames"),
        ("total_chunks", "total_chunks"),
        ("modality_missing", "modality"),
        ("modality_mapping", "task_index"),
        ("episode_index", "episode_index"),
        ("global_index", "global index"),
        ("task_resolution", "task_index"),
        ("task_runs", "task_index"),
        ("sensor", "payload column"),
        ("video_hash", "video hash"),
        ("video_symlink", "regular non-symlink"),
        ("provenance_mapping", "provenance"),
        ("provenance_schema", "provenance"),
        ("provenance_dataset_path", "provenance"),
        ("provenance_workbook_hash", "provenance"),
        ("source_hash", "source manifest"),
    ],
)
def test_validate_variant_rejects_corruption(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    annotation = _annotation(
        1,
        ("approach", "pick", "turn and approach", "drop"),
        "approach, pick, turn and approach, drop",
    )
    output = tmp_path / "subtasks"
    export_variant(
        source,
        output,
        [annotation],
        "subtasks",
        source_manifest_sha256=dataset_manifest_sha256(source),
        workbook_sha256="b" * 64,
    )
    parquet = output / "data/chunk-000/episode_000000.parquet"
    video = output / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    provenance_path = output / "meta/annotation_provenance.json"

    if corruption == "metadata":
        info_path = output / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_frames"] += 1
        info_path.write_text(json.dumps(info), encoding="utf-8")
    elif corruption == "total_chunks":
        info_path = output / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_chunks"] = 999
        info_path.write_text(json.dumps(info), encoding="utf-8")
    elif corruption == "modality_missing":
        (output / "meta/modality.json").unlink()
    elif corruption == "modality_mapping":
        (output / "meta/modality.json").write_text(
            json.dumps({"annotation": {"human.task_description": {"original_key": "wrong_column"}}}),
            encoding="utf-8",
        )
    elif corruption == "episode_index":
        _rewrite_column(parquet, "episode_index", [7] + [0] * 200)
    elif corruption == "global_index":
        _rewrite_column(parquet, "index", [9] + list(range(1, 201)))
    elif corruption == "task_resolution":
        _rewrite_column(parquet, "task_index", [99] + [0] * 49 + [1] * 50 + [2] * 50 + [3] * 51)
    elif corruption == "task_runs":
        _rewrite_column(parquet, "task_index", [1] + [0] * 49 + [1] * 50 + [2] * 50 + [3] * 51)
    elif corruption == "sensor":
        table = pq.read_table(parquet)
        values = table["observation.state"].to_pylist()
        values[0] = [999.0, 999.0]
        _rewrite_column(parquet, "observation.state", values)
    elif corruption == "video_hash":
        video.write_bytes(b"corrupt")
    elif corruption == "video_symlink":
        video.unlink()
        video.symlink_to(source / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4")
    elif corruption == "provenance_mapping":
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["episodes"][0]["source_episode_index"] = 0
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "provenance_schema":
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["schema_version"] = 999
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "provenance_dataset_path":
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["source"]["dataset_path"] = "/wrong/source"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "provenance_workbook_hash":
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["annotations"]["workbook_sha256"] = "c" * 64
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "source_hash":
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["source"]["manifest_sha256"] = "0" * 64
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    else:
        raise AssertionError(f"unknown test corruption {corruption}")

    with pytest.raises(DatasetValidationError, match=match):
        validate_variant(
            source,
            output,
            [annotation],
            "subtasks",
            workbook_sha256="b" * 64,
        )


def _write_fixture_annotations(source: Path) -> Path:
    return write_xlsx(
        source / "pnp_trash.xlsx",
        [HEADERS, [0, 0], valid_row(episode=1)],
    )


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


def test_export_both_rejects_release_marker_ancestor_of_output(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    marker = tmp_path / "release"
    subtasks = marker / "subtasks"
    full_prompt = tmp_path / "full_prompt"

    with pytest.raises(AnnotationError, match="release marker.*overlap"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert not marker.exists()
    assert not subtasks.exists()
    assert not full_prompt.exists()


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
    sub_provenance = json.loads((subtasks / "meta/annotation_provenance.json").read_text(encoding="utf-8"))
    full_provenance = json.loads((full_prompt / "meta/annotation_provenance.json").read_text(encoding="utf-8"))

    def mapping(value: dict[str, list[dict[str, int]]]) -> list[tuple[int, int, int]]:
        return [
            (row["source_episode_index"], row["output_episode_index"], row["length"]) for row in value["episodes"]
        ]

    assert mapping(sub_provenance) == mapping(full_provenance)

    value = json.loads(marker.read_text(encoding="utf-8"))
    value["datasets"]["subtasks"]["manifest_sha256"] = "0" * 64
    marker.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="manifest"):
        validate_release_marker(marker, subtasks, full_prompt)


def test_release_marker_rejects_cross_variant_episode_mapping_mismatch(tmp_path: Path) -> None:
    _source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    provenance_path = full_prompt / "meta/annotation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["episodes"][0]["source_episode_index"] = 1
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    value = json.loads(marker.read_text(encoding="utf-8"))
    value["datasets"]["subtasks"]["manifest_sha256"] = dataset_manifest_sha256(subtasks)
    value["datasets"]["full_prompt"]["manifest_sha256"] = dataset_manifest_sha256(full_prompt)
    marker.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="mapping"):
        validate_release_marker(marker, subtasks, full_prompt)


@pytest.mark.parametrize("variant", ["subtasks", "full_prompt"])
def test_release_marker_rejects_dataset_root_symlink(tmp_path: Path, variant: str) -> None:
    _source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    dataset = {"subtasks": subtasks, "full_prompt": full_prompt}[variant]
    target = tmp_path / f"{variant}-target"
    dataset.rename(target)
    dataset.symlink_to(target, target_is_directory=True)

    with pytest.raises(DatasetValidationError, match="dataset root.*symlink"):
        validate_release_marker(marker, subtasks, full_prompt)


def test_export_both_validates_temporary_marker_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    validated_markers: list[Path] = []

    def fail_precommit_validation(
        candidate: Path,
        _subtasks: Path,
        _full_prompt: Path,
    ) -> dict[str, object]:
        validated_markers.append(candidate)
        assert candidate != marker
        assert not annotations_module._path_lexists(marker)
        raise DatasetValidationError("forced precommit release validation failure")

    monkeypatch.setattr(annotations_module, "validate_release_marker", fail_precommit_validation)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, DatasetValidationError)
    assert "forced precommit" in str(error_info.value.__cause__)
    assert validated_markers
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_does_not_clobber_raced_release_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    raced_value = b"raced marker\n"
    real_link = annotations_module.os.link

    def race_marker_before_link(
        source_path: Path,
        destination_path: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        assert Path(destination_path) == marker
        marker.write_bytes(raced_value)
        real_link(source_path, destination_path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(annotations_module.os, "link", race_marker_before_link)

    with pytest.raises(DatasetValidationError, match="incomplete publication"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert marker.read_bytes() == raced_value
    assert subtasks.is_dir()
    assert full_prompt.is_dir()


def test_export_both_closes_marker_descriptor_and_preserves_primary_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    marker_descriptors: list[int] = []
    real_mkstemp = annotations_module.tempfile.mkstemp
    real_unlink = Path.unlink

    def record_marker_descriptor(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, temporary = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == f".{marker.name}.":
            marker_descriptors.append(descriptor)
        return descriptor, temporary

    def fail_fdopen(*_args: object, **_kwargs: object) -> None:
        raise OSError("forced marker fdopen failure")

    def fail_marker_temp_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{marker.name}."):
            raise OSError("forced marker temp cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(annotations_module.tempfile, "mkstemp", record_marker_descriptor)
    monkeypatch.setattr(annotations_module.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(Path, "unlink", fail_marker_temp_cleanup)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, OSError)
    assert "forced marker fdopen failure" in str(error_info.value.__cause__)
    assert marker_descriptors
    for descriptor in marker_descriptors:
        with pytest.raises(OSError):
            annotations_module.os.fstat(descriptor)
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


@pytest.mark.parametrize("failure_phase", ["tree", "parent"])
def test_export_both_does_not_commit_marker_before_dataset_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    marker_commit_attempted = False
    real_link = annotations_module.os.link

    def record_marker_link(*args: object, **kwargs: object) -> None:
        nonlocal marker_commit_attempted
        marker_commit_attempted = True
        real_link(*args, **kwargs)

    if failure_phase == "tree":

        def fail_tree_fsync(_path: Path) -> None:
            raise OSError("forced tree durability failure")

        monkeypatch.setattr(annotations_module, "_fsync_tree", fail_tree_fsync, raising=False)
    else:
        monkeypatch.setattr(annotations_module, "_fsync_tree", lambda _path: None, raising=False)
        directory_fsync_calls = 0

        def fail_first_parent_fsync(_path: Path) -> None:
            nonlocal directory_fsync_calls
            directory_fsync_calls += 1
            if directory_fsync_calls == 1:
                raise OSError("forced parent durability failure")

        monkeypatch.setattr(
            annotations_module,
            "_fsync_directory",
            fail_first_parent_fsync,
            raising=False,
        )
    monkeypatch.setattr(annotations_module.os, "link", record_marker_link)

    expected_error = OSError if failure_phase == "tree" else DatasetValidationError
    expected_message = f"forced {failure_phase}" if failure_phase == "tree" else "incomplete publication"
    with pytest.raises(expected_error, match=expected_message):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert not marker_commit_attempted
    assert subtasks.is_dir() is (failure_phase == "parent")
    assert not full_prompt.exists()
    assert not annotations_module._path_lexists(marker)


def test_export_both_preserves_outputs_when_marker_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_fsync_directory = annotations_module._fsync_directory
    marker_fsync_failed = False

    def fail_marker_fsync_once(path: Path) -> None:
        nonlocal marker_fsync_failed
        if annotations_module._path_lexists(marker) and not marker_fsync_failed:
            marker_fsync_failed = True
            raise OSError("forced marker durability failure")
        real_fsync_directory(path)

    monkeypatch.setattr(annotations_module, "_fsync_directory", fail_marker_fsync_once)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, OSError)
    assert "forced marker durability failure" in str(error_info.value.__cause__)
    assert marker_fsync_failed
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_leaves_changed_outputs_and_reports_incomplete_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    changed_file = subtasks / "changed-after-publication"

    def mutate_output_before_failure(
        _candidate: Path,
        _subtasks: Path,
        _full_prompt: Path,
    ) -> dict[str, object]:
        changed_file.write_text("external change", encoding="utf-8")
        raise DatasetValidationError("forced precommit release validation failure")

    monkeypatch.setattr(annotations_module, "validate_release_marker", mutate_output_before_failure)

    with pytest.raises(DatasetValidationError, match="incomplete publication"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert changed_file.read_text(encoding="utf-8") == "external change"
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_preserves_foreign_recreated_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    staging_paths: list[Path] = []
    real_mkdtemp = annotations_module.tempfile.mkdtemp

    def record_staging_path(*args: object, **kwargs: object) -> str:
        temporary = real_mkdtemp(*args, **kwargs)
        staging_paths.append(Path(temporary))
        return temporary

    def recreate_staging_then_fail(
        _candidate: Path,
        _subtasks: Path,
        _full_prompt: Path,
    ) -> dict[str, object]:
        former_staging = staging_paths[0]
        former_staging.mkdir()
        (former_staging / "foreign.txt").write_bytes(b"foreign staging contents\n")
        raise DatasetValidationError("forced precommit release validation failure")

    monkeypatch.setattr(annotations_module.tempfile, "mkdtemp", record_staging_path)
    monkeypatch.setattr(annotations_module, "validate_release_marker", recreate_staging_then_fail)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as caught:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert "incomplete cleanup/publication" not in str(caught.value)
    foreign = staging_paths[0] / "foreign.txt"
    assert foreign.read_bytes() == b"foreign staging contents\n"
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_surfaces_foreign_unpublished_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    staging_paths: list[Path] = []

    def replace_first_staging_then_fail(
        dataset_path: Path,
        output_path: Path,
        annotations: dict[int, object],
        variant: str,
        source_manifest_sha256: str,
        workbook_sha256: str,
        **kwargs: object,
    ) -> object:
        del dataset_path, annotations, variant, source_manifest_sha256, workbook_sha256, kwargs
        staging_paths.append(output_path)
        output_path.rename(output_path.with_name(f"{output_path.name}-moved"))
        output_path.mkdir()
        (output_path / "foreign.txt").write_bytes(b"foreign unpublished staging\n")
        raise DatasetValidationError("forced prepublication failure")

    monkeypatch.setattr(annotations_module, "export_variant", replace_first_staging_then_fail)

    with pytest.raises(DatasetValidationError, match="preserved foreign path"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert (staging_paths[0] / "foreign.txt").read_bytes() == b"foreign unpublished staging\n"
    assert not subtasks.exists()
    assert not full_prompt.exists()
    assert not annotations_module._path_lexists(marker)


def test_export_both_preserves_raced_output_directory_without_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_rename = annotations_module._rename_noreplace

    def race_second_output(staged: Path, output: Path) -> None:
        if output == full_prompt:
            output.mkdir()
            (output / "foreign.txt").write_bytes(b"raced output contents\n")
        real_rename(staged, output)

    monkeypatch.setattr(
        annotations_module,
        "_rename_noreplace",
        race_second_output,
        raising=False,
    )

    with pytest.raises(DatasetValidationError, match="incomplete publication"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert subtasks.is_dir()
    assert (full_prompt / "foreign.txt").read_bytes() == b"raced output contents\n"
    assert not annotations_module._path_lexists(marker)


def test_export_both_handles_keyboard_interrupt_during_marker_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_fsync_directory = annotations_module._fsync_directory

    def interrupt_marker_fsync(path: Path) -> None:
        if annotations_module._path_lexists(marker):
            raise KeyboardInterrupt("forced marker interrupt")
        real_fsync_directory(path)

    monkeypatch.setattr(annotations_module, "_fsync_directory", interrupt_marker_fsync)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, KeyboardInterrupt)
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_handles_keyboard_interrupt_after_marker_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_link = annotations_module.os.link

    def interrupt_after_link(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        raise KeyboardInterrupt("forced post-link interrupt")

    monkeypatch.setattr(annotations_module.os, "link", interrupt_after_link)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, KeyboardInterrupt)
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert not annotations_module._path_lexists(marker)


def test_export_both_handles_keyboard_interrupt_during_second_output_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_rename = annotations_module._rename_noreplace

    def interrupt_second_output(staged: Path, output: Path) -> None:
        if output == full_prompt:
            raise KeyboardInterrupt("forced output interrupt")
        real_rename(staged, output)

    monkeypatch.setattr(
        annotations_module,
        "_rename_noreplace",
        interrupt_second_output,
        raising=False,
    )

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, KeyboardInterrupt)
    assert subtasks.is_dir()
    assert not full_prompt.exists()
    assert not annotations_module._path_lexists(marker)


def test_export_both_handles_keyboard_interrupt_after_first_output_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "release.json"
    real_rename = annotations_module._rename_noreplace

    def interrupt_after_first_rename(staged: Path, output: Path) -> None:
        real_rename(staged, output)
        if output == subtasks:
            raise KeyboardInterrupt("forced post-rename interrupt")

    monkeypatch.setattr(annotations_module, "_rename_noreplace", interrupt_after_first_rename)

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert isinstance(error_info.value.__cause__, KeyboardInterrupt)
    assert subtasks.is_dir()
    assert not full_prompt.exists()
    assert not annotations_module._path_lexists(marker)


def test_export_both_fsyncs_new_ancestors_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "nested/subtasks-parent/subtasks"
    full_prompt = tmp_path / "other/full-prompt-parent/full_prompt"
    marker = tmp_path / "markers/releases/release.json"
    fsync_calls: list[Path] = []
    marker_commit_attempted = False
    real_fsync_directory = annotations_module._fsync_directory
    real_link = annotations_module.os.link

    def fail_second_ancestor_fsync(path: Path) -> None:
        fsync_calls.append(path)
        if path == tmp_path / "nested":
            raise OSError("forced ancestor durability failure")
        real_fsync_directory(path)

    def record_marker_link(*args: object, **kwargs: object) -> None:
        nonlocal marker_commit_attempted
        marker_commit_attempted = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(annotations_module, "_fsync_directory", fail_second_ancestor_fsync)
    monkeypatch.setattr(annotations_module.os, "link", record_marker_link)

    with pytest.raises(OSError, match="forced ancestor durability failure"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    assert fsync_calls[:2] == [tmp_path, tmp_path / "nested"]
    assert not marker_commit_attempted
    assert not subtasks.exists()
    assert not full_prompt.exists()
    assert not annotations_module._path_lexists(marker)


def test_export_both_rejects_symlink_publication_ancestor(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(AnnotationError, match="symlink ancestor"):
        export_both(
            source,
            workbook,
            linked_parent / "subtasks",
            tmp_path / "full_prompt",
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=tmp_path / "release.json",
        )

    assert not (actual_parent / "subtasks").exists()
    assert not (tmp_path / "full_prompt").exists()
    assert not (tmp_path / "release.json").exists()


@pytest.mark.parametrize("protected_path", ["subtasks", "marker"])
def test_export_both_rejects_symlink_component_before_parent_traversal(
    tmp_path: Path,
    protected_path: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    alternate_target = tmp_path / "alternate-parent" / "target"
    alternate_target.mkdir(parents=True)
    symlink = tmp_path / "namespace-link"
    symlink.symlink_to(alternate_target, target_is_directory=True)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    if protected_path == "subtasks":
        subtasks = symlink / ".." / "subtasks"
    else:
        marker = symlink / ".." / "pnp_trash_left_only.release.json"

    with pytest.raises(AnnotationError, match="symlink"):
        export_both(
            source,
            workbook,
            subtasks,
            full_prompt,
            direction_filter="left",
            expected_direction_counts=(1, 1),
            release_marker_path=marker,
        )

    alternate_parent = alternate_target.parent
    assert not (tmp_path / "subtasks").exists()
    assert not (tmp_path / "full_prompt").exists()
    assert not (tmp_path / "pnp_trash_left_only.release.json").exists()
    assert not (alternate_parent / "subtasks").exists()
    assert not (alternate_parent / "pnp_trash_left_only.release.json").exists()


def test_export_both_all_preserves_legacy_schema_without_classification(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    row = valid_row(episode=1)
    row[6] = "move toward the trash bin"
    workbook = write_xlsx(source / "pnp_trash.xlsx", [HEADERS, [0, 0], row])
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"

    export_both(
        source,
        workbook,
        subtasks,
        full_prompt,
        direction_filter="all",
    )

    for output in (subtasks, full_prompt):
        provenance = json.loads((output / "meta/annotation_provenance.json").read_text(encoding="utf-8"))
        assert provenance["schema_version"] == 1
        assert "selection" not in provenance


def test_export_both_all_rejects_release_marker_argument(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)

    with pytest.raises(AnnotationError, match="release marker"):
        export_both(
            source,
            workbook,
            tmp_path / "subtasks",
            tmp_path / "full_prompt",
            direction_filter="all",
            release_marker_path=tmp_path / "release.json",
        )


def test_export_both_refuses_preexisting_output(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    subtasks.mkdir()

    with pytest.raises(AnnotationError, match="already exists"):
        export_both(source, workbook, subtasks, tmp_path / "full_prompt")


def test_export_both_rejects_overlapping_output_paths_before_writing(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "outputs"
    full_prompt = subtasks / "full_prompt"

    with pytest.raises(AnnotationError, match="must not overlap"):
        export_both(source, workbook, subtasks, full_prompt)

    assert not subtasks.exists()


def test_export_both_cleans_first_staging_directory_if_second_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    real_mkdtemp = annotations_module.tempfile.mkdtemp
    calls = 0

    def fail_second_mkdtemp(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced second staging creation failure")
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(annotations_module.tempfile, "mkdtemp", fail_second_mkdtemp)

    with pytest.raises(OSError, match="forced second staging creation failure"):
        export_both(source, workbook, tmp_path / "subtasks", tmp_path / "full_prompt")

    assert not list(tmp_path.glob(".*.staging-*"))


def test_export_both_publishes_neither_output_when_second_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    real_export = annotations_module.export_variant

    def fail_full_prompt(*args: object, **kwargs: object):
        variant = args[3]
        if variant == "full_prompt":
            raise AnnotationError("forced full prompt failure")
        return real_export(*args, **kwargs)

    monkeypatch.setattr(annotations_module, "export_variant", fail_full_prompt)

    with pytest.raises(AnnotationError, match="forced full prompt failure"):
        export_both(source, workbook, subtasks, full_prompt)

    assert not subtasks.exists()
    assert not full_prompt.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_export_both_preserves_first_output_when_second_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    real_rename = annotations_module._rename_noreplace

    def fail_second_publish(path: Path, target: Path) -> None:
        if target == full_prompt:
            raise OSError("forced second publish failure")
        real_rename(path, target)

    monkeypatch.setattr(
        annotations_module,
        "_rename_noreplace",
        fail_second_publish,
        raising=False,
    )

    with pytest.raises(DatasetValidationError, match="incomplete publication") as error_info:
        export_both(source, workbook, subtasks, full_prompt)

    assert isinstance(error_info.value.__cause__, OSError)
    assert "forced second publish failure" in str(error_info.value.__cause__)
    assert subtasks.is_dir()
    assert not full_prompt.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_export_both_rejects_workbook_changed_during_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = write_xlsx(
        tmp_path / "external_annotations.xlsx",
        [HEADERS, [0, 0], valid_row(episode=1)],
    )
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    real_export = annotations_module.export_variant

    def mutate_workbook_after_first_export(*args: object, **kwargs: object):
        result = real_export(*args, **kwargs)
        if args[3] == "subtasks":
            changed = valid_row(episode=1)
            changed[-1] = InlineText("changed full prompt")
            write_xlsx(workbook, [HEADERS, [0, 0], changed])
        return result

    monkeypatch.setattr(
        annotations_module,
        "export_variant",
        mutate_workbook_after_first_export,
    )

    with pytest.raises(DatasetValidationError, match="workbook changed"):
        export_both(source, workbook, subtasks, full_prompt)

    assert not subtasks.exists()
    assert not full_prompt.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_export_both_publishes_two_validated_outputs(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"

    results = export_both(source, workbook, subtasks, full_prompt)

    assert [result.output_path for result in results] == [subtasks, full_prompt]
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    assert (
        validate_variant(
            source,
            subtasks,
            load_annotations(workbook, expected_episodes={0, 1}),
            "subtasks",
            workbook_sha256=annotations_module.file_sha256(workbook),
        )["episodes"]
        == 1
    )


def test_cli_validate_only_reports_both_existing_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    export_both(source, workbook, subtasks, full_prompt)

    reports = annotation_cli_main(
        AnnotatePnpTrashConfig(
            dataset_path=source,
            annotations_path=workbook,
            subtasks_output_path=subtasks,
            full_prompt_output_path=full_prompt,
            validate_only=True,
        )
    )

    assert [report["variant"] for report in reports] == ["subtasks", "full_prompt"]
    printed = capsys.readouterr().out
    assert '"episodes": 1' in printed
    assert '"expected_runs_per_episode": 4' in printed
    assert '"expected_runs_per_episode": 1' in printed


def test_cli_validates_complete_left_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)

    reports = annotation_cli_main(
        AnnotatePnpTrashConfig(
            dataset_path=source,
            annotations_path=source / "pnp_trash.xlsx",
            subtasks_output_path=subtasks,
            full_prompt_output_path=full_prompt,
            direction_filter="left",
            expected_left_episodes=1,
            expected_right_episodes=1,
            release_marker_path=marker,
            validate_only=True,
        )
    )

    assert [report["episodes"] for report in reports] == [1, 1]
    payload = json.loads(capsys.readouterr().out)
    assert payload["release"]["direction"] == "left"
    assert payload["release"]["state"] == "complete"


def test_cli_left_export_rejects_broken_marker_symlink(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    marker.symlink_to(tmp_path / "missing-release.json")

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
            )
        )

    assert not subtasks.exists()
    assert not full_prompt.exists()


def test_cli_all_export_ignores_release_marker_path(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    marker.symlink_to(tmp_path / "missing-release.json")

    reports = annotation_cli_main(
        AnnotatePnpTrashConfig(
            dataset_path=source,
            annotations_path=workbook,
            subtasks_output_path=subtasks,
            full_prompt_output_path=full_prompt,
            release_marker_path=marker,
        )
    )

    assert [report["variant"] for report in reports] == ["subtasks", "full_prompt"]
    assert subtasks.is_dir()
    assert full_prompt.is_dir()
    for output in (subtasks, full_prompt):
        provenance = json.loads((output / "meta/annotation_provenance.json").read_text(encoding="utf-8"))
        assert provenance["schema_version"] == 1


def test_cli_export_validates_canonical_published_output_paths(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    subtasks = tmp_path / "missing" / ".." / "subtasks"
    full_prompt = tmp_path / "other-missing" / ".." / "full_prompt"

    reports = annotation_cli_main(
        AnnotatePnpTrashConfig(
            dataset_path=source,
            annotations_path=workbook,
            subtasks_output_path=subtasks,
            full_prompt_output_path=full_prompt,
        )
    )

    assert [report["variant"] for report in reports] == ["subtasks", "full_prompt"]
    assert (tmp_path / "subtasks").is_dir()
    assert (tmp_path / "full_prompt").is_dir()


def test_cli_validate_only_rejects_output_symlink_ancestor(tmp_path: Path) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_fixture_annotations(source)
    published = tmp_path / "published"
    subtasks = published / "subtasks"
    full_prompt = published / "full_prompt"
    export_both(source, workbook, subtasks, full_prompt)
    alias = tmp_path / "published-alias"
    alias.symlink_to(published, target_is_directory=True)

    with pytest.raises(DatasetValidationError, match="symlink"):
        annotation_cli_main(
            AnnotatePnpTrashConfig(
                dataset_path=source,
                annotations_path=workbook,
                subtasks_output_path=alias / "subtasks",
                full_prompt_output_path=full_prompt,
                validate_only=True,
            )
        )


@pytest.mark.parametrize("protected_path", ["subtasks", "marker"])
def test_cli_left_export_rejects_symlink_component_before_parent_traversal(
    tmp_path: Path,
    protected_path: str,
) -> None:
    source = _make_source_dataset(tmp_path / "source")
    workbook = _write_two_direction_fixture_annotations(source)
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = tmp_path / "namespace-link"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"
    marker = tmp_path / "pnp_trash_left_only.release.json"
    if protected_path == "subtasks":
        subtasks = symlink / ".." / "subtasks"
    else:
        marker = symlink / ".." / "pnp_trash_left_only.release.json"

    with pytest.raises(AnnotationError, match="symlink"):
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
            )
        )

    assert not (tmp_path / "subtasks").exists()
    assert not (tmp_path / "full_prompt").exists()
    assert not (tmp_path / "pnp_trash_left_only.release.json").exists()


@pytest.mark.parametrize("protected_path", ["subtasks", "marker"])
def test_cli_validate_only_rejects_symlink_component_before_parent_traversal(
    tmp_path: Path,
    protected_path: str,
) -> None:
    source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = tmp_path / "namespace-link"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    if protected_path == "subtasks":
        subtasks = symlink / ".." / "subtasks"
    else:
        marker = symlink / ".." / "pnp_trash_left_only.release.json"

    with pytest.raises(DatasetValidationError, match="symlink"):
        annotation_cli_main(
            AnnotatePnpTrashConfig(
                dataset_path=source,
                annotations_path=source / "pnp_trash.xlsx",
                subtasks_output_path=subtasks,
                full_prompt_output_path=full_prompt,
                validate_only=True,
                direction_filter="left",
                expected_left_episodes=1,
                expected_right_episodes=1,
                release_marker_path=marker,
            )
        )


def test_cli_positional_config_preserves_validate_only_field(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workbook = tmp_path / "annotations.xlsx"
    subtasks = tmp_path / "subtasks"
    full_prompt = tmp_path / "full_prompt"

    config = AnnotatePnpTrashConfig(source, workbook, subtasks, full_prompt, True)

    assert config.validate_only is True
    assert config.direction_filter == "all"


def test_cli_parses_left_only_flags(tmp_path: Path) -> None:
    marker = tmp_path / "left.release.json"

    config = tyro.cli(
        AnnotatePnpTrashConfig,
        args=[
            "--direction-filter",
            "left",
            "--expected-left-episodes",
            "1",
            "--expected-right-episodes",
            "2",
            "--release-marker-path",
            str(marker),
        ],
    )

    assert config.direction_filter == "left"
    assert config.expected_left_episodes == 1
    assert config.expected_right_episodes == 2
    assert config.release_marker_path == marker


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("workbook", "workbook changed during validation"),
        ("source", "source changed during validation"),
    ],
)
def test_cli_rejects_inputs_changed_after_left_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    source, subtasks, full_prompt, marker = _publish_filtered_fixture(tmp_path)
    workbook = source / "pnp_trash.xlsx"
    original_validate_release_marker = annotation_cli_module.validate_release_marker

    def mutate_after_marker_validation(
        release_marker: Path,
        subtasks_output: Path,
        full_prompt_output: Path,
    ) -> dict[str, object]:
        report = original_validate_release_marker(release_marker, subtasks_output, full_prompt_output)
        if mutation == "workbook":
            workbook.write_bytes(workbook.read_bytes() + b"\n")
        elif mutation == "source":
            info = source / "meta/info.json"
            info.write_bytes(info.read_bytes() + b"\n")
        return report

    monkeypatch.setattr(annotation_cli_module, "validate_release_marker", mutate_after_marker_validation)

    with pytest.raises(DatasetValidationError, match=message):
        annotation_cli_main(
            AnnotatePnpTrashConfig(
                dataset_path=source,
                annotations_path=workbook,
                subtasks_output_path=subtasks,
                full_prompt_output_path=full_prompt,
                validate_only=True,
                direction_filter="left",
                expected_left_episodes=1,
                expected_right_episodes=1,
                release_marker_path=marker,
            )
        )

    assert capsys.readouterr().out == ""
