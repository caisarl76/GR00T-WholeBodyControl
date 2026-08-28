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

from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import (
    AnnotationError,
    EpisodeAnnotation,
    build_run_steps,
    build_task_map,
    export_variant,
    load_annotations,
    snap_boundary_frames,
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
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{escape(value.value)}</t></is></c>'
                )
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
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
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
    assert build_run_steps(len(timestamps), frames).tolist() == (
        [0] * 50 + [1] * 50 + [2] * 50 + [3] * 51
    )


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
                "timestamp": pa.array(
                    np.arange(episode_length, dtype=np.float32) * np.float32(0.02)
                ),
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
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
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
    (root / "meta/modality.json").write_text('{"annotation":{}}', encoding="utf-8")
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [
            {"episode_index": episode, "tasks": ["source task"], "length": episode_length}
            for episode in range(2)
        ],
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

    provenance = json.loads(
        (subtasks / "meta/annotation_provenance.json").read_text(encoding="utf-8")
    )
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
