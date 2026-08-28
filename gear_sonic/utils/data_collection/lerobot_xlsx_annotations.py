"""Build LeRobot v2.1 language annotations from a simple XLSX workbook."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Literal, Sequence
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"^([A-Z]+)[1-9][0-9]*$")

_REQUIRED_HEADERS = (
    "episode",
    "valid",
    "subtask1",
    "time1",
    "subtask2",
    "time2",
    "subtask3",
    "time3",
    "subtask4",
    "full prompt",
)


class AnnotationError(ValueError):
    """Raised when the annotation workbook violates the export contract."""


@dataclass(frozen=True)
class EpisodeAnnotation:
    """Validated annotations for one retained source episode."""

    episode: int
    subtasks: tuple[str, str, str, str]
    boundaries_s: tuple[float, float, float]
    full_prompt: str


AnnotationVariant = Literal["subtasks", "full_prompt"]


def _validate_nonempty_runs(length: int, starts: tuple[int, int, int]) -> None:
    if length < 4 or not 0 < starts[0] < starts[1] < starts[2] < length:
        raise AnnotationError(
            f"snapped boundaries must create four nonempty subtask runs, got length={length}, "
            f"starts={starts}"
        )


def snap_boundary_frames(
    timestamps: pa.Array | pa.ChunkedArray,
    boundaries_s: tuple[float, float, float],
    fps: int,
) -> tuple[int, int, int]:
    """Snap annotated seconds to nearest timestamp indices.

    Equal-distance ties use the earlier frame. The selected boundary frame is
    the first frame of the later subtask.
    """

    if fps <= 0:
        raise AnnotationError(f"fps must be positive, got {fps}")
    values = np.asarray(timestamps.to_pylist(), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise AnnotationError("timestamps must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise AnnotationError("timestamps must be finite and strictly increasing")

    maximum_error = 0.5 / fps + 1e-6
    selected: list[int] = []
    for boundary in boundaries_s:
        if not math.isfinite(boundary):
            raise AnnotationError(f"boundary must be finite, got {boundary!r}")
        insertion = int(np.searchsorted(values, boundary, side="left"))
        candidates = {max(0, insertion - 1), min(len(values) - 1, insertion)}
        frame = min(candidates, key=lambda index: (abs(values[index] - boundary), index))
        error = abs(values[frame] - boundary)
        if error > maximum_error:
            raise AnnotationError(
                f"boundary {boundary:g}s is farther than half a frame from a timestamp "
                f"(nearest={values[frame]:g}s, error={error:g}s, fps={fps})"
            )
        selected.append(frame)

    starts = tuple(selected)
    _validate_nonempty_runs(len(values), starts)
    return starts


def build_run_steps(length: int, starts: tuple[int, int, int]) -> np.ndarray:
    """Return subtask step numbers 0..3 for every frame."""

    _validate_nonempty_runs(length, starts)
    return np.repeat(np.arange(4, dtype=np.int64), np.diff((0, *starts, length)))


def build_task_map(
    annotations: Sequence[EpisodeAnnotation],
    variant: AnnotationVariant,
) -> dict[str, int]:
    """Build a deterministic exact-string prompt-to-index map."""

    if variant == "subtasks":
        minimum_step: dict[str, int] = {}
        for annotation in annotations:
            for step, prompt in enumerate(annotation.subtasks):
                minimum_step[prompt] = min(step, minimum_step.get(prompt, step))
        prompts = [
            prompt
            for prompt, _step in sorted(
                minimum_step.items(),
                key=lambda item: (item[1], item[0].encode("utf-8")),
            )
        ]
    elif variant == "full_prompt":
        prompts = sorted(
            {annotation.full_prompt for annotation in annotations},
            key=lambda prompt: prompt.encode("utf-8"),
        )
    else:
        raise AnnotationError(f"unknown annotation variant: {variant!r}")
    return {prompt: index for index, prompt in enumerate(prompts)}


@dataclass(frozen=True)
class ExportResult:
    """Summary of one emitted annotation variant."""

    output_path: Path
    variant: AnnotationVariant
    episodes: int
    frames: int
    tasks: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnnotationError(f"JSON file must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AnnotationError(
                        f"JSONL row must contain an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError(f"cannot read JSONL file {path}: {error}") from error
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=4)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _format_episode_path(info: dict[str, Any], key: str, episode_index: int) -> Path:
    chunks_size = _integer_cell(info.get("chunks_size", 1000), "chunks_size")
    if chunks_size <= 0:
        raise AnnotationError(f"chunks_size must be positive, got {chunks_size}")
    pattern = info[key]
    if not isinstance(pattern, str):
        raise AnnotationError(f"info.json {key} must be a string")
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    )


def _format_video_path(
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunks_size = _integer_cell(info.get("chunks_size", 1000), "chunks_size")
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    if not isinstance(pattern, str):
        raise AnnotationError("info.json video_path must be a string")
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise AnnotationError("info.json features must be an object")
    return sorted(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise AnnotationError(f"source parquet is missing required column {name!r}")
    field = table.schema.field(index)
    replacement = pa.array(values, type=field.type)
    return table.set_column(index, field, replacement)


def _assert_source_identifiers(table: pa.Table, source_episode: int) -> None:
    length = len(table)
    required = ("episode_index", "frame_index", "index", "task_index", "timestamp")
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise AnnotationError(f"source parquet is missing required columns: {missing}")
    episode_values = np.asarray(table["episode_index"].to_pylist())
    frame_values = np.asarray(table["frame_index"].to_pylist())
    if not np.array_equal(episode_values, np.full(length, source_episode)):
        raise AnnotationError(
            f"source episode {source_episode} has inconsistent episode_index values"
        )
    if not np.array_equal(frame_values, np.arange(length)):
        raise AnnotationError(f"source episode {source_episode} has noncontiguous frame_index")


def _scalar_stats(values: np.ndarray) -> dict[str, list[int | float]]:
    if len(values) == 0:
        raise AnnotationError("cannot compute statistics for an empty array")
    is_integer = np.issubdtype(values.dtype, np.integer)

    def scalar(value: np.generic) -> int | float:
        return int(value) if is_integer else float(value)

    return {
        "min": [scalar(values.min())],
        "max": [scalar(values.max())],
        "mean": [float(values.astype(np.float64).mean())],
        "std": [float(values.astype(np.float64).std())],
        "count": [len(values)],
    }


def _episode_prompt_indices(
    annotation: EpisodeAnnotation,
    variant: AnnotationVariant,
    task_map: dict[str, int],
    length: int,
    boundary_frames: tuple[int, int, int],
) -> tuple[np.ndarray, list[str]]:
    if variant == "subtasks":
        steps = build_run_steps(length, boundary_frames)
        prompt_ids = np.asarray(
            [task_map[annotation.subtasks[step]] for step in range(4)],
            dtype=np.int64,
        )
        return prompt_ids[steps], list(annotation.subtasks)
    if variant == "full_prompt":
        return (
            np.full(length, task_map[annotation.full_prompt], dtype=np.int64),
            [annotation.full_prompt],
        )
    raise AnnotationError(f"unknown annotation variant: {variant!r}")


def export_variant(
    source: Path,
    destination: Path,
    annotations: Sequence[EpisodeAnnotation],
    variant: AnnotationVariant,
    source_manifest_sha256: str,
    workbook_sha256: str,
) -> ExportResult:
    """Materialize one self-contained LeRobot annotation variant."""

    if variant not in {"subtasks", "full_prompt"}:
        raise AnnotationError(f"unknown annotation variant: {variant!r}")
    if not annotations:
        raise AnnotationError("cannot export an empty annotation set")
    if destination.exists():
        if any(destination.iterdir()):
            raise AnnotationError(f"destination already exists and is not empty: {destination}")
    else:
        destination.mkdir(parents=True)

    source_info = _read_json(source / "meta/info.json")
    if source_info.get("codebase_version") != "v2.1":
        raise AnnotationError("source dataset must use LeRobot codebase_version v2.1")
    fps = _integer_cell(source_info.get("fps"), "fps")
    source_episodes = {
        _integer_cell(row.get("episode_index"), "episode_index"): row
        for row in _read_jsonl(source / "meta/episodes.jsonl")
    }
    source_stats = {
        _integer_cell(row.get("episode_index"), "episode_index"): row
        for row in _read_jsonl(source / "meta/episodes_stats.jsonl")
    }
    ordered_annotations = sorted(annotations, key=lambda item: item.episode)
    annotation_episodes = [annotation.episode for annotation in ordered_annotations]
    if len(set(annotation_episodes)) != len(annotation_episodes):
        raise AnnotationError("annotations contain duplicate source episodes")
    missing_episodes = sorted(set(annotation_episodes) - source_episodes.keys())
    if missing_episodes:
        raise AnnotationError(f"annotations reference missing source episodes: {missing_episodes}")

    task_map = build_task_map(ordered_annotations, variant)
    video_keys = _video_keys(source_info)
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    provenance_episodes: list[dict[str, Any]] = []
    global_index = 0
    copied_videos = 0

    for output_episode, annotation in enumerate(ordered_annotations):
        source_data_path = source / _format_episode_path(
            source_info, "data_path", annotation.episode
        )
        if not source_data_path.is_file():
            raise AnnotationError(f"source parquet does not exist: {source_data_path}")
        table = pq.read_table(source_data_path)
        _assert_source_identifiers(table, annotation.episode)
        length = len(table)
        source_length = _integer_cell(
            source_episodes[annotation.episode].get("length"),
            f"source episode {annotation.episode} length",
        )
        if length != source_length:
            raise AnnotationError(
                f"source episode {annotation.episode} metadata length {source_length} "
                f"does not match parquet length {length}"
            )
        boundary_frames = snap_boundary_frames(
            table["timestamp"], annotation.boundaries_s, fps
        )
        task_indices, episode_prompts = _episode_prompt_indices(
            annotation, variant, task_map, length, boundary_frames
        )

        rewritten = _replace_column(
            table,
            "episode_index",
            np.full(length, output_episode, dtype=np.int64),
        )
        rewritten = _replace_column(
            rewritten,
            "index",
            np.arange(global_index, global_index + length, dtype=np.int64),
        )
        rewritten = _replace_column(rewritten, "task_index", task_indices)
        output_data_path = destination / _format_episode_path(
            source_info, "data_path", output_episode
        )
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rewritten, output_data_path)

        for video_key in video_keys:
            source_video = source / _format_video_path(
                source_info, annotation.episode, video_key
            )
            if not source_video.is_file() or source_video.is_symlink():
                raise AnnotationError(f"source video is missing or not regular: {source_video}")
            output_video = destination / _format_video_path(
                source_info, output_episode, video_key
            )
            output_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, output_video)
            copied_videos += 1

        output_episodes.append(
            {
                "episode_index": output_episode,
                "tasks": episode_prompts,
                "length": length,
            }
        )
        stats_row = deepcopy(source_stats.get(annotation.episode))
        if stats_row is None:
            raise AnnotationError(
                f"source statistics are missing episode {annotation.episode}"
            )
        stats = stats_row.get("stats")
        if not isinstance(stats, dict):
            raise AnnotationError(
                f"source statistics for episode {annotation.episode} have no stats object"
            )
        stats_row["episode_index"] = output_episode
        stats["episode_index"] = _scalar_stats(
            np.full(length, output_episode, dtype=np.int64)
        )
        stats["index"] = _scalar_stats(
            np.arange(global_index, global_index + length, dtype=np.int64)
        )
        stats["task_index"] = _scalar_stats(task_indices)
        output_stats.append(stats_row)

        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        provenance_episodes.append(
            {
                "source_episode_index": annotation.episode,
                "output_episode_index": output_episode,
                "length": length,
                "boundaries_s": list(annotation.boundaries_s),
                "boundary_frames": list(boundary_frames),
                "boundary_timestamps_s": [
                    float(timestamps[frame]) for frame in boundary_frames
                ],
                "prompts": episode_prompts,
            }
        )
        global_index += length

    output_info = deepcopy(source_info)
    chunks_size = _integer_cell(source_info.get("chunks_size", 1000), "chunks_size")
    output_info.update(
        {
            "total_episodes": len(ordered_annotations),
            "total_frames": global_index,
            "total_tasks": len(task_map),
            "total_videos": copied_videos,
            "total_chunks": max(1, math.ceil(len(ordered_annotations) / chunks_size)),
            "splits": {"train": f"0:{len(ordered_annotations)}"},
        }
    )
    _write_json(destination / "meta/info.json", output_info)
    _write_jsonl(destination / "meta/episodes.jsonl", output_episodes)
    _write_jsonl(
        destination / "meta/tasks.jsonl",
        [
            {"task_index": task_index, "task": prompt}
            for prompt, task_index in sorted(task_map.items(), key=lambda item: item[1])
        ],
    )
    _write_jsonl(destination / "meta/episodes_stats.jsonl", output_stats)
    source_modality = source / "meta/modality.json"
    if not source_modality.is_file():
        raise AnnotationError(f"source modality metadata does not exist: {source_modality}")
    shutil.copy2(source_modality, destination / "meta/modality.json")
    _write_json(
        destination / "meta/annotation_provenance.json",
        {
            "schema_version": 1,
            "variant": variant,
            "source": {
                "dataset_path": str(source.resolve()),
                "manifest_sha256": source_manifest_sha256,
            },
            "annotations": {"workbook_sha256": workbook_sha256},
            "tasks": [
                {"task_index": task_index, "task": prompt}
                for prompt, task_index in sorted(task_map.items(), key=lambda item: item[1])
            ],
            "episodes": provenance_episodes,
        },
    )
    return ExportResult(
        output_path=destination,
        variant=variant,
        episodes=len(ordered_annotations),
        frames=global_index,
        tasks=len(task_map),
    )


def _column_index(cell_reference: str) -> int:
    match = _CELL_REFERENCE.fullmatch(cell_reference)
    if match is None:
        raise AnnotationError(f"invalid XLSX cell reference: {cell_reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _read_shared_strings(archive: ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    namespace = {"m": _SHEET_NS}
    return [
        "".join(text.text or "" for text in item.findall(".//m:t", namespace))
        for item in root.findall("m:si", namespace)
    ]


def _first_worksheet_path(archive: ZipFile) -> str:
    namespace = {"m": _SHEET_NS, "r": _OFFICE_REL_NS}
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find("m:sheets", namespace)
    if sheets is None or len(sheets) == 0:
        raise AnnotationError("XLSX workbook has no worksheets")
    relationship_id = sheets[0].attrib.get(f"{{{_OFFICE_REL_NS}}}id")
    if not relationship_id:
        raise AnnotationError("first XLSX worksheet has no relationship id")

    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target")
            break
    if not target:
        raise AnnotationError("first XLSX worksheet relationship is missing")

    normalized = PurePosixPath(target.lstrip("/"))
    if not normalized.parts or normalized.parts[0] != "xl":
        normalized = PurePosixPath("xl") / normalized
    if ".." in normalized.parts:
        raise AnnotationError("XLSX worksheet path escapes the archive root")
    worksheet_path = normalized.as_posix()
    if worksheet_path not in archive.namelist():
        raise AnnotationError(f"XLSX worksheet is missing: {worksheet_path}")
    return worksheet_path


def _decode_cell(cell: ElementTree.Element, shared_strings: list[str]) -> Any:
    namespace = {"m": _SHEET_NS}
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//m:t", namespace))

    value_element = cell.find("m:v", namespace)
    if value_element is None or value_element.text is None:
        return None
    raw_value = value_element.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError) as error:
            raise AnnotationError(f"invalid XLSX shared string index: {raw_value!r}") from error
    if cell_type == "b":
        if raw_value not in {"0", "1"}:
            raise AnnotationError(f"invalid XLSX boolean value: {raw_value!r}")
        return raw_value == "1"
    if cell_type in {"str", "e"}:
        return raw_value
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _read_first_worksheet(archive: ZipFile) -> list[dict[int, Any]]:
    shared_strings = _read_shared_strings(archive)
    worksheet_path = _first_worksheet_path(archive)
    root = ElementTree.fromstring(archive.read(worksheet_path))
    namespace = {"m": _SHEET_NS}
    rows: list[dict[int, Any]] = []
    for row in root.findall(".//m:sheetData/m:row", namespace):
        values: dict[int, Any] = {}
        for cell in row.findall("m:c", namespace):
            reference = cell.attrib.get("r")
            if reference is None:
                raise AnnotationError("XLSX cell is missing its reference")
            values[_column_index(reference)] = _decode_cell(cell, shared_strings)
        rows.append(values)
    return rows


def _normalized_header(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split()) if value is not None else ""


def _integer_cell(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{name} must be an integer, got {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise AnnotationError(f"{name} must be an integer, got {value!r}")
    return int(numeric)


def _required_text(row: dict[str, Any], name: str, episode: int) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise AnnotationError(f"episode {episode}: {name} must be nonempty")
    return str(value).strip()


def _required_time(row: dict[str, Any], name: str, episode: int) -> float:
    value = row.get(name)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"episode {episode}: {name} must be finite") from error
    if not math.isfinite(result):
        raise AnnotationError(f"episode {episode}: {name} must be finite")
    return result


def _validate_and_select_rows(
    worksheet_rows: list[dict[int, Any]],
    expected_episodes: set[int] | None,
) -> list[EpisodeAnnotation]:
    if not worksheet_rows:
        raise AnnotationError("XLSX worksheet is empty")

    headers: dict[str, int] = {}
    for column, value in worksheet_rows[0].items():
        header = _normalized_header(value)
        if not header:
            continue
        if header in headers:
            raise AnnotationError(f"duplicate header: {header!r}")
        headers[header] = column
    missing_headers = sorted(set(_REQUIRED_HEADERS) - headers.keys())
    if missing_headers:
        raise AnnotationError(f"missing required columns: {missing_headers}")

    seen_episodes: set[int] = set()
    annotations: list[EpisodeAnnotation] = []
    for worksheet_row in worksheet_rows[1:]:
        episode_value = worksheet_row.get(headers["episode"])
        if episode_value is None or str(episode_value).strip() == "":
            continue
        episode = _integer_cell(episode_value, "episode")
        if episode < 0:
            raise AnnotationError(f"episode must be nonnegative, got {episode}")
        if episode in seen_episodes:
            raise AnnotationError(f"duplicate episode row: {episode}")
        seen_episodes.add(episode)

        values = {name: worksheet_row.get(column) for name, column in headers.items()}
        valid = _integer_cell(values["valid"], f"episode {episode}: valid")
        if valid not in {0, 1}:
            raise AnnotationError(f"episode {episode}: valid must be 0 or 1, got {valid}")
        if valid == 0:
            continue

        subtasks = tuple(
            _required_text(values, f"subtask{number}", episode) for number in range(1, 5)
        )
        boundaries = tuple(
            _required_time(values, f"time{number}", episode) for number in range(1, 4)
        )
        if not 0 < boundaries[0] < boundaries[1] < boundaries[2]:
            raise AnnotationError(
                f"episode {episode}: time1, time2, and time3 must be strictly increasing and positive"
            )
        annotations.append(
            EpisodeAnnotation(
                episode=episode,
                subtasks=subtasks,
                boundaries_s=boundaries,
                full_prompt=_required_text(values, "full prompt", episode),
            )
        )

    if expected_episodes is not None and seen_episodes != expected_episodes:
        missing = sorted(expected_episodes - seen_episodes)
        unexpected = sorted(seen_episodes - expected_episodes)
        details = []
        if missing:
            details.append(f"missing episode rows {missing}")
        if unexpected:
            details.append(f"unexpected episode rows {unexpected}")
        raise AnnotationError("; ".join(details))
    if not annotations:
        raise AnnotationError("workbook contains no rows with valid == 1")
    return sorted(annotations, key=lambda annotation: annotation.episode)


def load_annotations(
    path: Path,
    expected_episodes: set[int] | None = None,
) -> list[EpisodeAnnotation]:
    """Load, validate, and return rows whose ``valid`` value is one."""

    try:
        with ZipFile(path) as archive:
            worksheet_rows = _read_first_worksheet(archive)
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise AnnotationError(f"cannot read XLSX workbook {path}: {error}") from error
    return _validate_and_select_rows(worksheet_rows, expected_episodes)
