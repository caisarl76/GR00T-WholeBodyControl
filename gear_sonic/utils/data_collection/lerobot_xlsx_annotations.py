"""Build LeRobot v2.1 language annotations from a simple XLSX workbook."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
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


class DatasetValidationError(AnnotationError):
    """Raised when an emitted dataset fails structural validation."""


@dataclass(frozen=True)
class EpisodeAnnotation:
    """Validated annotations for one retained source episode."""

    episode: int
    subtasks: tuple[str, str, str, str]
    boundaries_s: tuple[float, float, float]
    full_prompt: str


DirectionFilter = Literal["all", "left"]


@dataclass(frozen=True)
class AnnotationSelection:
    """Summary of a validated left-only annotation selection."""

    direction: Literal["left"]
    candidate_episodes: int
    selected_episodes: int
    excluded_episodes: int


def _turn_direction(prompt: str) -> Literal["left", "right"]:
    normalized = " ".join(prompt.casefold().split())
    has_left = re.search(r"\bturn left\b", normalized) is not None
    has_right = re.search(r"\bturn right\b", normalized) is not None
    if has_left == has_right:
        raise AnnotationError(f"subtask3 must contain exactly one turn direction, got {prompt!r}")
    return "left" if has_left else "right"


def select_annotations_by_direction(
    annotations: Sequence[EpisodeAnnotation],
    direction_filter: DirectionFilter,
    *,
    expected_counts: tuple[int, int] | None = None,
) -> tuple[list[EpisodeAnnotation], AnnotationSelection | None]:
    """Select annotations with the supported ``all`` or ``left`` filter.

    Left selection classifies subtask 3 (``subtasks[2]``) and interprets
    ``expected_counts`` in ``(left, right)`` order.
    """

    if direction_filter == "all":
        return list(annotations), None
    if direction_filter != "left":
        raise AnnotationError(f"unknown direction filter: {direction_filter!r}")
    classified = [(row, _turn_direction(row.subtasks[2])) for row in annotations]
    left = sorted(
        (row for row, direction in classified if direction == "left"),
        key=lambda row: row.episode,
    )
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


def _selection_provenance(
    selection: AnnotationSelection,
    annotations: Sequence[EpisodeAnnotation],
) -> dict[str, object]:
    if type(selection.direction) is not str or selection.direction != "left":
        raise AnnotationError("selection direction must be built-in str 'left'")

    counts = {
        "candidate_episodes": selection.candidate_episodes,
        "selected_episodes": selection.selected_episodes,
        "excluded_episodes": selection.excluded_episodes,
    }
    for field, value in counts.items():
        if type(value) is not int:
            raise AnnotationError(f"selection {field} must be a built-in int")
        if value < 0:
            raise AnnotationError(f"selection {field} must be nonnegative")
    if selection.candidate_episodes != selection.selected_episodes + selection.excluded_episodes:
        raise AnnotationError("selection candidate_episodes must equal selected_episodes + excluded_episodes")
    if selection.selected_episodes != len(annotations):
        raise AnnotationError(
            "selection selected_episodes must match the retained annotation count: "
            f"selected={selection.selected_episodes}, retained={len(annotations)}"
        )

    for annotation in annotations:
        try:
            direction = _turn_direction(annotation.subtasks[2])
        except AnnotationError as error:
            raise AnnotationError(
                f"selection retained episode {annotation.episode} is not strictly left: {error}"
            ) from error
        if direction != "left":
            raise AnnotationError(f"selection retained episode {annotation.episode} is {direction}, expected left")

    return {"direction": selection.direction, **counts}


AnnotationVariant = Literal["subtasks", "full_prompt"]


def _validate_nonempty_runs(length: int, starts: tuple[int, int, int]) -> None:
    if length < 4 or not 0 < starts[0] < starts[1] < starts[2] < length:
        raise AnnotationError(
            f"snapped boundaries must create four nonempty subtask runs, got length={length}, starts={starts}"
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
                    raise AnnotationError(f"JSONL row must contain an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError(f"cannot read JSONL file {path}: {error}") from error
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=4)
        stream.write("\n")


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


def _format_safe_relative_path(pattern: str, label: str, **values: object) -> Path:
    try:
        formatted = pattern.format(**values)
    except (IndexError, KeyError, ValueError) as error:
        raise AnnotationError(f"{label} is not a valid path pattern: {error}") from error
    posix_path = PurePosixPath(formatted)
    if (
        not formatted
        or "\\" in formatted
        or posix_path.is_absolute()
        or not posix_path.parts
        or ".." in posix_path.parts
    ):
        raise AnnotationError(f"{label} must format to a safe relative POSIX path, got {formatted!r}")
    return Path(*posix_path.parts)


def _safe_dataset_path(root: Path, relative: Path, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root / relative
    if candidate.is_symlink():
        raise AnnotationError(f"{label} must be a regular non-symlink path: {candidate}")
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise AnnotationError(f"{label} must remain within dataset root {resolved_root}")
    return candidate


def _format_episode_path(info: dict[str, Any], key: str, episode_index: int) -> Path:
    chunks_size = _integer_cell(info.get("chunks_size", 1000), "chunks_size")
    if chunks_size <= 0:
        raise AnnotationError(f"chunks_size must be positive, got {chunks_size}")
    pattern = info[key]
    if not isinstance(pattern, str):
        raise AnnotationError(f"info.json {key} must be a string")
    return _format_safe_relative_path(
        pattern,
        f"info.json {key}",
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
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
    return _format_safe_relative_path(
        pattern,
        "info.json video_path",
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise AnnotationError("info.json features must be an object")
    return sorted(
        key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _task_description_original_key(modality: dict[str, Any]) -> object:
    annotation = modality.get("annotation")
    if not isinstance(annotation, dict):
        return None
    task_description = annotation.get("human.task_description")
    if not isinstance(task_description, dict):
        return None
    return task_description.get("original_key")


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
        raise AnnotationError(f"source episode {source_episode} has inconsistent episode_index values")
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
    *,
    selection: AnnotationSelection | None = None,
) -> ExportResult:
    """Materialize one self-contained LeRobot annotation variant."""

    if variant not in {"subtasks", "full_prompt"}:
        raise AnnotationError(f"unknown annotation variant: {variant!r}")
    if not annotations:
        raise AnnotationError("cannot export an empty annotation set")
    selection_payload = _selection_provenance(selection, annotations) if selection is not None else None

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
    source_modality = source / "meta/modality.json"
    if not source_modality.is_file() or source_modality.is_symlink():
        raise AnnotationError(f"source modality metadata does not exist: {source_modality}")
    modality = _read_json(source_modality)
    if _task_description_original_key(modality) != "task_index":
        raise AnnotationError("source modality.json must map annotation.human.task_description to task_index")

    source_data_paths: dict[int, Path] = {}
    output_data_paths: dict[int, Path] = {}
    source_video_paths: dict[tuple[int, str], Path] = {}
    output_video_paths: dict[tuple[int, str], Path] = {}
    output_targets: set[Path] = set()
    for output_episode, annotation in enumerate(ordered_annotations):
        source_data_paths[annotation.episode] = _safe_dataset_path(
            source,
            _format_episode_path(source_info, "data_path", annotation.episode),
            "source data path",
        )
        output_data_paths[output_episode] = _safe_dataset_path(
            destination,
            _format_episode_path(source_info, "data_path", output_episode),
            "output data path",
        )
        for video_key in video_keys:
            source_video_paths[(annotation.episode, video_key)] = _safe_dataset_path(
                source,
                _format_video_path(source_info, annotation.episode, video_key),
                "source video path",
            )
            output_video_paths[(output_episode, video_key)] = _safe_dataset_path(
                destination,
                _format_video_path(source_info, output_episode, video_key),
                "output video path",
            )
        episode_targets = [output_data_paths[output_episode]] + [
            output_video_paths[(output_episode, video_key)] for video_key in video_keys
        ]
        for target in episode_targets:
            if target in output_targets:
                raise AnnotationError(f"output path pattern produces duplicate target: {target}")
            output_targets.add(target)

    for source_path in source_data_paths.values():
        if not source_path.is_file() or source_path.is_symlink():
            raise AnnotationError(f"source parquet does not exist or is not regular: {source_path}")
    for source_path in source_video_paths.values():
        if not source_path.is_file() or source_path.is_symlink():
            raise AnnotationError(f"source video is missing or not regular: {source_path}")
    missing_stats = sorted(set(annotation_episodes) - source_stats.keys())
    if missing_stats:
        raise AnnotationError(f"source statistics are missing episodes: {missing_stats}")

    if destination.is_symlink():
        raise AnnotationError(f"destination must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise AnnotationError(f"destination must be a directory: {destination}")
        if any(destination.iterdir()):
            raise AnnotationError(f"destination already exists and is not empty: {destination}")
    else:
        destination.mkdir(parents=True)

    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    provenance_episodes: list[dict[str, Any]] = []
    global_index = 0
    copied_videos = 0

    for output_episode, annotation in enumerate(ordered_annotations):
        source_data_path = source_data_paths[annotation.episode]
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
        boundary_frames = snap_boundary_frames(table["timestamp"], annotation.boundaries_s, fps)
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
        output_data_path = output_data_paths[output_episode]
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rewritten, output_data_path)

        for video_key in video_keys:
            source_video = source_video_paths[(annotation.episode, video_key)]
            output_video = output_video_paths[(output_episode, video_key)]
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
            raise AnnotationError(f"source statistics are missing episode {annotation.episode}")
        stats = stats_row.get("stats")
        if not isinstance(stats, dict):
            raise AnnotationError(f"source statistics for episode {annotation.episode} have no stats object")
        stats_row["episode_index"] = output_episode
        stats["episode_index"] = _scalar_stats(np.full(length, output_episode, dtype=np.int64))
        stats["index"] = _scalar_stats(np.arange(global_index, global_index + length, dtype=np.int64))
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
                "boundary_timestamps_s": [float(timestamps[frame]) for frame in boundary_frames],
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
    if selection_payload is not None:
        provenance["selection"] = selection_payload

    _write_json(destination / "meta/info.json", output_info)
    _write_jsonl(destination / "meta/episodes.jsonl", output_episodes)
    _write_jsonl(destination / "meta/tasks.jsonl", task_rows)
    _write_jsonl(destination / "meta/episodes_stats.jsonl", output_stats)
    shutil.copy2(source_modality, destination / "meta/modality.json")
    _write_json(destination / "meta/annotation_provenance.json", provenance)
    return ExportResult(
        output_path=destination,
        variant=variant,
        episodes=len(ordered_annotations),
        frames=global_index,
        tasks=len(task_map),
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_sha256(dataset_path: Path) -> str:
    """Hash every regular dataset file and its POSIX relative path."""

    if not dataset_path.is_dir():
        raise AnnotationError(f"dataset path is not a directory: {dataset_path}")
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(dataset_path.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
        if path.is_symlink():
            raise AnnotationError(f"source dataset contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(dataset_path).as_posix()
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise AnnotationError(f"dataset contains no files: {dataset_path}")
    return digest.hexdigest()


def _release_sha256(value: object, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetValidationError(f"release marker {field} must be a lowercase SHA-256 digest")
    return value


def _release_selection(value: object, context: str) -> dict[str, str | int]:
    fields = {
        "direction",
        "candidate_episodes",
        "selected_episodes",
        "excluded_episodes",
    }
    if type(value) is not dict or value.keys() != fields:
        raise DatasetValidationError(f"{context} selection must contain exactly {sorted(fields)!r}")
    direction = value["direction"]
    if type(direction) is not str or direction != "left":
        raise DatasetValidationError(f"{context} selection direction must be built-in str 'left'")
    counts: dict[str, int] = {}
    for field in ("candidate_episodes", "selected_episodes", "excluded_episodes"):
        count = value[field]
        if type(count) is not int or count < 0:
            raise DatasetValidationError(f"{context} selection {field} must be a nonnegative built-in int")
        counts[field] = count
    if counts["candidate_episodes"] != counts["selected_episodes"] + counts["excluded_episodes"]:
        raise DatasetValidationError(
            f"{context} selection candidate_episodes must equal selected_episodes + excluded_episodes"
        )
    return {"direction": direction, **counts}


def _release_episode_mapping(provenance: dict[str, Any], variant: str) -> list[tuple[int, int, int]]:
    episodes = provenance.get("episodes")
    if type(episodes) is not list:
        raise DatasetValidationError(f"{variant} provenance episode mapping must be a list")
    mapping: list[tuple[int, int, int]] = []
    for row_number, row in enumerate(episodes):
        if type(row) is not dict:
            raise DatasetValidationError(
                f"{variant} provenance episode mapping row {row_number} must be an object"
            )
        values: list[int] = []
        for field in ("source_episode_index", "output_episode_index", "length"):
            item = row.get(field)
            if type(item) is not int or item < 0:
                raise DatasetValidationError(
                    f"{variant} provenance episode mapping {field} must be a nonnegative built-in int"
                )
            values.append(item)
        mapping.append((values[0], values[1], values[2]))
    return mapping


def validate_release_marker(
    marker: Path,
    subtasks: Path,
    full_prompt: Path,
) -> dict[str, object]:
    """Validate that a filtered dataset pair is a complete consumable release."""

    if not _path_lexists(marker):
        raise DatasetValidationError(f"release marker does not exist: {marker}")
    if marker.is_symlink() or not marker.is_file():
        raise DatasetValidationError(f"release marker must be a regular non-symlink file: {marker}")
    try:
        value = _read_json(marker)
    except AnnotationError as error:
        raise DatasetValidationError(f"cannot read release marker: {error}") from error

    marker_fields = {
        "format_version",
        "state",
        "direction",
        "candidate_episodes",
        "selected_episodes",
        "excluded_episodes",
        "source_manifest_sha256",
        "workbook_sha256",
        "datasets",
    }
    if value.keys() != marker_fields:
        raise DatasetValidationError(
            f"release marker fields must match exactly: expected={sorted(marker_fields)!r}"
        )
    format_version = value["format_version"]
    if type(format_version) is not int or format_version != 1:
        raise DatasetValidationError("release marker format_version must be integer 1")
    state = value["state"]
    if type(state) is not str or state != "complete":
        raise DatasetValidationError("release marker state must be built-in str 'complete'")
    marker_selection = _release_selection(
        {
            "direction": value["direction"],
            "candidate_episodes": value["candidate_episodes"],
            "selected_episodes": value["selected_episodes"],
            "excluded_episodes": value["excluded_episodes"],
        },
        "release marker",
    )
    source_manifest_sha256 = _release_sha256(
        value["source_manifest_sha256"],
        "source_manifest_sha256",
    )
    workbook_sha256 = _release_sha256(value["workbook_sha256"], "workbook_sha256")

    datasets = value["datasets"]
    if type(datasets) is not dict or datasets.keys() != {"subtasks", "full_prompt"}:
        raise DatasetValidationError("release marker datasets must contain exactly 'subtasks' and 'full_prompt'")
    dataset_paths = {"subtasks": subtasks, "full_prompt": full_prompt}
    for variant, dataset_path in dataset_paths.items():
        dataset = datasets[variant]
        if type(dataset) is not dict or dataset.keys() != {"path", "manifest_sha256"}:
            raise DatasetValidationError(
                f"release marker dataset {variant} must contain exactly path and manifest_sha256"
            )
        recorded_path = dataset["path"]
        if type(recorded_path) is not str or recorded_path != dataset_path.name:
            raise DatasetValidationError(
                f"release marker dataset {variant} path must equal basename {dataset_path.name!r}"
            )
        recorded_manifest = _release_sha256(
            dataset["manifest_sha256"],
            f"datasets.{variant}.manifest_sha256",
        )
        try:
            actual_manifest = dataset_manifest_sha256(dataset_path)
        except (AnnotationError, OSError) as error:
            raise DatasetValidationError(
                f"release marker cannot compute {variant} dataset manifest: {error}"
            ) from error
        if recorded_manifest != actual_manifest:
            raise DatasetValidationError(f"release marker {variant} dataset manifest does not match")

    mappings: dict[str, list[tuple[int, int, int]]] = {}
    for variant, dataset_path in dataset_paths.items():
        provenance_path = dataset_path / "meta/annotation_provenance.json"
        try:
            provenance = _read_json(provenance_path)
        except AnnotationError as error:
            raise DatasetValidationError(f"release marker cannot read {variant} provenance: {error}") from error
        schema_version = provenance.get("schema_version")
        if type(schema_version) is not int or schema_version != 2:
            raise DatasetValidationError(f"{variant} provenance selection requires schema_version integer 2")
        if provenance.get("variant") != variant:
            raise DatasetValidationError(f"{variant} provenance variant does not match its dataset")
        provenance_source = provenance.get("source")
        if type(provenance_source) is not dict:
            raise DatasetValidationError(f"{variant} provenance source must be an object")
        recorded_source_manifest = _release_sha256(
            provenance_source.get("manifest_sha256"),
            f"{variant} provenance source manifest",
        )
        if recorded_source_manifest != source_manifest_sha256:
            raise DatasetValidationError(f"{variant} provenance source manifest does not match the release marker")
        provenance_annotations = provenance.get("annotations")
        if type(provenance_annotations) is not dict:
            raise DatasetValidationError(f"{variant} provenance annotations must be an object")
        recorded_workbook = _release_sha256(
            provenance_annotations.get("workbook_sha256"),
            f"{variant} provenance workbook_sha256",
        )
        if recorded_workbook != workbook_sha256:
            raise DatasetValidationError(f"{variant} provenance workbook hash does not match the release marker")
        provenance_selection = _release_selection(
            provenance.get("selection"),
            f"{variant} provenance",
        )
        if provenance_selection != marker_selection:
            raise DatasetValidationError(
                f"{variant} provenance selection does not match the release marker selection"
            )
        mappings[variant] = _release_episode_mapping(provenance, variant)

    if mappings["subtasks"] != mappings["full_prompt"]:
        raise DatasetValidationError("cross-variant provenance episode mapping does not match")
    if len(mappings["subtasks"]) != marker_selection["selected_episodes"]:
        raise DatasetValidationError(
            "release marker selection selected_episodes does not match the provenance episode mapping"
        )
    return {"state": state, **marker_selection}


def _expected_task_rows(task_map: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"task_index": task_index, "task": prompt}
        for prompt, task_index in sorted(task_map.items(), key=lambda item: item[1])
    ]


def _expected_stats_row(
    source_row: dict[str, Any],
    output_episode: int,
    global_index: int,
    task_indices: np.ndarray,
) -> dict[str, Any]:
    expected = deepcopy(source_row)
    length = len(task_indices)
    expected["episode_index"] = output_episode
    stats = expected.get("stats")
    if not isinstance(stats, dict):
        raise DatasetValidationError("source episode statistics have no stats object")
    stats["episode_index"] = _scalar_stats(np.full(length, output_episode, dtype=np.int64))
    stats["index"] = _scalar_stats(np.arange(global_index, global_index + length, dtype=np.int64))
    stats["task_index"] = _scalar_stats(task_indices)
    return expected


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

    if variant not in {"subtasks", "full_prompt"}:
        raise DatasetValidationError(f"unknown annotation variant: {variant!r}")
    if not output.is_dir():
        raise DatasetValidationError(f"output dataset is not a directory: {output}")
    try:
        source_info = _read_json(source / "meta/info.json")
        output_info = _read_json(output / "meta/info.json")
        source_episode_rows = _read_jsonl(source / "meta/episodes.jsonl")
        output_episode_rows = _read_jsonl(output / "meta/episodes.jsonl")
        source_stats_rows = _read_jsonl(source / "meta/episodes_stats.jsonl")
        output_stats_rows = _read_jsonl(output / "meta/episodes_stats.jsonl")
        output_task_rows = _read_jsonl(output / "meta/tasks.jsonl")
        provenance = _read_json(output / "meta/annotation_provenance.json")
        source_modality = _read_json(source / "meta/modality.json")
        output_modality = _read_json(output / "meta/modality.json")
    except AnnotationError as error:
        raise DatasetValidationError(str(error)) from error

    if _task_description_original_key(output_modality) != "task_index":
        raise DatasetValidationError(
            "output modality.json must map annotation.human.task_description to task_index"
        )
    if output_modality != source_modality:
        raise DatasetValidationError("output modality.json differs from source modality.json")

    ordered_annotations = sorted(annotations, key=lambda item: item.episode)
    task_map = build_task_map(ordered_annotations, variant)
    expected_task_rows = _expected_task_rows(task_map)
    if output_task_rows != expected_task_rows:
        raise DatasetValidationError(
            f"tasks.jsonl does not match the expected task_index map: "
            f"expected={expected_task_rows!r}, actual={output_task_rows!r}"
        )

    source_episodes = {
        _integer_cell(row.get("episode_index"), "episode_index"): row for row in source_episode_rows
    }
    source_stats = {_integer_cell(row.get("episode_index"), "episode_index"): row for row in source_stats_rows}
    fps = _integer_cell(source_info.get("fps"), "fps")
    video_keys = _video_keys(source_info)
    expected_videos = len(ordered_annotations) * len(video_keys)
    expected_total_frames = 0
    for annotation in ordered_annotations:
        source_episode = source_episodes.get(annotation.episode)
        if source_episode is None:
            raise DatasetValidationError(f"source metadata is missing episode {annotation.episode}")
        expected_total_frames += _integer_cell(
            source_episode.get("length"),
            f"source episode {annotation.episode} length",
        )
    chunks_size = _integer_cell(source_info.get("chunks_size", 1000), "chunks_size")
    if chunks_size <= 0:
        raise DatasetValidationError(f"chunks_size must be positive, got {chunks_size}")
    expected_info = deepcopy(source_info)
    expected_info.update(
        {
            "total_episodes": len(ordered_annotations),
            "total_frames": expected_total_frames,
            "total_tasks": len(task_map),
            "total_videos": expected_videos,
            "total_chunks": max(1, math.ceil(len(ordered_annotations) / chunks_size)),
            "splits": {"train": f"0:{len(ordered_annotations)}"},
        }
    )
    if output_info.keys() != expected_info.keys():
        missing = sorted(expected_info.keys() - output_info.keys())
        unexpected = sorted(output_info.keys() - expected_info.keys())
        raise DatasetValidationError(f"info.json keys mismatch: missing={missing!r}, unexpected={unexpected!r}")
    for name, expected in expected_info.items():
        if output_info.get(name) != expected:
            raise DatasetValidationError(
                f"info.json {name} mismatch: expected={expected!r}, actual={output_info.get(name)!r}"
            )
    if source_info.get("codebase_version") != "v2.1":
        raise DatasetValidationError("source and output codebase_version must be v2.1")
    required_directories = [output / "meta", output / "data"]
    if video_keys:
        required_directories.append(output / "videos")
    for directory in required_directories:
        if not directory.is_dir() or directory.is_symlink():
            raise DatasetValidationError(f"required dataset directory is missing or a symlink: {directory}")
    if len(output_episode_rows) != len(ordered_annotations):
        raise DatasetValidationError("episodes.jsonl row count does not match retained episodes")
    if len(output_stats_rows) != len(ordered_annotations):
        raise DatasetValidationError("episodes_stats.jsonl row count does not match retained episodes")

    current_manifest = source_manifest_sha256 or dataset_manifest_sha256(source)
    expected_schema_version = 2 if selection is not None else 1
    recorded_schema_version = provenance.get("schema_version")
    if type(recorded_schema_version) is not int or recorded_schema_version != expected_schema_version:
        raise DatasetValidationError(f"provenance schema_version must be integer {expected_schema_version}")
    if selection is None:
        if "selection" in provenance:
            raise DatasetValidationError("provenance selection must be absent for a complete export")
    else:
        try:
            expected_selection = _selection_provenance(selection, ordered_annotations)
        except AnnotationError as error:
            raise DatasetValidationError(str(error)) from error
        recorded_selection = provenance.get("selection")
        if type(recorded_selection) is not dict or recorded_selection.keys() != expected_selection.keys():
            raise DatasetValidationError(
                "provenance selection does not match the expected selection: "
                f"expected={expected_selection!r}, actual={recorded_selection!r}"
            )
        for field, expected_value in expected_selection.items():
            recorded_value = recorded_selection[field]
            if type(recorded_value) is not type(expected_value) or recorded_value != expected_value:
                raise DatasetValidationError(
                    "provenance selection does not match the expected selection: "
                    f"field={field!r}, expected={expected_value!r}, actual={recorded_value!r}"
                )
    provenance_source = provenance.get("source")
    if not isinstance(provenance_source, dict):
        raise DatasetValidationError("provenance source must be an object")
    if provenance_source.get("dataset_path") != str(source.resolve()):
        raise DatasetValidationError("provenance source dataset_path does not match the current source")
    if provenance_source.get("manifest_sha256") != current_manifest:
        raise DatasetValidationError("provenance source manifest does not match the current source manifest")
    provenance_annotations = provenance.get("annotations")
    if not isinstance(provenance_annotations, dict):
        raise DatasetValidationError("provenance annotations must be an object")
    recorded_workbook_sha256 = provenance_annotations.get("workbook_sha256")
    if (
        not isinstance(recorded_workbook_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_workbook_sha256) is None
    ):
        raise DatasetValidationError("provenance workbook_sha256 must be a lowercase SHA-256 digest")
    if recorded_workbook_sha256 != workbook_sha256:
        raise DatasetValidationError("provenance workbook_sha256 does not match the current annotation workbook")
    if provenance.get("variant") != variant:
        raise DatasetValidationError("provenance variant does not match output variant")
    if provenance.get("tasks") != expected_task_rows:
        raise DatasetValidationError("provenance task map does not match tasks.jsonl")

    total_frames = 0
    validated_videos = 0
    expected_provenance_episodes: list[dict[str, Any]] = []
    expected_output_episodes: list[dict[str, Any]] = []
    expected_output_stats: list[dict[str, Any]] = []
    for output_episode, annotation in enumerate(ordered_annotations):
        source_episode_row = source_episodes.get(annotation.episode)
        if source_episode_row is None:
            raise DatasetValidationError(f"source metadata is missing episode {annotation.episode}")
        try:
            source_path = _safe_dataset_path(
                source,
                _format_episode_path(source_info, "data_path", annotation.episode),
                "source data path",
            )
            output_path = _safe_dataset_path(
                output,
                _format_episode_path(output_info, "data_path", output_episode),
                "output data path",
            )
        except AnnotationError as error:
            raise DatasetValidationError(str(error)) from error
        if not output_path.is_file() or output_path.is_symlink():
            raise DatasetValidationError(f"output parquet is not a regular non-symlink file: {output_path}")
        source_table = pq.read_table(source_path)
        output_table = pq.read_table(output_path)
        if not source_table.schema.equals(output_table.schema, check_metadata=True):
            raise DatasetValidationError(f"episode {output_episode} Arrow schema differs from its source")
        length = len(source_table)
        if len(output_table) != length:
            raise DatasetValidationError(f"episode {output_episode} output length differs from source")
        source_length = _integer_cell(
            source_episode_row.get("length"),
            f"source episode {annotation.episode} length",
        )
        if length != source_length:
            raise DatasetValidationError(f"source episode {annotation.episode} metadata/parquet length mismatch")

        boundary_frames = snap_boundary_frames(source_table["timestamp"], annotation.boundaries_s, fps)
        expected_task_indices, prompts = _episode_prompt_indices(
            annotation, variant, task_map, length, boundary_frames
        )
        actual_episode_indices = np.asarray(output_table["episode_index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(
            actual_episode_indices,
            np.full(length, output_episode, dtype=np.int64),
        ):
            raise DatasetValidationError(f"episode {output_episode} has noncontiguous episode_index values")
        actual_frames = np.asarray(output_table["frame_index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(actual_frames, np.arange(length, dtype=np.int64)):
            raise DatasetValidationError(f"episode {output_episode} has noncontiguous frame_index values")
        actual_global_indices = np.asarray(output_table["index"].to_pylist(), dtype=np.int64)
        expected_global_indices = np.arange(total_frames, total_frames + length, dtype=np.int64)
        if not np.array_equal(actual_global_indices, expected_global_indices):
            raise DatasetValidationError(f"episode {output_episode} has a noncontiguous global index")
        actual_task_indices = np.asarray(output_table["task_index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(actual_task_indices, expected_task_indices):
            raise DatasetValidationError(f"episode {output_episode} task_index values do not match annotations")
        unresolved = sorted(set(actual_task_indices.tolist()) - set(task_map.values()))
        if unresolved:
            raise DatasetValidationError(
                f"episode {output_episode} has unresolved task_index values: {unresolved}"
            )

        for column in source_table.column_names:
            if column in {"episode_index", "index", "task_index"}:
                continue
            if not source_table[column].equals(output_table[column]):
                raise DatasetValidationError(
                    f"episode {output_episode} payload column {column!r} differs from source"
                )

        expected_output_episodes.append({"episode_index": output_episode, "tasks": prompts, "length": length})
        source_stats_row = source_stats.get(annotation.episode)
        if source_stats_row is None:
            raise DatasetValidationError(f"source statistics are missing episode {annotation.episode}")
        expected_output_stats.append(
            _expected_stats_row(
                source_stats_row,
                output_episode,
                total_frames,
                expected_task_indices,
            )
        )
        timestamps = np.asarray(source_table["timestamp"].to_pylist(), dtype=np.float64)
        expected_provenance_episodes.append(
            {
                "source_episode_index": annotation.episode,
                "output_episode_index": output_episode,
                "length": length,
                "boundaries_s": list(annotation.boundaries_s),
                "boundary_frames": list(boundary_frames),
                "boundary_timestamps_s": [float(timestamps[frame]) for frame in boundary_frames],
                "prompts": prompts,
            }
        )

        for video_key in video_keys:
            try:
                source_video = _safe_dataset_path(
                    source,
                    _format_video_path(source_info, annotation.episode, video_key),
                    "source video path",
                )
                output_video = _safe_dataset_path(
                    output,
                    _format_video_path(output_info, output_episode, video_key),
                    "output video path",
                )
            except AnnotationError as error:
                raise DatasetValidationError(str(error)) from error
            if output_video.is_symlink() or not output_video.is_file():
                raise DatasetValidationError(f"output video is not a regular non-symlink file: {output_video}")
            source_stat = source_video.stat()
            output_stat = output_video.stat()
            if (source_stat.st_dev, source_stat.st_ino) == (
                output_stat.st_dev,
                output_stat.st_ino,
            ):
                raise DatasetValidationError(f"output video is not independent from source: {output_video}")
            if file_sha256(source_video) != file_sha256(output_video):
                raise DatasetValidationError(f"output video hash mismatch: {output_video}")
            validated_videos += 1
        total_frames += length

    if output_episode_rows != expected_output_episodes:
        raise DatasetValidationError("episodes.jsonl does not match expected episode metadata")
    if output_stats_rows != expected_output_stats:
        raise DatasetValidationError("episodes_stats.jsonl does not match rewritten identifier/task statistics")
    if provenance.get("episodes") != expected_provenance_episodes:
        raise DatasetValidationError("provenance episode mapping or boundaries are incorrect")
    if output_info.get("total_frames") != total_frames:
        raise DatasetValidationError(
            f"info.json total_frames mismatch: expected={total_frames}, actual={output_info.get('total_frames')!r}"
        )
    if validated_videos != expected_videos:
        raise DatasetValidationError(
            f"validated video count mismatch: expected={expected_videos}, actual={validated_videos}"
        )
    return {
        "variant": variant,
        "episodes": len(ordered_annotations),
        "frames": total_frames,
        "tasks": len(task_map),
        "videos": validated_videos,
        "expected_runs_per_episode": 4 if variant == "subtasks" else 1,
    }


def _path_is_within(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents


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
    """Stage, validate, and publish both requested dataset variants."""

    raw_outputs = [subtasks_output_path, full_prompt_output_path]
    for output in raw_outputs:
        if _path_lexists(output):
            raise AnnotationError(f"output path already exists: {output}")
    if release_marker_path is not None and _path_lexists(release_marker_path):
        raise AnnotationError(f"release marker already exists: {release_marker_path}")
    if direction_filter == "all":
        if release_marker_path is not None:
            raise AnnotationError("release marker path must be omitted for direction_filter='all'")
    elif direction_filter == "left":
        if release_marker_path is None:
            raise AnnotationError("left direction filter requires a release marker path")
    else:
        raise AnnotationError(f"unknown direction filter: {direction_filter!r}")

    source = dataset_path.resolve()
    workbook = annotations_path.resolve()
    outputs = [output.resolve() for output in raw_outputs]
    marker = release_marker_path.resolve() if release_marker_path is not None else None
    if outputs[0] == outputs[1]:
        raise AnnotationError("subtasks and full-prompt output paths must differ")
    if _path_is_within(outputs[0], outputs[1]) or _path_is_within(outputs[1], outputs[0]):
        raise AnnotationError("subtasks and full-prompt output paths must not overlap")
    for output in outputs:
        if _path_lexists(output):
            raise AnnotationError(f"output path already exists: {output}")
        if _path_is_within(output, source):
            raise AnnotationError(f"output path must be outside the source dataset: {output}")
    if marker is not None:
        if _path_lexists(marker):
            raise AnnotationError(f"release marker already exists: {marker}")
        if any(_path_is_within(marker, output) for output in outputs):
            raise AnnotationError("release marker path must be outside both output datasets")
        if _path_is_within(marker, source):
            raise AnnotationError("release marker path must be outside the source dataset")
    source_episode_rows = _read_jsonl(source / "meta/episodes.jsonl")
    expected_episodes = {_integer_cell(row.get("episode_index"), "episode_index") for row in source_episode_rows}
    try:
        workbook_bytes = workbook.read_bytes()
    except OSError as error:
        raise AnnotationError(f"cannot read XLSX workbook {workbook}: {error}") from error
    complete_annotations = load_annotations_bytes(workbook_bytes, expected_episodes=expected_episodes)
    annotations = complete_annotations
    selection: AnnotationSelection | None = None
    if direction_filter == "left":
        annotations, selection = select_annotations_by_direction(
            complete_annotations,
            direction_filter,
            expected_counts=expected_direction_counts,
        )
        if selection is None:
            raise AnnotationError("left direction filter did not produce a selection")
    before_manifest = dataset_manifest_sha256(source)
    workbook_sha256 = hashlib.sha256(workbook_bytes).hexdigest()

    staging: list[Path] = []
    published_first = False
    try:
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            staging.append(
                Path(
                    tempfile.mkdtemp(
                        prefix=f".{output.name}.staging-",
                        dir=output.parent,
                    )
                )
            )
        selection_kwargs = {} if selection is None else {"selection": selection}
        subtask_result = export_variant(
            source,
            staging[0],
            annotations,
            "subtasks",
            before_manifest,
            workbook_sha256,
            **selection_kwargs,
        )
        full_result = export_variant(
            source,
            staging[1],
            annotations,
            "full_prompt",
            before_manifest,
            workbook_sha256,
            **selection_kwargs,
        )
        validate_variant(
            source,
            staging[0],
            annotations,
            "subtasks",
            source_manifest_sha256=before_manifest,
            workbook_sha256=workbook_sha256,
            **selection_kwargs,
        )
        validate_variant(
            source,
            staging[1],
            annotations,
            "full_prompt",
            source_manifest_sha256=before_manifest,
            workbook_sha256=workbook_sha256,
            **selection_kwargs,
        )
        try:
            current_workbook_sha256 = file_sha256(workbook)
        except OSError as error:
            raise DatasetValidationError(
                f"annotation workbook changed or became unreadable during export: {error}"
            ) from error
        if current_workbook_sha256 != workbook_sha256:
            raise DatasetValidationError("annotation workbook changed during export")
        after_manifest = dataset_manifest_sha256(source)
        if before_manifest != after_manifest:
            raise DatasetValidationError("source manifest changed during export")

        staging[0].replace(outputs[0])
        published_first = True
        try:
            staging[1].replace(outputs[1])
        except Exception:
            outputs[0].replace(staging[0])
            published_first = False
            raise
        results = (
            ExportResult(
                output_path=outputs[0],
                variant=subtask_result.variant,
                episodes=subtask_result.episodes,
                frames=subtask_result.frames,
                tasks=subtask_result.tasks,
            ),
            ExportResult(
                output_path=outputs[1],
                variant=full_result.variant,
                episodes=full_result.episodes,
                frames=full_result.frames,
                tasks=full_result.tasks,
            ),
        )
        if marker is not None:
            if _path_lexists(marker):
                raise AnnotationError(f"release marker already exists: {marker}")
            if selection is None:
                raise AnnotationError("release marker requires a filtered annotation selection")
            marker_value = {
                "format_version": 1,
                "state": "complete",
                "direction": "left",
                "candidate_episodes": selection.candidate_episodes,
                "selected_episodes": selection.selected_episodes,
                "excluded_episodes": selection.excluded_episodes,
                "source_manifest_sha256": before_manifest,
                "workbook_sha256": workbook_sha256,
                "datasets": {
                    "subtasks": {
                        "path": outputs[0].name,
                        "manifest_sha256": dataset_manifest_sha256(outputs[0]),
                    },
                    "full_prompt": {
                        "path": outputs[1].name,
                        "manifest_sha256": dataset_manifest_sha256(outputs[1]),
                    },
                },
            }
            _atomic_write_json(marker, marker_value)
            validate_release_marker(marker, outputs[0], outputs[1])
        return results
    finally:
        for path in staging:
            if path.exists():
                shutil.rmtree(path)
        if published_first and not outputs[1].exists():
            raise DatasetValidationError("first output published without the second and rollback did not complete")


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

        subtasks = tuple(_required_text(values, f"subtask{number}", episode) for number in range(1, 5))
        boundaries = tuple(_required_time(values, f"time{number}", episode) for number in range(1, 4))
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


def load_annotations_bytes(
    data: bytes,
    expected_episodes: set[int] | None = None,
) -> list[EpisodeAnnotation]:
    """Parse one immutable XLSX byte snapshot and return its valid rows."""

    try:
        with ZipFile(BytesIO(data)) as archive:
            worksheet_rows = _read_first_worksheet(archive)
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise AnnotationError(f"cannot read XLSX workbook bytes: {error}") from error
    return _validate_and_select_rows(worksheet_rows, expected_episodes)


def load_annotations(
    path: Path,
    expected_episodes: set[int] | None = None,
) -> list[EpisodeAnnotation]:
    """Load, validate, and return rows whose ``valid`` value is one."""

    try:
        data = path.read_bytes()
    except OSError as error:
        raise AnnotationError(f"cannot read XLSX workbook {path}: {error}") from error
    return load_annotations_bytes(data, expected_episodes)
