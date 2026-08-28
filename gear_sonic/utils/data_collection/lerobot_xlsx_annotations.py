"""Build LeRobot v2.1 language annotations from a simple XLSX workbook."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Literal, Sequence
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np
import pyarrow as pa


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
        prompt_steps = {
            (step, annotation.subtasks[step])
            for annotation in annotations
            for step in range(4)
        }
        prompts = [
            prompt
            for _step, prompt in sorted(
                prompt_steps,
                key=lambda item: (item[0], item[1].encode("utf-8")),
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
