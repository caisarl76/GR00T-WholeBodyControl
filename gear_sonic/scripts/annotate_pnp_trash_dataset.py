"""Create or validate XLSX-annotated PnP trash LeRobot datasets.

Example:

    python gear_sonic/scripts/annotate_pnp_trash_dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Literal

import tyro

from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import (
    AnnotationError,
    DatasetValidationError,
    dataset_manifest_sha256,
    export_both,
    file_sha256,
    load_annotations_bytes,
    select_annotations_by_direction,
    validate_release_marker,
    validate_variant,
)


@dataclass(frozen=True)
class AnnotatePnpTrashConfig:
    """Paths and mode for the PnP trash annotation export."""

    dataset_path: Path = Path("outputs/pnp_trash")
    """Immutable source LeRobot v2.1 dataset."""

    annotations_path: Path = Path("outputs/pnp_trash/pnp_trash.xlsx")
    """XLSX file containing validity, prompts, and time boundaries."""

    subtasks_output_path: Path = Path("outputs/pnp_trash_subtasks")
    """Output containing four timestamped prompts per valid episode."""

    full_prompt_output_path: Path = Path("outputs/pnp_trash_full_prompt")
    """Output containing one episode-wide prompt per valid episode."""

    validate_only: bool = False
    """Validate existing outputs instead of creating them."""

    direction_filter: Literal["all", "left"] = "all"
    """Export all valid episodes or only the left-turn subset."""

    expected_left_episodes: int = 44
    """Expected number of valid left-turn episodes for the left-only release."""

    expected_right_episodes: int = 28
    """Expected number of valid right-turn episodes excluded from the left-only release."""

    release_marker_path: Path = Path("outputs/pnp_trash_left_only.release.json")
    """Completion marker published with the left-only release."""


def _source_episode_indices(dataset_path: Path) -> set[int]:
    episodes_path = dataset_path / "meta/episodes.jsonl"
    episodes: set[int] = set()
    with episodes_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.add(int(row["episode_index"]))
    return episodes


def _normalized_artifact_path(
    path: Path,
    *,
    validation: bool,
    allow_existing_leaf: bool,
) -> Path:
    """Normalize an artifact path while rejecting symlink components in raw order."""

    raw_path = path.absolute()
    components = raw_path.parts[1:]
    current = Path(raw_path.anchor)
    error_type = DatasetValidationError if validation else AnnotationError
    for index, component in enumerate(components):
        if component == ".":
            continue
        if component == "..":
            current = current.parent
            continue
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise error_type(f"cannot inspect artifact path {current}: {error}") from error
        if stat.S_ISLNK(mode) and not (allow_existing_leaf and index == len(components) - 1):
            raise error_type(f"artifact path has a symlink component: {current}")
    return current


def main(config: AnnotatePnpTrashConfig) -> tuple[dict[str, object], dict[str, object]]:
    """Create both datasets or validate the already-published pair."""

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
        _normalized_artifact_path(
            raw_subtasks,
            validation=False,
            allow_existing_leaf=True,
        )
        _normalized_artifact_path(
            raw_full_prompt,
            validation=False,
            allow_existing_leaf=True,
        )
        marker = (
            _normalized_artifact_path(
                marker_path,
                validation=False,
                allow_existing_leaf=True,
            )
            if marker_path is not None
            else None
        )
        results = export_both(
            source,
            workbook,
            raw_subtasks,
            raw_full_prompt,
            direction_filter=config.direction_filter,
            expected_direction_counts=expected_counts,
            release_marker_path=marker_path,
        )
        subtasks = results[0].output_path
        full_prompt = results[1].output_path
    else:
        subtasks = _normalized_artifact_path(
            raw_subtasks,
            validation=True,
            allow_existing_leaf=False,
        )
        full_prompt = _normalized_artifact_path(
            raw_full_prompt,
            validation=True,
            allow_existing_leaf=False,
        )
        marker = (
            _normalized_artifact_path(
                marker_path,
                validation=True,
                allow_existing_leaf=False,
            )
            if marker_path is not None
            else None
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
    reports = (
        validate_variant(
            source,
            subtasks,
            annotations,
            "subtasks",
            source_manifest_sha256=source_manifest,
            workbook_sha256=workbook_sha256,
            selection=selection,
        ),
        validate_variant(
            source,
            full_prompt,
            annotations,
            "full_prompt",
            source_manifest_sha256=source_manifest,
            workbook_sha256=workbook_sha256,
            selection=selection,
        ),
    )
    payload: dict[str, object] = {"outputs": reports}
    if marker is not None:
        payload["release"] = validate_release_marker(marker, subtasks, full_prompt)
    try:
        current_workbook_sha256 = file_sha256(workbook)
    except OSError as error:
        raise DatasetValidationError(f"workbook changed during validation: {error}") from error
    if current_workbook_sha256 != workbook_sha256:
        raise DatasetValidationError("workbook changed during validation")
    if dataset_manifest_sha256(source) != source_manifest:
        raise DatasetValidationError("source changed during validation")
    print(json.dumps(payload, indent=2))
    return reports


if __name__ == "__main__":
    main(tyro.cli(AnnotatePnpTrashConfig))
