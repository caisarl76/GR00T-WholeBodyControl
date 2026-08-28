"""Create or validate XLSX-annotated PnP trash LeRobot datasets.

Example:

    python gear_sonic/scripts/annotate_pnp_trash_dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import tyro

from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import (
    dataset_manifest_sha256,
    export_both,
    load_annotations,
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


def main(config: AnnotatePnpTrashConfig) -> tuple[dict[str, object], dict[str, object]]:
    """Create both datasets or validate the already-published pair."""

    source = config.dataset_path.resolve()
    workbook = config.annotations_path.resolve()
    subtasks = config.subtasks_output_path.resolve()
    full_prompt = config.full_prompt_output_path.resolve()

    if not config.validate_only:
        export_both(source, workbook, subtasks, full_prompt)

    annotations = load_annotations(
        workbook,
        expected_episodes=_source_episode_indices(source),
    )
    source_manifest = dataset_manifest_sha256(source)
    reports = (
        validate_variant(
            source,
            subtasks,
            annotations,
            "subtasks",
            source_manifest_sha256=source_manifest,
        ),
        validate_variant(
            source,
            full_prompt,
            annotations,
            "full_prompt",
            source_manifest_sha256=source_manifest,
        ),
    )
    print(json.dumps({"outputs": reports}, indent=2))
    return reports


if __name__ == "__main__":
    main(tyro.cli(AnnotatePnpTrashConfig))
