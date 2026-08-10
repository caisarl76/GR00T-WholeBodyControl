"""Strict adapter for revision-pinned Unitree G1 Dex3 LeRobot episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
from pathlib import Path, PurePosixPath
import string
from typing import Any

import numpy as np

from gear_sonic.data.unitree_conversion.contracts import CanonicalEpisode, SourceSpec
from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES, NOMINAL_G1_MUJOCO

DEX3_ARM_NAMES = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)

DEX3_HAND_NAMES = (
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
)

DEX3_FEATURE_NAMES = DEX3_ARM_NAMES + DEX3_HAND_NAMES

_ARM_TARGET_BY_SOURCE = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
}

_ROW_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "task_index",
    "frame_index",
    "episode_index",
)
_MISSING = object()


def _finite_float_array(value: object, *, field_name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric array")
    array = np.asarray(source, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _exact_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must equal the exact Dex3 feature names")
    try:
        names = tuple(feature_names)
    except TypeError as error:
        raise ValueError("feature_names must equal the exact Dex3 feature names") from error
    if names != DEX3_FEATURE_NAMES:
        raise ValueError("feature_names must equal the exact Dex3 feature names in documented source order")
    return names


def adapt_dex3_arrays(
    *,
    source_repo_id: str,
    source_revision: str,
    source_episode_id: int,
    observed: object,
    desired: object,
    feature_names: Sequence[str],
    timestamps: object,
    task_indices: object,
) -> CanonicalEpisode:
    """Map documented 28D Dex3 state/action arrays into one canonical episode."""
    names = _exact_feature_names(feature_names)
    observed_array = _finite_float_array(observed, field_name="observed")
    desired_array = _finite_float_array(desired, field_name="desired")
    if observed_array.ndim != 2 or observed_array.shape[1] != 28:
        raise ValueError(f"observed must have shape (N, 28); got {observed_array.shape}")
    if desired_array.ndim != 2 or desired_array.shape[1] != 28:
        raise ValueError(f"desired must have shape (N, 28); got {desired_array.shape}")
    if desired_array.shape[0] != observed_array.shape[0]:
        raise ValueError("observed and desired must have the same number of rows")

    row_count = observed_array.shape[0]
    if row_count < 2:
        raise ValueError("Dex3 episodes require at least two rows")
    timestamps_array = _finite_float_array(timestamps, field_name="timestamps")
    if timestamps_array.shape != (row_count,):
        raise ValueError(f"timestamps must have shape ({row_count},); got {timestamps_array.shape}")
    try:
        task_indices_array = np.asarray(task_indices)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task_indices must have shape ({row_count},)") from error
    if task_indices_array.shape != (row_count,):
        raise ValueError(f"task_indices must have shape ({row_count},); got {task_indices_array.shape}")

    observed_body_q = np.tile(NOMINAL_G1_MUJOCO, (row_count, 1))
    desired_body_q = np.tile(NOMINAL_G1_MUJOCO, (row_count, 1))
    for source_name, target_name in _ARM_TARGET_BY_SOURCE.items():
        source_index = names.index(source_name)
        target_index = G1_MUJOCO_NAMES.index(target_name)
        observed_body_q[:, target_index] = observed_array[:, source_index]
        desired_body_q[:, target_index] = desired_array[:, source_index]

    left_hand_indices = tuple(names.index(name) for name in DEX3_HAND_NAMES[:7])
    right_hand_indices = tuple(names.index(name) for name in DEX3_HAND_NAMES[7:])
    identity_roots = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (row_count, 1))
    return CanonicalEpisode(
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_episode_id=source_episode_id,
        source_fps=30,
        timestamps=timestamps_array,
        task_indices=task_indices_array,
        observed_root_wxyz=identity_roots,
        reference_root_wxyz=identity_roots,
        observed_body_q=observed_body_q,
        desired_body_q=desired_body_q,
        observed_left_hand=observed_array[:, left_hand_indices],
        observed_right_hand=observed_array[:, right_hand_indices],
        desired_left_hand=desired_array[:, left_hand_indices],
        desired_right_hand=desired_array[:, right_hand_indices],
    )


def _to_numpy(value: object, *, field_name: str) -> np.ndarray:
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        return np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} cannot be converted to a NumPy array") from error


def _integer_scalar(value: object, *, field_name: str) -> int:
    array = _to_numpy(value, field_name=field_name)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{field_name} must be an integer scalar")
    integer = int(array)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise ValueError(f"{field_name} must be an int64-compatible integer scalar")
    return integer


def _numeric_scalar(value: object, *, field_name: str) -> float:
    array = _to_numpy(value, field_name=field_name)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric scalar")
    scalar = float(array)
    if not math.isfinite(scalar):
        raise ValueError(f"{field_name} must be finite")
    return scalar


def _feature_names_from_meta(features: object, key: str) -> tuple[str, ...]:
    if not isinstance(features, Mapping) or key not in features or not isinstance(features[key], Mapping):
        raise ValueError(
            "dataset meta.features must be a mapping containing mapping entries for observation.state and action"
        )
    metadata = features[key]
    if "names" not in metadata:
        raise ValueError(f"dataset meta.features entry {key} must contain names")
    names = metadata["names"]
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names")
    try:
        names_tuple = tuple(names)
    except TypeError as error:
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names") from error
    if len(names_tuple) == 1 and isinstance(names_tuple[0], (list, tuple)):
        names_tuple = tuple(names_tuple[0])
    if names_tuple != DEX3_FEATURE_NAMES or not all(isinstance(name, str) for name in names_tuple):
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names")
    return names_tuple


def _default_lerobot_dataset_factory() -> Callable[..., Any]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _default_snapshot_downloader() -> Callable[..., str]:
    from huggingface_hub import snapshot_download

    return snapshot_download


def _default_lerobot_cache_base() -> Path:
    """Mirror LeRobot's Hugging Face cache-base convention without importing LeRobot."""
    lerobot_home = os.getenv("HF_LEROBOT_HOME")
    if lerobot_home is not None:
        return Path(lerobot_home).expanduser()

    default_home = Path.home() / ".cache"
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", str(default_home))
    hf_home = os.getenv("HF_HOME", str(Path(xdg_cache_home) / "huggingface"))
    return Path(os.path.expandvars(hf_home)).expanduser() / "lerobot"


def _revision_scoped_root(cache_base: str | Path, repo_id: str, revision: str) -> Path:
    """Return a cache-contained path unique to one repository identity and revision."""
    base = Path(cache_base).expanduser().resolve()
    repository_identity = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
    return base / f"repo-{repository_identity}" / f"revision-{revision}"


def _verify_snapshot_location(downloaded: object, scoped_root: Path) -> None:
    try:
        downloaded_root = Path(downloaded).expanduser().resolve()
    except TypeError as error:
        raise ValueError("snapshot downloader must return the revision-scoped root") from error
    if downloaded_root != scoped_root.resolve():
        raise ValueError("snapshot downloader must return the revision-scoped root")


def _reject_duplicate_metadata_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _regular_file_under_root(scoped_root: Path, relative_path: PurePosixPath) -> Path:
    candidate = scoped_root.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(scoped_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as error:
        message = f"materialized file is missing or outside revision-scoped root: {relative_path}"
        raise ValueError(message) from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"materialized file is not a regular file: {relative_path}")
    return candidate


def _read_prefetch_metadata(scoped_root: Path) -> tuple[str, str, tuple[str, ...]]:
    info_path = _regular_file_under_root(scoped_root, PurePosixPath("meta/info.json"))
    try:
        with info_path.open(encoding="utf-8") as stream:
            info = json.load(stream, object_pairs_hook=_reject_duplicate_metadata_keys)
    except json.JSONDecodeError as error:
        raise ValueError("metadata info.json must contain valid JSON") from error
    if not isinstance(info, dict):
        raise ValueError("metadata info.json must contain a JSON object")

    for field_name in ("codebase_version", "chunks_size", "data_path", "video_path", "features"):
        if field_name not in info:
            raise ValueError(f"metadata is missing required field {field_name}")

    if info["codebase_version"] != "v3.0":
        raise ValueError("metadata codebase_version must be exactly 'v3.0'")
    chunks_size = info["chunks_size"]
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("metadata chunks_size must be a positive integer")
    data_path = info["data_path"]
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("metadata data_path must be a nonempty string")
    video_path = info["video_path"]
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("metadata video_path must be a nonempty string")

    features = info["features"]
    if not isinstance(features, dict) or not features:
        raise ValueError("metadata features must be a nonempty mapping")
    video_keys: list[str] = []
    for feature_key, feature in features.items():
        if not isinstance(feature_key, str) or not feature_key:
            raise ValueError("metadata features keys must be nonempty strings")
        if not isinstance(feature, dict):
            raise ValueError(f"metadata features entry {feature_key!r} must be a mapping")
        dtype = feature.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"metadata features entry {feature_key!r} dtype must be a nonempty string")
        if dtype == "video":
            video_keys.append(feature_key)
    return data_path, video_path, tuple(video_keys)


def _episode_metadata_files(scoped_root: Path) -> tuple[Path, ...]:
    episodes_root = scoped_root / "meta/episodes"
    try:
        resolved_root = episodes_root.resolve(strict=True)
        resolved_root.relative_to(scoped_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("episode metadata must contain at least one parquet file") from error
    if episodes_root.is_symlink() or not episodes_root.is_dir():
        raise ValueError("episode metadata directory must be a safe regular directory")

    try:
        candidates = sorted(episodes_root.rglob("*.parquet"))
    except OSError as error:
        raise ValueError("episode metadata parquet files must be readable") from error
    if not candidates:
        raise ValueError("episode metadata must contain at least one parquet file")

    files: list[Path] = []
    for candidate in candidates:
        relative_path = PurePosixPath(candidate.relative_to(scoped_root).as_posix())
        files.append(_regular_file_under_root(scoped_root, relative_path))
    return tuple(files)


def _episode_metadata_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"episode metadata {field_name} must be a nonnegative integer")
    return value


def _selected_episode_row(
    scoped_root: Path,
    *,
    episode_id: int,
    video_keys: tuple[str, ...],
) -> dict[str, int]:
    import pyarrow.parquet as pq

    required_columns = (
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        *(f"videos/{video_key}/{suffix}" for video_key in video_keys for suffix in ("chunk_index", "file_index")),
    )
    seen_episode_ids: set[int] = set()
    selected: dict[str, int] | None = None
    for metadata_path in _episode_metadata_files(scoped_root):
        try:
            table = pq.read_table(metadata_path)
        except Exception as error:
            raise ValueError(f"episode metadata parquet must be readable: {metadata_path}") from error
        missing_columns = tuple(column for column in required_columns if column not in table.column_names)
        if missing_columns:
            raise ValueError(f"episode metadata parquet is missing required columns: {missing_columns}")

        for row_index in range(table.num_rows):
            row = {
                column: _episode_metadata_integer(
                    table[column][row_index].as_py(),
                    field_name=column,
                )
                for column in required_columns
            }
            source_episode_id = row["episode_index"]
            if source_episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_index in episode metadata: {source_episode_id}")
            seen_episode_ids.add(source_episode_id)
            if source_episode_id == episode_id:
                selected = row

    if selected is None:
        raise ValueError(f"no episode metadata row for selected episode {episode_id}")
    return selected


def _render_metadata_path(
    template: str,
    *,
    field_name: str,
    required_fields: frozenset[str],
    values: Mapping[str, object],
) -> PurePosixPath:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"metadata {field_name} template is malformed") from error

    fields: list[str] = []
    for _, template_field, format_spec, conversion in parsed:
        if template_field is None:
            continue
        if template_field not in required_fields or conversion is not None or "{" in format_spec:
            raise ValueError(f"metadata {field_name} template contains unsupported fields")
        fields.append(template_field)
    if len(fields) != len(set(fields)) or set(fields) != required_fields:
        raise ValueError(f"metadata {field_name} template must contain each required field exactly once")

    try:
        rendered = template.format(**values)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"metadata {field_name} template cannot be rendered") from error
    relative_path = PurePosixPath(rendered)
    if (
        not rendered
        or "\\" in rendered
        or relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise ValueError(f"metadata {field_name} template must render a safe relative path")
    return relative_path


def _selected_episode_paths(scoped_root: Path, episode_id: int) -> tuple[PurePosixPath, ...]:
    data_template, video_template, video_keys = _read_prefetch_metadata(scoped_root)
    episode_row = _selected_episode_row(
        scoped_root,
        episode_id=episode_id,
        video_keys=video_keys,
    )
    data_path = _render_metadata_path(
        data_template,
        field_name="data_path",
        required_fields=frozenset({"chunk_index", "file_index"}),
        values={
            "chunk_index": episode_row["data/chunk_index"],
            "file_index": episode_row["data/file_index"],
        },
    )
    if data_path.suffix != ".parquet":
        raise ValueError("metadata data_path template must render a parquet path")

    paths = [data_path]
    for video_key in video_keys:
        paths.append(
            _render_metadata_path(
                video_template,
                field_name="video_path",
                required_fields=frozenset({"chunk_index", "file_index", "video_key"}),
                values={
                    "chunk_index": episode_row[f"videos/{video_key}/chunk_index"],
                    "file_index": episode_row[f"videos/{video_key}/file_index"],
                    "video_key": video_key,
                },
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError("metadata templates render duplicate selected episode paths")
    return tuple(paths)


def _prefetch_exact_revision(
    *,
    source_spec: SourceSpec,
    episode_id: int,
    scoped_root: Path,
    snapshot_downloader: Callable[..., object],
) -> None:
    common_kwargs = {
        "repo_id": source_spec.repo_id,
        "repo_type": "dataset",
        "revision": source_spec.revision,
        "local_dir": scoped_root,
    }
    downloaded = snapshot_downloader(**common_kwargs, allow_patterns=["meta/**"])
    _verify_snapshot_location(downloaded, scoped_root)

    selected_paths = _selected_episode_paths(scoped_root, episode_id)
    selected_patterns = [str(path) for path in selected_paths]
    downloaded = snapshot_downloader(**common_kwargs, allow_patterns=selected_patterns)
    _verify_snapshot_location(downloaded, scoped_root)
    for relative_path in selected_paths:
        _regular_file_under_root(scoped_root, relative_path)


def load_dex3_episode(
    source_spec: SourceSpec,
    episode_id: int,
    root: str | Path | None = None,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., object] | None = None,
) -> CanonicalEpisode:
    """Load one selected episode from an approved, revision-pinned Dex3 source."""
    if not isinstance(source_spec, SourceSpec) or not source_spec.approved:
        raise ValueError("source_spec must be an approved SourceSpec")
    if source_spec.episode_count is None:
        raise ValueError("source_spec must have a pinned episode_count")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        raise ValueError("episode_id must be a nonnegative integer")
    if episode_id >= source_spec.episode_count:
        raise ValueError("episode_id must be smaller than pinned episode_count")
    if episode_id not in source_spec.episodes:
        raise ValueError("episode_id must be selected in source_spec.episodes")

    cache_base = _default_lerobot_cache_base() if root is None else root
    scoped_root = _revision_scoped_root(cache_base, source_spec.repo_id, source_spec.revision)
    downloader = _default_snapshot_downloader() if snapshot_downloader is None else snapshot_downloader
    _prefetch_exact_revision(
        source_spec=source_spec,
        episode_id=episode_id,
        scoped_root=scoped_root,
        snapshot_downloader=downloader,
    )
    factory = _default_lerobot_dataset_factory() if dataset_factory is None else dataset_factory
    dataset = factory(
        repo_id=source_spec.repo_id,
        root=scoped_root,
        episodes=[episode_id],
        revision=source_spec.revision,
        download_videos=True,
        video_backend="pyav",
    )
    meta = getattr(dataset, "meta", None)
    if getattr(dataset, "revision", None) != source_spec.revision:
        raise ValueError("dataset.revision must equal pinned source revision")
    if getattr(meta, "revision", None) != source_spec.revision:
        raise ValueError("dataset.meta.revision must equal pinned source revision")
    total_episodes = getattr(meta, "total_episodes", None)
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, numbers.Integral)
        or total_episodes != source_spec.episode_count
    ):
        raise ValueError("dataset metadata total_episodes must exactly match pinned episode_count")
    fps = getattr(meta, "fps", None)
    if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not math.isfinite(float(fps)) or fps != 30:
        raise ValueError("dataset metadata fps must be numeric 30")

    features = getattr(meta, "features", None)
    observation_names = _feature_names_from_meta(features, "observation.state")
    action_names = _feature_names_from_meta(features, "action")
    if observation_names != action_names:
        raise ValueError("observation.state and action feature names must match exactly")

    observed_rows: list[np.ndarray] = []
    desired_rows: list[np.ndarray] = []
    timestamps: list[float] = []
    task_indices: list[int] = []
    raw_rows = getattr(dataset, "hf_dataset", _MISSING)
    if raw_rows is _MISSING or isinstance(raw_rows, (str, bytes, Mapping)):
        raise ValueError("dataset.hf_dataset must be a non-mapping iterable of raw rows")
    try:
        row_iterator = iter(raw_rows)
    except TypeError as error:
        raise ValueError("dataset.hf_dataset must be a non-mapping iterable of raw rows") from error

    for row_number, row in enumerate(row_iterator):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_number} must be a mapping")
        for key in _ROW_KEYS:
            if key not in row:
                raise ValueError(f"row {row_number} is missing required key {key}")

        source_episode_id = _integer_scalar(
            row["episode_index"],
            field_name=f"row {row_number} episode_index",
        )
        if source_episode_id != episode_id:
            raise ValueError(
                f"source episode_index {source_episode_id} does not match requested episode {episode_id}"
            )
        frame_index = _integer_scalar(
            row["frame_index"],
            field_name=f"row {row_number} frame_index",
        )
        if frame_index != row_number:
            raise ValueError(f"expected frame_index {row_number}, got {frame_index}")

        observed = _to_numpy(row["observation.state"], field_name=f"row {row_number} observation.state")
        desired = _to_numpy(row["action"], field_name=f"row {row_number} action")
        if observed.shape != (28,):
            raise ValueError(f"row {row_number} observation.state must have shape (28,); got {observed.shape}")
        if desired.shape != (28,):
            raise ValueError(f"row {row_number} action must have shape (28,); got {desired.shape}")
        observed_rows.append(observed)
        desired_rows.append(desired)
        timestamps.append(_numeric_scalar(row["timestamp"], field_name=f"row {row_number} timestamp"))
        task_indices.append(_integer_scalar(row["task_index"], field_name=f"row {row_number} task_index"))

    if not observed_rows:
        raise ValueError(f"no rows for requested episode {episode_id}")
    return adapt_dex3_arrays(
        source_repo_id=source_spec.repo_id,
        source_revision=source_spec.revision,
        source_episode_id=episode_id,
        observed=np.stack(observed_rows),
        desired=np.stack(desired_rows),
        feature_names=observation_names,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        task_indices=np.asarray(task_indices),
    )
