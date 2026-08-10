"""Validated contracts for Unitree conversion provenance and canonical data."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_VALID_SCOPES = frozenset({"smoke", "full"})


def validate_revision(value: object, *, field_name: str = "revision") -> str:
    """Return an immutable Git revision or raise a contract error."""
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a 40-character immutable revision containing only hexadecimal characters"
        )
    return value


def validate_sha256(value: object, *, field_name: str = "sha256") -> str:
    """Return a SHA-256 digest or raise a contract error."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 containing only hexadecimal characters")
    return value


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _finite_float64_array(value: object, *, field_name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric array")
    array = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _task_index_array(value: object) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("task_indices must contain nonnegative int64-compatible integers") from error
    if source.dtype.kind not in "iu":
        raise ValueError("task_indices must contain nonnegative int64-compatible integers")
    if source.size:
        if np.any(source < 0) or np.any(source > np.iinfo(np.int64).max):
            raise ValueError("task_indices must contain nonnegative int64-compatible integers")
    return np.array(source, dtype=np.int64, order="C", copy=True)


@dataclass
class CanonicalEpisode:
    """Mutable, array-owning canonical representation of one source episode."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_fps: int
    timestamps: np.ndarray
    task_indices: np.ndarray
    observed_root_wxyz: np.ndarray
    reference_root_wxyz: np.ndarray
    observed_body_q: np.ndarray
    desired_body_q: np.ndarray
    observed_left_hand: np.ndarray
    observed_right_hand: np.ndarray
    desired_left_hand: np.ndarray
    desired_right_hand: np.ndarray

    def __post_init__(self) -> None:
        _nonempty_string(self.source_repo_id, field_name="source_repo_id")
        validate_revision(self.source_revision, field_name="source_revision")
        if (
            isinstance(self.source_episode_id, bool)
            or not isinstance(self.source_episode_id, int)
            or self.source_episode_id < 0
        ):
            raise ValueError("source_episode_id must be a nonnegative integer")
        if isinstance(self.source_fps, bool) or not isinstance(self.source_fps, int) or self.source_fps != 30:
            raise ValueError("source_fps must be exactly integer 30")

        timestamps = _finite_float64_array(self.timestamps, field_name="timestamps")
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must have shape (N,); got {timestamps.shape}")
        row_count = timestamps.shape[0]
        if row_count < 2:
            raise ValueError("canonical episodes require at least two rows")
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")

        task_indices = _task_index_array(self.task_indices)
        if task_indices.shape != (row_count,):
            raise ValueError(f"task_indices must have shape ({row_count},); got {task_indices.shape}")

        expected_shapes = {
            "observed_root_wxyz": (row_count, 4),
            "reference_root_wxyz": (row_count, 4),
            "observed_body_q": (row_count, 29),
            "desired_body_q": (row_count, 29),
            "observed_left_hand": (row_count, 7),
            "observed_right_hand": (row_count, 7),
            "desired_left_hand": (row_count, 7),
            "desired_right_hand": (row_count, 7),
        }
        copied_arrays: dict[str, np.ndarray] = {}
        for field_name, expected_shape in expected_shapes.items():
            array = _finite_float64_array(getattr(self, field_name), field_name=field_name)
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}; got {array.shape}")
            copied_arrays[field_name] = array

        self.timestamps = timestamps
        self.task_indices = task_indices
        for field_name, array in copied_arrays.items():
            setattr(self, field_name, array)


@dataclass(frozen=True)
class ArtifactSpec:
    """Byte-exact identity of an artifact in an immutable repository revision."""

    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.repo_id, field_name="repo_id")
        validate_revision(self.revision)
        _nonempty_string(self.filename, field_name="filename")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("artifact size must be a positive integer")
        validate_sha256(self.sha256)


@dataclass(frozen=True)
class SourceSpec:
    """Immutable dataset identity plus operator-reviewed conversion selection."""

    approved: bool
    repo_id: str
    revision: str
    episode_count: int | None = None
    episodes: tuple[int, ...] = ()
    primary_camera: str | None = None
    camera_map: Mapping[str, str] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")
        _nonempty_string(self.repo_id, field_name="repo_id")
        validate_revision(self.revision)
        if self.label is not None:
            _nonempty_string(self.label, field_name="source label")

        episode_count = self.episode_count
        if episode_count is not None and (
            isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0
        ):
            raise ValueError("episode_count must be a positive integer when provided")

        try:
            episodes = tuple(self.episodes)
        except TypeError as error:
            raise ValueError("episodes must be an iterable of integer episode IDs") from error
        if any(isinstance(item, bool) or not isinstance(item, int) for item in episodes):
            raise ValueError("every episode ID must be an integer")
        if any(item < 0 for item in episodes):
            raise ValueError("every episode ID must be nonnegative")
        if tuple(sorted(set(episodes))) != episodes:
            raise ValueError("episode IDs must be unique and strictly increasing")
        if episode_count is None and episodes:
            raise ValueError("episode_count is required when episode IDs are provided")
        if episode_count is not None and any(item >= episode_count for item in episodes):
            raise ValueError("every episode ID must be smaller than episode_count")
        object.__setattr__(self, "episodes", episodes)

        if self.primary_camera is not None:
            _nonempty_string(self.primary_camera, field_name="primary_camera")
        if not isinstance(self.camera_map, Mapping):
            raise ValueError("camera_map must be a mapping of source keys to target keys")
        camera_map = dict(self.camera_map)
        for source_key, target_key in camera_map.items():
            _nonempty_string(source_key, field_name="camera_map source key")
            _nonempty_string(target_key, field_name="camera_map target key")
        object.__setattr__(self, "camera_map", MappingProxyType(camera_map))

        if self.approved:
            if episode_count is None or not episodes:
                raise ValueError("approved sources require episode_count and at least one episode ID")
            if self.primary_camera is None:
                raise ValueError("approved sources require a primary_camera")
            if self.primary_camera not in camera_map:
                raise ValueError("primary_camera must be present in camera_map")


@dataclass(frozen=True)
class CollectionMembership:
    """Ordered dataset membership observed in one Hugging Face collection."""

    slug: str
    repo_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.slug, field_name="collection slug")
        if isinstance(self.repo_ids, (str, bytes)):
            raise ValueError("collection repo_ids must be a non-string iterable")
        try:
            repo_ids = tuple(self.repo_ids)
        except TypeError as error:
            raise ValueError("collection repo_ids must be an iterable") from error
        for repo_id in repo_ids:
            _nonempty_string(repo_id, field_name="collection repository ID")
        if len(set(repo_ids)) != len(repo_ids):
            raise ValueError("collection membership contains duplicate repositories")
        object.__setattr__(self, "repo_ids", repo_ids)


@dataclass(frozen=True)
class SourceLock:
    """Revision-locked artifact and ordered dataset provenance."""

    version: int
    scope: str
    semantic_repo_commit: str
    encoder: ArtifactSpec
    observation_config: ArtifactSpec
    sources: tuple[SourceSpec, ...]
    collections: tuple[CollectionMembership, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("source lock version must be an integer")
        if self.version != 1:
            raise ValueError(f"unsupported source lock version: {self.version!r}")
        if not isinstance(self.scope, str) or self.scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}, got {self.scope!r}")
        validate_revision(
            self.semantic_repo_commit,
            field_name="semantic_repo_commit",
        )
        if not isinstance(self.encoder, ArtifactSpec):
            raise ValueError("encoder must be an ArtifactSpec")
        if not isinstance(self.observation_config, ArtifactSpec):
            raise ValueError("observation_config must be an ArtifactSpec")
        for field_name in ("repo_id", "revision"):
            if getattr(self.encoder, field_name) != getattr(self.observation_config, field_name):
                raise ValueError(f"encoder and observation_config must share the same {field_name}")

        try:
            sources = tuple(self.sources)
            collections = tuple(self.collections)
        except TypeError as error:
            raise ValueError("sources and collections must be iterable") from error
        if not all(isinstance(source, SourceSpec) for source in sources):
            raise ValueError("sources must contain only SourceSpec entries")
        if not all(isinstance(membership, CollectionMembership) for membership in collections):
            raise ValueError("collections must contain only CollectionMembership entries")
        repo_ids = tuple(source.repo_id for source in sources)
        if len(set(repo_ids)) != len(repo_ids):
            raise ValueError("source lock contains duplicate dataset repositories")
        labels = tuple(source.label for source in sources if source.label is not None)
        if len(set(labels)) != len(labels):
            raise ValueError("source lock contains duplicate source labels")

        if self.scope == "full":
            if not sources:
                raise ValueError("full source locks require at least one source")
            if not collections:
                raise ValueError("full source locks require at least one collection membership")

        if collections:
            collection_slugs = tuple(membership.slug for membership in collections)
            if len(set(collection_slugs)) != len(collection_slugs):
                raise ValueError("source lock contains duplicate collection slugs")
            recorded_repo_ids = tuple(repo_id for membership in collections for repo_id in membership.repo_ids)
            if recorded_repo_ids != repo_ids:
                raise ValueError("ordered collection membership must exactly match ordered sources")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "collections", collections)

        if self.scope == "smoke":
            for label in ("dex3", "inspire"):
                if label not in labels:
                    raise ValueError(f"smoke source lock is missing {label!r}")

    def source(self, label: str) -> SourceSpec:
        """Return a named smoke source."""
        for source in self.sources:
            if source.label == label:
                return source
        raise AttributeError(f"source lock has no source named {label!r}")

    @property
    def dex3(self) -> SourceSpec:
        return self.source("dex3")

    @property
    def inspire(self) -> SourceSpec:
        return self.source("inspire")
