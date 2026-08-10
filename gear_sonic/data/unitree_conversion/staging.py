"""Immutable per-episode staging and deterministic Unitree dataset merge."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Protocol
from urllib.parse import quote

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear_sonic.data.unitree_conversion.contracts import validate_revision, validate_sha256
from gear_sonic.data.unitree_conversion.validation import (
    TARGET_FRAME_SCHEMA,
    TARGET_VIDEO_KEYS,
    ValidationReport,
    probe_video_50fps,
    validate_stage_payload,
    validate_target_rows,
)

_STAGE_SCHEMA_VERSION = "unitree-sonic-stage-v1"
_MERGE_SCHEMA_VERSION = "unitree-sonic-merge-v1"
_VIDEO_KEY_PATTERN = re.compile(r"^observation\.images\.[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPO_PART_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_source_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("source file paths must be safe canonical POSIX-relative paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("source file paths must be safe canonical POSIX-relative paths")
    return value


def _validated_sha(value: object, *, field_name: str) -> str:
    digest = validate_sha256(value, field_name=field_name)
    if digest != digest.lower():
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


@dataclass(frozen=True)
class StageIdentity:
    """All immutable inputs that give one episode stage its identity."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_file_sha256: Mapping[str, str]
    conversion_config_sha256: str
    converter_version: str
    encoder_sha256: str
    encoder_config_sha256: str
    source_lock_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_repo_id, str) or not self.source_repo_id.strip():
            raise ValueError("source_repo_id must be a nonempty repository identifier")
        parts = self.source_repo_id.split("/")
        if len(parts) < 2 or any(_REPO_PART_PATTERN.fullmatch(part) is None for part in parts):
            raise ValueError("source_repo_id must be a safe slash-separated repository identifier")
        revision = validate_revision(self.source_revision, field_name="source_revision")
        if revision != revision.lower():
            raise ValueError("source_revision must use lowercase hexadecimal")
        if type(self.source_episode_id) is not int or self.source_episode_id < 0:
            raise ValueError("source_episode_id must be a nonnegative integer")
        if not isinstance(self.source_file_sha256, Mapping) or not self.source_file_sha256:
            raise ValueError("source_file_sha256 must contain exact source file hashes")
        files: dict[str, str] = {}
        for source_path, digest in self.source_file_sha256.items():
            path = _safe_source_path(source_path)
            files[path] = _validated_sha(digest, field_name=f"source file {path} SHA-256")
        if not isinstance(self.converter_version, str) or not self.converter_version.strip():
            raise ValueError("converter_version must be a nonempty string")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in self.converter_version):
            raise ValueError("converter_version must not contain control characters")
        object.__setattr__(self, "source_file_sha256", MappingProxyType(dict(sorted(files.items()))))
        for field_name in (
            "conversion_config_sha256",
            "encoder_sha256",
            "encoder_config_sha256",
            "source_lock_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_sha(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_repo_id": self.source_repo_id,
            "source_revision": self.source_revision,
            "source_episode_id": self.source_episode_id,
            "source_file_sha256": dict(self.source_file_sha256),
            "conversion_config_sha256": self.conversion_config_sha256,
            "converter_version": self.converter_version,
            "encoder_sha256": self.encoder_sha256,
            "encoder_config_sha256": self.encoder_config_sha256,
            "source_lock_sha256": self.source_lock_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> StageIdentity:
        if not isinstance(value, dict):
            raise ValueError("stage identity must be an object")
        expected = {
            "source_repo_id",
            "source_revision",
            "source_episode_id",
            "source_file_sha256",
            "conversion_config_sha256",
            "converter_version",
            "encoder_sha256",
            "encoder_config_sha256",
            "source_lock_sha256",
        }
        if set(value) != expected:
            raise ValueError("stage identity contains missing or unexpected fields")
        return cls(**value)

    @property
    def conversion_key(self) -> tuple[object, ...]:
        return (
            self.source_repo_id,
            self.source_revision,
            self.conversion_config_sha256,
            self.converter_version,
            self.encoder_sha256,
            self.encoder_config_sha256,
            self.source_lock_sha256,
        )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    data = bytes(np.ascontiguousarray(value).tobytes(order="C"))
    return np.frombuffer(data, dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class StagePayload:
    """Detached typed target rows and explicit target-camera video sources."""

    rows: Sequence[Mapping[str, object]]
    videos: Mapping[str, Path | bytes]

    def __post_init__(self) -> None:
        if isinstance(self.rows, (str, bytes)):
            raise ValueError("stage rows must be a sequence of target mappings")
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise ValueError("stage rows must be a sequence of target mappings") from error
        validate_target_rows(rows)
        owned_rows: list[Mapping[str, object]] = []
        for row in rows:
            owned: dict[str, object] = {}
            for key in TARGET_FRAME_SCHEMA:
                value = row[key]
                owned[key] = _immutable_array(value) if isinstance(value, np.ndarray) else value
            owned_rows.append(MappingProxyType(owned))

        if not isinstance(self.videos, Mapping) or not self.videos:
            raise ValueError("stage videos must be a nonempty mapping")
        videos: dict[str, Path | bytes] = {}
        for key, source in self.videos.items():
            if not isinstance(key, str) or _VIDEO_KEY_PATTERN.fullmatch(key) is None:
                raise ValueError("video target keys must be safe observation.images.* feature names")
            if key not in TARGET_VIDEO_KEYS:
                raise ValueError(f"video target key {key!r} is unsupported")
            if isinstance(source, bytes):
                if not source:
                    raise ValueError(f"video target {key} bytes must be nonempty")
                videos[key] = bytes(source)
            elif isinstance(source, Path):
                if source.is_symlink():
                    raise ValueError(f"video target {key} must be a regular source file, not a symlink")
                try:
                    path = source.absolute()
                    metadata = path.stat()
                except OSError as error:
                    raise ValueError(f"video target {key} must be a readable regular file") from error
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"video target {key} must be a readable regular file")
                videos[key] = path
            else:
                raise ValueError(f"video target {key} must be supplied as pathlib.Path or bytes")
        if "observation.images.ego_view" not in videos:
            raise ValueError("stage videos require observation.images.ego_view")
        object.__setattr__(self, "rows", tuple(owned_rows))
        object.__setattr__(self, "videos", MappingProxyType(dict(sorted(videos.items()))))


@dataclass(frozen=True)
class StageResult:
    path: Path
    identity: StageIdentity
    manifest_digest: str
    reused: bool = False


@dataclass(frozen=True)
class StagedVideoArtifact:
    """A checksum-bound target video inside a revalidated immutable stage."""

    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class MergeEpisode:
    path: Path
    identity: StageIdentity
    rows: tuple[Mapping[str, object], ...]
    videos: Mapping[str, StagedVideoArtifact]
    stage_manifest_digest: str
    stage_checksums_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "videos", MappingProxyType(dict(sorted(self.videos.items()))))


@dataclass(frozen=True)
class MergeVerification:
    """Observed final-dataset facts returned by a merge backend."""

    source_episode_ids: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    global_indices: tuple[int, ...]
    episode_indices: tuple[int, ...]
    task_to_index: Mapping[str, int]
    frame_task_indices: tuple[int, ...]
    frame_timestamps: tuple[np.float32, ...]
    video_frame_counts: Mapping[int, Mapping[str, int]]
    stats_source_episode_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_to_index", MappingProxyType(dict(self.task_to_index)))
        object.__setattr__(
            self,
            "video_frame_counts",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(sorted(value.items())))
                    for key, value in sorted(self.video_frame_counts.items())
                }
            ),
        )


class MergeBackend(Protocol):
    def __call__(
        self,
        output: Path,
        episodes: tuple[MergeEpisode, ...],
        task_catalog: tuple[str, ...],
    ) -> MergeVerification: ...


def stage_path(output_root: str | Path, identity: StageIdentity) -> Path:
    """Return the deterministic immutable path for an episode identity."""
    if not isinstance(identity, StageIdentity):
        raise TypeError("identity must be a StageIdentity")
    repo_safe = quote(identity.source_repo_id, safe="-._")
    return (
        Path(output_root).absolute()
        / ".staging"
        / repo_safe
        / identity.source_revision
        / f"episode-{identity.source_episode_id}"
        / identity.conversion_config_sha256
    )


def _secure_read(path: Path) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path.name}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError(f"artifact changed while opening: {path.name}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read()
        finally:
            os.close(descriptor)
        final = path.lstat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ):
            raise ValueError(f"artifact changed while reading: {path.name}")
        return data
    except OSError as error:
        raise ValueError(f"cannot securely read artifact: {path}") from error


def _read_video_source(source: Path | bytes, *, key: str) -> bytes:
    if isinstance(source, bytes):
        return source
    try:
        return _secure_read(source)
    except ValueError as error:
        raise ValueError(f"video target {key} source changed or became unreadable") from error


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    directories = [path]
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in file_names:
            candidate = root_path / name
            metadata = candidate.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("publication tree must contain only regular files and directories")
            descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in directory_names:
            candidate = root_path / name
            if not stat.S_ISDIR(candidate.lstat().st_mode):
                raise ValueError("publication tree must contain only regular files and directories")
            directories.append(candidate)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _ensure_safe_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if os.path.lexists(cursor) and not stat.S_ISDIR(cursor.lstat().st_mode):
        raise ValueError(f"path component is not a real directory: {cursor}")
    for component in reversed(missing):
        component.mkdir()
        _fsync_directory(component)
        _fsync_directory(component.parent)
    cursor = path
    while True:
        if not stat.S_ISDIR(cursor.lstat().st_mode):
            raise ValueError(f"path component is not a real directory: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def _atomic_publish(source: Path, destination: Path) -> None:
    """Atomically rename a sibling directory without replacing any existing target."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError:
        renameat2 = None
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"immutable publication target appeared concurrently: {destination}")
        if error_number not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(error_number, os.strerror(error_number), destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"immutable publication target appeared concurrently: {destination}")
    source.rename(destination)


def _arrow_type(spec) -> pa.DataType:
    scalar = pa.from_numpy_dtype(spec.dtype)
    return pa.list_(scalar, list_size=spec.shape[0])


def _rows_to_parquet(rows: Sequence[Mapping[str, object]]) -> bytes:
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "task":
            array = pa.array([row[key] for row in rows], type=pa.string())
            field_type = pa.string()
        elif key == "timestamp":
            array = pa.array([row[key] for row in rows], type=pa.float32())
            field_type = pa.float32()
        else:
            assert spec is not None
            stacked = np.stack([row[key] for row in rows])
            values = pa.array(stacked.reshape(-1), type=pa.from_numpy_dtype(spec.dtype))
            array = pa.FixedSizeListArray.from_arrays(values, spec.shape[0])
            field_type = _arrow_type(spec)
        arrays.append(array)
        fields.append(pa.field(key, field_type, nullable=False))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="NONE", use_dictionary=False, write_statistics=True)
    return sink.getvalue().to_pybytes()


def _parquet_to_rows(data: bytes) -> tuple[Mapping[str, object], ...]:
    try:
        table = pq.read_table(pa.BufferReader(data))
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ValueError("frame-data.parquet must be a readable Arrow parquet file") from error
    if tuple(table.column_names) != tuple(TARGET_FRAME_SCHEMA):
        raise ValueError("frame-data.parquet columns must match deterministic target order")
    if table.num_rows < 1:
        raise ValueError("frame-data.parquet must contain at least one row")
    columns: dict[str, object] = {}
    for key, spec in TARGET_FRAME_SCHEMA.items():
        column = table.column(key).combine_chunks()
        expected_type = pa.string() if key == "task" else pa.float32() if key == "timestamp" else _arrow_type(spec)
        if column.type != expected_type or column.null_count:
            raise ValueError(f"frame-data.parquet column {key} has the wrong exact Arrow type")
        if key == "task":
            columns[key] = column.to_pylist()
        elif key == "timestamp":
            columns[key] = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        else:
            assert spec is not None
            values = column.values.to_numpy(zero_copy_only=False).reshape(table.num_rows, *spec.shape)
            columns[key] = values.astype(spec.dtype, copy=False)
    rows: list[Mapping[str, object]] = []
    for index in range(table.num_rows):
        row: dict[str, object] = {}
        for key in TARGET_FRAME_SCHEMA:
            values = columns[key]
            if key == "task":
                row[key] = values[index]
            elif key == "timestamp":
                row[key] = np.float32(values[index])
            else:
                row[key] = np.array(values[index], copy=True)
        rows.append(MappingProxyType(row))
    validate_target_rows(rows)
    return tuple(rows)


def _identity_manifest(identity: StageIdentity) -> dict[str, object]:
    return identity.to_dict()


def _artifact_paths(root: Path) -> tuple[str, ...]:
    artifacts: list[str] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if not stat.S_ISDIR((current_path / directory_name).lstat().st_mode):
                raise ValueError("stage contains a symlink or non-directory artifact")
        for file_name in file_names:
            candidate = current_path / file_name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError("stage contains a symlink or non-file artifact")
            artifacts.append(candidate.relative_to(root).as_posix())
    return tuple(sorted(artifacts))


def _reject_symlink_components(path: Path) -> None:
    cursor = path.absolute()
    while True:
        try:
            metadata = cursor.lstat()
        except OSError as error:
            raise ValueError(f"publication path component is missing: {cursor}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"publication path must not contain symlink components: {cursor}")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def _checksums_bytes(root: Path, artifacts: Iterable[str]) -> bytes:
    lines = [f"{_sha256(_secure_read(root / artifact))}  {artifact}\n" for artifact in sorted(artifacts)]
    return "".join(lines).encode("utf-8")


def _parse_json_canonical(data: bytes, *, artifact: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{artifact} must contain canonical JSON") from error
    if not isinstance(value, dict) or _canonical_json(value) != data:
        raise ValueError(f"{artifact} must contain canonical JSON")
    return value


@dataclass(frozen=True)
class _VerifiedStage:
    manifest: Mapping[str, object]
    identity: StageIdentity
    rows: tuple[Mapping[str, object], ...]
    videos: Mapping[str, StagedVideoArtifact]
    report: ValidationReport
    manifest_digest: str
    checksums_digest: str


def _verify_checksum_tree(root: Path, *, checksum_name: str) -> None:
    _reject_symlink_components(root)
    if not root.exists() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("publication root must be a real directory")
    actual_artifacts = _artifact_paths(root)
    if checksum_name not in actual_artifacts:
        raise ValueError(f"{checksum_name} is missing")
    checksum_data = _secure_read(root / checksum_name)
    try:
        text = checksum_data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{checksum_name} must be ASCII") from error
    entries: list[tuple[str, str]] = []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n") or len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"{checksum_name} is malformed")
        digest, relative = line[:64], line[66:-1]
        _validated_sha(digest, field_name=f"{checksum_name} digest")
        _safe_source_path(relative)
        entries.append((relative, digest))
    expected_artifacts = tuple(path for path in actual_artifacts if path != checksum_name)
    if tuple(path for path, _ in entries) != expected_artifacts:
        raise ValueError(f"{checksum_name} must list every artifact exactly once in sorted order")
    if len(set(path for path, _ in entries)) != len(entries):
        raise ValueError(f"{checksum_name} contains duplicate artifacts")
    if any(_sha256(_secure_read(root / path)) != digest for path, digest in entries):
        raise ValueError(f"{checksum_name} does not match artifact bytes")
    if _checksums_bytes(root, expected_artifacts) != checksum_data:
        raise ValueError(f"{checksum_name} is not canonical")


def _verify_stage(path: Path, expected_identity: StageIdentity | None = None) -> _VerifiedStage:
    _verify_checksum_tree(path, checksum_name="checksums.sha256")
    manifest_data = _secure_read(path / "manifest.json")
    manifest = _parse_json_canonical(manifest_data, artifact="manifest.json")
    if set(manifest) != {"schema_version", "identity", "row_count", "columns", "frame_data", "videos"}:
        raise ValueError("manifest.json contains missing or unexpected fields")
    if manifest["schema_version"] != _STAGE_SCHEMA_VERSION:
        raise ValueError("manifest.json schema version is unsupported")
    identity = StageIdentity.from_dict(manifest["identity"])
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("stage identity does not match")
    expected_columns = [
        {
            "name": key,
            "dtype": "string" if key == "task" else "float32" if key == "timestamp" else str(spec.dtype),
            "shape": [] if spec is None else list(spec.shape),
        }
        for key, spec in TARGET_FRAME_SCHEMA.items()
    ]
    if manifest["columns"] != expected_columns or manifest["frame_data"] != "frame-data.parquet":
        raise ValueError("manifest.json does not describe the exact staged frame schema")
    rows = _parquet_to_rows(_secure_read(path / "frame-data.parquet"))
    if manifest["row_count"] != len(rows):
        raise ValueError("manifest row_count differs from frame-data.parquet")
    video_manifest = manifest["videos"]
    if not isinstance(video_manifest, dict):
        raise ValueError("manifest videos must be an object")
    video_bytes: dict[str, bytes] = {}
    video_artifacts: dict[str, StagedVideoArtifact] = {}
    for key, metadata in sorted(video_manifest.items()):
        if key not in TARGET_VIDEO_KEYS or not isinstance(metadata, dict):
            raise ValueError("manifest contains an unsupported video target")
        expected_artifact = f"videos/{key}.mp4"
        if set(metadata) != {"artifact", "sha256", "size", "frame_count", "fps", "width", "height"}:
            raise ValueError("manifest video metadata contains missing or unexpected fields")
        if metadata["artifact"] != expected_artifact or metadata["fps"] != 50:
            raise ValueError("manifest video artifact or fps is invalid")
        data = _secure_read(path / expected_artifact)
        if metadata["sha256"] != _sha256(data) or metadata["size"] != len(data):
            raise ValueError("manifest video digest or size differs from artifact")
        video_bytes[key] = data
        video_artifacts[key] = StagedVideoArtifact(
            path=path / expected_artifact,
            sha256=metadata["sha256"],
            size=metadata["size"],
        )
    report = validate_stage_payload(rows, video_bytes)
    for key in report.video_frame_counts:
        metadata = video_manifest[key]
        width, height = report.video_dimensions[key]
        if metadata["frame_count"] != report.video_frame_counts[key] or (
            metadata["width"],
            metadata["height"],
        ) != (width, height):
            raise ValueError("manifest video structure differs from decoded artifact")
    validation_data = _secure_read(path / "validation.json")
    if _parse_json_canonical(validation_data, artifact="validation.json") != report.to_dict():
        raise ValueError("validation.json differs from complete structural validation")
    expected_artifacts = {
        "checksums.sha256",
        "manifest.json",
        "validation.json",
        "frame-data.parquet",
        *(f"videos/{key}.mp4" for key in video_artifacts),
    }
    if set(_artifact_paths(path)) != expected_artifacts:
        raise ValueError("stage contains missing or unexpected artifacts")
    return _VerifiedStage(
        manifest=MappingProxyType(manifest),
        identity=identity,
        rows=rows,
        videos=MappingProxyType(video_artifacts),
        report=report,
        manifest_digest=_sha256(manifest_data),
        checksums_digest=_sha256(_secure_read(path / "checksums.sha256")),
    )


def can_resume(path: str | Path, identity: StageIdentity) -> bool:
    """Return true only for a complete, exact, non-symlink immutable stage."""
    if not isinstance(identity, StageIdentity):
        return False
    try:
        _verify_stage(Path(path).absolute(), identity)
    except (OSError, TypeError, ValueError):
        return False
    return True


def write_stage(output_root: str | Path, identity: StageIdentity, payload: StagePayload) -> StageResult:
    """Validate and atomically publish one immutable episode stage."""
    if not isinstance(identity, StageIdentity):
        raise TypeError("identity must be a StageIdentity")
    if not isinstance(payload, StagePayload):
        raise TypeError("payload must be a StagePayload")
    final = stage_path(output_root, identity)
    _ensure_safe_directory(final.parent)
    if os.path.lexists(final):
        if can_resume(final, identity):
            verified = _verify_stage(final, identity)
            return StageResult(final, identity, verified.manifest_digest, reused=True)
        raise FileExistsError(f"immutable stage already exists but is incomplete or mismatched: {final}")

    temp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        video_bytes = {key: _read_video_source(source, key=key) for key, source in payload.videos.items()}
        report = validate_stage_payload(payload.rows, video_bytes)
        parquet_data = _rows_to_parquet(payload.rows)
        _write_file(temp / "frame-data.parquet", parquet_data)
        video_manifest: dict[str, dict[str, object]] = {}
        for key, data in video_bytes.items():
            artifact = f"videos/{key}.mp4"
            _write_file(temp / artifact, data)
            width, height = report.video_dimensions[key]
            video_manifest[key] = {
                "artifact": artifact,
                "sha256": _sha256(data),
                "size": len(data),
                "frame_count": report.video_frame_counts[key],
                "fps": 50,
                "width": width,
                "height": height,
            }
        manifest = {
            "schema_version": _STAGE_SCHEMA_VERSION,
            "identity": _identity_manifest(identity),
            "row_count": len(payload.rows),
            "columns": [
                {
                    "name": key,
                    "dtype": "string" if key == "task" else "float32" if key == "timestamp" else str(spec.dtype),
                    "shape": [] if spec is None else list(spec.shape),
                }
                for key, spec in TARGET_FRAME_SCHEMA.items()
            ],
            "frame_data": "frame-data.parquet",
            "videos": video_manifest,
        }
        _write_file(temp / "manifest.json", _canonical_json(manifest))
        _write_file(temp / "validation.json", _canonical_json(report.to_dict()))
        artifacts = _artifact_paths(temp)
        _write_file(temp / "checksums.sha256", _checksums_bytes(temp, artifacts))
        _fsync_tree(temp)
        verified = _verify_stage(temp, identity)
        _atomic_publish(temp, final)
        _fsync_directory(final.parent)
        return StageResult(final, identity, verified.manifest_digest, reused=False)
    except BaseException:
        if os.path.lexists(temp):
            shutil.rmtree(temp)
        raise


def load_stage_rows(path: str | Path) -> tuple[Mapping[str, object], ...]:
    """Return fully revalidated, detached rows from one immutable stage."""
    return _verify_stage(Path(path).absolute()).rows


def _stage_input(value: StageResult | str | Path) -> MergeEpisode:
    if isinstance(value, StageResult):
        path = value.path.absolute()
        expected_identity = value.identity
    elif isinstance(value, (str, Path)):
        path = Path(value).absolute()
        expected_identity = None
    else:
        raise TypeError("merge stage inputs must be StageResult or paths")
    try:
        verified = _verify_stage(path, expected_identity)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"stage is not complete and exact: {path}") from error
    if isinstance(value, StageResult) and value.manifest_digest != verified.manifest_digest:
        raise ValueError(f"stage is not complete and exact: {path}")
    return MergeEpisode(
        path=path,
        identity=verified.identity,
        rows=verified.rows,
        videos=verified.videos,
        stage_manifest_digest=verified.manifest_digest,
        stage_checksums_digest=verified.checksums_digest,
    )


def _merge_manifest(episodes: tuple[MergeEpisode, ...], task_catalog: tuple[str, ...]) -> dict[str, object]:
    first = episodes[0].identity
    return {
        "schema_version": _MERGE_SCHEMA_VERSION,
        "source_repo_id": first.source_repo_id,
        "source_revision": first.source_revision,
        "conversion_identity": {
            "conversion_config_sha256": first.conversion_config_sha256,
            "converter_version": first.converter_version,
            "encoder_sha256": first.encoder_sha256,
            "encoder_config_sha256": first.encoder_config_sha256,
            "source_lock_sha256": first.source_lock_sha256,
        },
        "source_episode_ids": [episode.identity.source_episode_id for episode in episodes],
        "episode_lengths": [len(episode.rows) for episode in episodes],
        "total_frames": sum(len(episode.rows) for episode in episodes),
        "video_keys": list(episodes[0].videos),
        "task_catalog": list(task_catalog),
        "stages": [
            {
                "source_episode_id": episode.identity.source_episode_id,
                "source_file_sha256": dict(episode.identity.source_file_sha256),
                "manifest_sha256": episode.stage_manifest_digest,
                "checksums_sha256": episode.stage_checksums_digest,
            }
            for episode in episodes
        ],
    }


def _expected_merge_verification(
    episodes: tuple[MergeEpisode, ...], task_catalog: tuple[str, ...]
) -> dict[str, object]:
    task_to_index = {task: index for index, task in enumerate(task_catalog)}
    global_indices: list[int] = []
    episode_indices: list[int] = []
    task_indices: list[int] = []
    timestamp_bits: list[int] = []
    video_counts: dict[str, dict[str, int]] = {}
    cursor = 0
    for target_episode_index, episode in enumerate(episodes):
        for row in episode.rows:
            global_indices.append(cursor)
            episode_indices.append(target_episode_index)
            task_indices.append(task_to_index[row["task"]])
            timestamp = row["timestamp"]
            assert isinstance(timestamp, np.float32)
            timestamp_bits.append(int(timestamp.view(np.uint32)))
            cursor += 1
        video_counts[str(episode.identity.source_episode_id)] = {key: len(episode.rows) for key in episode.videos}
    return {
        "success": True,
        "source_episode_ids": [episode.identity.source_episode_id for episode in episodes],
        "episode_lengths": [len(episode.rows) for episode in episodes],
        "global_indices": global_indices,
        "episode_indices": episode_indices,
        "task_to_index": task_to_index,
        "frame_task_indices": task_indices,
        "frame_timestamp_float32_bits": timestamp_bits,
        "video_frame_counts": video_counts,
        "stats_source_episode_ids": [episode.identity.source_episode_id for episode in episodes],
    }


def _verification_to_dict(report: MergeVerification) -> dict[str, object]:
    timestamp_bits: list[int] = []
    for index, timestamp in enumerate(report.frame_timestamps):
        if not isinstance(timestamp, np.float32):
            raise ValueError(f"merge backend timestamp {index} must be numpy.float32")
        timestamp_bits.append(int(timestamp.view(np.uint32)))
    return {
        "success": True,
        "source_episode_ids": list(report.source_episode_ids),
        "episode_lengths": list(report.episode_lengths),
        "global_indices": list(report.global_indices),
        "episode_indices": list(report.episode_indices),
        "task_to_index": dict(report.task_to_index),
        "frame_task_indices": list(report.frame_task_indices),
        "frame_timestamp_float32_bits": timestamp_bits,
        "video_frame_counts": {str(key): dict(value) for key, value in report.video_frame_counts.items()},
        "stats_source_episode_ids": list(report.stats_source_episode_ids),
    }


def _verify_merged_output(
    path: Path,
    expected_manifest: dict[str, object],
    expected_validation: dict[str, object] | None = None,
) -> None:
    _verify_checksum_tree(path, checksum_name="dataset-checksums.sha256")
    manifest = _parse_json_canonical(_secure_read(path / "source-manifest.json"), artifact="source-manifest.json")
    if manifest != expected_manifest:
        raise ValueError("source-manifest.json does not match requested stage inputs")
    validation = _parse_json_canonical(
        _secure_read(path / "merge-validation.json"), artifact="merge-validation.json"
    )
    if validation.get("success") is not True:
        raise ValueError("merge-validation.json does not record success")
    if expected_validation is not None and validation != expected_validation:
        raise ValueError("merge-validation.json does not match requested stage inputs")


def _validate_merge_inputs(
    stage_values: Iterable[StageResult | str | Path],
) -> tuple[MergeEpisode, ...]:
    values = tuple(stage_values)
    if not values:
        raise ValueError("merge requires at least one stage")
    episodes = tuple(_stage_input(value) for value in values)
    repositories = {episode.identity.source_repo_id for episode in episodes}
    if len(repositories) != 1:
        raise ValueError("merge requires exactly one source repository")
    conversion_keys = {episode.identity.conversion_key for episode in episodes}
    if len(conversion_keys) != 1:
        raise ValueError("merge stages must have the same conversion identity and artifact locks")
    episode_ids = [episode.identity.source_episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("merge contains a duplicate source episode ID")
    episodes = tuple(sorted(episodes, key=lambda episode: episode.identity.source_episode_id))
    video_keys = {tuple(episode.videos) for episode in episodes}
    if len(video_keys) != 1:
        raise ValueError("merge stages must have one cohort-wide target video schema")
    return episodes


def merge_stages(
    stages: Iterable[StageResult | str | Path],
    output: str | Path,
    *,
    merge_backend: MergeBackend | None = None,
    resume_existing: bool = False,
) -> Path:
    """Fully revalidate, deterministically merge, and atomically publish stages."""
    episodes = _validate_merge_inputs(stages)
    tasks = tuple(sorted({row["task"] for episode in episodes for row in episode.rows}))
    expected_manifest = _merge_manifest(episodes, tasks)
    expected_validation = _expected_merge_verification(episodes, tasks)
    final = Path(output).absolute()
    _ensure_safe_directory(final.parent)
    if os.path.lexists(final):
        if resume_existing:
            try:
                _verify_merged_output(final, expected_manifest, expected_validation)
            except (OSError, TypeError, ValueError):
                pass
            else:
                return final
        raise FileExistsError(f"existing output is immutable, incomplete, or mismatched: {final}")

    temp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    temp.rmdir()
    try:
        backend = _default_merge_backend if merge_backend is None else merge_backend
        report = backend(temp, episodes, tasks)
        if not isinstance(report, MergeVerification):
            raise ValueError("merge backend must return a MergeVerification")
        actual_validation = _verification_to_dict(report)
        if actual_validation != expected_validation:
            raise ValueError("merge backend verification differs from exact requested dataset")
        if not temp.exists() or not stat.S_ISDIR(temp.lstat().st_mode):
            raise ValueError("merge backend did not create a real output directory")
        _write_file(temp / "source-manifest.json", _canonical_json(expected_manifest))
        _write_file(temp / "merge-validation.json", _canonical_json(actual_validation))
        artifacts = _artifact_paths(temp)
        _write_file(temp / "dataset-checksums.sha256", _checksums_bytes(temp, artifacts))
        _fsync_tree(temp)
        _verify_merged_output(temp, expected_manifest, expected_validation)
        _atomic_publish(temp, final)
        _fsync_directory(final.parent)
        return final
    except BaseException:
        if os.path.lexists(temp):
            shutil.rmtree(temp)
        raise


def dataset_manifest_digest(path: str | Path) -> str:
    """Hash the canonical, path-independent logical merge manifest."""
    root = Path(path).absolute()
    _verify_checksum_tree(root, checksum_name="dataset-checksums.sha256")
    data = _secure_read(root / "source-manifest.json")
    _parse_json_canonical(data, artifact="source-manifest.json")
    return _sha256(data)


def _create_gr00t_exporter(
    output: Path,
    video_keys: tuple[str, ...],
    task_catalog: tuple[str, ...],
):
    """Lazily construct the repository exporter with the exact 50 Hz schema."""
    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_g1_robot_model,
        get_modality_config_sonic_vla,
        get_wrist_camera_features,
        get_wrist_camera_modality_config,
    )

    robot_model = get_g1_robot_model(waist_location="lower_and_upper_body")
    features = get_features_sonic_vla(robot_model)
    wrist_features = get_wrist_camera_features()
    for key in video_keys:
        if key in wrist_features:
            features[key] = wrist_features[key]
        if key not in features or features[key]["dtype"] != "video":
            raise ValueError(f"target video key {key!r} is not supported by the SONIC VLA schema")
    for key in tuple(features):
        if features[key]["dtype"] == "video" and key not in video_keys:
            del features[key]

    modality = get_modality_config_sonic_vla(robot_model)
    wrist_modality = get_wrist_camera_modality_config()["video"]
    target_to_modality = {value["original_key"]: name for name, value in wrist_modality.items()}
    for key in video_keys:
        if key in target_to_modality:
            name = target_to_modality[key]
            modality["video"][name] = wrist_modality[name]
    modality["video"] = {
        name: value for name, value in modality["video"].items() if value["original_key"] in video_keys
    }

    exporter = Gr00tDataExporter.create(
        save_root=output,
        fps=50,
        features=features,
        modality_config=modality,
        task=task_catalog[0],
        script_config={"unitree_conversion_schema": _MERGE_SCHEMA_VERSION},
        robot_type="g1",
        overwrite_existing=False,
    )
    for expected_index, task in enumerate(task_catalog):
        exporter.meta.add_task(task)
        if exporter.meta.get_task_index(task) != expected_index:
            raise ValueError("Gr00tDataExporter did not preregister lexicographic task IDs")
    return exporter


def _iter_staged_video(artifact: StagedVideoArtifact):
    """Yield detached RGB frames from one already-validated stage MP4."""
    data = _secure_read(artifact.path)
    if len(data) != artifact.size or _sha256(data) != artifact.sha256:
        raise ValueError("staged video changed after stage verification")
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1 or streams[0].average_rate is None:
                raise ValueError("staged video must contain exactly one video stream")
            if Fraction(streams[0].average_rate) != 50:
                raise ValueError("staged video must have nominal fps 50")
            for frame in container.decode(streams[0]):
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.dtype(np.uint8):
                    raise ValueError("staged video must decode as 640x480 RGB uint8")
                yield np.array(rgb, order="C", copy=True)
    except av.error.FFmpegError as error:
        raise ValueError("staged video became unreadable during merge") from error


def _close_unused_video_writers(exporter: object) -> None:
    writers = getattr(exporter, "video_writers", None)
    if isinstance(writers, Mapping) and writers and all(hasattr(writer, "cancel") for writer in writers.values()):
        for writer in writers.values():
            writer.cancel()
    else:
        exporter.stop_video_writers()


def _default_merge_backend(
    output: Path,
    episodes: tuple[MergeEpisode, ...],
    task_catalog: tuple[str, ...],
) -> MergeVerification:
    """Stream staged rows and decoded video frames through Gr00tDataExporter."""
    video_keys = tuple(episodes[0].videos)
    exporter = _create_gr00t_exporter(output, video_keys, task_catalog)
    completed = False
    try:
        for episode in episodes:
            iterators = {key: iter(_iter_staged_video(artifact)) for key, artifact in episode.videos.items()}
            for row in episode.rows:
                frame = dict(row)
                for key, iterator in iterators.items():
                    try:
                        frame[key] = next(iterator)
                    except StopIteration as error:
                        raise ValueError(f"staged video {key} ended before its frame rows") from error
                exporter.add_frame(frame)
            for key, iterator in iterators.items():
                try:
                    next(iterator)
                except StopIteration:
                    continue
                raise ValueError(f"staged video {key} contains more frames than its frame rows")
            exporter.save_episode()
        _close_unused_video_writers(exporter)
        completed = True
        return _inspect_gr00t_output(exporter, episodes, task_catalog)
    finally:
        if not completed:
            writers = getattr(exporter, "video_writers", {})
            for writer in writers.values() if isinstance(writers, Mapping) else ():
                if hasattr(writer, "cancel"):
                    writer.cancel()


def _inspect_gr00t_output(
    exporter: object,
    episodes: tuple[MergeEpisode, ...],
    task_catalog: tuple[str, ...],
) -> MergeVerification:
    """Read back exact indices, tasks, timestamps, videos, and stats from an export."""
    from lerobot.common.datasets.compute_stats import compute_episode_stats

    meta = exporter.meta
    if meta.total_episodes != len(episodes):
        raise ValueError("exported episode count differs from requested merge")
    expected_total = sum(len(episode.rows) for episode in episodes)
    if meta.total_frames != expected_total:
        raise ValueError("exported global frame count differs from requested merge")
    task_to_index = {task: meta.get_task_index(task) for task in task_catalog}
    if task_to_index != {task: index for index, task in enumerate(task_catalog)}:
        raise ValueError("exported global task catalog is not lexicographic")

    global_indices: list[int] = []
    episode_indices: list[int] = []
    frame_task_indices: list[int] = []
    frame_timestamps: list[np.float32] = []
    episode_lengths: list[int] = []
    video_counts: dict[int, dict[str, int]] = {}
    stats_ids: list[int] = []
    root = Path(exporter.root)
    for target_index, episode in enumerate(episodes):
        parquet_path = root / meta.get_data_file_path(target_index)
        table = pq.read_table(pa.BufferReader(_secure_read(parquet_path)))
        length = table.num_rows
        episode_lengths.append(length)
        required = {"index", "episode_index", "task_index", "timestamp"}
        if not required.issubset(table.column_names):
            raise ValueError("exported parquet is missing index, episode, task, or timestamp columns")
        global_indices.extend(int(value) for value in table.column("index").to_pylist())
        episode_indices.extend(int(value) for value in table.column("episode_index").to_pylist())
        frame_task_indices.extend(int(value) for value in table.column("task_index").to_pylist())
        frame_timestamps.extend(np.float32(value) for value in table.column("timestamp").to_pylist())
        counts: dict[str, int] = {}
        for key in episode.videos:
            video_path = root / meta.get_video_file_path(target_index, key)
            count, _ = probe_video_50fps(_secure_read(video_path), field_name=f"exported video {key}")
            counts[key] = count
        video_counts[episode.identity.source_episode_id] = counts
        if target_index not in meta.episodes or target_index not in meta.episodes_stats:
            raise ValueError("exported episode metadata or statistics are missing")
        episode_metadata = meta.episodes[target_index]
        expected_episode_tasks = list(dict.fromkeys(row["task"] for row in episode.rows))
        if episode_metadata.get("length") != length or episode_metadata.get("tasks") != expected_episode_tasks:
            raise ValueError("exported episode length or first-occurrence task order differs")
        nonvideo_features = {
            key: feature for key, feature in exporter.features.items() if feature["dtype"] != "video"
        }
        episode_data: dict[str, np.ndarray] = {}
        for key, feature in nonvideo_features.items():
            if key not in table.column_names:
                raise ValueError(f"exported parquet is missing statistics feature {key}")
            values = table.column(key).to_pylist()
            episode_data[key] = np.asarray(values, dtype=np.dtype(feature["dtype"]))
        expected_stats = compute_episode_stats(episode_data, nonvideo_features)
        actual_stats = meta.episodes_stats[target_index]
        if set(actual_stats) != set(expected_stats):
            raise ValueError("exported episode statistics feature keys differ")
        for feature_key in expected_stats:
            if set(actual_stats[feature_key]) != set(expected_stats[feature_key]) or any(
                not np.array_equal(
                    np.asarray(actual_stats[feature_key][statistic]),
                    np.asarray(expected_stats[feature_key][statistic]),
                )
                for statistic in expected_stats[feature_key]
            ):
                raise ValueError(f"exported episode statistics differ for {feature_key}")
        stats_ids.append(episode.identity.source_episode_id)
    return MergeVerification(
        source_episode_ids=tuple(episode.identity.source_episode_id for episode in episodes),
        episode_lengths=tuple(episode_lengths),
        global_indices=tuple(global_indices),
        episode_indices=tuple(episode_indices),
        task_to_index=task_to_index,
        frame_task_indices=tuple(frame_task_indices),
        frame_timestamps=tuple(frame_timestamps),
        video_frame_counts=video_counts,
        stats_source_episode_ids=tuple(stats_ids),
    )
