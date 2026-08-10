from dataclasses import fields
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion import dex3_adapter as dex3_adapter_module
from gear_sonic.data.unitree_conversion.contracts import CanonicalEpisode, SourceSpec
from gear_sonic.data.unitree_conversion.dex3_adapter import adapt_dex3_arrays, load_dex3_episode
from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES, NOMINAL_G1_MUJOCO

ARM_NAMES = (
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
HAND_NAMES = (
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
FEATURE_NAMES = ARM_NAMES + HAND_NAMES
ARM_TARGET_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CANONICAL_ARRAY_FIELDS = (
    "timestamps",
    "task_indices",
    "observed_root_wxyz",
    "reference_root_wxyz",
    "observed_body_q",
    "desired_body_q",
    "observed_left_hand",
    "observed_right_hand",
    "desired_left_hand",
    "desired_right_hand",
)


def _adapter_kwargs(n: int = 3) -> dict[str, object]:
    observed = np.arange(n * 28, dtype=np.float64).reshape(n, 28)
    return {
        "source_repo_id": "synthetic/dex3",
        "source_revision": "1" * 40,
        "source_episode_id": 7,
        "observed": observed,
        "desired": observed + 1000.0,
        "feature_names": FEATURE_NAMES,
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
        "task_indices": np.arange(n, dtype=np.int64),
    }


def _canonical_kwargs(n: int = 3) -> dict[str, object]:
    return {
        "source_repo_id": "synthetic/dex3",
        "source_revision": "a" * 40,
        "source_episode_id": 2,
        "source_fps": 30,
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
        "task_indices": np.arange(n, dtype=np.int64),
        "observed_root_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "reference_root_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "observed_body_q": np.tile(NOMINAL_G1_MUJOCO, (n, 1)),
        "desired_body_q": np.tile(NOMINAL_G1_MUJOCO, (n, 1)),
        "observed_left_hand": np.zeros((n, 7)),
        "observed_right_hand": np.zeros((n, 7)),
        "desired_left_hand": np.ones((n, 7)),
        "desired_right_hand": np.ones((n, 7)),
    }


def test_adapts_arms_by_semantic_name_and_preserves_exact_dds_hand_orders() -> None:
    kwargs = _adapter_kwargs()
    observed = kwargs["observed"]
    desired = kwargs["desired"]

    episode = adapt_dex3_arrays(**kwargs)

    expected_observed_body = np.tile(NOMINAL_G1_MUJOCO, (3, 1))
    expected_desired_body = np.tile(NOMINAL_G1_MUJOCO, (3, 1))
    for source_name, target_name in zip(ARM_NAMES, ARM_TARGET_NAMES, strict=True):
        source_index = FEATURE_NAMES.index(source_name)
        target_index = G1_MUJOCO_NAMES.index(target_name)
        expected_observed_body[:, target_index] = observed[:, source_index]
        expected_desired_body[:, target_index] = desired[:, source_index]

    np.testing.assert_array_equal(episode.observed_body_q, expected_observed_body)
    np.testing.assert_array_equal(episode.desired_body_q, expected_desired_body)
    np.testing.assert_array_equal(episode.observed_left_hand, observed[:, 14:21])
    np.testing.assert_array_equal(episode.desired_left_hand, desired[:, 14:21])
    np.testing.assert_array_equal(episode.observed_right_hand, observed[:, 21:28])
    np.testing.assert_array_equal(episode.desired_right_hand, desired[:, 21:28])
    assert HAND_NAMES[3:7] == (
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
    )
    assert HAND_NAMES[10:14] == (
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    )
    np.testing.assert_array_equal(episode.observed_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 3)
    np.testing.assert_array_equal(episode.reference_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 3)


def test_adapter_returns_mutable_owned_contiguous_arrays_isolated_from_every_source_buffer() -> None:
    observed_storage = np.arange(3 * 56, dtype=np.float32).reshape(3, 56)
    desired_storage = observed_storage + 500.0
    timestamp_storage = np.arange(6, dtype=np.float32) / 60.0
    task_storage = np.arange(6, dtype=np.int32)
    observed = observed_storage[:, ::2]
    desired = desired_storage[:, ::2]
    timestamps = timestamp_storage[::2]
    task_indices = task_storage[::2]
    kwargs = _adapter_kwargs()
    kwargs.update(
        observed=observed,
        desired=desired,
        timestamps=timestamps,
        task_indices=task_indices,
    )

    episode = adapt_dex3_arrays(**kwargs)
    snapshots = {name: getattr(episode, name).copy() for name in CANONICAL_ARRAY_FIELDS}
    observed_storage[:] = -1
    desired_storage[:] = -2
    timestamp_storage[:] = -3
    task_storage[:] = -4

    for name in CANONICAL_ARRAY_FIELDS:
        array = getattr(episode, name)
        np.testing.assert_array_equal(array, snapshots[name])
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert array.flags.writeable
    assert episode.task_indices.dtype == np.dtype(np.int64)
    assert all(
        getattr(episode, name).dtype == np.dtype(np.float64)
        for name in CANONICAL_ARRAY_FIELDS
        if name != "task_indices"
    )


@pytest.mark.parametrize(
    "feature_names",
    [
        tuple(f"feature_{index}" for index in range(28)),
        (ARM_NAMES[1], ARM_NAMES[0]) + ARM_NAMES[2:] + HAND_NAMES,
        ARM_NAMES + HAND_NAMES[:-2] + (HAND_NAMES[-1], HAND_NAMES[-2]),
        FEATURE_NAMES[:-1],
    ],
)
def test_adapter_rejects_nonexact_or_swapped_feature_names(feature_names: tuple[str, ...]) -> None:
    kwargs = _adapter_kwargs()
    kwargs["feature_names"] = feature_names

    with pytest.raises(ValueError, match="exact Dex3 feature names"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("observed", np.zeros((3, 27))),
        ("observed", np.zeros((3, 28, 1))),
        ("desired", np.zeros((3, 29))),
        ("desired", np.zeros(28)),
    ],
)
def test_adapter_rejects_wrong_state_and_action_shapes(field_name: str, value: np.ndarray) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        adapt_dex3_arrays(**kwargs)


def test_adapter_rejects_mismatched_observed_and_desired_row_counts() -> None:
    kwargs = _adapter_kwargs()
    kwargs["desired"] = np.zeros((2, 28))

    with pytest.raises(ValueError, match="same number of rows"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("timestamps", np.zeros(2)),
        ("timestamps", np.zeros((3, 1))),
        ("task_indices", np.zeros(2, dtype=np.int64)),
        ("task_indices", np.zeros((3, 1), dtype=np.int64)),
    ],
)
def test_adapter_rejects_wrong_timeline_shapes(field_name: str, value: np.ndarray) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize("field_name", ["observed", "desired", "timestamps"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_adapter_rejects_nonfinite_sources(field_name: str, bad_value: float) -> None:
    kwargs = _adapter_kwargs()
    value = np.array(kwargs[field_name], copy=True)
    value.flat[-1] = bad_value
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name}.*finite"):
        adapt_dex3_arrays(**kwargs)


def test_adapter_rejects_episodes_shorter_than_two_rows() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        adapt_dex3_arrays(**_adapter_kwargs(n=1))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_repo_id", "", "nonempty"),
        ("source_repo_id", "   ", "nonempty"),
        ("source_revision", "main", "40-character immutable revision"),
        ("source_episode_id", True, "nonnegative integer"),
        ("source_episode_id", -1, "nonnegative integer"),
    ],
)
def test_adapter_rejects_invalid_source_identity(field_name: str, value: object, message: str) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        adapt_dex3_arrays(**kwargs)


def test_canonical_episode_has_exact_mutable_contract_fields() -> None:
    assert tuple(field.name for field in fields(CanonicalEpisode)) == (
        "source_repo_id",
        "source_revision",
        "source_episode_id",
        "source_fps",
        "timestamps",
        "task_indices",
        "observed_root_wxyz",
        "reference_root_wxyz",
        "observed_body_q",
        "desired_body_q",
        "observed_left_hand",
        "observed_right_hand",
        "desired_left_hand",
        "desired_right_hand",
    )
    episode = CanonicalEpisode(**_canonical_kwargs())
    episode.observed_body_q[0, 0] = 123.0
    assert episode.observed_body_q[0, 0] == 123.0


@pytest.mark.parametrize("source_fps", [29, 30.1, "30", True, np.int64(30)])
def test_canonical_episode_requires_builtin_integer_30_hz(source_fps: object) -> None:
    kwargs = _canonical_kwargs()
    kwargs["source_fps"] = source_fps

    with pytest.raises(ValueError, match="source_fps must be exactly integer 30"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_repo_id", None, "nonempty"),
        ("source_revision", "f" * 39, "40-character immutable revision"),
        ("source_episode_id", 1.5, "nonnegative integer"),
        ("source_episode_id", True, "nonnegative integer"),
        ("source_episode_id", -5, "nonnegative integer"),
    ],
)
def test_canonical_episode_rejects_invalid_source_identity(
    field_name: str,
    value: object,
    message: str,
) -> None:
    kwargs = _canonical_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "shape"),
    [
        ("timestamps", (3, 1)),
        ("task_indices", (2,)),
        ("observed_root_wxyz", (3, 3)),
        ("reference_root_wxyz", (2, 4)),
        ("observed_body_q", (3, 28)),
        ("desired_body_q", (2, 29)),
        ("observed_left_hand", (3, 6)),
        ("observed_right_hand", (2, 7)),
        ("desired_left_hand", (3, 8)),
        ("desired_right_hand", (3, 7, 1)),
    ],
)
def test_canonical_episode_rejects_every_wrong_array_shape(field_name: str, shape: tuple[int, ...]) -> None:
    kwargs = _canonical_kwargs()
    dtype = np.int64 if field_name == "task_indices" else np.float64
    kwargs[field_name] = np.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        CanonicalEpisode(**kwargs)


def test_canonical_episode_rejects_fewer_than_two_timestamps() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        CanonicalEpisode(**_canonical_kwargs(n=1))


@pytest.mark.parametrize("timestamps", [np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.2, 0.1])])
def test_canonical_episode_requires_strictly_increasing_timestamps(timestamps: np.ndarray) -> None:
    kwargs = _canonical_kwargs()
    kwargs["timestamps"] = timestamps

    with pytest.raises(ValueError, match="strictly increasing"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    "task_indices",
    [
        np.array([0.0, 1.0, 2.0]),
        np.array([False, False, True]),
        np.array([0, -1, 2], dtype=np.int64),
        np.array([0, 1, np.iinfo(np.uint64).max], dtype=np.uint64),
        [0, 1.0, 2],
    ],
)
def test_canonical_episode_rejects_noninteger_or_nonsensical_task_values(task_indices: object) -> None:
    kwargs = _canonical_kwargs()
    kwargs["task_indices"] = task_indices

    with pytest.raises(ValueError, match="task_indices must contain nonnegative int64-compatible integers"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "timestamps",
        "observed_root_wxyz",
        "reference_root_wxyz",
        "observed_body_q",
        "desired_body_q",
        "observed_left_hand",
        "observed_right_hand",
        "desired_left_hand",
        "desired_right_hand",
    ],
)
def test_canonical_episode_rejects_nonfinite_values_in_every_float_array(field_name: str) -> None:
    kwargs = _canonical_kwargs()
    value = np.array(kwargs[field_name], copy=True)
    value.flat[-1] = np.nan
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name}.*finite"):
        CanonicalEpisode(**kwargs)


def test_canonical_episode_copies_noncontiguous_callers_into_owned_storage() -> None:
    kwargs = _canonical_kwargs()
    caller_arrays: dict[str, np.ndarray] = {}
    for field_name in CANONICAL_ARRAY_FIELDS:
        value = np.asarray(kwargs[field_name])
        storage = np.empty((value.shape[0] * 2, *value.shape[1:]), dtype=value.dtype)
        storage[::2] = value
        storage[1::2] = value
        view = storage[::2]
        caller_arrays[field_name] = view
        kwargs[field_name] = view

    episode = CanonicalEpisode(**kwargs)
    snapshots = {name: getattr(episode, name).copy() for name in CANONICAL_ARRAY_FIELDS}
    for value in caller_arrays.values():
        value[...] = 99

    for field_name, expected in snapshots.items():
        array = getattr(episode, field_name)
        np.testing.assert_array_equal(array, expected)
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert array.flags.writeable


class TorchLike:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value)

    def detach(self) -> "TorchLike":
        return self

    def cpu(self) -> "TorchLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class FakeDataset:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        total_episodes: object = 3,
        fps: object = 30,
        features: object | None = None,
        dataset_revision: object = "b" * 40,
        meta_revision: object = "b" * 40,
    ) -> None:
        if features is None:
            features = {
                "observation.state": {"names": list(FEATURE_NAMES)},
                "action": {"names": list(FEATURE_NAMES)},
            }
        self.meta = type(
            "FakeMeta",
            (),
            {
                "total_episodes": total_episodes,
                "fps": fps,
                "features": features,
                "revision": meta_revision,
            },
        )()
        self.revision = dataset_revision
        self.hf_dataset = rows
        self.top_level_iteration_attempts = 0

    def __iter__(self):
        self.top_level_iteration_attempts += 1
        raise AssertionError("top-level LeRobotDataset iteration may decode video or task strings")


def _source_spec(**changes: object) -> SourceSpec:
    values = {
        "approved": True,
        "repo_id": "unitreerobotics/synthetic-dex3",
        "revision": "b" * 40,
        "episode_count": 3,
        "episodes": (1,),
        "primary_camera": "observation.images.primary",
        "camera_map": {"observation.images.primary": "observation.images.ego_view"},
        "label": "dex3",
    }
    values.update(changes)
    return SourceSpec(**values)


def _corrupted_unpinned_source_spec() -> SourceSpec:
    source_spec = _source_spec()
    object.__setattr__(source_spec, "episode_count", None)
    return source_spec


def _rows(episode_id: int = 1, n: int = 2) -> list[dict[str, object]]:
    rows = []
    for frame_index in range(n):
        observed = np.arange(28, dtype=np.float32) + frame_index * 100
        desired = observed + 1000
        rows.append(
            {
                "observation.state": TorchLike(observed),
                "action": TorchLike(desired),
                "timestamp": TorchLike(frame_index / 30),
                "task_index": TorchLike(np.int32(frame_index)),
                "frame_index": TorchLike(np.int64(frame_index)),
                "episode_index": TorchLike(np.int64(episode_id)),
            }
        )
    return rows


def _recording_factory(dataset: FakeDataset, calls: list[dict[str, object]]):
    def factory(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    return factory


def test_loader_uses_exact_pinned_lerobot_constructor_and_adapts_torch_like_rows() -> None:
    calls: list[dict[str, object]] = []
    rows = _rows()
    dataset = FakeDataset(rows)

    episode = load_dex3_episode(
        _source_spec(),
        1,
        root=Path("/tmp/pinned-dex3"),
        dataset_factory=_recording_factory(dataset, calls),
    )

    assert calls == [
        {
            "repo_id": "unitreerobotics/synthetic-dex3",
            "root": Path("/tmp/pinned-dex3")
            / f"repo-{hashlib.sha256(b'unitreerobotics/synthetic-dex3').hexdigest()}"
            / f"revision-{'b' * 40}",
            "episodes": [1],
            "revision": "b" * 40,
            "download_videos": True,
            "video_backend": "pyav",
        }
    ]
    assert episode.source_repo_id == "unitreerobotics/synthetic-dex3"
    assert episode.source_revision == "b" * 40
    assert episode.source_episode_id == 1
    assert episode.source_fps == 30
    np.testing.assert_array_equal(episode.timestamps, [0.0, 1 / 30])
    np.testing.assert_array_equal(episode.task_indices, [0, 1])
    np.testing.assert_array_equal(episode.observed_left_hand[1], rows[1]["observation.state"].numpy()[14:21])
    np.testing.assert_array_equal(episode.desired_right_hand[1], rows[1]["action"].numpy()[21:28])
    assert dataset.top_level_iteration_attempts == 0


def test_revision_scoped_root_is_deterministic_repo_and_revision_identity(tmp_path: Path) -> None:
    same = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "a" * 40)
    same_again = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "a" * 40)
    other_revision = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "b" * 40)
    other_repo = dex3_adapter_module._revision_scoped_root(tmp_path, "other/repo", "a" * 40)

    assert same == same_again
    assert len({same, other_revision, other_repo}) == 3
    assert same.parent == other_revision.parent
    assert same.parent != other_repo.parent
    assert same.name != other_revision.name
    assert same.name == other_repo.name


@pytest.mark.parametrize("repo_id", ["../escape", "/absolute/repo", "a/../../outside", "odd ☃/repo"])
def test_revision_scoped_root_encodes_repo_as_one_safe_segment(tmp_path: Path, repo_id: str) -> None:
    scoped = dex3_adapter_module._revision_scoped_root(tmp_path, repo_id, "b" * 40)

    relative = scoped.relative_to(tmp_path)
    assert relative.parts == (
        f"repo-{hashlib.sha256(repo_id.encode()).hexdigest()}",
        f"revision-{'b' * 40}",
    )


def test_default_lerobot_cache_base_matches_hf_lerobot_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    assert dex3_adapter_module._default_lerobot_cache_base() == tmp_path


def test_loader_scopes_roots_by_both_repo_identity_and_full_revision(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        revision = kwargs["revision"]
        return FakeDataset(_rows(), dataset_revision=revision, meta_revision=revision)

    sources = (
        _source_spec(revision="a" * 40),
        _source_spec(revision="c" * 40),
        _source_spec(repo_id="other/repository", revision="a" * 40),
    )
    for source in sources:
        load_dex3_episode(source, 1, root=tmp_path, dataset_factory=factory)

    roots = [call["root"] for call in calls]
    assert len(set(roots)) == 3
    for root in roots:
        relative = Path(root).relative_to(tmp_path)
        assert len(relative.parts) == 2
        assert relative.parts[0].startswith("repo-")
        assert relative.parts[1].startswith("revision-")
    assert roots[0].parent == roots[1].parent
    assert roots[0].parent != roots[2].parent
    assert roots[0].name != roots[1].name
    assert roots[0].name == roots[2].name


@pytest.mark.parametrize("repo_id", ["../escape", "/absolute/repo", "a/../../outside", "odd ☃/repo"])
def test_loader_repo_identity_cannot_escape_or_add_segments_to_cache_base(
    tmp_path: Path,
    repo_id: str,
) -> None:
    calls: list[dict[str, object]] = []
    source = _source_spec(repo_id=repo_id)

    load_dex3_episode(
        source,
        1,
        root=tmp_path,
        dataset_factory=_recording_factory(FakeDataset(_rows()), calls),
    )

    relative = Path(calls[0]["root"]).relative_to(tmp_path)
    assert len(relative.parts) == 2
    assert relative.parts[0] == f"repo-{hashlib.sha256(repo_id.encode()).hexdigest()}"
    assert relative.parts[1] == f"revision-{'b' * 40}"


def test_loader_none_root_uses_lerobot_cache_convention_without_importing_lerobot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    load_dex3_episode(
        _source_spec(),
        1,
        dataset_factory=_recording_factory(FakeDataset(_rows()), calls),
    )

    assert Path(calls[0]["root"]).relative_to(tmp_path).parts == (
        f"repo-{hashlib.sha256(b'unitreerobotics/synthetic-dex3').hexdigest()}",
        f"revision-{'b' * 40}",
    )


@pytest.mark.parametrize(
    ("dataset_revision", "meta_revision", "message"),
    [
        ("c" * 40, "b" * 40, "dataset.revision must equal pinned source revision"),
        ("b" * 40, "c" * 40, "dataset.meta.revision must equal pinned source revision"),
        (None, "b" * 40, "dataset.revision must equal pinned source revision"),
        ("b" * 40, None, "dataset.meta.revision must equal pinned source revision"),
    ],
)
def test_loader_rejects_dataset_or_metadata_revision_mismatch(
    dataset_revision: object,
    meta_revision: object,
    message: str,
) -> None:
    dataset = FakeDataset(
        _rows(),
        dataset_revision=dataset_revision,
        meta_revision=meta_revision,
    )

    with pytest.raises(ValueError, match=message):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_iterates_only_raw_hf_dataset_rows() -> None:
    dataset = FakeDataset(_rows())

    episode = load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)

    assert episode.source_episode_id == 1
    assert dataset.top_level_iteration_attempts == 0


def test_loader_rejects_missing_raw_hf_dataset() -> None:
    dataset = FakeDataset(_rows())
    del dataset.hf_dataset

    with pytest.raises(ValueError, match="dataset.hf_dataset must be a non-mapping iterable of raw rows"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("hf_dataset", [None, "rows", b"rows", {"frame": 0}, 7])
def test_loader_rejects_malformed_raw_hf_dataset(hf_dataset: object) -> None:
    dataset = FakeDataset(_rows())
    dataset.hf_dataset = hf_dataset

    with pytest.raises(ValueError, match="dataset.hf_dataset must be a non-mapping iterable of raw rows"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("total_episodes", [2, 4, True, 3.0])
def test_loader_rejects_total_episode_metadata_mismatch(total_episodes: object) -> None:
    dataset = FakeDataset(_rows(), total_episodes=total_episodes)

    with pytest.raises(ValueError, match="total_episodes must exactly match pinned episode_count"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("fps", [29, 30.5, True, "30", np.nan])
def test_loader_rejects_non_30_hz_metadata(fps: object) -> None:
    dataset = FakeDataset(_rows(), fps=fps)

    with pytest.raises(ValueError, match="dataset metadata fps must be numeric 30"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("feature_key", ["observation.state", "action"])
def test_loader_rejects_mismatched_feature_names(feature_key: str) -> None:
    features = {
        "observation.state": {"names": list(FEATURE_NAMES)},
        "action": {"names": list(FEATURE_NAMES)},
    }
    features[feature_key] = {"names": [f"generic_{index}" for index in range(28)]}
    dataset = FakeDataset(_rows(), features=features)

    with pytest.raises(ValueError, match=rf"{feature_key} metadata must contain the exact Dex3 feature names"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    "features",
    [
        None,
        [],
        {"observation.state": {"names": list(FEATURE_NAMES)}},
        {"observation.state": list(FEATURE_NAMES), "action": {"names": list(FEATURE_NAMES)}},
    ],
)
def test_loader_rejects_undocumented_feature_metadata_schemas(features: object) -> None:
    dataset = FakeDataset(_rows())
    dataset.meta.features = features

    with pytest.raises(ValueError, match="dataset meta.features"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_rejects_episode_id_remapped_to_zero() -> None:
    dataset = FakeDataset(_rows(episode_id=0))

    with pytest.raises(ValueError, match="source episode_index 0 does not match requested episode 1"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_rejects_no_requested_rows() -> None:
    dataset = FakeDataset([])

    with pytest.raises(ValueError, match="no rows for requested episode 1"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    ("frame_indices", "message"),
    [
        ([0, 0], "expected frame_index 1, got 0"),
        ([0, 2], "expected frame_index 1, got 2"),
        ([1, 0], "expected frame_index 0, got 1"),
    ],
)
def test_loader_rejects_duplicate_gapped_or_out_of_order_frame_indices(
    frame_indices: list[int],
    message: str,
) -> None:
    rows = _rows()
    for row, frame_index in zip(rows, frame_indices, strict=True):
        row["frame_index"] = frame_index
    dataset = FakeDataset(rows)

    with pytest.raises(ValueError, match=message):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    "missing_key",
    ["observation.state", "action", "timestamp", "task_index", "frame_index", "episode_index"],
)
def test_loader_requires_every_documented_row_key(missing_key: str) -> None:
    rows = _rows()
    del rows[0][missing_key]
    dataset = FakeDataset(rows)

    with pytest.raises(ValueError, match=rf"row 0 is missing required key {missing_key}"):
        load_dex3_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    ("source_spec", "episode_id", "message"),
    [
        (object(), 1, "source_spec must be an approved SourceSpec"),
        (_source_spec(approved=False), 1, "source_spec must be an approved SourceSpec"),
        (_corrupted_unpinned_source_spec(), 1, "pinned episode_count"),
        (_source_spec(), True, "episode_id must be a nonnegative integer"),
        (_source_spec(), -1, "episode_id must be a nonnegative integer"),
        (_source_spec(), 3, "smaller than pinned episode_count"),
        (_source_spec(), 0, "selected in source_spec.episodes"),
    ],
)
def test_loader_rejects_unapproved_unpinned_or_ineligible_sources(
    source_spec: object,
    episode_id: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_dex3_episode(source_spec, episode_id, dataset_factory=lambda **_: FakeDataset(_rows()))
