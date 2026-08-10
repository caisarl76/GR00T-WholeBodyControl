from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import re

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.unitree_conversion import staging as staging_module
from gear_sonic.data.unitree_conversion.staging import (
    MergeVerification,
    StageIdentity,
    StagePayload,
    can_resume,
    dataset_manifest_digest,
    load_stage_rows,
    merge_stages,
    stage_path,
    write_stage,
)
from gear_sonic.data.unitree_conversion.validation import (
    TARGET_FRAME_SCHEMA,
    regenerate_eef_state,
    target_joint_limits,
)


def _write_video(
    path: Path,
    *,
    frame_count: int = 2,
    fps: int = 50,
    size: tuple[int, int] = (640, 480),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("libx264", "libopenh264", "h264"):
        try:
            av.codec.Codec(codec, "w")
        except (av.error.FFmpegError, ValueError):
            continue
        break
    else:
        pytest.skip("PyAV has no usable H.264 encoder")
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            rgb = np.full((size[1], size[0], 3), index * 50, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _row(index: int, task: str) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "timestamp":
            row[key] = np.float32(index / 50.0)
        elif key == "task":
            row[key] = task
        else:
            value = np.zeros(spec.shape, dtype=spec.dtype)
            if key == "teleop.smpl_frame_index":
                value[0] = index
            row[key] = value
    row["observation.root_orientation"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.cpp_rotation_offset"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.init_base_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.projected_gravity"][:] = [0.0, 0.0, -1.0]
    row["observation.eef_state"] = regenerate_eef_state(row["observation.state"])
    row["teleop.body_quat_w"][:] = [1.0, 0.0, 0.0, 0.0]
    row["teleop.target_body_orientation"][:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row["teleop.planner_facing"][:] = [1.0, 0.0, 0.0]
    row["teleop.planner_speed"][:] = [-1.0]
    row["teleop.planner_height"][:] = [-1.0]
    return row


def _identity(episode_id: int = 7, *, repo: str = "unitree/source") -> StageIdentity:
    return StageIdentity(
        source_repo_id=repo,
        source_revision="a" * 40,
        source_episode_id=episode_id,
        source_file_sha256={f"data/episode-{episode_id}.parquet": "b" * 64},
        conversion_config_sha256="c" * 64,
        converter_version="unitree-sonic-v1",
        encoder_sha256="d" * 64,
        encoder_config_sha256="e" * 64,
        source_lock_sha256="f" * 64,
    )


def _payload(tmp_path: Path, episode_id: int = 7, tasks: tuple[str, str] = ("pour", "place")) -> StagePayload:
    video = tmp_path / f"source-{episode_id}.mp4"
    _write_video(video)
    return StagePayload(
        rows=tuple(_row(index, task) for index, task in enumerate(tasks)),
        videos={"observation.images.ego_view": video},
    )


def test_stage_identity_is_deeply_immutable_and_path_is_safe(tmp_path: Path) -> None:
    identity = _identity()

    with pytest.raises(FrozenInstanceError):
        identity.source_episode_id = 8  # type: ignore[misc]
    with pytest.raises(TypeError):
        identity.source_file_sha256["new"] = "0" * 64  # type: ignore[index]

    assert stage_path(tmp_path, identity) == (
        tmp_path / ".staging" / "unitree%2Fsource" / ("a" * 40) / "episode-7" / ("c" * 64)
    )
    with pytest.raises(ValueError, match="source_repo_id"):
        replace(identity, source_repo_id="../escape")
    with pytest.raises(ValueError, match="source file"):
        replace(identity, source_file_sha256={"../escape": "b" * 64})


def test_write_stage_preserves_exact_arrays_and_resumes_matching_identity(tmp_path: Path) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))

    assert result.path == stage_path(tmp_path, identity)
    assert result.reused is False
    assert can_resume(result.path, identity) is True
    assert can_resume(result.path, replace(identity, encoder_sha256="0" * 64)) is False
    resumed = write_stage(tmp_path, identity, _payload(tmp_path, episode_id=8))
    assert resumed.path == result.path
    assert resumed.reused is True

    rows = load_stage_rows(result.path)
    assert tuple(rows[0]) == tuple(TARGET_FRAME_SCHEMA)
    for key, spec in TARGET_FRAME_SCHEMA.items():
        value = rows[1][key]
        if key == "task":
            assert value == "place"
        elif key == "timestamp":
            assert isinstance(value, np.float32)
            assert value.tobytes() == np.float32(1 / 50).tobytes()
        else:
            assert isinstance(value, np.ndarray)
            assert value.dtype == spec.dtype
            assert value.shape == spec.shape


def test_path_video_stage_never_uses_whole_file_secure_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    payload = _payload(tmp_path)
    real_secure_read = staging_module._secure_read

    def reject_mp4(path: Path) -> bytes:
        if path.suffix == ".mp4":
            raise AssertionError("whole MP4 read")
        return real_secure_read(path)

    monkeypatch.setattr(staging_module, "_secure_read", reject_mp4)
    result = write_stage(tmp_path, identity, payload)
    assert can_resume(result.path, identity)
    episode = staging_module._stage_input(result)
    frames = list(staging_module._iter_staged_video(next(iter(episode.videos.values()))))
    assert len(frames) == 2


@pytest.mark.parametrize(
    "artifact",
    (
        "manifest.json",
        "validation.json",
        "frame-data.parquet",
        "videos/observation.images.ego_view.mp4",
        "checksums.sha256",
    ),
)
def test_resume_rejects_corruption_in_every_stage_artifact(tmp_path: Path, artifact: str) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))
    target = result.path / artifact
    target.write_bytes(target.read_bytes() + b"corruption")

    assert can_resume(result.path, identity) is False
    with pytest.raises(FileExistsError, match="immutable stage"):
        write_stage(tmp_path, identity, _payload(tmp_path, episode_id=8))


def test_resume_rejects_unexpected_artifact_and_symlink(tmp_path: Path) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))
    (result.path / "unexpected").write_text("no")
    assert can_resume(result.path, identity) is False

    result2 = write_stage(tmp_path, replace(identity, source_episode_id=8), _payload(tmp_path, episode_id=8))
    (result2.path / "videos" / "observation.images.ego_view.mp4").unlink()
    (result2.path / "videos" / "observation.images.ego_view.mp4").symlink_to(tmp_path / "source-8.mp4")
    assert can_resume(result2.path, replace(identity, source_episode_id=8)) is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda rows: rows[:1], "video frame count"),
        (lambda rows: [{**rows[0], "task": ""}, rows[1]], "task"),
        (
            lambda rows: [rows[0], {**rows[1], "timestamp": np.float32(9.0)}],
            "timestamp",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "action.wbc": np.zeros((42,), dtype=np.float64)},
            ],
            "action.wbc",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "action.motion_token": np.full((64,), np.nan, dtype=np.float64)},
            ],
            "finite",
        ),
    ),
)
def test_write_stage_rejects_invalid_rows_or_video_counts_and_cleans_temp(
    tmp_path: Path, mutator, message: str
) -> None:
    payload = _payload(tmp_path)

    with pytest.raises(ValueError, match=message):
        invalid = StagePayload(rows=tuple(mutator(list(payload.rows))), videos=payload.videos)
        write_stage(tmp_path, _identity(), invalid)

    final = stage_path(tmp_path, _identity())
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


def test_stage_rejects_object_values_pickle_files_and_unsafe_video_targets(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    bad_row = dict(payload.rows[0])
    bad_row["action.wbc"] = np.array([object()] * 43, dtype=object)
    with pytest.raises(ValueError, match="action.wbc"):
        StagePayload(rows=(bad_row, payload.rows[1]), videos=payload.videos)
    with pytest.raises(ValueError, match="video target"):
        StagePayload(rows=payload.rows, videos={"../evil.pkl": b"pickle"})
    with pytest.raises(ValueError, match="unsupported"):
        StagePayload(rows=payload.rows, videos={"observation.images.unknown": b"video"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("teleop.planner_speed", np.array([0.0], dtype=np.float32), "neutral"),
        (
            "observation.root_orientation",
            np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "unit WXYZ",
        ),
        (
            "observation.projected_gravity",
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "inverse root rotation",
        ),
    ),
)
def test_stage_payload_rejects_corrupt_neutral_or_root_semantics(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1][field] = value
    with pytest.raises(ValueError, match=message):
        StagePayload(
            rows=tuple(rows),
            videos={"observation.images.ego_view": _payload(tmp_path).videos["observation.images.ego_view"]},
        )


def test_asset_free_target_kinematics_exposes_exact_43_joint_limits_and_fk() -> None:
    lower, upper = target_joint_limits()
    assert lower.shape == upper.shape == (43,)
    assert lower.dtype == upper.dtype == np.dtype(np.float64)
    assert np.all(lower <= 0.0)
    assert np.all(upper >= 0.0)
    eef = regenerate_eef_state(np.zeros(43, dtype=np.float64))
    assert eef.shape == (14,)
    assert np.isfinite(eef).all()
    assert np.linalg.norm(eef[3:7]) == pytest.approx(1.0)
    assert np.linalg.norm(eef[10:14]) == pytest.approx(1.0)


@pytest.mark.parametrize("field", ("observation.state", "action.wbc"))
def test_stage_payload_rejects_finite_joint_target_outside_urdf_limits(tmp_path: Path, field: str) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1][field] = rows[1][field].copy()
    rows[1][field][0] = 1e100
    with pytest.raises(ValueError, match=f"{re.escape(field)}.*joint limits"):
        StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)


def test_stage_payload_rejects_finite_but_fk_incorrect_eef_translation(tmp_path: Path) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1]["observation.eef_state"] = rows[1]["observation.eef_state"].copy()
    rows[1]["observation.eef_state"][0] += 0.01
    with pytest.raises(ValueError, match="observation.eef_state.*forward kinematics"):
        StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)


@pytest.mark.parametrize(
    ("fps", "size", "message"),
    ((30, (640, 480), "fps 50"), (50, (320, 240), "frame size")),
)
def test_stage_rejects_wrong_target_video_fps_or_dimensions(
    tmp_path: Path, fps: int, size: tuple[int, int], message: str
) -> None:
    video = tmp_path / "wrong.mp4"
    _write_video(video, fps=fps, size=size)
    payload = StagePayload(
        rows=(_row(0, "task"), _row(1, "task")),
        videos={"observation.images.ego_view": video},
    )
    with pytest.raises(ValueError, match=message):
        write_stage(tmp_path, _identity(), payload)


def test_stage_write_failure_cleans_unique_sibling_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    payload = _payload(tmp_path)
    real_write = staging_module._write_file
    calls = 0

    def fail_second(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        real_write(path, data)

    monkeypatch.setattr(staging_module, "_write_file", fail_second)
    with pytest.raises(OSError, match="simulated"):
        write_stage(tmp_path, identity, payload)
    final = stage_path(tmp_path, identity)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


def test_atomic_publication_fails_closed_when_rename_noreplace_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()

    def unavailable(source: Path, destination: Path) -> None:
        raise OSError("RENAME_NOREPLACE unavailable")

    monkeypatch.setattr(staging_module, "_rename_noreplace", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        write_stage(tmp_path, identity, _payload(tmp_path))
    final = stage_path(tmp_path, identity)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


class RecordingMergeBackend:
    def __init__(self) -> None:
        self.orders: list[tuple[int, ...]] = []
        self.task_catalogs: list[tuple[str, ...]] = []

    def __call__(self, output: Path, episodes, task_catalog: tuple[str, ...]) -> MergeVerification:
        output.mkdir()
        episode_ids = tuple(episode.identity.source_episode_id for episode in episodes)
        self.orders.append(episode_ids)
        self.task_catalogs.append(task_catalog)
        for episode in episodes:
            assert not hasattr(episode, "rows")
            assert len(tuple(staging_module.iter_stage_rows(episode, batch_size=1))) == episode.row_count
        (output / "backend.json").write_text(json.dumps({"episodes": episode_ids}))
        return self.validate(output, episodes, task_catalog)

    def validate(self, output, episodes, task_catalog) -> MergeVerification:
        return _exact_verification(episodes, task_catalog)


def _exact_verification(episodes, task_catalog: tuple[str, ...]) -> MergeVerification:
    return staging_module._expected_merge_verification(episodes, task_catalog)


def _two_stages(tmp_path: Path):
    first = write_stage(
        tmp_path,
        _identity(10),
        _payload(tmp_path, 10, tasks=("zeta", "alpha")),
    )
    second = write_stage(
        tmp_path,
        _identity(2),
        _payload(tmp_path, 2, tasks=("beta", "alpha")),
    )
    return first, second


def test_merge_is_deterministic_and_preregisters_lexicographic_tasks(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    first = merge_stages(
        reversed(stages), tmp_path / "first", merge_backend=backend, final_validator=backend.validate
    )
    second = merge_stages(stages, tmp_path / "second", merge_backend=backend, final_validator=backend.validate)

    assert backend.orders == [(2, 10), (2, 10)]
    assert backend.task_catalogs == [("alpha", "beta", "zeta")] * 2
    assert dataset_manifest_digest(first) == dataset_manifest_digest(second)
    manifest = json.loads((first / "source-manifest.json").read_text())
    assert manifest["source_episode_ids"] == [2, 10]
    assert manifest["task_catalog"] == ["alpha", "beta", "zeta"]


def test_default_merge_streams_decoded_video_frames_through_lazy_gr00t_exporter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = _two_stages(tmp_path)
    calls: list[tuple[str, object]] = []

    class FakeExporter:
        def add_frame(self, frame):
            calls.append(("frame", frame))

        def save_episode(self):
            calls.append(("save", None))

        def stop_video_writers(self):
            calls.append(("stop", None))

    def create(output, video_keys, task_catalog):
        output.mkdir()
        calls.append(("create", (tuple(video_keys), task_catalog)))
        return FakeExporter()

    def decode(artifact):
        calls.append(("decode", artifact.path.name))
        yield np.zeros((480, 640, 3), dtype=np.uint8)
        yield np.ones((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(staging_module, "_create_gr00t_exporter", create)
    monkeypatch.setattr(staging_module, "_iter_staged_video", decode)
    monkeypatch.setattr(
        staging_module,
        "_validate_gr00t_output",
        lambda output, episodes, tasks: _exact_verification(episodes, tasks),
    )

    merge_stages(stages, tmp_path / "default")

    assert calls[0] == (
        "create",
        (("observation.images.ego_view",), ("alpha", "beta", "zeta")),
    )
    frames = [value for event, value in calls if event == "frame"]
    assert len(frames) == 4
    assert all("observation.images.ego_view" in frame for frame in frames)
    assert all(frame["timestamp"].shape == (1,) for frame in frames)
    assert all(frame["timestamp"].dtype == np.dtype(np.float32) for frame in frames)
    assert [event for event, _ in calls].count("save") == 2
    assert [event for event, _ in calls].count("stop") == 1


def test_real_gr00t_exporter_adapts_timestamp_vector_to_exact_scalar_parquet(tmp_path: Path) -> None:
    output = tmp_path / "real-exporter"
    exporter = Gr00tDataExporter.create(
        save_root=output,
        fps=50,
        features={
            "observation.state": {
                "dtype": "float64",
                "shape": (2,),
                "names": ["a", "b"],
            }
        },
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
        task="task",
        script_config={},
    )
    for index in range(2):
        exporter.add_frame(
            {
                "observation.state": np.array([index, index + 1], dtype=np.float64),
                "timestamp": np.array([np.float32(index / 50.0)], dtype=np.float32),
                "task": "task",
            }
        )
        assert isinstance(exporter.episode_buffer["timestamp"][-1], np.float32)

    exporter.save_episode()

    table = pq.read_table(output / exporter.meta.get_data_file_path(0))
    timestamp = table.column("timestamp").combine_chunks()
    assert timestamp.type == pa.float32()
    assert timestamp.null_count == 0
    assert timestamp.to_numpy().tobytes() == np.array([0.0, 0.02], dtype=np.float32).tobytes()


def test_merge_rejects_mixed_repositories_duplicate_ids_and_conversion_locks(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    mixed = write_stage(
        tmp_path,
        _identity(20, repo="different/source"),
        _payload(tmp_path, 20),
    )
    backend = RecordingMergeBackend()
    with pytest.raises(ValueError, match="source repository"):
        merge_stages(
            (stages[0], mixed),
            tmp_path / "mixed",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    with pytest.raises(ValueError, match="duplicate source episode"):
        merge_stages(
            (stages[0], stages[0]),
            tmp_path / "duplicate",
            merge_backend=backend,
            final_validator=backend.validate,
        )

    changed_lock = write_stage(
        tmp_path,
        replace(_identity(21), encoder_sha256="1" * 64),
        _payload(tmp_path, 21),
    )
    with pytest.raises(ValueError, match="conversion identity"):
        merge_stages(
            (stages[0], changed_lock),
            tmp_path / "changed",
            merge_backend=backend,
            final_validator=backend.validate,
        )


def test_merge_rejects_output_inside_stage_or_staging_namespace_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    with pytest.raises(ValueError, match="input staging ancestry"):
        merge_stages(
            stages,
            stages[0].path / "nested-output",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    with pytest.raises(ValueError, match="input staging ancestry"):
        merge_stages(
            stages,
            tmp_path / ".staging" / "unrelated-output",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    assert all(can_resume(stage.path, stage.identity) for stage in stages)


def test_merge_revalidates_stages_and_never_overwrites_existing_output(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    output = merge_stages(stages, tmp_path / "dataset", merge_backend=backend, final_validator=backend.validate)
    assert (
        merge_stages(
            stages,
            output,
            merge_backend=backend,
            final_validator=backend.validate,
            resume_existing=True,
        )
        == output
    )

    (output / "backend.json").write_text("corrupt")
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(
            stages,
            output,
            merge_backend=backend,
            final_validator=backend.validate,
            resume_existing=True,
        )
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(stages, output, merge_backend=backend, final_validator=backend.validate)

    (stages[0].path / "frame-data.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="stage is not complete"):
        merge_stages(
            stages,
            tmp_path / "bad-stage",
            merge_backend=backend,
            final_validator=backend.validate,
        )


def test_merge_failure_cleans_sibling_temp_and_leaves_output_absent(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)

    def fail(output, episodes, tasks):
        output.mkdir()
        (output / "partial").write_text("partial")
        raise RuntimeError("boom")

    final = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="boom"):
        merge_stages(
            stages,
            final,
            merge_backend=fail,
            final_validator=lambda output, episodes, tasks: _exact_verification(episodes, tasks),
        )
    assert not final.exists()
    assert not list(tmp_path.glob(".dataset.*"))


def test_merge_rejects_independent_final_validation_mismatch_and_cleans_temp(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)

    def backend(output, episodes, tasks):
        output.mkdir()

    def wrong_validator(output, episodes, tasks):
        report = _exact_verification(episodes, tasks)
        return replace(report, frame_content_sha256="0" * 64)

    final = tmp_path / "wrong-verification"
    with pytest.raises(ValueError, match="verification differs"):
        merge_stages(
            stages,
            final,
            merge_backend=backend,
            final_validator=wrong_validator,
        )
    assert not final.exists()
    assert not list(tmp_path.glob(".wrong-verification.*"))


def test_injected_backend_requires_explicit_independent_final_validator(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    with pytest.raises(ValueError, match="independent final_validator"):
        merge_stages(stages, tmp_path / "unsafe", merge_backend=RecordingMergeBackend())


def test_production_merge_reopens_and_validates_real_gr00t_output(tmp_path: Path) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    output = merge_stages((stage,), tmp_path / "real-dataset")

    assert output.is_dir()
    assert len(dataset_manifest_digest(output)) == 64
    validation = json.loads((output / "merge-validation.json").read_text())
    assert validation["total_frames"] == 2
    assert validation["source_episode_ids"] == [3]


def test_independent_validator_rejects_corrupt_nonvideo_despite_consistent_backend_report(
    tmp_path: Path,
) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    def corrupting_backend(output, episodes, tasks):
        staging_module._default_merge_backend(output, episodes, tasks)
        from gear_sonic.data.exporter import Gr00tDatasetMetadata

        meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
        parquet_path = output / meta.get_data_file_path(0)
        table = pq.read_table(parquet_path)
        column = table.column("observation.state").combine_chunks()
        values = column.values.to_numpy(zero_copy_only=False).copy()
        values[0] += 1e-6
        changed = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float64()), 43)
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", changed)
        pq.write_table(table, parquet_path)
        return _exact_verification(episodes, tasks)

    final = tmp_path / "corrupt-dataset"
    with pytest.raises(ValueError, match="nonvideo value differs"):
        merge_stages(
            (stage,),
            final,
            merge_backend=corrupting_backend,
            final_validator=staging_module._validate_gr00t_output,
        )
    assert not final.exists()


def test_independent_validator_rejects_extra_global_task_catalog_entry(tmp_path: Path) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    def extra_task_backend(output, episodes, tasks):
        staging_module._default_merge_backend(output, episodes, tasks)
        from gear_sonic.data.exporter import Gr00tDatasetMetadata

        meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
        meta.add_task("unexpected extra task")

    with pytest.raises(ValueError, match="global task catalog"):
        merge_stages(
            (stage,),
            tmp_path / "extra-task",
            merge_backend=extra_task_backend,
            final_validator=staging_module._validate_gr00t_output,
        )


def test_empty_merge_and_output_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        merge_stages((), tmp_path / "empty", merge_backend=RecordingMergeBackend())
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(_two_stages(tmp_path), link, merge_backend=RecordingMergeBackend())
