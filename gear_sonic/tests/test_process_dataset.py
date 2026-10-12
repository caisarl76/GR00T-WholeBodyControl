import json

import pytest


def _mixed_dataset(tmp_path, include_modes=True):
    import numpy as np
    import pandas as pd

    root = tmp_path / "recording"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    modes = [1, 1, 1, 2, 2, 3, 5, 4, 0, 99, 1, 1]
    poses = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    frame = pd.DataFrame({
        "episode_index": [0] * len(modes),
        "frame_index": range(len(modes)),
        "index": range(len(modes)),
        "timestamp": np.arange(len(modes)) / 50,
        "teleop.smpl_pose": [np.full(63, x, dtype=np.float32) for x in poses],
        "action.motion_token": [np.full(64, x) for x in range(len(modes))],
        "teleop.left_hand_joints": [np.full(7, x / 20) for x in range(len(modes))],
    })
    if include_modes:
        frame["teleop.stream_mode"] = [np.array([x]) for x in modes]
    frame.to_parquet(root / "data/chunk-000/episode_000000.parquet")
    (root / "meta/info.json").write_text(json.dumps({"fps": 50, "features": {}}))
    (root / "meta/episodes.jsonl").write_text(json.dumps({
        "episode_index": 0, "length": len(modes), "tasks": ["walk and grasp"],
    }) + "\n")
    return root, frame


def test_cleaning_preserves_planner_pause_and_stationary_pose(tmp_path):
    import numpy as np

    from gear_sonic.scripts.process_dataset import process_single_dataset

    root, original = _mixed_dataset(tmp_path)
    stats, episodes, _ = process_single_dataset(root, remove_stale_smpl=True)
    kept = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11]
    assert episodes[0]["valid_indices"].tolist() == kept
    assert stats["frames_removed"] == stats["zero_frames"] == 2
    assert stats["frozen_leadin_frames"] == 0
    for key in ("action.motion_token", "teleop.left_hand_joints"):
        np.testing.assert_array_equal(
            np.stack(episodes[0]["df"][key]), np.stack(original.iloc[kept][key]),
        )


def test_cleaning_requires_mode_labels_unless_legacy_explicit(tmp_path):
    from gear_sonic.scripts.process_dataset import process_single_dataset

    root, _ = _mixed_dataset(tmp_path, include_modes=False)
    with pytest.raises(ValueError, match="teleop.stream_mode"):
        process_single_dataset(root, remove_stale_smpl=True)
    stats, _, _ = process_single_dataset(root, remove_stale_smpl=True, stale_smpl_policy="legacy")
    assert stats["frames_removed"] == 11
    stats, _, _ = process_single_dataset(root, remove_stale_smpl=False)
    assert stats["frames_removed"] == 0


def test_dry_run_does_not_rewrite_source_or_create_output(tmp_path, capsys):
    from gear_sonic.scripts.process_dataset import ProcessDatasetConfig, main

    root, _ = _mixed_dataset(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "cleaned"
    main(ProcessDatasetConfig(dataset_path=[str(root)], output_path=str(output), dry_run=True))
    assert not output.exists()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert "2 removed" in capsys.readouterr().out


@pytest.mark.parametrize("modes", [[[1, 2]], [[float("nan")]], [[1.5]]])
def test_invalid_mode_labels_are_rejected(modes):
    import numpy as np

    from gear_sonic.scripts.process_dataset import build_stale_mask

    with pytest.raises(ValueError, match="teleop.stream_mode"):
        build_stale_mask(np.zeros((1, 63)), np.array(modes))


def test_cleaned_copy_keeps_video_token_and_hand_alignment(tmp_path):
    import av
    import numpy as np
    import pandas as pd

    from gear_sonic.scripts.process_dataset import ProcessDatasetConfig, main

    root, original = _mixed_dataset(tmp_path)
    key = "observation.images.ego_view"
    info = json.loads((root / "meta/info.json").read_text())
    info["features"][key] = {"dtype": "video"}
    (root / "meta/info.json").write_text(json.dumps(info))
    video = root / "videos" / key / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("h264", rate=50)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(len(original)):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "cleaned"
    main(ProcessDatasetConfig(dataset_path=[str(root)], output_path=str(output)))
    kept = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11]
    cleaned = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
    assert cleaned["frame_index"].tolist() == list(range(len(kept)))
    np.testing.assert_allclose(cleaned["timestamp"], np.arange(len(kept)) / 50)
    for key in ("action.motion_token", "teleop.left_hand_joints", "teleop.stream_mode"):
        np.testing.assert_array_equal(np.stack(cleaned[key]), np.stack(original.iloc[kept][key]))
    with av.open(str(output / video.relative_to(root))) as container:
        intensities = [frame.to_ndarray(format="rgb24").mean() for frame in container.decode(video=0)]
    np.testing.assert_allclose(intensities, np.array(kept) * 20, atol=4)
    assert json.loads((output / "meta/info.json").read_text())["total_frames"] == len(kept)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_write_output_dataset_remaps_task_indices_by_task_text(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("av")
    pytest.importorskip("tyro")
    from gear_sonic.scripts.process_dataset import write_output_dataset

    episodes = [
        {
            "df": pd.DataFrame(
                {
                    "episode_index": [10, 10],
                    "index": [0, 1],
                    "frame_index": [0, 1],
                    "timestamp": [0.0, 0.02],
                    "task_index": [0, 0],
                }
            ),
            "source_video_paths": {},
            "valid_indices": None,
            "episode_meta": {"episode_index": 10, "tasks": ["point blue"], "length": 2},
            "fps": 50,
        },
        {
            "df": pd.DataFrame(
                {
                    "episode_index": [0, 0, 0],
                    "index": [0, 1, 2],
                    "frame_index": [0, 1, 2],
                    "timestamp": [0.0, 0.02, 0.04],
                    "task_index": [0, 0, 0],
                }
            ),
            "source_video_paths": {},
            "valid_indices": None,
            "episode_meta": {"episode_index": 0, "tasks": ["point green"], "length": 3},
            "fps": 50,
        },
    ]
    info = {
        "fps": 50,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"task_index": {"dtype": "int64", "shape": [1], "names": None}},
    }

    write_output_dataset(
        tmp_path / "merged",
        episodes,
        info,
        [
            {"task_index": 0, "task": "point blue"},
            {"task_index": 0, "task": "point green"},
        ],
        script_config=None,
    )

    tasks = [
        json.loads(line)
        for line in (tmp_path / "merged/meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first = pd.read_parquet(tmp_path / "merged/data/chunk-000/episode_000000.parquet")
    second = pd.read_parquet(tmp_path / "merged/data/chunk-000/episode_000001.parquet")
    merged_info = json.loads((tmp_path / "merged/meta/info.json").read_text(encoding="utf-8"))

    assert tasks == [
        {"task_index": 0, "task": "point blue"},
        {"task_index": 1, "task": "point green"},
    ]
    assert first["task_index"].tolist() == [0, 0]
    assert second["task_index"].tolist() == [1, 1, 1]
    assert merged_info["total_tasks"] == 2


def test_freshness_routes_modes_and_inclusive_boundaries():
    import numpy as np
    import pandas as pd

    from gear_sonic.scripts.process_dataset import build_stale_input_mask

    modes = np.array([[1], [1], [2], [2], [3], [4], [5], [0]])
    df = pd.DataFrame({
        "teleop.stream_mode": list(modes),
        "teleop.pose_receive_age_ms": [[100.0], [100.1], [999.0], [999.0], [999.0], [999.0], [999.0], [999.0]],
        "teleop.planner_receive_age_ms": [[999.0], [999.0], [200.0], [200.1], [200.1], [999.0], [200.0], [999.0]],
    })
    np.testing.assert_array_equal(build_stale_input_mask(df, True, 100, 200), [0, 1, 0, 1, 1, 0, 0, 0])


@pytest.mark.parametrize("mode", [1, 2, 3, 5])
def test_freshness_rejects_invalid_values_and_shapes(mode):
    import pandas as pd

    from gear_sonic.scripts.process_dataset import build_stale_input_mask

    column = "teleop.pose_receive_age_ms" if mode == 1 else "teleop.planner_receive_age_ms"
    for value in (-1.0, float("nan"), float("inf"), None):
        df = pd.DataFrame({"teleop.stream_mode": [[mode]], column: [[value]]})
        assert build_stale_input_mask(df, True, 100, 200).tolist() == [True]
    bad = pd.DataFrame({"teleop.stream_mode": [[[1]]], "teleop.pose_receive_age_ms": [[0.0]]})
    with pytest.raises(ValueError, match="stream_mode"):
        build_stale_input_mask(bad, True, 100, 200)


def test_freshness_missing_columns_and_disable_preserve():
    import pandas as pd

    from gear_sonic.scripts.process_dataset import build_stale_input_mask

    df = pd.DataFrame({"teleop.stream_mode": [[1], [2]], "teleop.pose_receive_age_ms": [[500.0], [500.0]]})
    assert build_stale_input_mask(df, True, 100, 200).tolist() == [True, False]
    no_ages = df.drop(columns=["teleop.pose_receive_age_ms"])
    assert build_stale_input_mask(no_ages, True, 100, 200).tolist() == [False, False]
    assert build_stale_input_mask(df, False, 100, 200).tolist() == [False, False]


def test_freshness_empty_frames_and_invalid_age_shape():
    import pandas as pd

    from gear_sonic.scripts.process_dataset import build_stale_input_mask

    assert build_stale_input_mask(pd.DataFrame(), True, 100, 200).size == 0
    df = pd.DataFrame({"teleop.stream_mode": [[2]], "teleop.planner_receive_age_ms": [[0, 1]]})
    with pytest.raises(ValueError, match="scalar or shape"):
        build_stale_input_mask(df, True, 100, 200)
    df = pd.DataFrame({"teleop.planner_receive_age_ms": [0]})
    with pytest.raises(ValueError, match="stream_mode"):
        build_stale_input_mask(df, True, 100, 200)


@pytest.mark.parametrize("parameter", ["pose_max_age_ms", "planner_max_age_ms"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_age_threshold_is_rejected(tmp_path, parameter, value):
    from gear_sonic.scripts.process_dataset import process_single_dataset

    with pytest.raises(ValueError, match=parameter):
        process_single_dataset(tmp_path, False, **{parameter: value})


def test_freshness_only_cleanup_and_exclusive_counts(tmp_path):
    import numpy as np

    from gear_sonic.scripts.process_dataset import process_single_dataset

    root, original = _mixed_dataset(tmp_path)
    original["teleop.pose_receive_age_ms"] = [np.array([500 if i == 2 else 0]) for i in range(12)]
    original["teleop.planner_receive_age_ms"] = [np.array([500 if i == 3 else 0]) for i in range(12)]
    original.to_parquet(root / "data/chunk-000/episode_000000.parquet")
    stats, episodes, _ = process_single_dataset(root, True)
    assert stats["frames_removed"] == 3
    assert stats["zero_frames"] == 2
    assert stats["stale_input_frames"] == 1
    assert 3 not in episodes[0]["valid_indices"]
    stats, episodes, _ = process_single_dataset(root, False)
    assert stats["frames_removed"] == stats["stale_input_frames"] == 2
    assert 10 in episodes[0]["valid_indices"]  # Zero SMPL alone is now ignored.
    stats, _, _ = process_single_dataset(root, False, remove_stale_input=False)
    assert stats["frames_removed"] == 0


def test_merge_rejects_different_features_even_with_same_directory_names(tmp_path):
    from gear_sonic.scripts.process_dataset import validate_script_configs

    roots = [tmp_path / prefix / "recording" for prefix in ("old", "new")]
    for root in roots:
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text(json.dumps({"features": {}, "script_config": {}}))
    assert validate_script_configs(roots) == {}
    (roots[1] / "meta/info.json").write_text(json.dumps({
        "features": {"teleop.pose_receive_age_ms": {"dtype": "float64", "shape": [1]}},
        "script_config": {},
    }))
    with pytest.raises(ValueError, match="feature schemas"):
        validate_script_configs(roots)


def test_native_inspire_fields_survive_cleaning_without_q7_relabeling(tmp_path):
    import numpy as np
    import pandas as pd

    from gear_sonic.data.pico_hand_features import hand_episode_features
    from gear_sonic.scripts.process_dataset import process_single_dataset

    root, original = _mixed_dataset(tmp_path)
    original = original.drop(columns=["teleop.left_hand_joints"])
    for key, feature in hand_episode_features("inspire_ftp").items():
        original[key] = [np.full(feature["shape"], i, dtype=feature["dtype"]) for i in range(len(original))]
    path = root / "data/chunk-000/episode_000000.parquet"
    original.to_parquet(path)
    _, episodes, _ = process_single_dataset(root, True)
    kept = episodes[0]["valid_indices"]
    cleaned = episodes[0]["df"]
    assert "teleop.left_hand_joints" not in cleaned
    for key in hand_episode_features("inspire_ftp"):
        np.testing.assert_array_equal(np.stack(cleaned[key]), np.stack(original.iloc[kept][key]))
    # Cleaning never modifies the recorded source in its processing pass.
    assert pd.read_parquet(path).shape == original.shape


def test_discarded_episode_filter_remains_available_with_mode_aware_cleaning(tmp_path):
    from gear_sonic.scripts.process_dataset import process_single_dataset

    root, _ = _mixed_dataset(tmp_path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["discarded_episode_indices"] = [0]
    info_path.write_text(json.dumps(info))
    stats, episodes, _ = process_single_dataset(root, True, remove_discarded=True)
    assert stats["episodes_discarded"] == 1
    assert episodes == []
    stats, episodes, _ = process_single_dataset(root, True, remove_discarded=False)
    assert stats["episodes_discarded"] == 0
    assert len(episodes[0]["df"]) == 10


@pytest.mark.parametrize("old_count,new_count", [(43, 41), (41, 43)])
def test_merge_rejects_incompatible_native_and_controller_joint_schemas(tmp_path, old_count, new_count):
    from gear_sonic.scripts.process_dataset import validate_script_configs

    roots = [tmp_path / name for name in ("controller", "optical")]
    for root, count in zip(roots, (old_count, new_count)):
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text(json.dumps({
            "features": {"observation.state": {"dtype": "float64", "shape": [count]}},
            "script_config": {"hand_profile": "inspire_ftp"},
        }))
    with pytest.raises(ValueError, match="feature schemas"):
        validate_script_configs(roots)
