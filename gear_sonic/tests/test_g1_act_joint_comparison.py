"""Regression coverage for reference transport and asynchronous ACT alignment."""

import json

import numpy as np
import pytest

from gear_sonic.scripts.compare_g1_act_sonic_joints import _metric, load_comparison
from gear_sonic.utils.inference.reference_adapter.g1_act import ACTION_NAMES, BODY_NAMES

# Independently written Isaac ordering, intentionally not the production constant.
ISAAC_MOTOR_INDICES = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
RIGHT_RUNTIME = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def run(tmp_path):
    replay, sim = tmp_path / "replay", tmp_path / "sim"
    replay.mkdir()
    sim.mkdir()
    frames = np.arange(12)[:, None]
    body = frames + np.arange(29)[None, :] / 100
    left = frames + np.arange(7)[None, :] / 10 + 10
    right = frames + np.arange(7)[None, :] / 10 + 20
    actions = np.arange(3)[:, None] * 2 + np.arange(28)[None, :] / 100
    np.save(replay / "actions.npy", actions)
    np.savez(
        replay / "reference.npz",
        joint_pos=body[:, ISAAC_MOTOR_INDICES],
        left_hand_joints=left,
        right_hand_joints=right,
    )
    (replay / "composition.json").write_text(
        json.dumps(
            {
                "source_hz": 25,
                "reference_hz": 50,
                "initial_transition_seconds": 0.1,
                "source_action_names": list(ACTION_NAMES),
            }
        )
    )
    (sim / "joint_contract.json").write_text(
        json.dumps(
            {
                group: [{"name": name, "range": [-100, 100]} for name in names]
                for group, names in (("body", BODY_NAMES), ("left", ACTION_NAMES[14:21]), ("right", RIGHT_RUNTIME))
            }
        )
    )

    def native(ms, frame, chunk=2, session=1):
        return {
            "monotonic_ns": ms * 1_000_000,
            "mode": "active",
            "session_id": session,
            "chunk_id": chunk,
            "reference_frame": frame,
            "body_reference_isaac": body[frame, ISAAC_MOTOR_INDICES].tolist(),
            "left_hand_reference": left[frame].tolist(),
            "right_hand_reference": right[frame].tolist(),
        }

    rows = [
        native(100, 4),
        native(120, 5),
        {"monotonic_ns": 125_000_000, "mode": "inactive"},
        native(130, 5, chunk=1),
        native(140, 7),
        native(200, 10),
        native(220, 9),
        native(230, 6, chunk=3, session=2),
        native(240, 8),
    ]
    obs = [
        {
            "monotonic_ns": ms * 1_000_000,
            "body_q": (body[0] + 2).tolist(),
            "body_command_target": (body[0] + 1).tolist() + [0] * 6,
            "left_hand_q": (left[0] + 2).tolist(),
            "right_hand_q": (right[0] + 2).tolist(),
            "left_command_target": (left[0] + 1).tolist(),
            "right_command_target": (right[0] + 1).tolist(),
        }
        for ms in (90, 110, 120, 126, 131, 145, 181, 201, 225, 235, 245)
    ]
    obs.append({"event": "keyboard", "key": "9", "monotonic_ns": 115_000_000})
    # Both files are deliberately unsorted.
    write_jsonl(tmp_path / "native.jsonl", rows[::-1])
    write_jsonl(sim / "observations.jsonl", obs[::-1])
    return tmp_path


def test_latest_prior_all_streams_source_timeline_and_name_orders(run):
    result = load_comparison(run)
    np.testing.assert_array_equal(result["monotonic_ns"], np.array([120, 145, 225, 245]) * 1_000_000)
    np.testing.assert_array_equal(result["reference_frames"], [5, 7, 9, 8])
    np.testing.assert_allclose(result["source_time_s"], [0, 0.04, 0.08, 0.06], atol=1e-15)
    # Correct interpolation includes entry only once and uses actual native frame.
    np.testing.assert_allclose(result["act_raw"][:, 0], [0, 2, 4, 3])
    np.testing.assert_allclose(result["saved_reference"][:, 0], [5.15, 7.15, 9.15, 8.15])
    np.testing.assert_allclose(result["saved_reference"][:, 7], [5.22, 7.22, 9.22, 8.22])
    np.testing.assert_allclose(result["native_reference"], result["saved_reference"])
    # ACT right index precedes middle, opposite to runtime arrays; left does not.
    np.testing.assert_allclose(result["saved_reference"][0, 21:], [25, 25.1, 25.2, 25.5, 25.6, 25.3, 25.4])
    np.testing.assert_allclose(result["issued_command"][0, 21:], [21, 21.1, 21.2, 21.5, 21.6, 21.3, 21.4])
    np.testing.assert_allclose(result["measured"][0, 21:], [22, 22.1, 22.2, 22.5, 22.6, 22.3, 22.4])
    np.testing.assert_allclose(result["issued_command"][0, 14:21], np.arange(7) / 10 + 11)
    np.testing.assert_allclose(result["measured"][0, :14], np.arange(15, 29) / 100 + 2)
    assert result["summary"]["alignment"]["excluded_observations"] == {
        "before_first_native": 1,
        "inactive_or_other_session_chunk": 3,
        "stale": 1,
        "outside_act_segment": 2,
    }
    assert result["summary"]["alignment"]["median_age_ms"] == 5
    assert result["summary"]["transport_reference"]["active_rows_checked"] == 6
    assert result["summary"]["transport_reference"]["passed"]
    assert len(result["summary"]["inputs"]) == 6
    assert all(len(digest) == 64 for digest in result["summary"]["inputs"].values())


def test_metrics_have_y_minus_x_bias_and_correct_units():
    metrics = _metric(np.array([1.0, 3.0]), np.array([4.0, 2.0]))
    assert metrics["mean_bias_rad"] == 1
    assert metrics["maxabs_rad"] == 3
    assert metrics["rms_rad"] == pytest.approx(np.sqrt(5))
    assert metrics["rms_deg"] == pytest.approx(np.rad2deg(np.sqrt(5)))
    assert metrics["mean_bias_deg"] == pytest.approx(np.rad2deg(1))


def test_transport_checks_unselected_transition_and_lower_body(run):
    path = run / "native.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transition = next(row for row in rows if row.get("reference_frame") == 4)
    transition["body_reference_isaac"][0] += 0.25
    write_jsonl(path, rows)
    result = load_comparison(run)
    check = result["summary"]["transport_reference"]
    assert not check["passed"]
    assert check["maxabs_rad"] == 0.25
    assert check["values_over_atol"] == 1
    # Selected samples still match: checking only selected ACT joints would miss this.
    np.testing.assert_array_equal(result["native_reference"], result["saved_reference"])


def test_transport_checks_saved_actual_frame(run):
    path = run / "native.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row.get("reference_frame") == 7)["reference_frame"] = 6
    write_jsonl(path, rows)
    check = load_comparison(run)["summary"]["transport_reference"]
    assert not check["passed"]
    assert check["values_over_atol"] == 43


def test_ambiguous_session_requires_selection(run):
    path = run / "native.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row.get("session_id") == 2)["chunk_id"] = 2
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="ambiguous"):
        load_comparison(run)
    assert load_comparison(run, session_id=1)["summary"]["scope"]["samples"] == 4


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("missing_timestamp", "monotonic_ns"),
        ("float_timestamp", "monotonic_ns"),
        ("duplicate_timestamp", "duplicate timestamps"),
        ("bad_shape", "expected shape"),
        ("nonfinite", "nonfinite"),
        ("missing_joint", "expected shape"),
    ],
)
def test_invalid_observation_has_clear_error(run, kind, expected):
    path = run / "sim/observations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if "body_q" in row)
    if kind == "missing_timestamp":
        row.pop("monotonic_ns")
    elif kind == "float_timestamp":
        row["monotonic_ns"] = 12.5
    elif kind == "duplicate_timestamp":
        rows.append(dict(row))
    elif kind == "bad_shape":
        row["right_hand_q"].pop()
    elif kind == "nonfinite":
        row["body_q"][0] = float("nan")
    else:
        row.pop("body_q")
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match=expected):
        load_comparison(run)


def test_missing_malformed_and_no_aligned_samples(run):
    path = run / "sim/observations.jsonl"
    path.write_text("{broken\n")
    with pytest.raises(ValueError, match="malformed"):
        load_comparison(run)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_comparison(run)
    write_jsonl(path, [{"event": "keyboard", "monotonic_ns": 1}])
    with pytest.raises(ValueError, match="no aligned samples"):
        load_comparison(run)
