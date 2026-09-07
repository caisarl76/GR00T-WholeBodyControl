"""Compare ACT, transported references, received DDS commands and joint observations.

All returned joint matrices use ACTION_NAMES and radians. Observation timestamps
are associated with the latest prior native row across *all* native modes/chunks.
This association cannot identify the exact asynchronous DDS command generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.utils.inference.reference_adapter.g1_act import (
    ACTION_NAMES,
    BODY_NAMES,
    ISAAC_FROM_MOTOR,
)

TRANSPORT_ATOL_RAD = 1e-6
MAX_AGE_NS = 40_000_000
PAIRS = {
    "native_minus_saved": ("saved_reference", "native_reference"),
    "raw_to_reference": ("act_raw", "saved_reference"),
    "reference_to_command": ("saved_reference", "issued_command"),
    "raw_to_command": ("act_raw", "issued_command"),
    "reference_to_measured": ("saved_reference", "measured"),
    "raw_to_measured": ("act_raw", "measured"),
}
GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 21), "right_hand": (21, 28)}


def _json(path, lines=False):
    try:
        content = path.read_text()
        value = (
            [json.loads(line) for line in content.splitlines() if line.strip()] if lines else json.loads(content)
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing or malformed input {path}: {exc}") from exc
    if lines and (not value or not all(isinstance(row, dict) for row in value)):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    if not lines and not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected numeric array") from exc
    if result.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: nonfinite values")
    return result


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name}: expected nonnegative int64 integer")
    return value


def _metric(x, y):
    """Error for X to Y is Y - X, including signed mean bias."""
    delta = np.asarray(y) - np.asarray(x)
    rad = {
        "rms_rad": float(np.sqrt(np.mean(delta**2))),
        "maxabs_rad": float(np.max(np.abs(delta))),
        "mean_bias_rad": float(np.mean(delta)),
    }
    return {**rad, **{key.replace("_rad", "_deg"): float(np.rad2deg(value)) for key, value in rad.items()}}


def _metrics(data):
    return {
        "all_joints": {label: _metric(data[x], data[y]) for label, (x, y) in PAIRS.items()},
        "groups": {
            group: {
                label: _metric(data[x][:, start:end], data[y][:, start:end]) for label, (x, y) in PAIRS.items()
            }
            for group, (start, end) in GROUPS.items()
        },
        "per_joint": {
            name: {label: _metric(data[x][:, j], data[y][:, j]) for label, (x, y) in PAIRS.items()}
            for j, name in enumerate(ACTION_NAMES)
        },
    }


def load_comparison(run_dir, session_id=None, chunk_id=2):
    """Load validated records and return observation-aligned ACT-order arrays.

    ``source_time_s`` starts at the ACT segment (reference frame / reference_hz
    minus initial_transition_seconds); ``monotonic_ns`` is the real observation
    timestamp. ``observations`` and ``native_rows`` retain their original fields.
    All matching active native rows, including transition and terminal padding,
    are checked against their saved reference frame before observation filtering.
    """
    run_dir = Path(run_dir)
    paths = [
        run_dir / item
        for item in (
            "replay/actions.npy",
            "replay/reference.npz",
            "replay/composition.json",
            "sim/joint_contract.json",
            "sim/observations.jsonl",
            "native.jsonl",
        )
    ]
    try:
        actions = np.load(paths[0], allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 28 or not len(actions):
            raise ValueError("actions.npy must have nonempty shape [N,28]")
        actions = _array(actions, actions.shape, "actions.npy")
        with np.load(paths[1], allow_pickle=False) as archive:
            body = archive["joint_pos"]
            if body.ndim != 2 or body.shape[1] != 29 or not len(body):
                raise ValueError("joint_pos must have nonempty shape [N,29]")
            count = len(body)
            body = _array(body, (count, 29), "saved joint_pos")
            left = _array(archive["left_hand_joints"], (count, 7), "saved left hand")
            right = _array(archive["right_hand_joints"], (count, 7), "saved right hand")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid replay arrays: {exc}") from exc
    composition = _json(paths[2])
    rates = {}
    for key in ("source_hz", "reference_hz", "initial_transition_seconds"):
        try:
            value = float(composition[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"composition.json: missing or invalid {key}") from exc
        if not np.isfinite(value) or value < 0 or (key != "initial_transition_seconds" and value == 0):
            raise ValueError(f"composition.json: invalid {key}")
        rates[key] = value
    if "source_action_names" in composition and list(composition["source_action_names"]) != list(ACTION_NAMES):
        raise ValueError("composition source_action_names disagree with ACT joint order")
    hz, reference_hz, entry = (rates[key] for key in ("source_hz", "reference_hz", "initial_transition_seconds"))
    contract = _json(paths[3])
    contract_names = {}
    for group, size in (("body", 29), ("left", 7), ("right", 7)):
        rows = contract.get(group)
        if not isinstance(rows, list) or len(rows) != size or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"joint contract: malformed {group} group")
        names = [row.get("name") for row in rows]
        if not all(isinstance(name, str) for name in names) or len(set(names)) != size:
            raise ValueError(f"joint contract: invalid {group} names")
        for row in rows:
            bounds = _array(row.get("range"), (2,), f"{group} {row['name']} range")
            if bounds[0] > bounds[1]:
                raise ValueError(f"joint contract: reversed range for {row['name']}")
        contract_names[group] = names
    if contract_names["body"] != list(BODY_NAMES):
        raise ValueError("joint contract body order disagrees with BODY_NAMES")
    hand_indices = {}
    for side, selected in (("left", ACTION_NAMES[14:21]), ("right", ACTION_NAMES[21:28])):
        if set(contract_names[side]) != set(selected):
            raise ValueError(f"joint contract {side} hand names disagree with ACT")
        hand_indices[side] = [contract_names[side].index(name) for name in selected]
    motor_from_isaac = np.argsort(ISAAC_FROM_MOTOR)

    def act_order(body_values, left_values, right_values, isaac=False):
        if isaac:
            body_values = body_values[..., motor_from_isaac]
        return np.concatenate(
            (
                body_values[..., 15:29],
                left_values[..., hand_indices["left"]],
                right_values[..., hand_indices["right"]],
            ),
            axis=-1,
        )

    saved = act_order(body, left, right, isaac=True)
    native = _json(paths[5], lines=True)
    observation_log = _json(paths[4], lines=True)
    for i, row in enumerate(observation_log):
        _integer(row.get("monotonic_ns"), f"observation log row {i} monotonic_ns")
    # Keyboard/event markers share this JSONL file but are not sensor samples.
    observations = [row for row in observation_log if "event" not in row]
    for i, row in enumerate(native):
        _integer(row.get("monotonic_ns"), f"native row {i} monotonic_ns")
        if not isinstance(row.get("mode"), str):
            raise ValueError(f"native row {i}: missing mode")
        if row["mode"] == "active":
            for key in ("session_id", "chunk_id", "reference_frame"):
                _integer(row.get(key), f"native row {i} {key}")
            for key, size in (
                ("body_reference_isaac", 29),
                ("left_hand_reference", 7),
                ("right_hand_reference", 7),
            ):
                _array(row.get(key), (size,), f"native row {i} {key}")
    for i, row in enumerate(observations):
        _integer(row.get("monotonic_ns"), f"observation row {i} monotonic_ns")
        for key, size in (
            ("body_q", 29),
            ("body_command_target", 35),
            ("left_hand_q", 7),
            ("right_hand_q", 7),
            ("left_command_target", 7),
            ("right_command_target", 7),
        ):
            _array(row.get(key), (size,), f"observation row {i} {key}")
    native.sort(key=lambda row: row["monotonic_ns"])
    observations.sort(key=lambda row: row["monotonic_ns"])
    for label, rows in (("native", native), ("observations", observations)):
        timestamps = [row["monotonic_ns"] for row in rows]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f"{label}: duplicate timestamps make temporal ordering ambiguous")
    active = [
        row
        for row in native
        if row["mode"] == "active"
        and row["chunk_id"] == chunk_id
        and (session_id is None or row["session_id"] == session_id)
    ]
    if not active:
        raise ValueError("no active native records match requested session/chunk")
    sessions = {row["session_id"] for row in active}
    if len(sessions) != 1:
        raise ValueError("session is ambiguous; pass --session-id")
    session_id = next(iter(sessions))
    frames = np.asarray([row["reference_frame"] for row in active], dtype=np.int64)
    if np.any(frames >= count):
        raise ValueError("active native reference_frame exceeds saved reference array")
    native_body = np.asarray([row["body_reference_isaac"] for row in active])
    native_left = np.asarray([row["left_hand_reference"] for row in active])
    native_right = np.asarray([row["right_hand_reference"] for row in active])
    all_saved = np.concatenate((body[frames], left[frames], right[frames]), axis=1)
    all_native = np.concatenate((native_body, native_left, native_right), axis=1)
    transport_errors = np.abs(all_native - all_saved)
    transport = {
        "passed": bool(np.all(transport_errors <= TRANSPORT_ATOL_RAD)),
        "atol_rad": TRANSPORT_ATOL_RAD,
        "rtol": 0,
        "active_rows_checked": len(active),
        "distinct_reference_frames": len(set(frames.tolist())),
        "joint_values_checked": int(transport_errors.size),
        "maxabs_rad": float(transport_errors.max()),
        "values_over_atol": int(np.count_nonzero(transport_errors > TRANSPORT_ATOL_RAD)),
        "scope": "all 29 body and 14 hand joints in every active requested session/chunk row",
    }
    native_times = np.asarray([row["monotonic_ns"] for row in native], dtype=np.int64)
    obs_times = np.asarray([row["monotonic_ns"] for row in observations], dtype=np.int64)
    prior = np.searchsorted(native_times, obs_times, side="right") - 1
    picked_obs, picked_native, ages = [], [], []
    excluded = {
        key: 0
        for key in ("before_first_native", "inactive_or_other_session_chunk", "stale", "outside_act_segment")
    }
    source_duration = (len(actions) - 1) / hz
    for observation, index in zip(observations, prior):
        if index < 0:
            excluded["before_first_native"] += 1
            continue
        row = native[index]
        if row["mode"] != "active" or row["session_id"] != session_id or row["chunk_id"] != chunk_id:
            excluded["inactive_or_other_session_chunk"] += 1
            continue
        age = observation["monotonic_ns"] - row["monotonic_ns"]
        if age > MAX_AGE_NS:
            excluded["stale"] += 1
            continue
        source_time = row["reference_frame"] / reference_hz - entry
        if source_time < -1e-12 or source_time > source_duration + 1e-12:
            excluded["outside_act_segment"] += 1
            continue
        picked_obs.append(observation)
        picked_native.append(row)
        ages.append(age / 1e6)
    if not picked_obs:
        raise ValueError("no aligned samples after session, chunk, age and ACT segment filters")
    frames = np.asarray([row["reference_frame"] for row in picked_native], dtype=np.int64)
    source_time = frames / reference_hz - entry
    source_axis = np.arange(len(actions)) / hz
    data = {
        "act_raw": np.column_stack([np.interp(source_time, source_axis, actions[:, j]) for j in range(28)]),
        "saved_reference": saved[frames],
        "native_reference": act_order(
            np.asarray([row["body_reference_isaac"] for row in picked_native]),
            np.asarray([row["left_hand_reference"] for row in picked_native]),
            np.asarray([row["right_hand_reference"] for row in picked_native]),
            isaac=True,
        ),
        "issued_command": act_order(
            np.asarray([row["body_command_target"][:29] for row in picked_obs]),
            np.asarray([row["left_command_target"] for row in picked_obs]),
            np.asarray([row["right_command_target"] for row in picked_obs]),
        ),
        "measured": act_order(
            np.asarray([row["body_q"] for row in picked_obs]),
            np.asarray([row["left_hand_q"] for row in picked_obs]),
            np.asarray([row["right_hand_q"] for row in picked_obs]),
        ),
        "observations": picked_obs,
        "native_rows": picked_native,
        "source_time_s": source_time,
        "reference_frames": frames,
        "monotonic_ns": np.asarray([row["monotonic_ns"] for row in picked_obs], dtype=np.int64),
    }
    data["summary"] = {
        "inputs": {
            str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "scope": {
            "session_id": session_id,
            "chunk_id": chunk_id,
            "samples": len(picked_obs),
            "source_action_frames": len(actions),
            "saved_reference_frames": count,
            "selected_reference_frame_min": int(frames.min()),
            "selected_reference_frame_max": int(frames.max()),
        },
        "alignment": {
            **rates,
            "max_allowed_age_ms": MAX_AGE_NS / 1e6,
            "median_age_ms": float(np.median(ages)),
            "max_age_ms": float(np.max(ages)),
            "excluded_observations": excluded,
            "event_markers_skipped": len(observation_log) - len(observations),
            "method": "each observation uses latest prior row from all timestamp-sorted native records",
            "source_time_formula": "reference_frame / reference_hz - initial_transition_seconds",
            "timing_caveat": (
                "Native and observation/DDS streams are asynchronous; timestamp association does not prove "
                "exact command causality or distinguish transport delay from control error."
            ),
        },
        "transport_reference": transport,
        "body_target_label": "received scaled decoder command; normalized raw network output was not logged",
        "hand_target_label": "direct DDS reference command; fingers are not learned decoder outputs",
        "metric_definition": (
            "X_to_Y signed error = Y - X; radians and degrees; "
            "observation-weighted; no physical fidelity pass/fail tolerance"
        ),
        "mappings": {
            "joint_names": list(ACTION_NAMES),
            "isaac_from_motor": ISAAC_FROM_MOTOR.tolist(),
            "motor_from_isaac": motor_from_isaac.tolist(),
            "runtime_joint_names": contract_names,
            "hand_runtime_to_act_indices": hand_indices,
        },
        "metrics": _metrics(data),
    }
    return data


def compare(run_dir, output_dir, session_id=None, chunk_id=2):
    data = load_comparison(run_dir, session_id=session_id, chunk_id=chunk_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = data["summary"]["metrics"]["per_joint"]
    metric_names = list(next(iter(next(iter(metrics.values())).values())))
    with (output_dir / "per_joint.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["joint", "group", *[f"{pair}_{metric}" for pair in PAIRS for metric in metric_names]])
        for j, name in enumerate(ACTION_NAMES):
            group = next(group for group, (start, end) in GROUPS.items() if start <= j < end)
            writer.writerow(
                [name, group, *[metrics[name][pair][metric] for pair in PAIRS for metric in metric_names]]
            )
    with (output_dir / "joint_timeseries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "monotonic_ns",
                "native_monotonic_ns",
                "native_age_ms",
                "reference_frame",
                "source_time_s",
                "joint",
                "act_raw_rad",
                "saved_reference_rad",
                "native_reference_rad",
                "issued_command_rad",
                "measured_rad",
            ]
        )
        for i, timestamp in enumerate(data["monotonic_ns"]):
            native_timestamp = data["native_rows"][i]["monotonic_ns"]
            for j, name in enumerate(ACTION_NAMES):
                writer.writerow(
                    [
                        int(timestamp),
                        native_timestamp,
                        (int(timestamp) - native_timestamp) / 1e6,
                        int(data["reference_frames"][i]),
                        data["source_time_s"][i],
                        name,
                        *[
                            data[key][i, j]
                            for key in (
                                "act_raw",
                                "saved_reference",
                                "native_reference",
                                "issued_command",
                                "measured",
                            )
                        ],
                    ]
                )
    (output_dir / "summary.json").write_text(json.dumps(data["summary"], indent=2, allow_nan=False) + "\n")
    _plots(output_dir, data)
    return data


def _plots(output_dir, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data["source_time_s"]
    for group, (start, end) in GROUPS.items():
        fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
        for axis, joint in zip(axes, range(start, end)):
            for key, label in (
                ("act_raw", "raw ACT"),
                ("saved_reference", "reference"),
                ("issued_command", "received issued target"),
                ("measured", "measured"),
            ):
                axis.plot(x, np.rad2deg(data[key][:, joint]), label=label, linewidth=1)
            axis.set_ylabel("degrees")
            axis.set_title(ACTION_NAMES[joint], fontsize=10)
            axis.grid(True, alpha=0.3)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel("ACT source time (s)")
        fig.suptitle(f"{group.replace('_', ' ').title()} — asynchronous latest-prior alignment")
        fig.tight_layout()
        fig.savefig(output_dir / f"{group}.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--chunk-id", type=int, default=2)
    args = parser.parse_args()
    try:
        data = compare(args.run_dir, args.output_dir, args.session_id, args.chunk_id)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "samples": data["summary"]["scope"]["samples"],
                "transport_reference": data["summary"]["transport_reference"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    if not data["summary"]["transport_reference"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
