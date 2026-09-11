"""Offline replay of captured PICO snapshots through the production hand tracker.

Example: python -m gear_sonic.scripts.replay_pico_hands --profile dex3 \
    --input outputs/hand_capture/capture_*.npz --output-dir outputs/hand_replay

Inputs use pico_hand_log.INPUT_SCHEMA plus measured float64[T,2,7 or 6].
Optional evidence: pose_label unicode[T], body_sent bool[T], hand_sent bool[T,2].
Each invocation starts fresh trackers; supply all ordered shards for a full run.
The tool writes replay.npz, transitions.jsonl and report.json. Replay checks are
not a complete D0 qualification: finger mapping, live body safety and host
qualification still require their original artifacts and review.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.teleop.pico_hand_log import INPUT_SCHEMA, snapshot_at, validate_capture
from gear_sonic.utils.teleop.pico_hand_tracking import HandReason, HandTracker, TrackingState

D0_LABELS = (
    "open",
    "fist",
    "pinch",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "crossing_hands",
    "occlusion",
    "controller_put_down",
    "controller_pickup",
)


def load_captures(paths, profile):
    """Read ordered shards without pickle, sorting or silently repairing clocks."""
    shards, kinds = [], []
    required = set(INPUT_SCHEMA) | {"measured"}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            shard = {name: archive[name].copy() for name in archive.files}
        validate_capture(shard, profile)
        # Older captures contain only device timestamps. Preserve provenance when mixing shards.
        shard.setdefault("timestamp_source", np.zeros((len(shard["tick_ns"]), 2), np.int8))
        if "profile" in shard and shard["profile"].tolist() != [profile]:
            raise ValueError(f"capture profile mismatch: {path}")
        if "schema_version" in shard and shard["schema_version"].tolist() != [1]:
            raise ValueError(f"unsupported capture schema: {path}")
        kinds.append(str(shard["source_kind"].item()) if "source_kind" in shard else "unspecified")
        shards.append(shard)
    if not shards:
        raise ValueError("at least one capture file is required")
    optional = set.intersection(*(set(shard) for shard in shards)) - {"schema_version", "profile", "source_kind"}
    arrays = {name: np.concatenate([shard[name] for shard in shards], axis=0) for name in required | optional}
    validate_capture(arrays, profile)
    source_kind = kinds[0] if len(set(kinds)) == 1 else "mixed"
    return arrays, source_kind


def _checks(arrays, result, limits, max_rate, processing_ns, *, synthetic, source_kind):
    tick = arrays["tick_ns"]
    count = len(tick)
    duration = (int(tick[-1]) - int(tick[0])) / 1e9
    emitted, present = result["emitted"], result["command_present"]
    finite = np.isfinite(emitted).all(axis=2)
    in_bounds = ((emitted >= limits[0][None] - 1e-6) & (emitted <= limits[1][None] + 1e-6)).all(axis=2)
    nonfinite = int(np.count_nonzero(present & ~finite))
    out_of_bounds = int(np.count_nonzero(present & finite & ~in_bounds))
    rate_violations, max_observed_rate, hold_changes, admission_violations = 0, 0.0, 0, 0
    freeze_events, recovery_events = [], []
    early_tracking_transitions = 0
    for side in range(2):
        valid_streak = 0
        convergence_streak = 0
        last_stamp = 0
        for index in range(count):
            state = int(result["state"][index, side])
            reason = int(result["reason"][index, side])
            advanced = int(result["source_timestamp_ns"][index, side]) != last_stamp
            last_stamp = int(result["source_timestamp_ns"][index, side])
            if reason == HandReason.OK:
                valid_streak += int(advanced)
            else:
                valid_streak = 0
            close_to_target = (
                reason == HandReason.OK
                and state in (TrackingState.RECOVERING, TrackingState.TRACKING)
                and result["target_present"][index, side]
                and np.max(np.abs(emitted[index, side] - result["target"][index, side]))
                <= (0.02 if emitted.shape[-1] == 7 else 0.01)
            )
            convergence_streak = convergence_streak + 1 if close_to_target else 0
            if index == 0:
                continue
            previous_state = int(result["state"][index - 1, side])
            if state == TrackingState.TRACKING and previous_state != TrackingState.TRACKING:
                early_tracking_transitions += int(convergence_streak < 5)
            if state == TrackingState.RECOVERING and previous_state in (
                TrackingState.WAITING,
                TrackingState.HOLDING,
            ):
                admission_violations += int(valid_streak < 5)
                recovery_events.append({"side": side, "tick_ns": int(tick[index]), "valid_advances": valid_streak})
            if reason == HandReason.SOURCE_STALE and int(result["reason"][index - 1, side]) != reason:
                freeze_events.append(
                    {
                        "side": side,
                        "tick_ns": int(tick[index]),
                        "age_ns": int(result["age_ns"][index, side]),
                        "holding": state == TrackingState.HOLDING,
                    }
                )
            if present[index, side] and present[index - 1, side]:
                delta = float(np.max(np.abs(emitted[index, side].astype(np.float64) - emitted[index - 1, side])))
                if state == TrackingState.HOLDING:
                    hold_changes += int(delta > 1e-6)
                # WAITING follows measured feedback, not a published retarget command.
                if previous_state != TrackingState.WAITING and state != TrackingState.WAITING:
                    dt = min((int(tick[index]) - int(tick[index - 1])) / 1e9, 0.04)
                    rate_violations += int(delta > max_rate * dt + 1e-6)
                    max_observed_rate = max(max_observed_rate, delta / dt)
    freeze_failures = sum(not event["holding"] or event["age_ns"] > 120_000_000 for event in freeze_events)
    labels = Counter(str(value) for value in arrays.get("pose_label", []))
    missing_labels = {label: max(0, 1000 - labels[label]) for label in D0_LABELS if labels[label] < 1000}
    body_cadence = None
    if "body_sent" in arrays:
        times = tick[arrays["body_sent"]]
        if len(times) >= 2:
            gaps = np.diff(times) / 1e6
            body_cadence = {
                "gap_p99_ms": float(np.percentile(gaps, 99)),
                "gap_max_ms": float(np.max(gaps)),
                "pass": bool(np.percentile(gaps, 99) <= 25 and np.max(gaps) <= 60),
            }
    opposite_evidence = []
    if "hand_sent" in arrays:
        for side in range(2):
            stale = result["reason"][:, side] == HandReason.SOURCE_STALE
            # Assess each contiguous stale interval; isolated samples cannot prove rate.
            starts = np.flatnonzero(stale & np.r_[True, ~stale[:-1]])
            ends = np.flatnonzero(stale & np.r_[~stale[1:], True])
            for start, end in zip(starts, ends):
                seconds = (int(tick[end]) - int(tick[start])) / 1e9
                sent = arrays["hand_sent"][start : end + 1, 1 - side]
                hz = max(0, int(np.count_nonzero(sent)) - 1) / seconds if seconds >= 0.1 else None
                opposite_evidence.append(
                    {
                        "frozen_side": side,
                        "duration_s": seconds,
                        "other_hand_hz": hz,
                        "pass": hz is not None and hz >= 45,
                    }
                )
    missing = []
    if "timestamp_source" in arrays and np.any(arrays["timestamp_source"] == 1):
        missing.append("Original APK uses host content-change timing, not device sample freshness evidence.")
    if synthetic or source_kind != "runtime_capture":
        missing.append(
            "Input or retargeter is synthetic/unspecified; no recorded real-source qualification claim."
        )
    if duration < 600:
        missing.append("Capture duration is below 600 seconds.")
    if missing_labels:
        missing.append("At least one required pose label has fewer than 1000 frames.")
    if body_cadence is None:
        missing.append("No sufficient actual body-publication cadence evidence.")
    if not opposite_evidence:
        missing.append("No actual opposite-hand publication-frequency evidence during source loss.")
    if not freeze_events:
        missing.append("No observable source-freeze transition exercised the hold deadline.")
    missing.extend(
        [
            "Finger identity/order and absence of hand-caused body mode transitions require qualification review.",
            "Offline processing timing is not proof of the live deployment-host timing gate.",
        ]
    )
    counts = {
        "nonfinite_emitted": nonfinite,
        "out_of_bounds_emitted": out_of_bounds,
        "rate_violations": rate_violations,
        "held_command_changes": hold_changes,
        "late_or_nonholding_freezes": int(freeze_failures),
        "early_recovery_admissions": admission_violations,
        "early_tracking_transitions": early_tracking_transitions,
        "retarget_rejections": int(np.count_nonzero(result["reason"] == HandReason.RETARGET_FAILED)),
    }
    checks_pass = not any(counts[key] for key in counts if key != "retarget_rejections")
    failed_evidence = (body_cadence is not None and not body_cadence["pass"]) or any(
        item["other_hand_hz"] is not None and not item["pass"] for item in opposite_evidence
    )
    return {
        "schema_version": 1,
        "source_kind": source_kind,
        "synthetic": bool(synthetic),
        "d0_qualified": False,
        "qualification_status": "incomplete" if checks_pass and not failed_evidence else "failed",
        "replay_safety_checks_pass": bool(checks_pass),
        "frames": count,
        "duration_s": duration,
        "max_rate": max_rate,
        "max_observed_rate": max_observed_rate,
        "counts": counts,
        "reason_codes": {reason.name: int(reason) for reason in HandReason},
        "state_codes": {state.name: int(state) for state in TrackingState},
        "freeze_events": freeze_events,
        "recovery_events": recovery_events,
        "pose_label_counts": dict(labels),
        "pose_label_frame_deficits": missing_labels,
        "body_cadence": body_cadence,
        "opposite_hand_evidence": opposite_evidence,
        "processing_p99_ms": float(np.percentile(processing_ns, 99) / 1e6),
        "processing_p99_within_5ms": bool(np.percentile(processing_ns, 99) <= 5_000_000),
        "missing_qualification_evidence": missing,
    }


class _ObservedRetargeter:
    """Record candidate diagnostics without changing the production limiter."""

    def __init__(self, implementation):
        self.implementation = implementation
        self.lower, self.upper = implementation.lower, implementation.upper
        self.raw_target = None
        self.out_of_bounds_targets = 0
        self.nonfinite_targets = 0

    def retarget(self, points):
        candidate = self.implementation.retarget(points)
        try:
            value = np.asarray(candidate, dtype=np.float64)
            if value.shape == np.asarray(self.lower).shape:
                self.raw_target = value.copy()
                self.nonfinite_targets += int(not np.isfinite(value).all())
                self.out_of_bounds_targets += int(
                    np.isfinite(value).all() and np.any((value < self.lower) | (value > self.upper))
                )
        except (ValueError, TypeError, OverflowError):
            pass
        return candidate


def replay(arrays, profile="dex3", *, retargeters=None, max_rate=None, source_kind="unspecified"):
    """Return (output arrays, JSON-serializable report) using production HandTracker.

    An injected retargeter pair is always reported as synthetic. Real retargeting
    dependencies/assets are loaded only when no pair is injected.
    """
    count = validate_capture(arrays, profile)
    synthetic = retargeters is not None or source_kind == "synthetic"
    if retargeters is None:
        from gear_sonic.utils.teleop.pico_hand_retargeting import HandRetargeter

        retargeters = [HandRetargeter(profile, side) for side in ("left", "right")]
    if len(retargeters) != 2:
        raise ValueError("replay requires two retargeters")
    observed = [_ObservedRetargeter(retargeter) for retargeter in retargeters]
    trackers = [HandTracker(retargeter, profile, max_rate) for retargeter in observed]
    size = 7 if profile == "dex3" else 6
    result = {
        "tick_ns": arrays["tick_ns"].copy(),
        "emitted": np.full((count, 2, size), np.nan, np.float32),
        "target": np.full((count, 2, size), np.nan, np.float32),
        "command_present": np.zeros((count, 2), np.bool_),
        "target_present": np.zeros((count, 2), np.bool_),
        "raw_target": np.full((count, 2, size), np.nan, np.float64),
        "raw_target_present": np.zeros((count, 2), np.bool_),
        "state": np.zeros((count, 2), np.int32),
        "reason": np.zeros((count, 2), np.int32),
        "source_epoch": np.zeros((count, 2), np.int64),
        "source_timestamp_ns": np.zeros((count, 2), np.int64),
        "age_ns": np.zeros((count, 2), np.int64),
        "binding_generation": arrays["binding_generation"].copy(),
    }
    processing_ns = np.empty(count, np.int64)
    for index, tick in enumerate(arrays["tick_ns"]):
        started = time.perf_counter_ns()
        for side, tracker in enumerate(trackers):
            observed[side].raw_target = None
            output = tracker.step(
                snapshot_at(arrays, index, side),
                arrays["body_wrist"][index, side],
                arrays["measured"][index, side],
                int(tick),
                enabled=bool(arrays["enabled"][index]) if "enabled" in arrays else True,
            )
            if observed[side].raw_target is not None:
                result["raw_target"][index, side] = observed[side].raw_target
                result["raw_target_present"][index, side] = True
            for key, attr in (("emitted", "command"), ("target", "target")):
                value = getattr(output, attr)
                if value is not None:
                    result[key][index, side] = value
                    result["command_present" if key == "emitted" else "target_present"][index, side] = True
            for key in ("state", "reason", "source_epoch", "source_timestamp_ns", "age_ns"):
                result[key][index, side] = int(getattr(output, key))
        processing_ns[index] = time.perf_counter_ns() - started
    result["processing_ns"] = processing_ns
    for provenance in ("pv", "sample_generation"):
        if provenance in arrays:
            result[provenance] = arrays[provenance].copy()
    limits = (np.stack([tracker.lower for tracker in trackers]), np.stack([tracker.upper for tracker in trackers]))
    report = _checks(
        arrays, result, limits, trackers[0].max_rate, processing_ns, synthetic=synthetic, source_kind=source_kind
    )
    report["profile"] = profile
    report["counts"]["out_of_bounds_targets_rejected"] = sum(item.out_of_bounds_targets for item in observed)
    report["counts"]["nonfinite_targets_rejected"] = sum(item.nonfinite_targets for item in observed)
    if "captured_command" in arrays and arrays["captured_command"].shape == result["emitted"].shape:
        comparable = np.isfinite(arrays["captured_command"]) & np.isfinite(result["emitted"])
        report["captured_command_comparison"] = {
            "comparable_values": int(np.count_nonzero(comparable)),
            "max_abs_difference": float(
                np.max(np.abs(arrays["captured_command"][comparable] - result["emitted"][comparable]))
            )
            if comparable.any()
            else None,
        }
    return result, report


def write_results(output_dir, arrays, report):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("replay.npz", "report.json", "transitions.jsonl")):
        raise FileExistsError("use a new replay output directory")
    np.savez(directory / "replay.npz", **arrays)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (directory / "transitions.jsonl").open("w") as stream:
        previous = [None, None]
        for index, tick in enumerate(arrays["tick_ns"]):
            for side in range(2):
                state = (int(arrays["state"][index, side]), int(arrays["reason"][index, side]))
                if state != previous[side]:
                    stream.write(
                        json.dumps(
                            {
                                "tick_ns": int(tick),
                                "side": ("left", "right")[side],
                                "state": TrackingState(state[0]).name,
                                "reason": HandReason(state[1]).name,
                            }
                        )
                        + "\n"
                    )
                    previous[side] = state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("dex3", "inspire_ftp"))
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rate", type=float, default=None)
    args = parser.parse_args(argv)
    arrays, kind = load_captures(args.input, args.profile)
    result, report = replay(arrays, args.profile, max_rate=args.max_rate, source_kind=kind)
    report["input_files"] = [str(path) for path in args.input]
    write_results(args.output_dir, result, report)
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "replay_safety_checks_pass": report["replay_safety_checks_pass"],
                "d0_qualified": False,
            }
        )
    )
    return 0 if report["replay_safety_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
