#!/usr/bin/env python3
"""Collect and evaluate fail-closed GPU evidence for GR00T training gates."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time

EXPECTED_TOTAL_MIB = 81559
BASELINE_MAX_FRACTION = 0.25
GATE_MAX_FRACTION = 0.95
PROCESS_RANGE_MAX_MIB = 256
AGGREGATE_RANGE_MAX_MIB = 512

_SAMPLE_KEYS = {
    "timestamp_utc",
    "gpu_index",
    "gpu_uuid",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_percent",
    "processes",
}
_PROCESS_KEYS = {"pid", "name", "used_memory_mib"}
_RESULT_KEYS = {
    "status",
    "reasons",
    "mode",
    "thresholds",
    "summary",
    "health_before",
    "health_after",
}
_INTEGER_RE = re.compile(r"-?[0-9]+\n?\Z")


class GpuGateError(RuntimeError):
    """Raised when evidence cannot be collected or parsed safely."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_processes(
    processes_by_gpu: Mapping[str | int, Iterable[Mapping[str, object]]] | None,
) -> dict[str, list[dict[str, object]]]:
    normalized: dict[str, list[dict[str, object]]] = {}
    for raw_gpu, processes in (processes_by_gpu or {}).items():
        gpu = str(raw_gpu)
        values = [{"pid": int(process["pid"]), "name": str(process["name"])} for process in processes]
        normalized[gpu] = sorted(values, key=lambda value: (value["pid"], value["name"]))
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _sample_process_sets(
    samples: Sequence[Mapping[str, object]],
) -> dict[int, list[tuple[tuple[int, str], ...]]]:
    result: dict[int, list[tuple[tuple[int, str], ...]]] = {}
    for sample in samples:
        gpu = int(sample["gpu_index"])
        process_set = tuple(
            sorted(
                (int(process["pid"]), str(process["name"]))
                for process in sample["processes"]  # type: ignore[union-attr]
            )
        )
        result.setdefault(gpu, []).append(process_set)
    return result


def _processes_from_last_samples(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    latest: dict[int, Mapping[str, object]] = {}
    for sample in samples:
        latest[int(sample["gpu_index"])] = sample
    return _normalize_processes(
        {
            str(gpu): sample["processes"]  # type: ignore[dict-item]
            for gpu, sample in latest.items()
        }
    )


def _validate_samples(samples: Sequence[Mapping[str, object]]) -> list[str]:
    reasons: list[str] = []
    if not samples:
        return ["sample schema: at least one sample is required"]
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != _SAMPLE_KEYS:
            reasons.append(f"sample schema: sample {sample_index} must use exact keys")
            continue
        scalar_ints = (
            "gpu_index",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_percent",
        )
        if (
            not isinstance(sample["timestamp_utc"], str)
            or not sample["timestamp_utc"]
            or not isinstance(sample["gpu_uuid"], str)
            or not sample["gpu_uuid"]
            or any(not _is_int(sample[key]) for key in scalar_ints)
            or int(sample["gpu_index"]) < 0
            or int(sample["memory_used_mib"]) < 0
            or int(sample["memory_total_mib"]) <= 0
            or int(sample["memory_total_mib"]) != EXPECTED_TOTAL_MIB
        ):
            reasons.append(f"sample schema: invalid scalar in sample {sample_index}")
            continue
        processes = sample["processes"]
        if not isinstance(processes, list):
            reasons.append(f"sample schema: processes in sample {sample_index} must be a list")
            continue
        seen_pids: set[int] = set()
        for process_index, process in enumerate(processes):
            if not isinstance(process, Mapping) or set(process) != _PROCESS_KEYS:
                reasons.append(f"sample schema: process {sample_index}:{process_index} must use exact keys")
                continue
            pid = process["pid"]
            memory = process["used_memory_mib"]
            name = process["name"]
            if (
                not _is_int(pid)
                or int(pid) <= 0
                or not isinstance(name, str)
                or not name
                or not _is_int(memory)
                or int(memory) < 0
            ):
                reasons.append(f"sample schema: invalid process {sample_index}:{process_index}")
            elif int(pid) in seen_pids:
                reasons.append(f"sample schema: duplicate PID {pid} in sample {sample_index}")
            else:
                seen_pids.add(int(pid))
    return reasons


def _health_regressed(
    health_before: Mapping[str, object] | None,
    health_after: Mapping[str, object] | None,
) -> bool:
    if health_before is None and health_after is None:
        return False
    if not isinstance(health_before, Mapping) or not isinstance(health_after, Mapping):
        return True
    if set(health_before) != set(health_after):
        return True
    for key, before in health_before.items():
        after = health_after[key]
        if isinstance(before, Mapping) or isinstance(after, Mapping):
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                return True
            if _health_regressed(before, after):
                return True
        elif (
            isinstance(before, bool)
            or isinstance(after, bool)
            or not isinstance(before, (int, float))
            or not isinstance(after, (int, float))
            or not math.isfinite(float(before))
            or not math.isfinite(float(after))
            or float(after) > float(before)
        ):
            return True
    return False


def _base_result(
    *,
    mode: str,
    reasons: list[str],
    thresholds: dict[str, object],
    summary: dict[str, object],
    health_before: Mapping[str, object] | None,
    health_after: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "mode": mode,
        "thresholds": thresholds,
        "summary": summary,
        "health_before": dict(health_before or {}),
        "health_after": dict(health_after or {}),
    }


def evaluate_baseline(
    *,
    samples: Sequence[Mapping[str, object]],
    total_mib: int = EXPECTED_TOTAL_MIB,
    health_before: Mapping[str, object] | None = None,
    health_after: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the immutable 30-second prelaunch baseline contract."""
    reasons = _validate_samples(samples)
    thresholds: dict[str, object] = {
        "expected_total_mib": EXPECTED_TOTAL_MIB,
        "maximum_memory_fraction": BASELINE_MAX_FRACTION,
        "maximum_memory_used_mib_exclusive": EXPECTED_TOTAL_MIB * BASELINE_MAX_FRACTION,
        "maximum_process_range_mib": PROCESS_RANGE_MAX_MIB,
        "maximum_aggregate_range_mib": AGGREGATE_RANGE_MAX_MIB,
    }
    if not _is_int(total_mib) or total_mib != EXPECTED_TOTAL_MIB:
        reasons.append(f"expected H100 total memory is {EXPECTED_TOTAL_MIB} MiB")
    valid_samples = [] if _validate_samples(samples) else list(samples)
    if valid_samples:
        for sample in valid_samples:
            if int(sample["memory_total_mib"]) != EXPECTED_TOTAL_MIB:
                reasons.append(f"GPU {sample['gpu_index']} total memory is not {EXPECTED_TOTAL_MIB} MiB")
            if int(sample["memory_used_mib"]) >= EXPECTED_TOTAL_MIB * BASELINE_MAX_FRACTION:
                reasons.append(f"GPU {sample['gpu_index']} exceeded baseline memory ceiling")

        process_sets = _sample_process_sets(valid_samples)
        for gpu, sets in process_sets.items():
            if any(value != sets[0] for value in sets[1:]):
                reasons.append(f"GPU {gpu} process set changed during baseline")

        uuid_sets: dict[int, set[str]] = {}
        aggregate_values: dict[int, list[int]] = {}
        process_values: dict[tuple[int, int, str], list[int]] = {}
        for sample in valid_samples:
            gpu = int(sample["gpu_index"])
            uuid_sets.setdefault(gpu, set()).add(str(sample["gpu_uuid"]))
            aggregate_values.setdefault(gpu, []).append(int(sample["memory_used_mib"]))
            for process in sample["processes"]:  # type: ignore[union-attr]
                key = (gpu, int(process["pid"]), str(process["name"]))
                process_values.setdefault(key, []).append(int(process["used_memory_mib"]))
        for gpu, uuids in uuid_sets.items():
            if len(uuids) != 1:
                reasons.append(f"GPU {gpu} UUID changed during baseline")
        for (gpu, pid, _), values in process_values.items():
            if max(values) - min(values) > PROCESS_RANGE_MAX_MIB:
                reasons.append(f"GPU {gpu} PID {pid} process memory range exceeded limit")
        for gpu, values in aggregate_values.items():
            if max(values) - min(values) > AGGREGATE_RANGE_MAX_MIB:
                reasons.append(f"GPU {gpu} aggregate memory range exceeded limit")
    if _health_regressed(health_before, health_after):
        reasons.append("GPU health regressed")
    summary = {
        "sample_count": len(samples),
        "baseline_processes_by_gpu": _processes_from_last_samples(valid_samples),
        "final_processes_by_gpu": _processes_from_last_samples(valid_samples),
    }
    return _base_result(
        mode="baseline",
        reasons=reasons,
        thresholds=thresholds,
        summary=summary,
        health_before=health_before,
        health_after=health_after,
    )


def _expected_pid_contract(
    expected_pids: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    normalized: dict[str, dict[str, int]] = {}
    reasons: list[str] = []
    for label, value in expected_pids.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(value, Mapping)
            or set(value) != {"gpu_index", "pid"}
            or not _is_int(value.get("gpu_index"))
            or int(value.get("gpu_index", -1)) < 0
            or not _is_int(value.get("pid"))
            or int(value.get("pid", 0)) <= 0
        ):
            reasons.append(f"invalid expected PID contract for {label!r}")
            continue
        normalized[label] = {
            "gpu_index": int(value["gpu_index"]),
            "pid": int(value["pid"]),
        }
    if len({value["pid"] for value in normalized.values()}) != len(normalized):
        reasons.append("expected PID values must be unique")
    return normalized, reasons


def _partition_sample_cycles(
    samples: Sequence[Mapping[str, object]],
) -> list[list[Mapping[str, object]]]:
    """Recover contiguous snapshot batches without trusting timestamp precision."""
    cycles: list[list[Mapping[str, object]]] = []
    current: list[Mapping[str, object]] = []
    current_gpus: set[int] = set()
    for sample in samples:
        gpu = int(sample["gpu_index"])
        if gpu in current_gpus:
            cycles.append(current)
            current = []
            current_gpus = set()
        current.append(sample)
        current_gpus.add(gpu)
    if current:
        cycles.append(current)
    return cycles


def _sample_cycle_contract(
    samples: Sequence[Mapping[str, object]],
    sample_cycles: Sequence[Sequence[Mapping[str, object]]] | None,
) -> tuple[list[list[Mapping[str, object]]], list[str]]:
    cycles = (
        _partition_sample_cycles(samples) if sample_cycles is None else [list(cycle) for cycle in sample_cycles]
    )
    reasons: list[str] = []
    if not cycles or any(not cycle for cycle in cycles):
        return [], ["sample cycle contract requires nonempty cycles"]
    flattened: list[Mapping[str, object]] = []
    for cycle_index, cycle in enumerate(cycles):
        gpu_indices = [int(sample["gpu_index"]) for sample in cycle]
        if len(gpu_indices) != len(set(gpu_indices)):
            reasons.append(f"sample cycle {cycle_index} repeats a GPU")
        if len({str(sample["timestamp_utc"]) for sample in cycle}) != 1:
            reasons.append(f"sample cycle {cycle_index} has inconsistent timestamps")
        flattened.extend(cycle)
    if flattened != list(samples):
        reasons.append("sample cycles do not exactly partition the ordered samples")
    return ([] if reasons else cycles), reasons


def evaluate_concurrent_gate(
    *,
    samples: Sequence[Mapping[str, object]],
    sample_cycles: Sequence[Sequence[Mapping[str, object]]] | None = None,
    expected_pids: Mapping[str, Mapping[str, object]],
    total_mib: int = EXPECTED_TOTAL_MIB,
    baseline_processes_by_gpu: Mapping[str | int, Iterable[Mapping[str, object]]] | None = None,
    exit_codes: Mapping[str, int] | None = None,
    health_before: Mapping[str, object] | None = None,
    health_after: Mapping[str, object] | None = None,
    timed_out: bool = False,
) -> dict[str, object]:
    """Evaluate concurrent smoke overlap, capacity, and process authority."""
    reasons = _validate_samples(samples)
    expected, expected_reasons = _expected_pid_contract(expected_pids)
    reasons.extend(expected_reasons)
    baseline = _normalize_processes(baseline_processes_by_gpu)
    valid_samples = [] if _validate_samples(samples) else list(samples)
    cycles, cycle_reasons = _sample_cycle_contract(valid_samples, sample_cycles)
    reasons.extend(cycle_reasons)
    if not _is_int(total_mib) or total_mib != EXPECTED_TOTAL_MIB:
        reasons.append(f"expected H100 total memory is {EXPECTED_TOTAL_MIB} MiB")
    if timed_out:
        reasons.append("concurrent gate timeout")
    allowed: dict[int, set[tuple[int, str]]] = {
        int(gpu): {(int(value["pid"]), str(value["name"])) for value in values} for gpu, values in baseline.items()
    }
    expected_by_pid = {value["pid"]: (label, value["gpu_index"]) for label, value in expected.items()}
    peaks: dict[int, int] = {}
    for sample in valid_samples:
        gpu = int(sample["gpu_index"])
        used = int(sample["memory_used_mib"])
        peaks[gpu] = max(peaks.get(gpu, 0), used)
        if int(sample["memory_total_mib"]) != EXPECTED_TOTAL_MIB:
            reasons.append(f"GPU {gpu} total memory is not {EXPECTED_TOTAL_MIB} MiB")
        if used >= EXPECTED_TOTAL_MIB * GATE_MAX_FRACTION:
            reasons.append(f"GPU {gpu} reached the concurrent memory ceiling")
        for process in sample["processes"]:  # type: ignore[union-attr]
            pid = int(process["pid"])
            name = str(process["name"])
            if pid in expected_by_pid:
                label, expected_gpu = expected_by_pid[pid]
                if gpu != expected_gpu:
                    reasons.append(f"GPU mismatch for {label}: expected {expected_gpu}, observed {gpu}")
            elif (pid, name) not in allowed.get(gpu, set()):
                reasons.append(f"unexpected process PID {pid} on GPU {gpu}")
    expected_pid_set = set(expected_by_pid)
    overlap_cycle_indices: list[int] = []
    for cycle_index, cycle in enumerate(cycles):
        live = {
            int(process["pid"])
            for sample in cycle
            for process in sample["processes"]  # type: ignore[union-attr]
            if int(process["pid"]) in expected_by_pid
            and int(sample["gpu_index"]) == expected_by_pid[int(process["pid"])][1]
        }
        if expected_pid_set and expected_pid_set <= live:
            overlap_cycle_indices.append(cycle_index)
    overlap = bool(overlap_cycle_indices)
    if not overlap:
        reasons.append("no sample proves overlap of all declared gate PIDs")
    if exit_codes is not None:
        if set(exit_codes) != set(expected):
            reasons.append("exit code labels do not match expected PID labels")
        for label, code in exit_codes.items():
            if not _is_int(code) or code != 0:
                reasons.append(f"{label} exit code is not zero")
    if _health_regressed(health_before, health_after):
        reasons.append("GPU health regressed")
    final_processes = _processes_from_last_samples(valid_samples)
    summary: dict[str, object] = {
        "sample_count": len(samples),
        "sample_cycle_count": len(cycles),
        "overlap_observed": overlap,
        "overlap_cycle_indices": overlap_cycle_indices,
        "expected_pids": expected,
        "baseline_processes_by_gpu": baseline,
        "final_processes_by_gpu": final_processes,
        "peak_memory_used_mib_by_gpu": {str(gpu): value for gpu, value in sorted(peaks.items())},
    }
    thresholds = {
        "expected_total_mib": EXPECTED_TOTAL_MIB,
        "maximum_memory_fraction": GATE_MAX_FRACTION,
        "maximum_memory_used_mib_exclusive": EXPECTED_TOTAL_MIB * GATE_MAX_FRACTION,
    }
    return _base_result(
        mode="concurrent",
        reasons=reasons,
        thresholds=thresholds,
        summary=summary,
        health_before=health_before,
        health_after=health_after,
    )


def _load_json_regular(path: Path) -> object:
    try:
        return json.loads(_read_regular_text_no_follow(path))
    except json.JSONDecodeError as error:
        raise GpuGateError(f"cannot parse JSON file {path}: {error}") from error


def _read_regular_text_no_follow(path: Path) -> str:
    """Open a control file once and read only the verified regular-file descriptor."""
    path = Path(path)
    if not path.is_absolute():
        raise GpuGateError(f"path must be absolute: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GpuGateError(f"cannot open regular file without following links: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GpuGateError(f"expected a regular file: {path}")
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                return stream.read()
        except (OSError, UnicodeError) as error:
            raise GpuGateError(f"cannot read regular file: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_expected_pid_files(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    seen_paths: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path in seen_paths:
            raise GpuGateError(f"duplicate expected PID path: {path}")
        seen_paths.add(path)
        value = _load_json_regular(path)
        if not isinstance(value, dict) or set(value) != {"label", "gpu_index", "pid"}:
            raise GpuGateError(f"expected PID file {path} must use exact keys")
        label = value["label"]
        gpu_index = value["gpu_index"]
        pid = value["pid"]
        if not isinstance(label, str) or not label:
            raise GpuGateError(f"expected PID label in {path} must be a nonempty string")
        if label in result:
            raise GpuGateError(f"duplicate label in expected PID files: {label}")
        if not _is_int(gpu_index) or int(gpu_index) < 0:
            raise GpuGateError(f"GPU index in {path} must be a nonnegative integer")
        if not _is_int(pid) or int(pid) <= 0:
            raise GpuGateError(f"PID in {path} must be a positive integer")
        result[label] = {"gpu_index": int(gpu_index), "pid": int(pid)}
    if len({value["pid"] for value in result.values()}) != len(result):
        raise GpuGateError("expected PID values must be unique")
    return result


def parse_exit_file_specs(specs: Sequence[str], *, expected_labels: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise GpuGateError("each exit-file must be label=/absolute/path")
        path = Path(raw_path)
        if not path.is_absolute():
            raise GpuGateError(f"exit-file path must be absolute: {path}")
        if label not in expected_labels:
            raise GpuGateError(f"unexpected exit-file label: {label}")
        if label in result:
            raise GpuGateError(f"duplicate exit-file label: {label}")
        if path in seen_paths:
            raise GpuGateError(f"duplicate exit-file path: {path}")
        if path.is_symlink():
            raise GpuGateError(f"exit-file path may not be a symlink: {path}")
        result[label] = path
        seen_paths.add(path)
    if set(result) != expected_labels:
        raise GpuGateError("exit-file labels must exactly match expected labels")
    return result


def read_exit_codes(specs: Mapping[str, Path]) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, path in specs.items():
        value = _read_regular_text_no_follow(path)
        if not _INTEGER_RE.fullmatch(value):
            raise GpuGateError(f"exit file {path} must contain exactly one integer")
        result[label] = int(value)
    return result


def collect_samples(
    *,
    gpu_indices: tuple[int, ...],
    duration_seconds: int,
    timeout_seconds: int,
    snapshot: Callable[[tuple[int, ...]], list[dict[str, object]]],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, object]], bool]:
    """Sample on monotonic one-second deadlines without signaling workloads."""
    if duration_seconds <= 0 or timeout_seconds <= 0:
        raise GpuGateError("duration and timeout must be positive")
    start = monotonic()
    deadline = start
    samples: list[dict[str, object]] = []
    for sample_index in range(duration_seconds):
        if monotonic() - start >= timeout_seconds:
            return samples, True
        samples.extend(snapshot(gpu_indices))
        if monotonic() - start >= timeout_seconds:
            return samples, True
        if sample_index + 1 < duration_seconds:
            deadline += 1.0
            delay = deadline - monotonic()
            if delay > 0:
                sleep(delay)
    return samples, False


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise GpuGateError(f"artifact destination already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise GpuGateError(f"artifact destination already exists: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def write_artifacts(
    samples: Sequence[Mapping[str, object]],
    result: Mapping[str, object],
    csv_path: Path,
    jsonl_path: Path,
    result_path: Path,
) -> None:
    if set(result) != _RESULT_KEYS:
        raise GpuGateError("result artifact must use the exact field set")
    destinations = [Path(csv_path), Path(jsonl_path), Path(result_path)]
    if len(set(destinations)) != len(destinations):
        raise GpuGateError("artifact destinations must be distinct")
    for destination in destinations:
        if os.path.lexists(destination):
            raise GpuGateError(f"artifact destination already exists: {destination}")
    csv_fields = [
        "timestamp_utc",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
    ]
    csv_buffer: list[str] = []
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=csv_fields)
    writer.writeheader()
    for sample in samples:
        writer.writerow({key: sample[key] for key in csv_fields})
    csv_buffer.append(stream.getvalue())
    process_lines = []
    for sample in samples:
        for process in sample["processes"]:  # type: ignore[union-attr]
            process_lines.append(
                json.dumps(
                    {
                        "timestamp_utc": sample["timestamp_utc"],
                        "gpu_index": sample["gpu_index"],
                        "gpu_uuid": sample["gpu_uuid"],
                        "pid": process["pid"],
                        "name": process["name"],
                        "used_memory_mib": process["used_memory_mib"],
                    },
                    sort_keys=True,
                )
            )
    jsonl_content = "" if not process_lines else "\n".join(process_lines) + "\n"
    _atomic_write_text(Path(csv_path), csv_buffer[0])
    _atomic_write_text(Path(jsonl_path), jsonl_content)
    _atomic_write_text(Path(result_path), json.dumps(result, indent=2, sort_keys=True) + "\n")


def _parse_int(value: str) -> int:
    value = value.strip()
    match = re.fullmatch(r"-?[0-9]+", value)
    if match is None:
        raise GpuGateError(f"nvidia-smi returned a noninteger value: {value!r}")
    return int(match.group())


def _run_nvidia_smi(
    query: str,
    *,
    gpu_indices: tuple[int, ...] | None = None,
) -> list[list[str]]:
    command = ["nvidia-smi"]
    if gpu_indices is not None:
        command.append(f"--id={','.join(str(index) for index in gpu_indices)}")
    command.extend(
        [
            f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuGateError(f"nvidia-smi failed: {error}") from error
    return [
        [field.strip() for field in row]
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        if row
    ]


def snapshot_gpus(gpu_indices: tuple[int, ...]) -> list[dict[str, object]]:
    gpu_rows = _run_nvidia_smi("gpu:index,uuid,memory.used,memory.total,utilization.gpu")
    process_rows = _run_nvidia_smi("compute-apps:gpu_uuid,pid,process_name,used_gpu_memory")
    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    for row in process_rows:
        if len(row) != 4:
            raise GpuGateError("unexpected nvidia-smi compute-apps schema")
        processes_by_uuid.setdefault(row[0], []).append(
            {"pid": _parse_int(row[1]), "name": row[2], "used_memory_mib": _parse_int(row[3])}
        )
    requested = set(gpu_indices)
    samples: list[dict[str, object]] = []
    timestamp = _utc_now()
    for row in gpu_rows:
        if len(row) != 5:
            raise GpuGateError("unexpected nvidia-smi GPU schema")
        gpu = _parse_int(row[0])
        if gpu not in requested:
            continue
        samples.append(
            {
                "timestamp_utc": timestamp,
                "gpu_index": gpu,
                "gpu_uuid": row[1],
                "memory_used_mib": _parse_int(row[2]),
                "memory_total_mib": _parse_int(row[3]),
                "utilization_gpu_percent": _parse_int(row[4]),
                "processes": sorted(processes_by_uuid.get(row[1], []), key=lambda value: int(value["pid"])),
            }
        )
    if {int(sample["gpu_index"]) for sample in samples} != requested:
        raise GpuGateError("nvidia-smi did not return every requested GPU")
    return sorted(samples, key=lambda sample: int(sample["gpu_index"]))


def _health_nonnegative_int(value: str, label: str) -> int:
    parsed = _parse_int(value)
    if parsed < 0:
        raise GpuGateError(f"nvidia-smi {label} must be nonnegative")
    return parsed


def _retired_page_counts(
    rows: Sequence[Sequence[str]],
    uuid_to_index: Mapping[str, int],
) -> dict[str, int]:
    causes = ("Single Bit ECC", "Double Bit ECC")
    entries = {
        uuid: {cause: {"placeholder_count": 0, "addresses": set()} for cause in causes} for uuid in uuid_to_index
    }
    for row in rows:
        if len(row) != 4:
            raise GpuGateError("unexpected nvidia-smi retired-page health schema")
        uuid, address, timestamp, cause = row
        if uuid not in entries:
            raise GpuGateError(f"retired-page health returned unexpected GPU UUID: {uuid}")
        if cause not in causes:
            raise GpuGateError(f"retired-page health returned unexpected cause: {cause}")
        entry = entries[uuid][cause]
        address_is_placeholder = address == "N/A"
        timestamp_is_placeholder = timestamp == "N/A"
        if address_is_placeholder or timestamp_is_placeholder:
            if not address_is_placeholder or not timestamp_is_placeholder:
                raise GpuGateError("retired-page health returned a partial N/A placeholder")
            entry["placeholder_count"] += 1
            continue
        if re.fullmatch(r"0x[0-9A-Fa-f]+", address) is None or not timestamp:
            raise GpuGateError("retired-page health returned a malformed address row")
        addresses = entry["addresses"]
        if address in addresses:
            raise GpuGateError("retired-page health returned a duplicate address")
        addresses.add(address)

    counts: dict[str, int] = {}
    for uuid, by_cause in entries.items():
        count = 0
        for cause in causes:
            placeholder_count = by_cause[cause]["placeholder_count"]
            addresses = by_cause[cause]["addresses"]
            exact_zero_placeholder = placeholder_count == 1 and not addresses
            actual_rows = placeholder_count == 0 and bool(addresses)
            if not exact_zero_placeholder and not actual_rows:
                raise GpuGateError(f"retired-page health has an invalid placeholder contract for {uuid} {cause}")
            count += len(addresses)
        counts[uuid] = count
    return counts


def snapshot_health(gpu_indices: tuple[int, ...]) -> dict[str, object]:
    if (
        not gpu_indices
        or len(set(gpu_indices)) != len(gpu_indices)
        or any(not _is_int(index) or index < 0 for index in gpu_indices)
    ):
        raise GpuGateError("health GPU indices must be unique nonnegative integers")
    requested = set(gpu_indices)
    gpu_rows = _run_nvidia_smi(
        "gpu:index,uuid,ecc.errors.uncorrected.volatile.total",
        gpu_indices=gpu_indices,
    )
    uuid_by_index: dict[int, str] = {}
    ecc_by_uuid: dict[str, int] = {}
    for row in gpu_rows:
        if len(row) != 3:
            raise GpuGateError("unexpected nvidia-smi GPU health schema")
        gpu = _health_nonnegative_int(row[0], "GPU index")
        uuid = row[1]
        if gpu not in requested:
            raise GpuGateError(f"GPU health returned unexpected index: {gpu}")
        if not uuid or uuid in {"N/A", "[N/A]"}:
            raise GpuGateError("GPU health returned an unavailable UUID")
        if gpu in uuid_by_index or uuid in ecc_by_uuid:
            raise GpuGateError("GPU health returned duplicate index or UUID rows")
        uuid_by_index[gpu] = uuid
        ecc_by_uuid[uuid] = _health_nonnegative_int(row[2], "uncorrected ECC")
    if set(uuid_by_index) != requested:
        raise GpuGateError("nvidia-smi did not return health for every requested GPU")

    remap_rows = _run_nvidia_smi(
        "remapped-rows:gpu_uuid,correctable,uncorrectable,pending,failure",
        gpu_indices=gpu_indices,
    )
    remap_by_uuid: dict[str, dict[str, int]] = {}
    for row in remap_rows:
        if len(row) != 5:
            raise GpuGateError("unexpected nvidia-smi row-remap health schema")
        uuid = row[0]
        if uuid not in ecc_by_uuid:
            raise GpuGateError(f"row-remap health returned unexpected GPU UUID: {uuid}")
        if uuid in remap_by_uuid:
            raise GpuGateError(f"row-remap health returned duplicate GPU UUID: {uuid}")
        remap_by_uuid[uuid] = {
            "row_remap_correctable": _health_nonnegative_int(row[1], "correctable row remaps"),
            "row_remap_uncorrectable": _health_nonnegative_int(row[2], "uncorrectable row remaps"),
            "row_remap_pending": _health_nonnegative_int(row[3], "pending row remaps"),
            "row_remap_failure": _health_nonnegative_int(row[4], "row-remap failures"),
        }
    if set(remap_by_uuid) != set(ecc_by_uuid):
        raise GpuGateError("row-remap health did not return every requested GPU UUID")

    retired_rows = _run_nvidia_smi(
        "retired-pages:gpu_uuid,address,timestamp,cause",
        gpu_indices=gpu_indices,
    )
    retired_by_uuid = _retired_page_counts(retired_rows, {uuid: gpu for gpu, uuid in uuid_by_index.items()})

    result: dict[str, object] = {}
    for gpu in sorted(requested):
        uuid = uuid_by_index[gpu]
        result[str(gpu)] = {
            "ecc_uncorrected": ecc_by_uuid[uuid],
            "retired_pages": retired_by_uuid[uuid],
            **remap_by_uuid[uuid],
        }
    return result


def _collect_concurrent(
    *,
    gpu_indices: tuple[int, ...],
    prelaunch_seconds: int,
    post_exit_seconds: int,
    timeout_seconds: int,
    pid_paths: Sequence[Path],
    exit_specs_raw: Sequence[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[list[dict[str, object]]],
    dict[str, dict[str, int]],
    dict[str, int] | None,
    bool,
    list[str],
]:
    start = time.monotonic()
    next_deadline = start
    samples: list[dict[str, object]] = []
    prelaunch: list[dict[str, object]] = []
    sample_cycles: list[list[dict[str, object]]] = []
    expected: dict[str, dict[str, int]] = {}
    exit_specs: dict[str, Path] | None = None
    exit_codes: dict[str, int] | None = None
    exit_seen_at: float | None = None
    errors: list[str] = []
    tick = 0
    while True:
        now = time.monotonic()
        if now - start >= timeout_seconds:
            errors.append("timeout while waiting for PID or exit files")
            return samples, prelaunch, sample_cycles, expected, exit_codes, True, errors
        current = snapshot_gpus(gpu_indices)
        samples.extend(current)
        sample_cycles.append(current)
        if time.monotonic() - start >= timeout_seconds:
            errors.append("timeout while waiting for PID or exit files")
            return samples, prelaunch, sample_cycles, expected, exit_codes, True, errors
        if tick < prelaunch_seconds:
            prelaunch.extend(current)
        elif not expected and all(path.exists() for path in pid_paths):
            try:
                expected = load_expected_pid_files(pid_paths)
                exit_specs = parse_exit_file_specs(exit_specs_raw, expected_labels=set(expected))
            except GpuGateError as error:
                errors.append(str(error))
                return samples, prelaunch, sample_cycles, expected, exit_codes, False, errors
        if expected and exit_specs is not None and all(path.exists() for path in exit_specs.values()):
            if exit_seen_at is None:
                try:
                    exit_codes = read_exit_codes(exit_specs)
                except GpuGateError as error:
                    errors.append(str(error))
                    return samples, prelaunch, sample_cycles, expected, exit_codes, False, errors
                exit_seen_at = now
            if now - exit_seen_at >= post_exit_seconds:
                if time.monotonic() - start >= timeout_seconds:
                    errors.append("timeout while waiting for PID or exit files")
                    return (
                        samples,
                        prelaunch,
                        sample_cycles,
                        expected,
                        exit_codes,
                        True,
                        errors,
                    )
                return samples, prelaunch, sample_cycles, expected, exit_codes, False, errors
        tick += 1
        next_deadline += 1.0
        delay = next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "concurrent"), required=True)
    parser.add_argument("--gpu-indices", nargs="+", type=int, required=True)
    parser.add_argument("--duration-seconds", type=_positive_int)
    parser.add_argument("--timeout-seconds", type=_positive_int, required=True)
    parser.add_argument("--expected-pid-file", action="append", type=_absolute_path, default=[])
    parser.add_argument("--exit-file", action="append", default=[])
    parser.add_argument("--prelaunch-seconds", type=_positive_int)
    parser.add_argument("--post-exit-seconds", type=_positive_int)
    parser.add_argument("--csv-path", type=_absolute_path, required=True)
    parser.add_argument("--process-jsonl-path", type=_absolute_path, required=True)
    parser.add_argument("--result-json-path", type=_absolute_path, required=True)
    parser.add_argument("--health-before-path", type=_absolute_path, required=True)
    parser.add_argument("--health-after-path", type=_absolute_path, required=True)
    return parser


def _failure_result(mode: str, reason: str) -> dict[str, object]:
    return _base_result(
        mode=mode,
        reasons=[reason],
        thresholds={},
        summary={},
        health_before=None,
        health_after=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gpu_indices = tuple(args.gpu_indices)
    if len(set(gpu_indices)) != len(gpu_indices) or any(value < 0 for value in gpu_indices):
        raise SystemExit("GPU indices must be unique nonnegative integers")
    samples: list[dict[str, object]] = []
    result = _failure_result(args.mode, "monitor did not complete")
    health_before: dict[str, object] = {}
    health_after: dict[str, object] = {}
    try:
        health_before = snapshot_health(gpu_indices)
        _atomic_write_text(
            args.health_before_path,
            json.dumps(health_before, indent=2, sort_keys=True) + "\n",
        )
        if args.mode == "baseline":
            if args.duration_seconds is None:
                raise GpuGateError("baseline mode requires --duration-seconds")
            if args.expected_pid_file or args.exit_file:
                raise GpuGateError("baseline mode does not accept PID or exit files")
            samples, timed_out = collect_samples(
                gpu_indices=gpu_indices,
                duration_seconds=args.duration_seconds,
                timeout_seconds=args.timeout_seconds,
                snapshot=snapshot_gpus,
            )
            health_after = snapshot_health(gpu_indices)
            result = evaluate_baseline(
                samples=samples,
                total_mib=EXPECTED_TOTAL_MIB,
                health_before=health_before,
                health_after=health_after,
            )
            if timed_out:
                result["reasons"].append("baseline monitor timeout")  # type: ignore[union-attr]
                result["status"] = "fail"
        else:
            if args.duration_seconds is not None:
                raise GpuGateError("concurrent mode does not accept --duration-seconds")
            if args.prelaunch_seconds is None or args.post_exit_seconds is None:
                raise GpuGateError("concurrent mode requires --prelaunch-seconds and --post-exit-seconds")
            if not args.expected_pid_file or not args.exit_file:
                raise GpuGateError("concurrent mode requires PID and exit files")
            (
                samples,
                prelaunch,
                sample_cycles,
                expected,
                exit_codes,
                timed_out,
                collection_errors,
            ) = _collect_concurrent(
                gpu_indices=gpu_indices,
                prelaunch_seconds=args.prelaunch_seconds,
                post_exit_seconds=args.post_exit_seconds,
                timeout_seconds=args.timeout_seconds,
                pid_paths=args.expected_pid_file,
                exit_specs_raw=args.exit_file,
            )
            health_after = snapshot_health(gpu_indices)
            baseline_result = evaluate_baseline(
                samples=prelaunch,
                total_mib=EXPECTED_TOTAL_MIB,
                health_before=health_before,
                health_after=health_before,
            )
            result = evaluate_concurrent_gate(
                samples=samples,
                sample_cycles=sample_cycles,
                expected_pids=expected,
                baseline_processes_by_gpu=baseline_result["summary"][  # type: ignore[index]
                    "baseline_processes_by_gpu"
                ],
                exit_codes=exit_codes,
                health_before=health_before,
                health_after=health_after,
                timed_out=timed_out,
            )
            extra_reasons = collection_errors
            if baseline_result["status"] != "pass":
                extra_reasons.extend(
                    f"prelaunch baseline: {reason}"
                    for reason in baseline_result["reasons"]  # type: ignore[union-attr]
                )
            if exit_codes is None:
                extra_reasons.append("missing exit files")
            if not expected:
                extra_reasons.append("missing expected PID files")
            result["reasons"].extend(extra_reasons)  # type: ignore[union-attr]
            if result["reasons"]:
                result["status"] = "fail"
        _atomic_write_text(
            args.health_after_path,
            json.dumps(health_after, indent=2, sort_keys=True) + "\n",
        )
    except (GpuGateError, OSError, ValueError) as error:
        result = _failure_result(args.mode, str(error))
        result["health_before"] = health_before
        result["health_after"] = health_after
    try:
        write_artifacts(
            samples,
            result,
            args.csv_path,
            args.process_jsonl_path,
            args.result_json_path,
        )
    except (GpuGateError, OSError, ValueError) as error:
        print(f"GPU gate artifact publication failed: {error}", flush=True)
        return 2
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
