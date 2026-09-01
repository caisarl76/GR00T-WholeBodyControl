from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import monitor_gr00t_gpu_gate as gpu_gate
from gear_sonic.scripts.monitor_gr00t_gpu_gate import (
    GpuGateError,
    collect_samples,
    evaluate_baseline,
    evaluate_concurrent_gate,
    load_expected_pid_files,
    parse_exit_file_specs,
    read_exit_codes,
    write_artifacts,
)


def _sample(
    used_mib: int,
    processes: list[tuple[int, str, int]],
    *,
    second: int = 0,
    gpu_index: int = 7,
) -> dict[str, object]:
    return {
        "timestamp_utc": f"2026-08-31T00:00:{second:02d}Z",
        "gpu_index": gpu_index,
        "gpu_uuid": f"GPU-fixture-{gpu_index}",
        "memory_used_mib": used_mib,
        "memory_total_mib": 81559,
        "utilization_gpu_percent": 50,
        "processes": [{"pid": pid, "name": name, "used_memory_mib": memory} for pid, name, memory in processes],
    }


def test_evaluate_baseline_accepts_stable_low_memory() -> None:
    samples = [_sample(13000, [(101, "eval", 6500)], second=i) for i in range(30)]
    result = evaluate_baseline(samples=samples, total_mib=81559)
    assert result["status"] == "pass"
    assert result["thresholds"] == {
        "expected_total_mib": 81559,
        "maximum_memory_fraction": 0.25,
        "maximum_memory_used_mib_exclusive": 20389.75,
        "maximum_process_range_mib": 256,
        "maximum_aggregate_range_mib": 512,
    }


def test_evaluate_baseline_rejects_wrong_h100_total() -> None:
    result = evaluate_baseline(samples=[_sample(13000, [(101, "eval", 6500)])], total_mib=80000)
    assert result["status"] == "fail"
    assert any("81559" in reason for reason in result["reasons"])


def test_evaluate_baseline_rejects_health_regression() -> None:
    result = evaluate_baseline(
        samples=[_sample(13000, [(101, "eval", 6500)])],
        total_mib=81559,
        health_before={"ecc_uncorrected": 0, "retired_pages": 0, "row_remap_pending": 0},
        health_after={"ecc_uncorrected": 1, "retired_pages": 0, "row_remap_pending": 0},
    )
    assert result["status"] == "fail"
    assert any("health" in reason for reason in result["reasons"])


def test_evaluate_concurrent_gate_rejects_95_percent_sample() -> None:
    samples = []
    for second in range(3):
        samples.extend(
            [
                _sample(77482, [(101, "full", 76000)], second=second, gpu_index=7),
                _sample(30000, [(202, "subtasks", 29000)], second=second, gpu_index=6),
            ]
        )
    result = evaluate_concurrent_gate(
        samples=samples,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("memory ceiling" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    ("samples", "reason"),
    [
        (
            [
                _sample(13000, [(101, "eval", 6500)]),
                _sample(13000, [(102, "eval", 6500)], second=1),
            ],
            "process set",
        ),
        (
            [
                _sample(13000, [(101, "eval", 1000)]),
                _sample(13000, [(101, "eval", 1257)], second=1),
            ],
            "process memory range",
        ),
        (
            [
                _sample(13000, [(101, "eval", 6500)]),
                _sample(13513, [(101, "eval", 6500)], second=1),
            ],
            "aggregate memory range",
        ),
    ],
)
def test_evaluate_baseline_rejects_instability(
    samples: list[dict[str, object]],
    reason: str,
) -> None:
    result = evaluate_baseline(samples=samples, total_mib=81559)
    assert result["status"] == "fail"
    assert any(reason in item for item in result["reasons"])


def test_concurrent_gate_requires_overlap_and_exact_processes() -> None:
    no_overlap = [_sample(20000, [(101, "full", 10000)])]
    result = evaluate_concurrent_gate(
        samples=no_overlap,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("overlap" in item for item in result["reasons"])

    unexpected = [
        _sample(30000, [(101, "full", 10000), (303, "other", 1000)], gpu_index=7),
        _sample(30000, [(202, "subtasks", 10000)], gpu_index=6),
    ]
    result = evaluate_concurrent_gate(
        samples=unexpected,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )
    assert result["status"] == "fail"
    assert any("unexpected process" in item for item in result["reasons"])


def test_concurrent_gate_does_not_union_distinct_cycles_with_duplicate_timestamps() -> None:
    first_cycle = [
        _sample(30000, [(202, "subtasks", 10000)], gpu_index=6),
        _sample(30000, [], gpu_index=7),
    ]
    second_cycle = [
        _sample(30000, [], gpu_index=6),
        _sample(30000, [(101, "full", 10000)], gpu_index=7),
    ]

    result = evaluate_concurrent_gate(
        samples=[*first_cycle, *second_cycle],
        sample_cycles=[first_cycle, second_cycle],
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
    )

    assert result["status"] == "fail"
    assert result["summary"]["sample_cycle_count"] == 2
    assert result["summary"]["overlap_cycle_indices"] == []
    assert any("overlap" in reason for reason in result["reasons"])


def test_concurrent_gate_accepts_baseline_processes_and_normalizes_summaries() -> None:
    baseline = {
        "7": [{"pid": 9, "name": "eval"}],
        "6": [{"pid": 8, "name": "display"}],
    }
    samples = [
        _sample(
            30000,
            [(9, "eval", 5000), (101, "full", 20000)],
            gpu_index=7,
        ),
        _sample(
            30000,
            [(8, "display", 5000), (202, "subtasks", 20000)],
            gpu_index=6,
        ),
        _sample(5000, [(9, "eval", 5000)], second=1, gpu_index=7),
        _sample(5000, [(8, "display", 5000)], second=1, gpu_index=6),
    ]
    result = evaluate_concurrent_gate(
        samples=samples,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        baseline_processes_by_gpu=baseline,
        exit_codes={"full": 0, "subtasks": 0},
        total_mib=81559,
    )

    assert result["status"] == "pass"
    assert result["summary"]["baseline_processes_by_gpu"] == {
        "6": [{"pid": 8, "name": "display"}],
        "7": [{"pid": 9, "name": "eval"}],
    }
    assert result["summary"]["final_processes_by_gpu"] == {
        "6": [{"pid": 8, "name": "display"}],
        "7": [{"pid": 9, "name": "eval"}],
    }


def test_concurrent_gate_rejects_gpu_mismatch_nonzero_exit_and_health_regression() -> None:
    samples = [
        _sample(30000, [(202, "subtasks", 10000)], gpu_index=7),
        _sample(30000, [(101, "full", 10000)], gpu_index=6),
    ]
    result = evaluate_concurrent_gate(
        samples=samples,
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
        exit_codes={"full": 1, "subtasks": 0},
        health_before={"ecc_uncorrected": 0},
        health_after={"ecc_uncorrected": 1},
    )

    assert result["status"] == "fail"
    assert any("GPU mismatch" in item for item in result["reasons"])
    assert any("exit code" in item for item in result["reasons"])
    assert any("health" in item for item in result["reasons"])


def test_concurrent_gate_rejects_timeout() -> None:
    result = evaluate_concurrent_gate(
        samples=[
            _sample(30000, [(101, "full", 10000)], gpu_index=7),
            _sample(30000, [(202, "subtasks", 10000)], gpu_index=6),
        ],
        expected_pids={
            "full": {"gpu_index": 7, "pid": 101},
            "subtasks": {"gpu_index": 6, "pid": 202},
        },
        total_mib=81559,
        timed_out=True,
    )
    assert result["status"] == "fail"
    assert any("timeout" in item for item in result["reasons"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda sample: sample.update(extra="forbidden"),
        lambda sample: sample.update(memory_total_mib=81558),
        lambda sample: sample["processes"][0].update(extra="forbidden"),
        lambda sample: sample.update(memory_used_mib=True),
    ],
)
def test_evaluators_reject_noncanonical_sample_schema(mutation) -> None:
    sample = _sample(13000, [(101, "eval", 6500)])
    mutation(sample)
    result = evaluate_baseline(samples=[sample], total_mib=81559)
    assert result["status"] == "fail"
    assert any("sample schema" in reason for reason in result["reasons"])


def test_load_expected_pid_files_requires_exact_unique_contract(tmp_path: Path) -> None:
    full = tmp_path / "full.pid.json"
    subtasks = tmp_path / "subtasks.pid.json"
    full.write_text(
        json.dumps({"label": "full", "gpu_index": 7, "pid": 101}),
        encoding="utf-8",
    )
    subtasks.write_text(
        json.dumps({"label": "subtasks", "gpu_index": 6, "pid": 202}),
        encoding="utf-8",
    )

    assert load_expected_pid_files([full, subtasks]) == {
        "full": {"gpu_index": 7, "pid": 101},
        "subtasks": {"gpu_index": 6, "pid": 202},
    }

    full.write_text(
        json.dumps({"label": "full", "gpu_index": 7, "pid": 101, "extra": True}),
        encoding="utf-8",
    )
    with pytest.raises(GpuGateError, match="exact keys"):
        load_expected_pid_files([full, subtasks])


def test_load_expected_pid_files_rejects_duplicate_or_type_confused_values(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({"label": "full", "gpu_index": 7, "pid": 101}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"label": "full", "gpu_index": 6, "pid": 202}),
        encoding="utf-8",
    )
    with pytest.raises(GpuGateError, match="duplicate label"):
        load_expected_pid_files([first, second])

    second.write_text(
        json.dumps({"label": "subtasks", "gpu_index": 6, "pid": True}),
        encoding="utf-8",
    )
    with pytest.raises(GpuGateError, match="positive integer"):
        load_expected_pid_files([first, second])


def test_exit_file_contract_requires_absolute_unique_paths_and_exact_integer(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.exit"
    subtasks = tmp_path / "subtasks.exit"
    specs = parse_exit_file_specs(
        [f"full={full}", f"subtasks={subtasks}"],
        expected_labels={"full", "subtasks"},
    )
    full.write_text("0\n", encoding="utf-8")
    subtasks.write_text("1\n", encoding="utf-8")
    assert read_exit_codes(specs) == {"full": 0, "subtasks": 1}

    with pytest.raises(GpuGateError, match="absolute"):
        parse_exit_file_specs(
            ["full=relative.exit", f"subtasks={subtasks}"],
            expected_labels={"full", "subtasks"},
        )

    subtasks.write_text("0 trailing\n", encoding="utf-8")
    with pytest.raises(GpuGateError, match="integer"):
        read_exit_codes(specs)


def test_collect_samples_uses_one_hz_deadline_schedule() -> None:
    now = [100.0]
    sleep_calls: list[float] = []
    calls: list[tuple[int, ...]] = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        now[0] += seconds

    def snapshot(gpu_indices: tuple[int, ...]) -> list[dict[str, object]]:
        calls.append(gpu_indices)
        return [_sample(1000, [], second=len(calls) - 1)]

    samples, timed_out = collect_samples(
        gpu_indices=(6, 7),
        duration_seconds=3,
        timeout_seconds=10,
        snapshot=snapshot,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert not timed_out
    assert len(samples) == 3
    assert calls == [(6, 7), (6, 7), (6, 7)]
    assert sleep_calls == [1.0, 1.0]


def test_collect_samples_times_out_fail_closed_without_signaling() -> None:
    now = [0.0]

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    samples, timed_out = collect_samples(
        gpu_indices=(7,),
        duration_seconds=5,
        timeout_seconds=2,
        snapshot=lambda _: [_sample(1000, [])],
        monotonic=monotonic,
        sleep=sleep,
    )

    assert timed_out
    assert len(samples) == 2
    source = Path(gpu_gate.__file__).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "send_signal" not in source
    assert "terminate(" not in source


def test_collect_samples_checks_timeout_after_slow_final_snapshot() -> None:
    now = [0.0]

    def snapshot(_: tuple[int, ...]) -> list[dict[str, object]]:
        now[0] = 2.0
        return [_sample(1000, [])]

    samples, timed_out = collect_samples(
        gpu_indices=(7,),
        duration_seconds=1,
        timeout_seconds=1,
        snapshot=snapshot,
        monotonic=lambda: now[0],
        sleep=lambda _: None,
    )

    assert len(samples) == 1
    assert timed_out


def test_concurrent_collector_checks_timeout_after_slow_success_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "full.pid.json"
    exit_path = tmp_path / "full.exit"
    pid_path.write_text(
        json.dumps({"label": "full", "gpu_index": 7, "pid": 101}),
        encoding="utf-8",
    )
    exit_path.write_text("0\n", encoding="utf-8")
    now = [0.0]
    snapshots = [0]

    def snapshot(_: tuple[int, ...]) -> list[dict[str, object]]:
        snapshots[0] += 1
        if snapshots[0] == 3:
            now[0] = 6.0
        return [_sample(1000, [(101, "full", 500)])]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(gpu_gate, "snapshot_gpus", snapshot)
    monkeypatch.setattr(gpu_gate.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gpu_gate.time, "sleep", sleep)

    result = gpu_gate._collect_concurrent(
        gpu_indices=(7,),
        prelaunch_seconds=1,
        post_exit_seconds=1,
        timeout_seconds=5,
        pid_paths=[pid_path],
        exit_specs_raw=[f"full={exit_path}"],
    )

    assert result[-2] is True
    assert any("timeout" in reason for reason in result[-1])


@pytest.mark.parametrize("value", ["N/A", "[N/A]", "", "12 MiB", "x12", "12x", "1.0"])
def test_parse_int_rejects_missing_or_partial_values(value: str) -> None:
    with pytest.raises(GpuGateError, match="integer"):
        gpu_gate._parse_int(value)


@pytest.mark.parametrize("query", ["process", "health"])
def test_snapshot_rejects_unavailable_process_and_health_values(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    if query == "process":
        responses = iter(
            [
                [["7", "GPU-7", "1000", "81559", "0"]],
                [["GPU-7", "101", "full", "N/A"]],
            ]
        )
        monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _: next(responses))
    else:
        monkeypatch.setattr(
            gpu_gate,
            "_run_nvidia_smi",
            lambda *_args, **_kwargs: [["7", "GPU-7", "N/A"]],
        )

    with pytest.raises(GpuGateError, match="integer"):
        if query == "process":
            gpu_gate.snapshot_gpus((7,))
        else:
            gpu_gate.snapshot_health((7,))


def test_snapshot_health_uses_three_selective_h100_query_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        "--query-gpu=": "6, GPU-six, 0\n7, GPU-seven, 0\n",
        "--query-remapped-rows=": ("GPU-six, 0, 0, 0, 0\nGPU-seven, 0, 0, 0, 0\n"),
        "--query-retired-pages=": (
            "GPU-six, [N/A], [N/A], Single Bit ECC\n"
            "GPU-six, [N/A], [N/A], Double Bit ECC\n"
            "GPU-seven, [N/A], [N/A], Single Bit ECC\n"
            "GPU-seven, [N/A], [N/A], Double Bit ECC\n"
        ),
    }
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        query_flag = next(key for key in outputs if any(arg.startswith(key) for arg in command))
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return SimpleNamespace(stdout=outputs[query_flag])

    monkeypatch.setattr(gpu_gate.subprocess, "run", run)

    assert gpu_gate.snapshot_health((6, 7)) == {
        "6": {
            "ecc_uncorrected": 0,
            "retired_pages": 0,
            "row_remap_correctable": 0,
            "row_remap_uncorrectable": 0,
            "row_remap_pending": 0,
            "row_remap_failure": 0,
        },
        "7": {
            "ecc_uncorrected": 0,
            "retired_pages": 0,
            "row_remap_correctable": 0,
            "row_remap_uncorrectable": 0,
            "row_remap_pending": 0,
            "row_remap_failure": 0,
        },
    }
    assert commands == [
        [
            "nvidia-smi",
            "--id=6,7",
            "--query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--id=6,7",
            "--query-remapped-rows=gpu_uuid,remapped_rows.correctable,"
            "remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--id=6,7",
            "--query-retired-pages=gpu_uuid,retired_pages.address,retired_pages.timestamp,retired_pages.cause",
            "--format=csv,noheader,nounits",
        ],
    ]


def test_snapshot_health_counts_real_retired_page_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "gpu": [["7", "GPU-seven", "0"]],
        "remapped-rows": [["GPU-seven", "1", "2", "0", "0"]],
        "retired-pages": [
            [
                "GPU-seven",
                "0x0000000000000010",
                "2026/09/01 00:00:00.000",
                "Single Bit ECC",
            ],
            ["GPU-seven", "N/A", "N/A", "Double Bit ECC"],
        ],
    }
    monkeypatch.setattr(
        gpu_gate,
        "_run_nvidia_smi",
        lambda query, **_kwargs: responses[query.split(":", 1)[0]],
    )

    health = gpu_gate.snapshot_health((7,))

    assert health["7"]["retired_pages"] == 1
    assert health["7"]["row_remap_correctable"] == 1
    assert health["7"]["row_remap_uncorrectable"] == 2


@pytest.mark.parametrize(
    ("case", "gpu_rows", "remap_rows", "retired_rows"),
    [
        (
            "partial placeholder",
            [["7", "GPU-seven", "0"]],
            [["GPU-seven", "0", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "2026/09/01 00:00:00.000", "Single Bit ECC"],
                ["GPU-seven", "N/A", "N/A", "Double Bit ECC"],
            ],
        ),
        (
            "mixed placeholder forms",
            [["7", "GPU-seven", "0"]],
            [["GPU-seven", "0", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "[N/A]", "Single Bit ECC"],
                ["GPU-seven", "[N/A]", "[N/A]", "Double Bit ECC"],
            ],
        ),
        (
            "duplicate zero cause",
            [["7", "GPU-seven", "0"]],
            [["GPU-seven", "0", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "N/A", "Single Bit ECC"],
                ["GPU-seven", "N/A", "N/A", "Single Bit ECC"],
            ],
        ),
        (
            "unknown retired cause",
            [["7", "GPU-seven", "0"]],
            [["GPU-seven", "0", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "N/A", "Unknown"],
                ["GPU-seven", "N/A", "N/A", "Double Bit ECC"],
            ],
        ),
        (
            "unavailable remap",
            [["7", "GPU-seven", "0"]],
            [["GPU-seven", "N/A", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "N/A", "Single Bit ECC"],
                ["GPU-seven", "N/A", "N/A", "Double Bit ECC"],
            ],
        ),
        (
            "unexpected UUID",
            [["7", "GPU-seven", "0"]],
            [["GPU-attacker", "0", "0", "0", "0"]],
            [
                ["GPU-seven", "N/A", "N/A", "Single Bit ECC"],
                ["GPU-seven", "N/A", "N/A", "Double Bit ECC"],
            ],
        ),
    ],
)
def test_snapshot_health_rejects_malformed_h100_rows(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    gpu_rows: list[list[str]],
    remap_rows: list[list[str]],
    retired_rows: list[list[str]],
) -> None:
    responses = {
        "gpu": gpu_rows,
        "remapped-rows": remap_rows,
        "retired-pages": retired_rows,
    }
    monkeypatch.setattr(
        gpu_gate,
        "_run_nvidia_smi",
        lambda query, **_kwargs: responses[query.split(":", 1)[0]],
    )

    with pytest.raises(GpuGateError, match="health|retired|remap|UUID|integer"):
        gpu_gate.snapshot_health((7,))


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
@pytest.mark.parametrize("destination_index", [0, 1, 2])
def test_write_artifacts_rejects_every_preexisting_destination_type(
    tmp_path: Path,
    kind: str,
    destination_index: int,
) -> None:
    paths = [tmp_path / "samples.csv", tmp_path / "processes.jsonl", tmp_path / "result.json"]
    destination = paths[destination_index]
    if kind == "file":
        destination.write_text("accepted", encoding="utf-8")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing-target")

    with pytest.raises(GpuGateError, match="already exists"):
        write_artifacts(
            [_sample(13000, [(101, "eval", 6500)])],
            {
                "status": "pass",
                "reasons": [],
                "mode": "baseline",
                "thresholds": {},
                "summary": {},
                "health_before": {},
                "health_after": {},
            },
            *paths,
        )

    assert os.path.lexists(destination)
    for index, path in enumerate(paths):
        if index != destination_index:
            assert not os.path.lexists(path)


@pytest.mark.parametrize("destination_index", [0, 1, 2])
def test_write_artifacts_does_not_overwrite_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_index: int,
) -> None:
    paths = [tmp_path / "samples.csv", tmp_path / "processes.jsonl", tmp_path / "result.json"]
    raced_path = paths[destination_index]
    real_link = os.link

    def racing_link(
        source: str | Path,
        destination: str | Path,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ) -> None:
        if Path(destination) == raced_path:
            raced_path.write_text("racer", encoding="utf-8")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(gpu_gate.os, "link", racing_link)
    with pytest.raises(GpuGateError, match="already exists"):
        write_artifacts(
            [_sample(13000, [(101, "eval", 6500)])],
            {
                "status": "pass",
                "reasons": [],
                "mode": "baseline",
                "thresholds": {},
                "summary": {},
                "health_before": {},
                "health_after": {},
            },
            *paths,
        )

    assert raced_path.read_text(encoding="utf-8") == "racer"
    if destination_index < 2:
        assert not os.path.lexists(paths[2])


@pytest.mark.parametrize("reader", ["pid", "exit"])
def test_control_files_reject_initial_symlinks(
    tmp_path: Path,
    reader: str,
) -> None:
    target = tmp_path / "target"
    path = tmp_path / ("control.json" if reader == "pid" else "exit")
    target.write_text(
        json.dumps({"label": "full", "gpu_index": 7, "pid": 101}) if reader == "pid" else "0\n",
        encoding="utf-8",
    )
    path.symlink_to(target)

    with pytest.raises(GpuGateError, match="without following links"):
        if reader == "pid":
            load_expected_pid_files([path])
        else:
            read_exit_codes({"full": path})


@pytest.mark.parametrize("reader", ["pid", "exit"])
def test_control_files_are_read_once_from_no_follow_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
) -> None:
    path = tmp_path / ("control.json" if reader == "pid" else "exit")
    replacement = tmp_path / "replacement"
    if reader == "pid":
        path.write_text(
            json.dumps({"label": "full", "gpu_index": 7, "pid": 101}),
            encoding="utf-8",
        )
        replacement.write_text(
            json.dumps({"label": "attacker", "gpu_index": 0, "pid": 999}),
            encoding="utf-8",
        )
    else:
        path.write_text("0\n", encoding="utf-8")
        replacement.write_text("1\n", encoding="utf-8")
    real_open = os.open
    calls: list[int] = []

    def replacing_open(raw_path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(raw_path, flags, mode, dir_fd=dir_fd)
        if Path(raw_path) == path:
            calls.append(flags)
            path.unlink()
            path.symlink_to(replacement)
        return descriptor

    monkeypatch.setattr(gpu_gate.os, "open", replacing_open)
    if reader == "pid":
        assert load_expected_pid_files([path]) == {"full": {"gpu_index": 7, "pid": 101}}
    else:
        assert read_exit_codes({"full": path}) == {"full": 0}

    assert len(calls) == 1
    assert calls[0] & os.O_NOFOLLOW
    assert calls[0] & os.O_NONBLOCK
    assert stat.S_ISLNK(path.lstat().st_mode)


def test_write_artifacts_uses_exact_field_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [_sample(13000, [(101, "eval", 6500)])]
    result = {
        "status": "pass",
        "reasons": [],
        "mode": "baseline",
        "thresholds": {},
        "summary": {},
        "health_before": {},
        "health_after": {},
    }
    csv_path = tmp_path / "samples.csv"
    jsonl_path = tmp_path / "processes.jsonl"
    result_path = tmp_path / "result.json"
    linked: list[tuple[Path, Path]] = []
    real_link = os.link

    def recording_link(source: str | Path, destination: str | Path, **kwargs) -> None:
        linked.append((Path(source), Path(destination)))
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(gpu_gate.os, "link", recording_link)
    write_artifacts(samples, result, csv_path, jsonl_path, result_path)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    process = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    stored_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(row) == {
        "timestamp_utc",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
    }
    assert set(process) == {
        "timestamp_utc",
        "gpu_index",
        "gpu_uuid",
        "pid",
        "name",
        "used_memory_mib",
    }
    assert set(stored_result) == {
        "status",
        "reasons",
        "mode",
        "thresholds",
        "summary",
        "health_before",
        "health_after",
    }
    assert linked[-1][1] == result_path
    assert linked[-1][0].parent == result_path.parent
    assert not list(tmp_path.glob("*.tmp"))
