#!/usr/bin/env python3
"""Verify one bounded GR00T training attempt and publish immutable evidence."""

import argparse
import ast
from collections.abc import Callable, Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import NoReturn


class AttemptVerificationError(RuntimeError):
    """The training-attempt evidence does not satisfy the accepted contract."""


_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_NONFINITE_TOKEN = re.compile(
    r"(?<![a-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![a-z0-9_])",
    re.IGNORECASE,
)
_REQUIRED_METRICS = frozenset(
    {
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
    }
)
_MODEL_SAVED_ANCHOR = "Model saved to"
_TRAINING_COMPLETED_ANCHOR = "Training completed"
_WANDB_SETUP = re.compile(r"\bsetting up run ([A-Za-z0-9_-]+)\b", re.IGNORECASE)
_WANDB_URL = re.compile(
    r"https?://[^\s\"'<>]+/runs/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_WANDB_LOCAL_RUN = re.compile(
    r"(?P<path>(?:\.?/)?[^\s\"'<>]*wandb/run-[^/\s\"'<>]*-(?P<id>[A-Za-z0-9_-]+))(?:/logs)?",
    re.IGNORECASE,
)


def _error(message: str) -> NoReturn:
    raise AttemptVerificationError(message)


def _required_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _error(f"{label} must be a regular non-symlink file: {path}")
    return path


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _error(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    _required_regular_file(path, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: _error(f"{label} contains nonfinite value: {value}"),
        )
    except AttemptVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttemptVerificationError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        _error(f"{label} must contain a JSON object")
    return value


def _read_exit_status(root: Path) -> None:
    path = _required_regular_file(root / "exit", label="exit status")
    try:
        status = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise AttemptVerificationError("exit status cannot be read") from exc
    if status != "0":
        _error(f"training attempt exit status must be exactly 0, got {status!r}")


def _normalize_log(log_bytes: bytes) -> str:
    text = log_bytes.decode("utf-8", errors="replace").replace("\r", "\n")
    return _ANSI_ESCAPE.sub("", text)


def _validate_no_fatal_log_indicators(clean_log: str) -> None:
    lowered = clean_log.casefold()
    for indicator in (
        "resuming from checkpoint",
        "traceback",
        "out of memory",
    ):
        if indicator in lowered:
            _error(f"train.log contains fatal indicator: {indicator}")
    if re.search(r"(?<![a-z0-9_])oom(?![a-z0-9_])", lowered):
        _error("train.log contains fatal indicator: OOM")
    if _NONFINITE_TOKEN.search(lowered):
        _error("train.log contains a nonfinite token")


def _parse_terminal_metrics(clean_log: str) -> tuple[dict[str, object], int]:
    candidates: list[tuple[dict[str, object], int]] = []
    offset = 0
    for line in clean_log.splitlines(keepends=True):
        candidate_text = line.strip()
        if candidate_text.startswith("{") and candidate_text.endswith("}"):
            try:
                candidate = ast.literal_eval(candidate_text)
            except (SyntaxError, ValueError):
                candidate = None
            if type(candidate) is dict and _REQUIRED_METRICS <= set(candidate):
                candidates.append((candidate, offset))
        offset += len(line)
    if len(candidates) != 1:
        _error(f"train.log must contain exactly one terminal metrics dictionary, got {len(candidates)}")

    metrics, position = candidates[0]
    for field in _REQUIRED_METRICS:
        value = metrics[field]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            _error(f"terminal metric {field} must be numeric, non-bool, and finite")
    if float(metrics["train_runtime"]) <= 0.0:
        _error("terminal metric train_runtime must be positive")
    for field in ("train_samples_per_second", "train_steps_per_second", "train_loss"):
        if float(metrics[field]) < 0.0:
            _error(f"terminal metric {field} must be nonnegative")
    if "epoch" in metrics:
        epoch = metrics["epoch"]
        if type(epoch) not in (int, float) or not math.isfinite(float(epoch)) or float(epoch) < 0.0:
            _error("optional terminal metric epoch must be numeric, non-bool, finite, and nonnegative")
    return metrics, position


def _validate_success_anchors(clean_log: str, metrics_position: int) -> None:
    model_saved_count = clean_log.count(_MODEL_SAVED_ANCHOR)
    training_completed_count = clean_log.count(_TRAINING_COMPLETED_ANCHOR)
    if model_saved_count != 1:
        _error(f"train.log must contain exactly one successful Model saved anchor, got {model_saved_count}")
    if training_completed_count != 1:
        _error(
            "train.log must contain exactly one successful Training completed anchor, "
            f"got {training_completed_count}"
        )
    model_saved_position = clean_log.index(_MODEL_SAVED_ANCHOR)
    training_completed_position = clean_log.index(_TRAINING_COMPLETED_ANCHOR)
    if not model_saved_position < training_completed_position < metrics_position:
        _error("train.log success evidence order must be Model saved, Training completed, terminal metrics")


def _parse_wandb_identity(clean_log: str) -> dict[str, str]:
    setup_ids = _WANDB_SETUP.findall(clean_log)
    if len(setup_ids) != 1:
        _error(f"train.log must contain exactly one W&B setting up run ID, got {len(setup_ids)}")

    urls = {match.group(0): match.group(1) for match in _WANDB_URL.finditer(clean_log)}
    if len(urls) != 1:
        _error(f"train.log must contain exactly one unique online run URL, got {len(urls)}")

    local_paths = {match.group("path"): match.group("id") for match in _WANDB_LOCAL_RUN.finditer(clean_log)}
    if len(local_paths) != 1:
        _error(f"train.log must contain exactly one unique W&B local run path, got {len(local_paths)}")

    setup_id = setup_ids[0]
    run_url, url_id = next(iter(urls.items()))
    local_run_path, local_id = next(iter(local_paths.items()))
    if setup_id != url_id or setup_id != local_id:
        _error("W&B setup, online URL, and local run path IDs do not match")
    return {
        "run_id": setup_id,
        "run_url": run_url,
        "local_run_path": local_run_path,
    }


def _validate_trainer_state(checkpoint: Path, expected_steps: int) -> tuple[int, int]:
    state = _load_json_object(checkpoint / "trainer_state.json", label="trainer_state.json")
    if type(state.get("global_step")) is not int or state["global_step"] != expected_steps:
        _error(f"trainer_state global_step must be exactly {expected_steps}")
    history = state.get("log_history", [])
    if type(history) is not list:
        _error("trainer_state log_history must be a list when present")
    loss_rows = 0
    for index, row in enumerate(history):
        if type(row) is not dict:
            _error(f"trainer_state log_history row {index} must be a dictionary")
        if "loss" not in row:
            continue
        loss_rows += 1
        loss = row["loss"]
        if type(loss) not in (int, float) or not math.isfinite(float(loss)):
            _error(f"trainer_state log_history loss at row {index} must be numeric, non-bool, and finite")
    return state["global_step"], loss_rows


def _validate_training_arguments(
    root: Path,
    checkpoint: Path,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[Path], object],
) -> dict[str, object]:
    arguments_path = _required_regular_file(
        checkpoint / "training_args.bin",
        label="training_args.bin",
    )
    try:
        arguments = load_training_arguments(arguments_path)
    except AttemptVerificationError:
        raise
    except Exception as exc:
        raise AttemptVerificationError("training_args.bin cannot be loaded") from exc
    expected = {
        "max_steps": expected_steps,
        "save_steps": expected_save_steps,
        "logging_steps": 10,
        "deepspeed": None,
        "report_to": ["wandb"],
    }
    for field, value in expected.items():
        if getattr(arguments, field, object()) != value:
            _error(f"effective TrainingArguments.{field} does not match the attempt contract")

    audit = _load_json_object(
        root / "training-arguments.json",
        label="training-arguments.json",
    )
    for field in ("save_steps", "logging_steps", "deepspeed", "report_to"):
        if audit.get(field, object()) != expected[field]:
            _error(f"training arguments audit field {field} does not match the attempt contract")
    return expected


def verify_training_attempt(
    root: Path,
    checkpoint: Path,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[Path], object],
) -> dict[str, object]:
    """Validate one completed attempt without publishing or mutating its inputs."""
    if type(expected_steps) is not int or expected_steps <= 0:
        _error("expected_steps must be a positive integer")
    if type(expected_save_steps) is not int or expected_save_steps <= 0:
        _error("expected_save_steps must be a positive integer")
    if root.is_symlink() or not root.is_dir():
        _error(f"attempt root must be a regular directory: {root}")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        _error(f"checkpoint must be a regular directory: {checkpoint}")

    _read_exit_status(root)
    freshness = _load_json_object(
        root / "freshness-runtime.json",
        label="freshness-runtime.json",
    )
    if "get_last_checkpoint" not in freshness or freshness["get_last_checkpoint"] is not None:
        _error("freshness-runtime get_last_checkpoint must be explicitly null")

    log_path = _required_regular_file(root / "train.log", label="train.log")
    try:
        log_bytes = log_path.read_bytes()
    except OSError as exc:
        raise AttemptVerificationError("train.log cannot be read") from exc
    clean_log = _normalize_log(log_bytes)
    _validate_no_fatal_log_indicators(clean_log)
    metrics, metrics_position = _parse_terminal_metrics(clean_log)
    _validate_success_anchors(clean_log, metrics_position)
    wandb = _parse_wandb_identity(clean_log)
    global_step, loss_rows = _validate_trainer_state(checkpoint, expected_steps)
    training_arguments = _validate_training_arguments(
        root,
        checkpoint,
        expected_steps=expected_steps,
        expected_save_steps=expected_save_steps,
        load_training_arguments=load_training_arguments,
    )
    return {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "global_step": global_step,
        "terminal_metrics": metrics,
        "trainer_state_loss_rows": loss_rows,
        "training_arguments": training_arguments,
        "wandb": wandb,
        "train_log": {
            "path": str(log_path),
            "bytes": len(log_bytes),
            "sha256": sha256(log_bytes).hexdigest(),
        },
    }


def publish_training_attempt_verdict(
    destination: Path,
    evidence: Mapping[str, object],
) -> None:
    """Atomically publish a final no-clobber verdict marker."""
    if not destination.is_absolute():
        _error("training-attempt verdict destination must be absolute")
    if os.path.lexists(destination):
        _error(f"training-attempt verdict destination already exists: {destination}")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        _error("training-attempt verdict parent must be a regular directory")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise AttemptVerificationError(
                f"training-attempt verdict destination already exists: {destination}"
            ) from exc
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("expected_steps", type=int)
    parser.add_argument("expected_save_steps", type=int)
    args = parser.parse_args(argv)

    import torch

    evidence = verify_training_attempt(
        args.attempt_root,
        args.checkpoint,
        expected_steps=args.expected_steps,
        expected_save_steps=args.expected_save_steps,
        load_training_arguments=lambda path: torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        ),
    )
    destination = args.attempt_root / "training-attempt-verdict.json"
    publish_training_attempt_verdict(destination, evidence)
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
