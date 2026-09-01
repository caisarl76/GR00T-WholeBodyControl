#!/usr/bin/env python3
"""Verify one bounded GR00T training attempt and publish immutable evidence."""

import argparse
import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
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
_WANDB_RUN_URL = re.compile(
    r"https?://[^\s\"'<>]+/runs/[^\s\"'<>]+",
    re.IGNORECASE,
)
_CANONICAL_WANDB_RUN_URL = re.compile(
    r"https://wandb\.ai/[^/\s\"'<>]+/gr00t-n1\.7-pnp-trash/runs/([A-Za-z0-9_-]+)",
)
_WANDB_LOCAL_RUN = re.compile(
    r"(?P<path>(?:\.?/)?[^\s\"'<>]*wandb/run-[^/\s\"'<>]*-"
    r"(?P<id>[A-Za-z0-9_-]+)(?:/logs)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _CapturedInput:
    label: str
    path: Path
    data: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    digest: str

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns

    def manifest_entry(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path": str(self.path),
            "bytes": self.size,
            "sha256": self.digest,
            "identity": {
                "device": self.device,
                "inode": self.inode,
                "mtime_ns": self.mtime_ns,
                "ctime_ns": self.ctime_ns,
            },
        }


def _error(message: str) -> NoReturn:
    raise AttemptVerificationError(message)


def _capture_regular_file(path: Path, *, label: str) -> _CapturedInput:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _error("O_NOFOLLOW is required for training-attempt evidence capture")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise AttemptVerificationError(f"{label} must be a regular non-symlink file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _error(f"{label} must be a regular non-symlink file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise AttemptVerificationError(f"{label} cannot be captured: {path}") from exc
    finally:
        os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    data = b"".join(chunks)
    if before_identity != after_identity or len(data) != before.st_size:
        _error(f"{label} changed while it was captured: {path}")
    return _CapturedInput(
        label=label,
        path=path,
        data=data,
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        digest=sha256(data).hexdigest(),
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _error(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _load_json_object(captured: _CapturedInput) -> dict[str, object]:
    try:
        value = json.loads(
            captured.data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: _error(f"{captured.label} contains nonfinite value: {value}"),
        )
    except AttemptVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttemptVerificationError(f"{captured.label} is not valid JSON") from exc
    if type(value) is not dict:
        _error(f"{captured.label} must contain a JSON object")
    return value


def _read_exit_status(captured: _CapturedInput) -> None:
    try:
        status = captured.data.decode("utf-8").strip()
    except UnicodeError as exc:
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
                expression = ast.parse(candidate_text, mode="eval")
            except (SyntaxError, ValueError):
                expression = None
            if expression is not None and isinstance(expression.body, ast.Dict):
                keys = [
                    key.value if isinstance(key, ast.Constant) and type(key.value) is str else None
                    for key in expression.body.keys
                ]
                key_set = {key for key in keys if key is not None}
                if _REQUIRED_METRICS <= key_set:
                    if len(keys) != len(key_set):
                        _error("terminal metrics dictionary contains duplicate or non-string keys")
                    allowed_keys = _REQUIRED_METRICS | {"epoch"}
                    if key_set not in (_REQUIRED_METRICS, allowed_keys):
                        _error("terminal metrics dictionary does not use the exact key set")
                    try:
                        candidate = ast.literal_eval(expression)
                    except (SyntaxError, ValueError) as exc:
                        raise AttemptVerificationError(
                            "terminal metrics dictionary contains non-literal values"
                        ) from exc
                    if type(candidate) is not dict:
                        _error("terminal metrics dictionary is not a dictionary")
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
    for field in ("train_samples_per_second", "train_steps_per_second"):
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

    run_urls = list(dict.fromkeys(_WANDB_RUN_URL.findall(clean_log)))
    if len(run_urls) != 1:
        _error(f"train.log must contain exactly one unique online run URL, got {len(run_urls)}")
    canonical_url = _CANONICAL_WANDB_RUN_URL.fullmatch(run_urls[0])
    if canonical_url is None:
        _error("W&B canonical online run URL must use HTTPS wandb.ai for gr00t-n1.7-pnp-trash")

    local_paths: list[str] = []
    local_ids = set()
    for match in _WANDB_LOCAL_RUN.finditer(clean_log):
        path = match.group("path")
        if path not in local_paths:
            local_paths.append(path)
        local_ids.add(match.group("id"))
    if not local_paths:
        _error("train.log must contain at least one W&B local run path")
    if len(local_ids) != 1:
        _error(f"train.log W&B local paths must resolve to exactly one run ID, got {len(local_ids)}")

    setup_id = setup_ids[0]
    run_url = run_urls[0]
    url_id = canonical_url.group(1)
    local_id = next(iter(local_ids))
    if setup_id != url_id or setup_id != local_id:
        _error("W&B setup, online URL, and local run path IDs do not match")
    return {
        "run_id": setup_id,
        "run_url": run_url,
        "local_run_paths": local_paths,
    }


def _validate_trainer_state(
    captured: _CapturedInput,
    expected_steps: int,
) -> tuple[int, int]:
    state = _load_json_object(captured)
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
    arguments_capture: _CapturedInput,
    audit_capture: _CapturedInput,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[BytesIO], object],
) -> dict[str, object]:
    try:
        arguments = load_training_arguments(BytesIO(arguments_capture.data))
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

    audit = _load_json_object(audit_capture)
    for field in ("save_steps", "logging_steps", "deepspeed", "report_to"):
        if audit.get(field, object()) != expected[field]:
            _error(f"training arguments audit field {field} does not match the attempt contract")
    return expected


def _capture_attempt_inputs(root: Path, checkpoint: Path) -> tuple[_CapturedInput, ...]:
    specifications = (
        ("exit status", root / "exit"),
        ("freshness-runtime.json", root / "freshness-runtime.json"),
        ("train.log", root / "train.log"),
        ("trainer_state.json", checkpoint / "trainer_state.json"),
        ("training_args.bin", checkpoint / "training_args.bin"),
        ("training-arguments.json", root / "training-arguments.json"),
    )
    return tuple(_capture_regular_file(path, label=label) for label, path in specifications)


def _revalidate_captured_inputs(captured_inputs: tuple[_CapturedInput, ...]) -> None:
    for captured in captured_inputs:
        current = _capture_regular_file(captured.path, label=captured.label)
        if current.identity != captured.identity or current.digest != captured.digest:
            _error(f"{captured.label} changed after validation: {captured.path}")


def _verify_training_attempt_with_captures(
    root: Path,
    checkpoint: Path,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[BytesIO], object],
) -> tuple[dict[str, object], tuple[_CapturedInput, ...]]:
    if type(expected_steps) is not int or expected_steps <= 0:
        _error("expected_steps must be a positive integer")
    if type(expected_save_steps) is not int or expected_save_steps <= 0:
        _error("expected_save_steps must be a positive integer")
    if root.is_symlink() or not root.is_dir():
        _error(f"attempt root must be a regular directory: {root}")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        _error(f"checkpoint must be a regular directory: {checkpoint}")

    captured_inputs = _capture_attempt_inputs(root, checkpoint)
    captures = {captured.label: captured for captured in captured_inputs}
    _read_exit_status(captures["exit status"])
    freshness = _load_json_object(captures["freshness-runtime.json"])
    if "get_last_checkpoint" not in freshness or freshness["get_last_checkpoint"] is not None:
        _error("freshness-runtime get_last_checkpoint must be explicitly null")

    log_capture = captures["train.log"]
    log_bytes = log_capture.data
    clean_log = _normalize_log(log_bytes)
    _validate_no_fatal_log_indicators(clean_log)
    metrics, metrics_position = _parse_terminal_metrics(clean_log)
    _validate_success_anchors(clean_log, metrics_position)
    wandb = _parse_wandb_identity(clean_log)
    global_step, loss_rows = _validate_trainer_state(
        captures["trainer_state.json"],
        expected_steps,
    )
    training_arguments = _validate_training_arguments(
        captures["training_args.bin"],
        captures["training-arguments.json"],
        expected_steps=expected_steps,
        expected_save_steps=expected_save_steps,
        load_training_arguments=load_training_arguments,
    )
    evidence = {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "global_step": global_step,
        "terminal_metrics": metrics,
        "trainer_state_loss_rows": loss_rows,
        "training_arguments": training_arguments,
        "wandb": wandb,
        "input_manifest": [captured.manifest_entry() for captured in captured_inputs],
        "train_log": {
            "path": str(log_capture.path),
            "bytes": len(log_bytes),
            "sha256": log_capture.digest,
        },
    }
    return evidence, captured_inputs


def verify_training_attempt(
    root: Path,
    checkpoint: Path,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[BytesIO], object],
) -> dict[str, object]:
    """Validate one completed attempt without publishing or mutating its inputs."""
    evidence, _captured_inputs = _verify_training_attempt_with_captures(
        root,
        checkpoint,
        expected_steps=expected_steps,
        expected_save_steps=expected_save_steps,
        load_training_arguments=load_training_arguments,
    )
    return evidence


def publish_training_attempt_verdict(
    destination: Path,
    evidence: Mapping[str, object],
    *,
    captured_inputs: tuple[_CapturedInput, ...] = (),
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
        _revalidate_captured_inputs(captured_inputs)
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


def verify_and_publish_training_attempt(
    root: Path,
    checkpoint: Path,
    *,
    expected_steps: int,
    expected_save_steps: int,
    load_training_arguments: Callable[[BytesIO], object],
) -> dict[str, object]:
    """Validate captured bytes, revalidate their identity, and publish once."""
    evidence, captured_inputs = _verify_training_attempt_with_captures(
        root,
        checkpoint,
        expected_steps=expected_steps,
        expected_save_steps=expected_save_steps,
        load_training_arguments=load_training_arguments,
    )
    publish_training_attempt_verdict(
        root / "training-attempt-verdict.json",
        evidence,
        captured_inputs=captured_inputs,
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("expected_steps", type=int)
    parser.add_argument("expected_save_steps", type=int)
    args = parser.parse_args(argv)

    import torch

    evidence = verify_and_publish_training_attempt(
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
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
