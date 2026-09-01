from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import verify_gr00t_training_attempt as verifier
from gear_sonic.scripts.verify_gr00t_training_attempt import (
    AttemptVerificationError,
    main,
    publish_training_attempt_verdict,
    verify_training_attempt,
)

RUN_ID = "uiqz3f58"
RUN_URL = f"https://wandb.ai/example/gr00t-n1.7-pnp-trash/runs/{RUN_ID}"
LOCAL_RUN_PATH = f"/outputs/wandb/run-20260901_010203-{RUN_ID}"
REQUIRED_METRICS = {
    "train_runtime": 12.5,
    "train_samples_per_second": 0.4,
    "train_steps_per_second": 0.08,
    "train_loss": 0.25,
}


def training_log(
    metrics: dict[str, object] | None = None,
    *,
    terminal_line: str | None = None,
    prefix: str = "",
    suffix: str = "",
) -> str:
    metrics_line = repr(REQUIRED_METRICS if metrics is None else metrics)
    if terminal_line is not None:
        metrics_line = terminal_line
    return (
        f"wandb: setting up run {RUN_ID}\n"
        f"wandb: Run data is saved locally in {LOCAL_RUN_PATH}\n"
        f"wandb: View run at {RUN_URL}\n"
        f"{prefix}"
        "Model saved to /outputs/checkpoint-1\n"
        "Training completed!\n"
        "Copied experiment_cfg into checkpoint\n"
        "wandb: View run at:\n"
        f"{metrics_line}\n"
        f"{suffix}"
    )


def write_attempt(
    tmp_path: Path,
    *,
    log: str | None = None,
    log_history: object = None,
    argument_overrides: dict[str, object] | None = None,
    audit_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, object]:
    root = tmp_path / "attempt"
    checkpoint = root / "experiment" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (root / "exit").write_text("0\n", encoding="utf-8")
    (root / "freshness-runtime.json").write_text(
        json.dumps({"get_last_checkpoint": None}),
        encoding="utf-8",
    )
    (root / "train.log").write_text(
        training_log() if log is None else log,
        encoding="utf-8",
    )
    state = {"global_step": 1}
    if log_history is not None:
        state["log_history"] = log_history
    (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    (checkpoint / "training_args.bin").write_bytes(b"fixture")
    audit = {
        "save_steps": 1,
        "logging_steps": 10,
        "deepspeed": None,
        "report_to": ["wandb"],
        **(audit_overrides or {}),
    }
    (root / "training-arguments.json").write_text(json.dumps(audit), encoding="utf-8")
    argument_values = {
        "max_steps": 1,
        "save_steps": 1,
        "logging_steps": 10,
        "deepspeed": None,
        "report_to": ["wandb"],
        **(argument_overrides or {}),
    }
    arguments = SimpleNamespace(**argument_values)
    return root, checkpoint, arguments


def verify_fixture(
    tmp_path: Path,
    *,
    log: str | None = None,
    log_history: object = None,
) -> dict[str, object]:
    root, checkpoint, arguments = write_attempt(
        tmp_path,
        log=log,
        log_history=log_history,
    )
    return verify_training_attempt(
        root,
        checkpoint,
        expected_steps=1,
        expected_save_steps=1,
        load_training_arguments=lambda path: (
            arguments if path == checkpoint / "training_args.bin" else pytest.fail(f"unexpected load path: {path}")
        ),
    )


def test_accepts_real_terminal_metrics_without_epoch_and_zero_loss(tmp_path: Path) -> None:
    metrics = {**REQUIRED_METRICS, "train_loss": 0.0}

    evidence = verify_fixture(tmp_path, log=training_log(metrics))

    assert evidence["terminal_metrics"] == metrics
    assert evidence["wandb"] == {
        "run_id": RUN_ID,
        "run_url": RUN_URL,
        "local_run_path": LOCAL_RUN_PATH,
    }
    log_bytes = training_log(metrics).encode()
    assert evidence["train_log"]["bytes"] == len(log_bytes)
    assert evidence["train_log"]["sha256"] == sha256(log_bytes).hexdigest()


def test_accepts_optional_zero_epoch(tmp_path: Path) -> None:
    metrics = {**REQUIRED_METRICS, "epoch": 0.0}

    evidence = verify_fixture(tmp_path, log=training_log(metrics))

    assert evidence["terminal_metrics"] == metrics


def test_accepts_zero_throughput_rates(tmp_path: Path) -> None:
    metrics = {
        **REQUIRED_METRICS,
        "train_samples_per_second": 0.0,
        "train_steps_per_second": 0.0,
    }

    assert verify_fixture(tmp_path, log=training_log(metrics))["terminal_metrics"] == metrics


@pytest.mark.parametrize(
    "metrics",
    [
        {**REQUIRED_METRICS, "train_runtime": 0.0},
        {**REQUIRED_METRICS, "train_loss": -0.1},
        {**REQUIRED_METRICS, "train_loss": True},
        {**REQUIRED_METRICS, "train_steps_per_second": -0.1},
        {**REQUIRED_METRICS, "epoch": -0.1},
        {**REQUIRED_METRICS, "epoch": math.inf},
    ],
)
def test_rejects_invalid_terminal_metric_values(
    tmp_path: Path,
    metrics: dict[str, object],
) -> None:
    with pytest.raises(AttemptVerificationError, match="metric|nonfinite"):
        verify_fixture(tmp_path, log=training_log(metrics))


def test_rejects_nonfinite_terminal_loss_token(tmp_path: Path) -> None:
    line = repr(REQUIRED_METRICS).replace("0.25", "nan")

    with pytest.raises(AttemptVerificationError, match="nonfinite"):
        verify_fixture(tmp_path, log=training_log(terminal_line=line))


@pytest.mark.parametrize(
    "indicator",
    ["Resuming from checkpoint", "Traceback", "CUDA out of memory", "OOM"],
)
def test_rejects_fatal_log_indicators(tmp_path: Path, indicator: str) -> None:
    with pytest.raises(AttemptVerificationError, match="fatal indicator"):
        verify_fixture(tmp_path, log=training_log(prefix=indicator + "\n"))


def test_rejects_duplicate_terminal_metrics_dicts(tmp_path: Path) -> None:
    log = training_log(suffix=repr(REQUIRED_METRICS) + "\n")

    with pytest.raises(AttemptVerificationError, match="exactly one terminal"):
        verify_fixture(tmp_path, log=log)


def test_strips_ansi_and_carriage_returns_before_parsing(tmp_path: Path) -> None:
    metrics = {**REQUIRED_METRICS, "epoch": 0}
    log = training_log(terminal_line=f"\x1b[32m{metrics!r}\x1b[0m").replace("\n", "\r")

    evidence = verify_fixture(tmp_path, log=log)

    assert evidence["terminal_metrics"] == metrics


def test_rejects_malformed_terminal_metrics(tmp_path: Path) -> None:
    malformed = "{'train_runtime': 1.0, 'train_loss': 0.2"

    with pytest.raises(AttemptVerificationError, match="exactly one terminal"):
        verify_fixture(tmp_path, log=training_log(terminal_line=malformed))


def test_rejects_terminal_metrics_before_success_anchors(tmp_path: Path) -> None:
    log = training_log(
        terminal_line="{'unrelated': 1}",
        prefix=repr(REQUIRED_METRICS) + "\n",
    )

    with pytest.raises(AttemptVerificationError, match="order"):
        verify_fixture(tmp_path, log=log)


@pytest.mark.parametrize(
    "log",
    [
        training_log(prefix="Model saved to duplicate\n"),
        training_log(prefix="Training completed!\n"),
        training_log(suffix="Model saved to too late\n"),
        training_log(suffix="Training completed!\n"),
        training_log(prefix=repr(REQUIRED_METRICS) + "\n"),
    ],
)
def test_requires_unique_ordered_success_anchors_and_terminal_metrics(
    tmp_path: Path,
    log: str,
) -> None:
    with pytest.raises(AttemptVerificationError, match="exactly one|order"):
        verify_fixture(tmp_path, log=log)


def test_accepts_finite_present_trainer_state_losses(tmp_path: Path) -> None:
    evidence = verify_fixture(
        tmp_path,
        log_history=[{"loss": 0.0}, {"loss": 0.5}, {"epoch": 0.1}],
    )

    assert evidence["trainer_state_loss_rows"] == 2


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_rejects_invalid_present_trainer_state_losses(tmp_path: Path, loss: object) -> None:
    with pytest.raises(AttemptVerificationError, match="trainer_state.*(?:loss|nonfinite)"):
        verify_fixture(tmp_path, log_history=[{"loss": loss}])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 2),
        ("save_steps", 2),
        ("logging_steps", 1),
        ("deepspeed", {"stage": 2}),
        ("report_to", []),
    ],
)
def test_rejects_wrong_effective_training_arguments(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, checkpoint, arguments = write_attempt(
        tmp_path,
        argument_overrides={field: value},
    )

    with pytest.raises(AttemptVerificationError, match=f"TrainingArguments.{field}"):
        verify_training_attempt(
            root,
            checkpoint,
            expected_steps=1,
            expected_save_steps=1,
            load_training_arguments=lambda _path: arguments,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("save_steps", 2),
        ("logging_steps", 1),
        ("deepspeed", {"stage": 2}),
        ("report_to", []),
    ],
)
def test_rejects_wrong_training_arguments_audit(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root, checkpoint, arguments = write_attempt(
        tmp_path,
        audit_overrides={field: value},
    )

    with pytest.raises(AttemptVerificationError, match=f"audit field {field}"):
        verify_training_attempt(
            root,
            checkpoint,
            expected_steps=1,
            expected_save_steps=1,
            load_training_arguments=lambda _path: arguments,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate-setup", "setting up run"),
        ("wrong-setup", "W&B.*ID"),
        ("missing-url", "online run URL"),
        ("wrong-url", "W&B.*ID"),
        ("missing-local", "local run path"),
        ("wrong-local", "W&B.*ID"),
    ],
)
def test_rejects_missing_duplicate_or_mismatched_wandb_identity(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    log = training_log()
    if case == "duplicate-setup":
        log += f"wandb: setting up run {RUN_ID}\n"
    elif case == "wrong-setup":
        log = log.replace(f"setting up run {RUN_ID}", "setting up run otherid")
    elif case == "missing-url":
        log = log.replace(RUN_URL, "")
    elif case == "wrong-url":
        log = log.replace(f"runs/{RUN_ID}", "runs/otherid")
    elif case == "missing-local":
        log = log.replace(LOCAL_RUN_PATH, "/outputs/not-a-wandb-run")
    else:
        log = log.replace(f"run-20260901_010203-{RUN_ID}", "run-20260901_010203-otherid")

    with pytest.raises(AttemptVerificationError, match=match):
        verify_fixture(tmp_path, log=log)


def test_accepts_blank_final_wandb_url_and_repeated_same_online_url(tmp_path: Path) -> None:
    log = training_log(suffix=f"wandb: View run at {RUN_URL}\nwandb: logs at {LOCAL_RUN_PATH}/logs\n")

    evidence = verify_fixture(tmp_path, log=log)

    assert evidence["wandb"]["run_id"] == RUN_ID


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_rejects_nonregular_train_log(tmp_path: Path, kind: str) -> None:
    root, checkpoint, arguments = write_attempt(tmp_path)
    train_log = root / "train.log"
    train_log.unlink()
    if kind == "symlink":
        target = tmp_path / "outside.log"
        target.write_text(training_log(), encoding="utf-8")
        train_log.symlink_to(target)
    else:
        train_log.mkdir()

    with pytest.raises(AttemptVerificationError, match="train.log.*regular non-symlink"):
        verify_training_attempt(
            root,
            checkpoint,
            expected_steps=1,
            expected_save_steps=1,
            load_training_arguments=lambda _path: arguments,
        )


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_publish_rejects_every_preexisting_destination(tmp_path: Path, kind: str) -> None:
    destination = tmp_path / "training-attempt-verdict.json"
    if kind == "file":
        destination.write_text("accepted evidence", encoding="utf-8")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing")

    with pytest.raises(AttemptVerificationError, match="already exists"):
        publish_training_attempt_verdict(destination, {"status": "pass"})

    if kind == "file":
        assert destination.read_text(encoding="utf-8") == "accepted evidence"


def test_publish_loses_race_without_overwriting_accepted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "training-attempt-verdict.json"

    def lose_race(_source: object, target: object) -> None:
        Path(target).write_text("winner", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(verifier.os, "link", lose_race)

    with pytest.raises(AttemptVerificationError, match="already exists"):
        publish_training_attempt_verdict(destination, {"status": "pass"})

    assert destination.read_text(encoding="utf-8") == "winner"
    assert not list(tmp_path.glob(".*.tmp"))


def test_main_verifies_and_publishes_final_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, checkpoint, arguments = write_attempt(tmp_path)
    loads = []

    def load(path: Path, *, map_location: str, weights_only: bool) -> object:
        loads.append((path, map_location, weights_only))
        return arguments

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))

    assert main([str(root), str(checkpoint), "1", "1"]) == 0

    destination = root / "training-attempt-verdict.json"
    evidence = json.loads(destination.read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["wandb"]["run_id"] == RUN_ID
    assert loads == [(checkpoint / "training_args.bin", "cpu", False)]
