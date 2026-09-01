from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from safetensors.torch import save_file
import torch

from gear_sonic.scripts import verify_gr00t_n17_checkpoint as checkpoint_verifier
from gear_sonic.scripts.verify_gr00t_n17_checkpoint import (
    COSMOS_MODEL_ID,
    COSMOS_REVISION,
    CheckpointError,
    _cache_is_read_only,
    _verify_offline_load_worker,
    main,
    verify_checkpoint_structure,
    verify_offline_load,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checkpoint_fixture(root: Path, step: int = 5) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    shards = {
        "model-00001-of-00002.safetensors": {"action.weight": torch.ones(1)},
        "model-00002-of-00002.safetensors": {"projector.weight": torch.ones(1)},
    }
    for filename, tensors in shards.items():
        save_file(tensors, checkpoint / filename)
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {
            "action.weight": "model-00001-of-00002.safetensors",
            "projector.weight": "model-00002-of-00002.safetensors",
        },
    }
    _write_json(checkpoint / "model.safetensors.index.json", index)
    torch.save(
        {"state": {0: {"step": torch.tensor(step)}}, "param_groups": [{"params": [0]}]},
        checkpoint / "optimizer.pt",
    )
    torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
    torch.save(
        {
            "python": (3, (1,), None),
            "numpy": ("MT19937",),
            "cpu": b"cpu",
            "cuda": [b"cuda"],
        },
        checkpoint / "rng_state.pth",
    )
    _write_json(checkpoint / "trainer_state.json", {"global_step": step})
    for filename in (
        "training_args.bin",
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ):
        (checkpoint / filename).write_bytes(b"fixture")
    experiment_cfg = checkpoint / "experiment_cfg"
    experiment_cfg.mkdir()
    for filename in (
        "config.yaml",
        "conf.yaml",
        "dataset_statistics.json",
        "final_model_config.json",
        "final_processor_config.json",
    ):
        (experiment_cfg / filename).write_bytes(b"fixture")
    return checkpoint


def test_valid_sharded_checkpoint_passes_payload_verification(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    verdict = verify_checkpoint_structure(checkpoint, expected_step=5)

    assert verdict["status"] == "pass"
    assert verdict["tensor_keys"] == ["action.weight", "projector.weight"]
    assert verdict["tensor_key_map"] == {
        "action.weight": "model-00001-of-00002.safetensors",
        "projector.weight": "model-00002-of-00002.safetensors",
    }
    assert verdict["total_tensor_bytes"] == 8
    paths = [entry["path"] for entry in verdict["files"]]
    assert paths == sorted(paths)
    index_entry = next(entry for entry in verdict["files"] if entry["path"] == "model.safetensors.index.json")
    expected_digest = hashlib.sha256((checkpoint / index_entry["path"]).read_bytes()).hexdigest()
    assert index_entry == {
        "path": "model.safetensors.index.json",
        "sha256": expected_digest,
        "size": (checkpoint / index_entry["path"]).stat().st_size,
    }


def test_checkpoint_rejects_content_replaced_after_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    original_validate = checkpoint_verifier._validate_trainer_state

    def validate_then_replace(path: Path, expected_step: int) -> None:
        original_validate(path, expected_step)
        _write_json(
            path / "trainer_state.json",
            {"global_step": expected_step, "replacement": True},
        )

    monkeypatch.setattr(
        checkpoint_verifier,
        "_validate_trainer_state",
        validate_then_replace,
    )

    with pytest.raises(CheckpointError, match="content changed"):
        verify_checkpoint_structure(checkpoint, expected_step=5)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("corrupt_shard", "Safetensors|safetensors"),
        ("index_mismatch", "tensor key map"),
        ("unreferenced_shard", "unreferenced"),
        ("unsharded", "sharded-only"),
        ("pytorch_bin", "PyTorch|bin"),
        ("trainer_step", "global_step"),
        ("scheduler_step", "last_epoch"),
        ("rng_key", "RNG"),
        ("partial", "partial"),
        ("temporary", "temporary|tmp"),
        ("incomplete", "incomplete"),
        ("zero_length", "zero-length"),
        ("duplicate_tensor", "duplicate tensor key"),
        ("unexpected_tensor", "unexpected"),
        ("missing_tensor", "missing"),
        ("nested_index", "exactly one"),
        ("nested_pytorch_bin", "PyTorch|bin"),
    ],
)
def test_checkpoint_corruption_fails_closed(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    if corruption == "corrupt_shard":
        (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"corrupt")
    elif corruption == "index_mismatch":
        path = checkpoint / "model.safetensors.index.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["weight_map"]["action.weight"] = "model-00002-of-00002.safetensors"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif corruption == "unreferenced_shard":
        save_file({"extra": torch.ones(1)}, checkpoint / "model-00003-of-00003.safetensors")
    elif corruption == "unsharded":
        save_file({"unexpected": torch.ones(1)}, checkpoint / "model.safetensors")
    elif corruption == "pytorch_bin":
        (checkpoint / "pytorch_model.bin").write_bytes(b"forbidden")
    elif corruption == "trainer_step":
        _write_json(checkpoint / "trainer_state.json", {"global_step": 4})
    elif corruption == "scheduler_step":
        torch.save({"last_epoch": 4}, checkpoint / "scheduler.pt")
    elif corruption == "rng_key":
        torch.save({"python": (), "numpy": (), "cpu": b"cpu"}, checkpoint / "rng_state.pth")
    elif corruption == "partial":
        (checkpoint / "leftover.part").write_bytes(b"partial")
    elif corruption == "temporary":
        (checkpoint / "leftover.tmp").write_bytes(b"temporary")
    elif corruption == "incomplete":
        (checkpoint / "leftover.incomplete").write_bytes(b"incomplete")
    elif corruption == "zero_length":
        (checkpoint / "zero.txt").touch()
    elif corruption == "duplicate_tensor":
        save_file(
            {"action.weight": torch.ones(1), "projector.weight": torch.ones(1)},
            checkpoint / "model-00002-of-00002.safetensors",
        )
        _set_index_total_size(checkpoint, 12)
    elif corruption == "unexpected_tensor":
        save_file(
            {"action.weight": torch.ones(1), "extra.weight": torch.ones(1)},
            checkpoint / "model-00001-of-00002.safetensors",
        )
        _set_index_total_size(checkpoint, 12)
    elif corruption == "missing_tensor":
        path = checkpoint / "model.safetensors.index.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["weight_map"]["missing.weight"] = "model-00001-of-00002.safetensors"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif corruption == "nested_index":
        (checkpoint / "experiment_cfg/model.safetensors.index.json").write_text("{}", encoding="utf-8")
    elif corruption == "nested_pytorch_bin":
        (checkpoint / "experiment_cfg/pytorch_model.bin").write_bytes(b"forbidden")
    else:
        raise AssertionError(corruption)

    with pytest.raises(CheckpointError, match=match):
        verify_checkpoint_structure(checkpoint, expected_step=5)


def _set_index_total_size(checkpoint: Path, size: int) -> None:
    path = checkpoint / "model.safetensors.index.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["metadata"]["total_size"] = size
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda path: _write_json(path / "model.safetensors.index.json", []), "index.*object"),
        (
            lambda path: _set_index_total_size(path, True),
            "total_size.*integer",
        ),
        (lambda path: _write_json(path / "trainer_state.json", {"global_step": True}), "global_step.*integer"),
        (lambda path: torch.save({"state": {}, "param_groups": []}, path / "optimizer.pt"), "optimizer"),
    ],
)
def test_serialized_payloads_require_strict_valid_types(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    mutate(checkpoint)

    with pytest.raises(CheckpointError, match=match):
        verify_checkpoint_structure(checkpoint, expected_step=5)


@pytest.mark.parametrize(
    ("checkpoint_step", "expected_step", "match"),
    [(5, 4, "checkpoint name"), (5, True, "expected_step.*integer")],
)
def test_checkpoint_name_and_expected_step_must_cohere_exactly(
    tmp_path: Path,
    checkpoint_step: int,
    expected_step: int,
    match: str,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path, step=checkpoint_step)

    with pytest.raises(CheckpointError, match=match):
        verify_checkpoint_structure(checkpoint, expected_step=expected_step)


def test_checkpoint_rejects_symlink_artifacts(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    (checkpoint / "linked.json").symlink_to(checkpoint / "config.json")

    with pytest.raises(CheckpointError, match="symlink"):
        verify_checkpoint_structure(checkpoint, expected_step=5)


def test_checkpoint_rejects_special_artifacts(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    os.mkfifo(checkpoint / "writer.pipe")

    with pytest.raises(CheckpointError, match="special"):
        verify_checkpoint_structure(checkpoint, expected_step=5)


def test_checkpoint_accepts_single_gpu_cuda_rng_tensor(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    rng_path = checkpoint / "rng_state.pth"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng["cuda"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    torch.save(rng, rng_path)

    assert verify_checkpoint_structure(checkpoint, expected_step=5)["status"] == "pass"


def _offline_dependencies(
    cache_root: Path,
    events: list[object],
    *,
    model_info: dict[str, object] | None = None,
    model_name: str = COSMOS_MODEL_ID,
    model_revision: str = COSMOS_REVISION,
    selector_name: str = "Qwen3Backbone",
    snapshot_result: Path | None = None,
    cache_read_only: bool = True,
    hub_cache_read_only: bool = True,
    model_error: BaseException | None = None,
    processor_model_name: str = COSMOS_MODEL_ID,
    model_info_repo: str = COSMOS_MODEL_ID,
    snapshot_config: dict[str, object] | None = None,
    snapshot_config_link: str | None = None,
) -> SimpleNamespace:
    snapshot = cache_root / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots" / COSMOS_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "transformers_version": "4.57.0.dev0",
    }
    if snapshot_config is not None:
        config_payload.update(snapshot_config)
    config_path = snapshot / "config.json"
    if snapshot_config_link is None:
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    elif snapshot_config_link == "contained":
        blobs = snapshot.parent.parent / "blobs"
        blobs.mkdir()
        blob = blobs / "pinned-config-blob"
        blob.write_text(json.dumps(config_payload), encoding="utf-8")
        config_path.symlink_to(Path("../../blobs") / blob.name)
    elif snapshot_config_link == "dangling":
        (snapshot.parent.parent / "blobs").mkdir()
        config_path.symlink_to(Path("../../blobs/missing-config-blob"))
    elif snapshot_config_link == "escaping":
        outside = cache_root / "outside-config.json"
        outside.write_text(json.dumps(config_payload), encoding="utf-8")
        config_path.symlink_to(outside)
    else:
        raise AssertionError(f"unsupported snapshot_config_link: {snapshot_config_link}")

    def network_model_info(*args: object, **kwargs: object) -> object:
        events.append(("network-model-info", args, kwargs))
        raise RuntimeError("HF_HUB_OFFLINE: model_info network access is forbidden")

    huggingface_hub = SimpleNamespace(model_info=network_model_info)

    class Config:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> SimpleNamespace:
            events.append(("config", path, kwargs))
            return SimpleNamespace(model_name=model_name, model_revision=model_revision)

    class LoadedModel:
        def __del__(self) -> None:
            events.append("model-released")

    class Model:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> tuple[object, dict[str, object]]:
            events.append(("model", path, kwargs))
            info_response = huggingface_hub.model_info(model_info_repo)
            events.append(("model-info-tags", info_response.tags))
            if model_error is not None:
                raise model_error
            info = model_info
            if info is None:
                info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            return LoadedModel(), info

    class LoadedProcessor:
        model_name = processor_model_name

        def __del__(self) -> None:
            events.append("processor-released")

    class Processor:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> object:
            events.append(("processor", path, kwargs))
            info_response = huggingface_hub.model_info(COSMOS_MODEL_ID)
            events.append(("processor-model-info-tags", info_response.tags))
            return LoadedProcessor()

    backbone = type(selector_name, (), {})

    def snapshot_download(**kwargs: object) -> str:
        events.append(("snapshot", kwargs))
        return str(snapshot if snapshot_result is None else snapshot_result)

    def is_read_only(path: Path) -> bool:
        events.append(("cache-mode", path))
        return hub_cache_read_only if path == cache_root / "hub" else cache_read_only

    return SimpleNamespace(
        config_class=Config,
        model_class=Model,
        processor_class=Processor,
        get_backbone_cls=lambda config: events.append(("selector", config)) or backbone,
        snapshot_download=snapshot_download,
        huggingface_hub=huggingface_hub,
        original_model_info=network_model_info,
        cache_is_read_only=is_read_only,
        gc_collect=lambda: events.append("gc"),
        empty_cache=lambda: events.append("empty-cache"),
    )


def test_cache_read_only_rejects_mode_0555_on_writable_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = SimpleNamespace(stat=lambda: SimpleNamespace(st_mode=0o40555))
    monkeypatch.setattr(os, "statvfs", lambda _path: SimpleNamespace(f_flag=0))

    assert _cache_is_read_only(fake_path) is False


def test_cache_read_only_accepts_genuine_read_only_filesystem_for_root_and_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True)
    inspected: list[Path] = []
    read_only_flag = getattr(os, "ST_RDONLY", 1)

    def statvfs(path: Path) -> SimpleNamespace:
        inspected.append(path)
        return SimpleNamespace(f_flag=read_only_flag)

    monkeypatch.setattr(os, "statvfs", statvfs)

    assert _cache_is_read_only(cache_root) is True
    assert _cache_is_read_only(hub_cache) is True
    assert inspected == [cache_root, hub_cache]


def _set_exact_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "NO_ALBUMENTATIONS_UPDATE",
    ):
        monkeypatch.setenv(key, "1")


def test_offline_worker_requires_environment_before_import_and_uses_exact_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(cache_root, events)
    _set_exact_offline_environment(monkeypatch)

    def dependency_loader() -> SimpleNamespace:
        events.append(
            (
                "loader",
                {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                        "HF_DATASETS_OFFLINE",
                        "NO_ALBUMENTATIONS_UPDATE",
                    )
                },
            )
        )
        return dependencies

    result = _verify_offline_load_worker(
        checkpoint,
        cache_root,
        COSMOS_REVISION,
        dependency_loader=dependency_loader,
    )

    assert result["status"] == "pass"
    assert result["model_class"] == "LoadedModel"
    assert result["processor_class"] == "LoadedProcessor"
    assert events[0] == (
        "loader",
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        },
    )
    hub_cache = cache_root / "hub"
    snapshot_call = next(event for event in events if isinstance(event, tuple) and event[0] == "snapshot")
    assert snapshot_call[1] == {
        "repo_id": COSMOS_MODEL_ID,
        "revision": COSMOS_REVISION,
        "cache_dir": str(hub_cache),
        "local_files_only": True,
    }
    for label in ("model", "processor"):
        call = next(event for event in events if isinstance(event, tuple) and event[0] == label)
        kwargs = call[2]
        assert kwargs["local_files_only"] is True
        assert kwargs["cache_dir"] == str(hub_cache)
        assert kwargs["revision"] == COSMOS_REVISION
        assert kwargs["transformers_loading_kwargs"] == {
            "trust_remote_code": True,
            "local_files_only": True,
            "revision": COSMOS_REVISION,
            "cache_dir": str(hub_cache),
        }
    assert events[-2:] == ["gc", "empty-cache"]
    assert events.index("model-released") < events.index("gc")
    assert events.index("processor-released") < events.index("gc")
    assert ("model-info-tags", ["qwen3_vl"]) in events
    assert ("processor-model-info-tags", ["qwen3_vl"]) in events
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


@pytest.mark.parametrize(
    ("snapshot_config", "match"),
    [
        ({"model_type": "mistral"}, "model_type"),
        ({"architectures": ["MistralForCausalLM"]}, "architectures"),
        ({"transformers_version": "4.57.1"}, "transformers_version"),
    ],
)
def test_offline_worker_rejects_inexact_cosmos_snapshot_identity_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_config: dict[str, object],
    match: str,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(
        cache_root,
        events,
        snapshot_config=snapshot_config,
    )

    with pytest.raises(CheckpointError, match=match):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            COSMOS_REVISION,
            dependency_loader=lambda: dependencies,
        )

    assert not any(isinstance(event, tuple) and event[0] == "model" for event in events)
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


def test_offline_worker_accepts_config_symlink_to_canonical_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(
        cache_root,
        events,
        snapshot_config_link="contained",
    )

    result = _verify_offline_load_worker(
        checkpoint,
        cache_root,
        COSMOS_REVISION,
        dependency_loader=lambda: dependencies,
    )

    assert result["status"] == "pass"
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


@pytest.mark.parametrize("link_kind", ["dangling", "escaping"])
def test_offline_worker_rejects_invalid_config_symlink_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(
        cache_root,
        events,
        snapshot_config_link=link_kind,
    )

    with pytest.raises(CheckpointError, match="config.json"):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            COSMOS_REVISION,
            dependency_loader=lambda: dependencies,
        )

    assert not any(isinstance(event, tuple) and event[0] == "model" for event in events)
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


def test_offline_worker_rejects_unexpected_model_info_repo_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(
        cache_root,
        events,
        model_info_repo="other/model",
    )

    with pytest.raises(CheckpointError, match="only supports.*Cosmos"):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            COSMOS_REVISION,
            dependency_loader=lambda: dependencies,
        )

    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


@pytest.mark.parametrize(
    ("dependency_overrides", "revision", "match"),
    [
        ({}, "wrong-revision", "revision"),
        ({"model_name": "other/model"}, COSMOS_REVISION, "canonical Cosmos"),
        ({"model_revision": "wrong"}, COSMOS_REVISION, "config revision"),
        ({"selector_name": "OtherBackbone"}, COSMOS_REVISION, "Qwen3Backbone"),
        ({"cache_read_only": False}, COSMOS_REVISION, "read-only"),
        ({"hub_cache_read_only": False}, COSMOS_REVISION, "hub.*read-only"),
        ({"snapshot_result": Path("/wrong/snapshot")}, COSMOS_REVISION, "snapshot"),
        (
            {"processor_model_name": "other/model"},
            COSMOS_REVISION,
            "processor.*canonical Cosmos",
        ),
    ],
)
def test_offline_load_rejects_pin_or_runtime_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_overrides: dict[str, object],
    revision: str,
    match: str,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(cache_root, events, **dependency_overrides)

    with pytest.raises(CheckpointError, match=match):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            revision,
            dependency_loader=lambda: dependencies,
        )


@pytest.mark.parametrize("loading_key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_offline_load_rejects_nonempty_loading_information(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loading_key: str,
) -> None:
    _set_exact_offline_environment(monkeypatch)
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    info[loading_key] = ["bad.weight"]
    dependencies = _offline_dependencies(cache_root, events, model_info=info)

    with pytest.raises(CheckpointError, match=loading_key):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            COSMOS_REVISION,
            dependency_loader=lambda: dependencies,
        )

    assert events[-2:] == ["gc", "empty-cache"]


def _offline_process_result(cache_root: Path) -> dict[str, object]:
    snapshot = cache_root / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots" / COSMOS_REVISION
    return {
        "status": "pass",
        "backbone_class": "Qwen3Backbone",
        "cosmos_model_id": COSMOS_MODEL_ID,
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_snapshot": str(snapshot),
        "model_class": "LoadedModel",
        "processor_class": "LoadedProcessor",
        "log_lines": ["offline worker passed"],
    }


def test_verify_offline_load_launches_fresh_child_with_exact_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = (tmp_path / "cache").resolve()
    monkeypatch.delenv("NO_ALBUMENTATIONS_UPDATE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-visible")
    monkeypatch.setenv("HF_HUB_OFFLINE", "conflicting")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def process_runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_offline_process_result(cache_root)),
            stderr="ignored child diagnostics",
        )

    result = verify_offline_load(
        checkpoint,
        cache_root,
        COSMOS_REVISION,
        process_runner=process_runner,
    )

    assert result["status"] == "pass"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-visible"
    assert os.environ.get("NO_ALBUMENTATIONS_UPDATE") is None
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        str((Path(__file__).parents[1] / "scripts/verify_gr00t_n17_checkpoint.py").resolve()),
        "--_offline-worker",
    ]
    child_environment = kwargs["env"]
    assert child_environment["CUDA_VISIBLE_DEVICES"] == ""
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "NO_ALBUMENTATIONS_UPDATE",
    ):
        assert child_environment[key] == "1"
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert 0 < kwargs["timeout"] <= 300
    request = json.loads(kwargs["input"])
    assert request == {
        "cache_root": str(cache_root),
        "checkpoint": str(checkpoint),
        "cosmos_revision": COSMOS_REVISION,
    }


def test_verify_offline_load_rejects_albumentations_conflict_before_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "0")
    launched = False

    def process_runner(*_args: object, **_kwargs: object) -> object:
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    with pytest.raises(CheckpointError, match="NO_ALBUMENTATIONS_UPDATE"):
        verify_offline_load(
            checkpoint,
            (tmp_path / "cache").resolve(),
            COSMOS_REVISION,
            process_runner=process_runner,
        )

    assert launched is False


@pytest.mark.parametrize("failure", ["nonzero", "signal", "invalid_json", "invalid_result", "timeout"])
def test_verify_offline_load_fails_closed_on_child_failure(tmp_path: Path, failure: str) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)

    def process_runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            return SimpleNamespace(returncode=2, stdout='{"status":"fail"}', stderr="")
        if failure == "signal":
            return SimpleNamespace(returncode=-9, stdout="", stderr="")
        if failure == "invalid_json":
            return SimpleNamespace(returncode=0, stdout="not-json", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"status":"pass"}', stderr="")

    with pytest.raises(CheckpointError, match="offline child"):
        verify_offline_load(
            checkpoint,
            (tmp_path / "cache").resolve(),
            COSMOS_REVISION,
            process_runner=process_runner,
        )


def test_offline_worker_rejects_inexact_environment_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    events: list[object] = []
    dependencies = _offline_dependencies(cache_root, events)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "NO_ALBUMENTATIONS_UPDATE",
    ):
        monkeypatch.setenv(key, "1")

    with pytest.raises(CheckpointError, match="offline child environment"):
        _verify_offline_load_worker(
            checkpoint,
            cache_root,
            COSMOS_REVISION,
            dependency_loader=lambda: events.append("loader") or dependencies,
        )

    assert "loader" not in events


def _cli_arguments(checkpoint: Path, cache_root: Path, output_dir: Path, *, offline: bool = False) -> list[str]:
    arguments = [
        str(checkpoint),
        "--expected-step",
        "5",
        "--cache-root",
        str(cache_root),
        "--cosmos-revision",
        COSMOS_REVISION,
        "--output-dir",
        str(output_dir),
    ]
    if offline:
        arguments.append("--offline-load")
    return arguments


def test_cli_atomically_publishes_complete_structural_evidence_with_verdict_last(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    output_dir = (tmp_path / "verdict").resolve()
    published: list[str] = []

    def atomic_writer(path: Path, payload: object, *, text: bool = False) -> None:
        published.append(path.name)
        if text:
            path.write_text(str(payload), encoding="utf-8")
        else:
            _write_json(path, payload)

    exit_code = main(
        _cli_arguments(checkpoint, tmp_path / "unused-cache", output_dir),
        atomic_writer=atomic_writer,
    )

    assert exit_code == 0
    assert published == ["files.json", "tensor-keys.json", "offline-load.log", "verdict.json"]
    assert json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))["status"] == "pass"
    assert json.loads((output_dir / "files.json").read_text(encoding="utf-8"))
    assert json.loads((output_dir / "tensor-keys.json").read_text(encoding="utf-8"))["tensor_keys"] == [
        "action.weight",
        "projector.weight",
    ]
    assert (output_dir / "offline-load.log").read_text(encoding="utf-8") == "offline load not requested\n"


def test_cli_returns_nonzero_and_publishes_failure_without_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    cache_root = tmp_path / "cache"
    output_dir = (tmp_path / "failed-verdict").resolve()
    secret = "hf_secret-token-123"

    def process_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout='{"status":"fail"}', stderr=secret)

    exit_code = main(
        _cli_arguments(checkpoint, cache_root, output_dir, offline=True),
        offline_process_runner=process_runner,
    )

    assert exit_code == 1
    combined = "\n".join(
        [
            capsys.readouterr().out,
            *(path.read_text(encoding="utf-8") for path in output_dir.iterdir()),
        ]
    )
    assert secret not in combined
    assert json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))["status"] == "fail"


def test_cli_runs_offline_child_only_after_structural_success(tmp_path: Path) -> None:
    cache_root = (tmp_path / "cache").resolve()
    successful_checkpoint = _write_checkpoint_fixture(tmp_path / "successful")
    failed_checkpoint = _write_checkpoint_fixture(tmp_path / "failed")
    (failed_checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"corrupt")
    launched_for: list[str] = []

    def process_runner(_command: list[str], **kwargs: object) -> SimpleNamespace:
        request = json.loads(kwargs["input"])
        launched_for.append(request["checkpoint"])
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_offline_process_result(cache_root)),
            stderr="",
        )

    successful_exit = main(
        _cli_arguments(
            successful_checkpoint,
            cache_root,
            (tmp_path / "successful-verdict").resolve(),
            offline=True,
        ),
        offline_process_runner=process_runner,
    )
    failed_exit = main(
        _cli_arguments(
            failed_checkpoint,
            cache_root,
            (tmp_path / "failed-structural-verdict").resolve(),
            offline=True,
        ),
        offline_process_runner=process_runner,
    )

    assert successful_exit == 0
    assert failed_exit == 1
    assert launched_for == [str(successful_checkpoint)]


@pytest.mark.parametrize("output_kind", ["relative", "existing"])
def test_cli_requires_an_absolute_nonexistent_output_directory(tmp_path: Path, output_kind: str) -> None:
    checkpoint = _write_checkpoint_fixture(tmp_path)
    if output_kind == "relative":
        output_dir = Path("relative-verdict")
    else:
        output_dir = (tmp_path / "existing").resolve()
        output_dir.mkdir()
        (output_dir / "keep.txt").write_text("preserve", encoding="utf-8")

    exit_code = main(_cli_arguments(checkpoint, tmp_path / "unused-cache", output_dir))

    assert exit_code == 2
    if output_kind == "relative":
        assert not output_dir.exists()
    else:
        assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "preserve"
