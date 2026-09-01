#!/usr/bin/env python3
"""Fail-closed verification for complete GR00T N1.7 training checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

COSMOS_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT_RELATIVE = Path("models--nvidia--Cosmos-Reason2-2B/snapshots") / COSMOS_REVISION
COSMOS_SNAPSHOT_CONFIG_IDENTITY: dict[str, object] = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "transformers_version": "4.57.0.dev0",
}

_OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "NO_ALBUMENTATIONS_UPDATE": "1",
}
_CHECKPOINT_NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)")
_SHARD_NAME = re.compile(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors")
_INTERNAL_OFFLINE_WORKER_FLAG = "--_offline-worker"
_OFFLINE_CHILD_TIMEOUT_SECONDS = 180
_ROOT_JSON_CONFIGS = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
)
_EXPERIMENT_JSON_CONFIGS = (
    "dataset_statistics.json",
    "final_model_config.json",
    "final_processor_config.json",
)


class CheckpointError(RuntimeError):
    """The candidate checkpoint does not satisfy the completeness contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise CheckpointError(f"JSON is not valid JSON: nonstandard numeric constant {value}")


def _load_json(path: Path, *, label: str) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_json,
            )
    except CheckpointError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{label} is not valid JSON") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    value = _load_json(path, label=label)
    if type(value) is not dict:
        raise CheckpointError(f"{label} must contain a JSON object")
    return value


def _checkpoint_step(checkpoint: Path, expected_step: int) -> int:
    if type(expected_step) is not int:
        raise CheckpointError("expected_step must be an integer")
    if expected_step < 0:
        raise CheckpointError("expected_step must be nonnegative")
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise CheckpointError("checkpoint name must be exactly checkpoint-N")
    named_step = int(match.group(1))
    if named_step != expected_step:
        raise CheckpointError(f"checkpoint name step {named_step} differs from expected_step {expected_step}")
    return named_step


def _scan_checkpoint_files(checkpoint: Path) -> list[Path]:
    if os.path.lexists(checkpoint) and checkpoint.is_symlink():
        raise CheckpointError("checkpoint directory must not be a symlink")
    if not checkpoint.is_dir():
        raise CheckpointError("checkpoint must be an existing directory")

    files: list[Path] = []
    pending = [checkpoint]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise CheckpointError("checkpoint directory cannot be enumerated") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(checkpoint).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise CheckpointError(f"checkpoint artifact cannot be inspected: {relative}") from exc
            if stat.S_ISLNK(mode):
                raise CheckpointError(f"symlink artifact is forbidden: {relative}")
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise CheckpointError(f"special checkpoint artifact is forbidden: {relative}")
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError as exc:
                raise CheckpointError(f"checkpoint artifact cannot be inspected: {relative}") from exc
            if size == 0:
                raise CheckpointError(f"zero-length checkpoint artifact is forbidden: {relative}")
            lowered = entry.name.lower()
            if lowered.endswith(".part") or lowered.endswith(".partial"):
                raise CheckpointError(f"partial checkpoint artifact is forbidden: {relative}")
            if lowered.endswith(".tmp"):
                raise CheckpointError(f"temporary/tmp checkpoint artifact is forbidden: {relative}")
            if lowered.endswith(".incomplete"):
                raise CheckpointError(f"incomplete checkpoint artifact is forbidden: {relative}")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(checkpoint).as_posix())


def _required_file(checkpoint: Path, name: str) -> Path:
    path = checkpoint / name
    if not path.is_file() or path.is_symlink():
        raise CheckpointError(f"required checkpoint file is missing: {name}")
    return path


def _load_torch_payload(torch_module: Any, path: Path, *, label: str) -> object:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"{label} cannot be loaded") from exc


def _validate_optimizer(torch_module: Any, checkpoint: Path) -> None:
    optimizer = _load_torch_payload(
        torch_module,
        _required_file(checkpoint, "optimizer.pt"),
        label="optimizer.pt",
    )
    if type(optimizer) is not dict:
        raise CheckpointError("optimizer.pt must contain an optimizer dictionary")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if type(state) is not dict or not state:
        raise CheckpointError("optimizer state must be present and nonempty")
    if type(groups) is not list or not groups:
        raise CheckpointError("optimizer parameter groups must be present and nonempty")
    for index, group in enumerate(groups):
        if type(group) is not dict:
            raise CheckpointError(f"optimizer parameter group {index} must be a dictionary")
        parameters = group.get("params")
        if type(parameters) is not list or not parameters:
            raise CheckpointError(f"optimizer parameter group {index} has no parameters")


def _validate_scheduler(torch_module: Any, checkpoint: Path, expected_step: int) -> None:
    scheduler = _load_torch_payload(
        torch_module,
        _required_file(checkpoint, "scheduler.pt"),
        label="scheduler.pt",
    )
    if type(scheduler) is not dict:
        raise CheckpointError("scheduler.pt must contain a scheduler dictionary")
    last_epoch = scheduler.get("last_epoch")
    if type(last_epoch) is not int:
        raise CheckpointError("scheduler last_epoch must be an integer")
    if last_epoch != expected_step:
        raise CheckpointError(f"scheduler last_epoch {last_epoch} differs from expected step {expected_step}")


def _validate_rng(torch_module: Any, checkpoint: Path) -> None:
    rng = _load_torch_payload(
        torch_module,
        _required_file(checkpoint, "rng_state.pth"),
        label="rng_state.pth",
    )
    if type(rng) is not dict:
        raise CheckpointError("RNG state must be a dictionary")
    missing = sorted({"python", "numpy", "cpu", "cuda"} - set(rng))
    if missing:
        raise CheckpointError(f"RNG state is missing required keys: {missing}")
    if type(rng["python"]) is not tuple or type(rng["numpy"]) is not tuple:
        raise CheckpointError("RNG Python and NumPy states must be tuples")
    tensor_or_bytes = (bytes, bytearray, torch_module.Tensor)
    if not isinstance(rng["cpu"], tensor_or_bytes):
        raise CheckpointError("RNG CPU state has an invalid payload")
    cuda = rng["cuda"]
    if isinstance(cuda, tensor_or_bytes):
        return
    if type(cuda) not in (list, tuple) or not cuda:
        raise CheckpointError("RNG CUDA state must be a tensor or nonempty sequence")
    if any(not isinstance(value, tensor_or_bytes) for value in cuda):
        raise CheckpointError("RNG CUDA state has an invalid payload")


def _validate_trainer_state(checkpoint: Path, expected_step: int) -> None:
    trainer = _load_json_object(
        _required_file(checkpoint, "trainer_state.json"),
        label="trainer_state.json",
    )
    global_step = trainer.get("global_step")
    if type(global_step) is not int:
        raise CheckpointError("trainer_state global_step must be an integer")
    if global_step != expected_step:
        raise CheckpointError(
            f"trainer_state global_step {global_step} differs from expected step {expected_step}"
        )


def _validate_required_configuration_artifacts(checkpoint: Path) -> None:
    _required_file(checkpoint, "training_args.bin")
    for filename in _ROOT_JSON_CONFIGS:
        _required_file(checkpoint, filename)
    experiment = checkpoint / "experiment_cfg"
    if experiment.is_symlink() or not experiment.is_dir():
        raise CheckpointError("required experiment_cfg directory is missing")
    for filename in ("config.yaml", "conf.yaml"):
        _required_file(experiment, filename)
    for filename in _EXPERIMENT_JSON_CONFIGS:
        _required_file(experiment, filename)


def _validate_shards(checkpoint: Path, files: list[Path]) -> dict[str, object]:
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError as exc:
        raise CheckpointError("safetensors is required to verify checkpoint shards") from exc

    root_files = {path.name for path in files if path.parent == checkpoint}
    index_paths = [
        path.relative_to(checkpoint).as_posix() for path in files if path.name == "model.safetensors.index.json"
    ]
    if index_paths != ["model.safetensors.index.json"]:
        raise CheckpointError("checkpoint must contain exactly one root model.safetensors.index.json")
    nested_safetensors = [
        path.relative_to(checkpoint).as_posix()
        for path in files
        if path.suffix == ".safetensors" and path.parent != checkpoint
    ]
    if nested_safetensors:
        raise CheckpointError(f"sharded-only checkpoint has nested Safetensors: {nested_safetensors}")
    if "model.safetensors" in root_files:
        raise CheckpointError("sharded-only checkpoint forbids model.safetensors")
    forbidden_bins = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in files
        if path.suffix == ".bin" and not (path.parent == checkpoint and path.name == "training_args.bin")
    )
    if forbidden_bins:
        raise CheckpointError(f"PyTorch model bin payloads are forbidden: {forbidden_bins}")

    safetensor_names = {name for name in root_files if name.endswith(".safetensors")}
    present_shards = {name for name in safetensor_names if _SHARD_NAME.fullmatch(name)}
    unexpected_safetensors = sorted(safetensor_names - present_shards)
    if unexpected_safetensors:
        raise CheckpointError(f"sharded-only checkpoint contains unexpected Safetensors: {unexpected_safetensors}")
    if not present_shards:
        raise CheckpointError("sharded-only checkpoint requires one or more model shards")

    index_path = _required_file(checkpoint, "model.safetensors.index.json")
    index = _load_json_object(index_path, label="Safetensors index")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if type(metadata) is not dict:
        raise CheckpointError("Safetensors index metadata must be a JSON object")
    if type(weight_map) is not dict:
        raise CheckpointError("Safetensors index weight_map must be a JSON object")
    if not weight_map:
        raise CheckpointError("Safetensors index weight_map must be nonempty")
    total_size = metadata.get("total_size")
    if type(total_size) is not int:
        raise CheckpointError("Safetensors index total_size must be an integer")
    if total_size < 0:
        raise CheckpointError("Safetensors index total_size must be nonnegative")
    for key, shard in weight_map.items():
        if type(key) is not str or not key:
            raise CheckpointError("Safetensors index tensor keys must be nonempty strings")
        if type(shard) is not str or _SHARD_NAME.fullmatch(shard) is None:
            raise CheckpointError("Safetensors index shard names must use model-NNNNN-of-NNNNN.safetensors")

    referenced_shards = set(weight_map.values())
    unreferenced = sorted(present_shards - referenced_shards)
    if unreferenced:
        raise CheckpointError(f"tensor key map has unreferenced model shards: {unreferenced}")
    missing_shards = sorted(referenced_shards - present_shards)
    if missing_shards:
        raise CheckpointError(f"referenced model shards are missing: {missing_shards}")

    shard_parts = [_SHARD_NAME.fullmatch(name) for name in sorted(present_shards)]
    shard_totals = {int(match.group(2)) for match in shard_parts if match is not None}
    if len(shard_totals) != 1:
        raise CheckpointError("model shard filenames disagree on shard count")
    shard_total = shard_totals.pop()
    expected_names = {
        f"model-{index_number:05d}-of-{shard_total:05d}.safetensors" for index_number in range(1, shard_total + 1)
    }
    if present_shards != expected_names:
        raise CheckpointError("model shard filename set is incomplete or noncontiguous")

    actual_weight_map: dict[str, str] = {}
    total_tensor_bytes = 0
    for shard in sorted(referenced_shards):
        try:
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
                for key in sorted(handle.keys()):
                    if key in actual_weight_map:
                        raise CheckpointError(f"duplicate tensor key: {key}")
                    tensor = handle.get_tensor(key)
                    total_tensor_bytes += tensor.numel() * tensor.element_size()
                    actual_weight_map[key] = shard
        except CheckpointError:
            raise
        except (SafetensorError, OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError(f"Safetensors shard cannot be read: {shard}") from exc

    indexed_keys = set(weight_map)
    actual_keys = set(actual_weight_map)
    missing_keys = sorted(indexed_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - indexed_keys)
    misplaced_keys = sorted(key for key in indexed_keys & actual_keys if weight_map[key] != actual_weight_map[key])
    if missing_keys or unexpected_keys or misplaced_keys:
        raise CheckpointError(
            "actual tensor key map differs from index: "
            f"missing={missing_keys}, unexpected={unexpected_keys}, misplaced={misplaced_keys}"
        )
    if total_size != total_tensor_bytes:
        raise CheckpointError("indexed total_size differs from tensor payload")
    return {
        "tensor_keys": sorted(actual_weight_map),
        "tensor_key_map": {key: actual_weight_map[key] for key in sorted(actual_weight_map)},
        "total_tensor_bytes": total_tensor_bytes,
    }


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError("checkpoint artifact cannot be opened without following links") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointError("checkpoint artifact changed to a special file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise CheckpointError("checkpoint artifact changed while hashing")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _build_file_manifest(checkpoint: Path, files: list[Path]) -> list[dict[str, object]]:
    manifest = []
    for path in files:
        size, digest = _hash_regular_file(path)
        manifest.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "sha256": digest,
                "size": size,
            }
        )
    if [path.relative_to(checkpoint).as_posix() for path in _scan_checkpoint_files(checkpoint)] != [
        entry["path"] for entry in manifest
    ]:
        raise CheckpointError("checkpoint artifact set changed during verification")
    return manifest


def verify_checkpoint_structure(checkpoint: Path, expected_step: int) -> dict[str, object]:
    """Read and validate every required checkpoint payload and return its manifests."""
    checkpoint = Path(checkpoint)
    _checkpoint_step(checkpoint, expected_step)
    files = _scan_checkpoint_files(checkpoint)
    pre_validation_manifest = _build_file_manifest(checkpoint, files)

    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("torch is required to verify checkpoint state") from exc

    tensor_manifest = _validate_shards(checkpoint, files)
    _validate_optimizer(torch, checkpoint)
    _validate_scheduler(torch, checkpoint, expected_step)
    _validate_rng(torch, checkpoint)
    _validate_trainer_state(checkpoint, expected_step)
    _validate_required_configuration_artifacts(checkpoint)
    file_manifest = _build_file_manifest(checkpoint, files)
    if file_manifest != pre_validation_manifest:
        raise CheckpointError("checkpoint content changed during semantic validation")
    return {
        "status": "pass",
        "expected_step": expected_step,
        "files": file_manifest,
        **tensor_manifest,
    }


def _assert_offline_child_environment() -> None:
    mismatches = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in _OFFLINE_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise CheckpointError("offline child environment is not exact")


def _cache_is_read_only(path: Path) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except (AttributeError, OSError, TypeError):
        return False


def _validate_cosmos_snapshot_config(snapshot: Path) -> None:
    """Fail closed unless the local snapshot declares the pinned Qwen3 VL identity."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise CheckpointError("exact resolved Cosmos snapshot must be a real directory")
    config_path = snapshot / "config.json"
    if config_path.is_symlink():
        blobs_path = snapshot.parent.parent / "blobs"
        if blobs_path.is_symlink() or not blobs_path.is_dir():
            raise CheckpointError("Cosmos snapshot config.json symlink requires a real canonical blobs directory")
        try:
            resolved_config_path = config_path.resolve(strict=True)
            resolved_blobs_path = blobs_path.resolve(strict=True)
            resolved_config_path.relative_to(resolved_blobs_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError(
                "Cosmos snapshot config.json symlink must resolve inside the canonical blobs directory"
            ) from exc
        if not resolved_config_path.is_file():
            raise CheckpointError("Cosmos snapshot config.json symlink must resolve to a regular file")
        config_path = resolved_config_path
    elif not config_path.is_file():
        raise CheckpointError("exact resolved Cosmos snapshot config.json must be a regular file")
    payload = _load_json_object(config_path, label="Cosmos snapshot config.json")
    for field, expected in COSMOS_SNAPSHOT_CONFIG_IDENTITY.items():
        actual = payload.get(field)
        if actual != expected:
            raise CheckpointError(
                f"Cosmos snapshot config {field} mismatch: expected {expected!r}, got {actual!r}"
            )


@contextmanager
def _offline_cosmos_model_info_shim(
    snapshot: Path,
    huggingface_hub_module: object,
) -> Iterator[None]:
    """Classify only the pinned Cosmos repo locally during official checkpoint loading."""
    _validate_cosmos_snapshot_config(snapshot)
    original_model_info = getattr(huggingface_hub_module, "model_info", None)
    if not callable(original_model_info):
        raise CheckpointError("huggingface_hub.model_info must be callable before installing offline shim")

    def pinned_model_info(repo_id: str, *args: object, **kwargs: object) -> SimpleNamespace:
        if repo_id != COSMOS_MODEL_ID:
            raise CheckpointError("offline model_info shim only supports the exact canonical Cosmos model ID")
        if args or kwargs:
            raise CheckpointError("offline Cosmos model_info shim does not accept additional arguments")
        return SimpleNamespace(tags=["qwen3_vl"])

    setattr(huggingface_hub_module, "model_info", pinned_model_info)
    try:
        yield
    finally:
        setattr(huggingface_hub_module, "model_info", original_model_info)


def _load_offline_dependencies() -> SimpleNamespace:
    import gc

    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, get_backbone_cls
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    import huggingface_hub
    import torch

    return SimpleNamespace(
        config_class=Gr00tN1d7Config,
        model_class=Gr00tN1d7,
        processor_class=Gr00tN1d7Processor,
        get_backbone_cls=get_backbone_cls,
        snapshot_download=huggingface_hub.snapshot_download,
        huggingface_hub=huggingface_hub,
        cache_is_read_only=_cache_is_read_only,
        gc_collect=gc.collect,
        empty_cache=torch.cuda.empty_cache,
    )


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_loading_info(value: object) -> None:
    if type(value) is not dict:
        raise CheckpointError("model loading_info must be a dictionary")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if key not in value:
            raise CheckpointError(f"model loading_info is missing field: {key}")
        entries = value[key]
        if type(entries) is not list:
            raise CheckpointError(f"model loading_info {key} must be a list")
        if entries:
            raise CheckpointError(f"model loading_info {key} is nonempty")


def _verify_offline_load_worker(
    checkpoint: Path,
    cache_root: Path,
    cosmos_revision: str,
    *,
    dependency_loader: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Load pinned GR00T objects inside an already isolated child process."""
    _assert_offline_child_environment()
    if cosmos_revision != COSMOS_REVISION:
        raise CheckpointError("Cosmos revision differs from the pinned exact revision")

    checkpoint = Path(checkpoint)
    cache_root = Path(cache_root)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise CheckpointError("offline checkpoint must be a real directory")
    if not cache_root.is_absolute():
        raise CheckpointError("cache_root must be absolute")
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise CheckpointError("cache_root must be a real existing directory")
    hub_cache = cache_root / "hub"
    if hub_cache.is_symlink() or not hub_cache.is_dir():
        raise CheckpointError("effective Hugging Face hub cache is missing")

    load_dependencies = _load_offline_dependencies if dependency_loader is None else dependency_loader
    runtime = load_dependencies()
    failure: BaseException | None = None
    result: dict[str, object] | None = None
    loaded: object | None = None
    model: object | None = None
    processor: object | None = None
    try:
        if not runtime.cache_is_read_only(cache_root):
            raise CheckpointError("Hugging Face cache_root is not read-only")
        if not runtime.cache_is_read_only(hub_cache):
            raise CheckpointError("effective Hugging Face hub cache is not read-only")
        expected_snapshot = hub_cache / COSMOS_SNAPSHOT_RELATIVE
        if expected_snapshot.is_symlink() or not expected_snapshot.is_dir():
            raise CheckpointError("exact pinned Cosmos snapshot is missing")
        snapshot_raw = runtime.snapshot_download(
            repo_id=COSMOS_MODEL_ID,
            revision=COSMOS_REVISION,
            cache_dir=str(hub_cache),
            local_files_only=True,
        )
        snapshot = Path(snapshot_raw)
        if snapshot.is_symlink() or _absolute_without_resolving(snapshot) != _absolute_without_resolving(
            expected_snapshot
        ):
            raise CheckpointError("offline Cosmos snapshot resolution mismatch")

        with _offline_cosmos_model_info_shim(snapshot, runtime.huggingface_hub):
            checkpoint_loading_kwargs = {
                "local_files_only": True,
                "cache_dir": str(hub_cache),
            }
            model_config = runtime.config_class.from_pretrained(
                checkpoint,
                **checkpoint_loading_kwargs,
            )
            if getattr(model_config, "model_name", None) != COSMOS_MODEL_ID:
                raise CheckpointError("model config does not use the canonical Cosmos model ID")
            if getattr(model_config, "model_revision", None) != COSMOS_REVISION:
                raise CheckpointError("model config revision differs from the pinned Cosmos revision")
            backbone_class = runtime.get_backbone_cls(model_config)
            if getattr(backbone_class, "__name__", None) != "Qwen3Backbone":
                raise CheckpointError("pinned selector did not return Qwen3Backbone")

            transformers_loading_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "revision": COSMOS_REVISION,
                "cache_dir": str(hub_cache),
            }
            loaded = runtime.model_class.from_pretrained(
                checkpoint,
                transformers_loading_kwargs=transformers_loading_kwargs,
                output_loading_info=True,
                **transformers_loading_kwargs,
            )
            if type(loaded) is not tuple or len(loaded) != 2:
                raise CheckpointError("model load did not return loading_info")
            model, loading_info = loaded
            _validate_loading_info(loading_info)
            processor = runtime.processor_class.from_pretrained(
                checkpoint,
                transformers_loading_kwargs=transformers_loading_kwargs,
                **transformers_loading_kwargs,
            )
            if processor is None:
                raise CheckpointError("processor load returned no object")
            if getattr(processor, "model_name", None) != COSMOS_MODEL_ID:
                raise CheckpointError("processor does not use the canonical Cosmos model ID")
            result = {
                "status": "pass",
                "backbone_class": backbone_class.__name__,
                "cosmos_model_id": COSMOS_MODEL_ID,
                "cosmos_revision": COSMOS_REVISION,
                "cosmos_snapshot": str(_absolute_without_resolving(expected_snapshot)),
                "model_class": type(model).__name__,
                "processor_class": type(processor).__name__,
                "log_lines": [
                    "offline environment active",
                    "pinned Cosmos snapshot resolved locally",
                    "Qwen3Backbone selector verified",
                    f"GR00T model loaded: {type(model).__name__}",
                    f"GR00T processor loaded: {type(processor).__name__}",
                    "model loading_info is empty",
                ],
            }
    except BaseException as exc:
        if isinstance(exc, CheckpointError):
            failure = exc
        else:
            failure = CheckpointError(f"offline load failed ({type(exc).__name__})")
            failure.__cause__ = exc

    loaded = None
    model = None
    processor = None
    cleanup_failure: BaseException | None = None
    for cleanup in (runtime.gc_collect, runtime.empty_cache):
        try:
            cleanup()
        except BaseException as exc:
            if cleanup_failure is None:
                cleanup_failure = exc
    if cleanup_failure is not None:
        raise CheckpointError(
            f"offline object cleanup failed ({type(cleanup_failure).__name__})"
        ) from cleanup_failure
    if failure is not None:
        raise failure
    if result is None:
        raise CheckpointError("offline load returned without a verdict")
    return result


def _load_child_json(text: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except CheckpointError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise CheckpointError("offline child returned invalid JSON") from exc


def _validate_offline_child_result(value: object, cache_root: Path) -> dict[str, object]:
    if type(value) is not dict or value.get("status") != "pass":
        raise CheckpointError("offline child returned an invalid result")
    exact_fields = {
        "cosmos_model_id": COSMOS_MODEL_ID,
        "cosmos_revision": COSMOS_REVISION,
        "backbone_class": "Qwen3Backbone",
        "cosmos_snapshot": str(cache_root / "hub" / COSMOS_SNAPSHOT_RELATIVE),
    }
    if any(value.get(key) != expected for key, expected in exact_fields.items()):
        raise CheckpointError("offline child returned an invalid pinned identity")
    for key in ("model_class", "processor_class"):
        if type(value.get(key)) is not str or not value[key]:
            raise CheckpointError("offline child returned an invalid class record")
    log_lines = value.get("log_lines")
    if type(log_lines) is not list or any(type(line) is not str for line in log_lines):
        raise CheckpointError("offline child returned invalid log records")
    return value


def verify_offline_load(
    checkpoint: Path,
    cache_root: Path,
    cosmos_revision: str,
    *,
    process_runner: Callable[..., object] | None = None,
    timeout_seconds: int = _OFFLINE_CHILD_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Launch the model/processor gate in a fresh CPU-only child process."""
    if cosmos_revision != COSMOS_REVISION:
        raise CheckpointError("Cosmos revision differs from the pinned exact revision")
    albumentations_value = os.environ.get("NO_ALBUMENTATIONS_UPDATE")
    if albumentations_value not in (None, "1"):
        raise CheckpointError("NO_ALBUMENTATIONS_UPDATE must be absent or exactly '1'")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise CheckpointError("offline child timeout must be a positive integer")

    checkpoint = Path(checkpoint)
    cache_root = Path(cache_root)
    child_environment = os.environ.copy()
    child_environment.update(_OFFLINE_ENVIRONMENT)
    request = {
        "cache_root": str(cache_root),
        "checkpoint": str(checkpoint),
        "cosmos_revision": cosmos_revision,
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _INTERNAL_OFFLINE_WORKER_FLAG,
    ]
    run_process = subprocess.run if process_runner is None else process_runner
    try:
        completed = run_process(
            command,
            input=json.dumps(request, allow_nan=False, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=child_environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckpointError("offline child timed out") from exc
    except BaseException as exc:
        raise CheckpointError(f"offline child launch failed ({type(exc).__name__})") from exc

    return_code = getattr(completed, "returncode", None)
    if type(return_code) is not int or return_code != 0:
        raise CheckpointError("offline child exited abnormally")
    stdout = getattr(completed, "stdout", None)
    if type(stdout) is not str:
        raise CheckpointError("offline child returned no structured result")
    try:
        value = _load_child_json(stdout)
        return _validate_offline_child_result(value, cache_root)
    except CheckpointError as exc:
        raise CheckpointError(f"offline child result rejected ({type(exc).__name__})") from exc


def _offline_worker_entrypoint() -> int:
    """Read one request from stdin and emit one secret-free JSON response."""
    try:
        _assert_offline_child_environment()
        request = json.load(
            sys.stdin,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
        if type(request) is not dict or set(request) != {
            "cache_root",
            "checkpoint",
            "cosmos_revision",
        }:
            raise CheckpointError("offline child request has invalid fields")
        if any(type(request[key]) is not str for key in request):
            raise CheckpointError("offline child request values must be strings")
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                result = _verify_offline_load_worker(
                    Path(request["checkpoint"]),
                    Path(request["cache_root"]),
                    request["cosmos_revision"],
                )
    except BaseException as exc:
        print(
            json.dumps(
                {"status": "fail", "error_type": type(exc).__name__},
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--cosmos-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--offline-load", action="store_true")
    return parser


def _prepare_output_directory(output_dir: Path) -> None:
    if not output_dir.is_absolute():
        raise CheckpointError("output directory must be absolute")
    if os.path.lexists(output_dir):
        raise CheckpointError("output directory must not already exist")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CheckpointError("output directory parent must be a real existing directory")
    try:
        output_dir.mkdir()
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise CheckpointError("output directory could not be created") from exc


def _atomic_publish(path: Path, payload: object, *, text: bool = False) -> None:
    if os.path.lexists(path):
        raise CheckpointError("evidence path already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if text:
                if type(payload) is not str:
                    raise CheckpointError("text evidence payload must be a string")
                handle.write(payload)
            else:
                json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CheckpointError("evidence path already exists") from exc
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _failure_record(gate: str, error: BaseException) -> dict[str, str]:
    message = str(error) if gate == "structure" else f"{type(error).__name__}"
    return {"gate": gate, "error_type": type(error).__name__, "message": message}


def main(
    argv: list[str] | None = None,
    *,
    offline_process_runner: Callable[..., object] | None = None,
    atomic_writer: Callable[..., None] | None = None,
) -> int:
    """Run requested gates and publish durable evidence, with the verdict last."""
    arguments = _build_parser().parse_args(argv)
    try:
        _prepare_output_directory(arguments.output_dir)
    except CheckpointError as exc:
        print(f"checkpoint verifier setup failed ({type(exc).__name__})", file=sys.stderr)
        return 2

    writer = _atomic_publish if atomic_writer is None else atomic_writer
    files: list[dict[str, object]] = []
    tensor_keys: dict[str, object] = {
        "tensor_keys": [],
        "tensor_key_map": {},
        "total_tensor_bytes": 0,
    }
    failures: list[dict[str, str]] = []
    structural: dict[str, object] | None = None
    offline: dict[str, object] | None = None
    offline_log = "offline load not requested\n"

    try:
        structural = verify_checkpoint_structure(arguments.checkpoint, arguments.expected_step)
        files = structural["files"]
        tensor_keys = {
            "tensor_keys": structural["tensor_keys"],
            "tensor_key_map": structural["tensor_key_map"],
            "total_tensor_bytes": structural["total_tensor_bytes"],
        }
    except BaseException as exc:
        failures.append(_failure_record("structure", exc))

    if structural is not None and arguments.offline_load:
        try:
            offline = verify_offline_load(
                arguments.checkpoint,
                arguments.cache_root,
                arguments.cosmos_revision,
                process_runner=offline_process_runner,
            )
            offline_log = "\n".join(offline["log_lines"]) + "\n"
        except BaseException as exc:
            failures.append(_failure_record("offline_load", exc))
            offline_log = f"offline load failed ({type(exc).__name__})\n"
    elif arguments.offline_load:
        offline_log = "offline load not run because structural verification failed\n"

    status = "fail" if failures else "pass"
    verdict = {
        "status": status,
        "expected_step": arguments.expected_step,
        "offline_load_requested": arguments.offline_load,
        "structural_status": "pass" if structural is not None else "fail",
        "offline_load_status": (
            "pass" if offline is not None else "fail" if arguments.offline_load else "not_requested"
        ),
        "failures": failures,
    }
    try:
        writer(arguments.output_dir / "files.json", files)
        writer(arguments.output_dir / "tensor-keys.json", tensor_keys)
        writer(arguments.output_dir / "offline-load.log", offline_log, text=True)
        writer(arguments.output_dir / "verdict.json", verdict)
    except BaseException as exc:
        print(f"checkpoint evidence publication failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    if sys.argv[1:] == [_INTERNAL_OFFLINE_WORKER_FLAG]:
        raise SystemExit(_offline_worker_entrypoint())
    raise SystemExit(main())
