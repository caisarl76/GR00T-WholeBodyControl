#!/usr/bin/env python3
"""Launch the pinned GR00T N1.7 fine-tune recipe without network fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

COSMOS_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT_RELATIVE = "models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

DEFAULT_CACHE_ROOT = Path("/root/.cache/huggingface")
FRESHNESS_AUDIT_ENV = "GR00T_FRESHNESS_AUDIT_PATH"
PREFLIGHT_ONLY_ENV = "GR00T_PINNED_PREFLIGHT_ONLY"
TRAINING_ARGS_AUDIT_ENV = "GR00T_TRAINING_ARGS_AUDIT_PATH"

PINNED_RECIPE: dict[str, object] = {
    "training.optim": "adamw_torch",
    "training.global_batch_size": 32,
    "training.batch_size": None,
    "training.dataloader_num_workers": 4,
    "training.learning_rate": 1e-4,
    "training.gradient_accumulation_steps": 1,
    "training.lr_scheduler_type": "cosine",
    "training.weight_decay": 1e-5,
    "training.warmup_ratio": 0.05,
    "training.warmup_steps": 0,
    "training.max_grad_norm": 1.0,
    "training.logging_steps": 10,
    "training.num_gpus": 1,
    "training.use_ddp": False,
    "training.tf32": True,
    "training.fp16": False,
    "training.bf16": True,
    "training.gradient_checkpointing": False,
    "training.save_only_model": False,
    "training.save_total_limit": 5,
    "training.wandb_project": "gr00t-n1.7-pnp-trash",
    "data.shard_size": 1024,
    "data.episode_sampling_rate": 0.1,
    "data.num_shards_per_epoch": 100000,
    "data.shuffle": True,
    "data.seed": 42,
    "model.tune_llm": False,
    "model.tune_visual": False,
    "model.tune_projector": True,
    "model.tune_diffusion_model": True,
    "model.tune_vlln": True,
    "model.tune_top_llm_layers": 0,
    "model.state_dropout_prob": 0.2,
    "model.random_rotation_angle": None,
    "model.extra_augmentation_config": None,
    "model.color_jitter_params": {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    },
}

TRAINING_ARGUMENT_AUDIT_FIELDS = (
    "deepspeed",
    "report_to",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "logging_steps",
    "save_steps",
    "save_total_limit",
    "save_only_model",
    "fp16",
    "bf16",
    "tf32",
    "gradient_checkpointing",
    "optim",
    "dataloader_num_workers",
    "seed",
)


def apply_runtime_pins(config: object, *, cache_root: Path) -> None:
    """Apply the immutable Cosmos identity and offline Transformers settings."""
    hub_cache = cache_root / "hub"
    config.model.model_name = COSMOS_MODEL_ID
    config.model.model_revision = COSMOS_REVISION
    config.training.transformers_local_files_only = True
    config.training.transformers_cache_dir = str(hub_cache)


def assert_offline_environment(environ: Mapping[str, str] | None = None) -> None:
    """Require every offline flag to have its exact fail-closed value."""
    environment = os.environ if environ is None else environ
    mismatches = {
        key: {"expected": expected, "actual": environment.get(key)}
        for key, expected in REQUIRED_OFFLINE_ENV.items()
        if environment.get(key) != expected
    }
    if mismatches:
        raise RuntimeError("required offline environment is not active: " + json.dumps(mismatches, sort_keys=True))


def resolve_snapshot(
    cache_root: Path,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Resolve the exact Cosmos revision locally and verify its pinned location."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download

        snapshot_download = huggingface_snapshot_download

    hub_cache = cache_root / "hub"
    expected = hub_cache / COSMOS_SNAPSHOT_RELATIVE
    if expected.is_symlink() or not expected.is_dir():
        raise RuntimeError(f"exact Cosmos snapshot is missing or not a directory: {expected}")

    resolved_raw = snapshot_download(
        repo_id=COSMOS_MODEL_ID,
        revision=COSMOS_REVISION,
        cache_dir=str(hub_cache),
        local_files_only=True,
    )
    returned_path = Path(os.path.abspath(resolved_raw))
    expected_path = Path(os.path.abspath(expected))
    if returned_path != expected_path or returned_path.is_symlink():
        raise RuntimeError(
            f"exact Cosmos snapshot resolution mismatch: expected {expected_path}, returned {returned_path}"
        )
    try:
        expected_resolved = expected_path.resolve(strict=True)
        resolved = returned_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("exact Cosmos snapshot could not be resolved locally") from exc
    if resolved != expected_resolved:
        raise RuntimeError(
            f"exact Cosmos snapshot resolution mismatch: expected {expected_resolved}, resolved {resolved}"
        )
    return resolved


def _validate_new_absolute_path(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise RuntimeError(f"{label} audit path must be absolute: {path}")
    if os.path.lexists(path):
        raise RuntimeError(f"{label} audit path must not already exist: {path}")
    if not path.parent.is_dir():
        raise RuntimeError(f"{label} audit path parent must exist: {path.parent}")


def _atomic_write_new_json(path: Path, payload: Mapping[str, object], *, label: str) -> None:
    """Publish JSON atomically without ever replacing a preexisting path."""
    _validate_new_absolute_path(path, label=label)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise RuntimeError(f"{label} audit path already exists: {path}") from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def assert_fresh_experiment(
    experiment: Path,
    *,
    get_last_checkpoint: Callable[[str], str | None],
    audit_path: Path,
) -> None:
    """Create a never-before-used experiment directory and audit checkpoint discovery."""
    _validate_new_absolute_path(audit_path, label="freshness")
    if os.path.lexists(experiment):
        raise RuntimeError(f"fresh experiment path must be absent: {experiment}")

    experiment.mkdir(parents=True, exist_ok=False)
    checkpoint = get_last_checkpoint(str(experiment))
    if checkpoint is not None:
        raise RuntimeError(f"fresh experiment checkpoint discovery returned a checkpoint: {checkpoint}")
    if any(experiment.iterdir()):
        raise RuntimeError(
            f"fresh experiment directory must remain empty after checkpoint discovery: {experiment}"
        )
    _atomic_write_new_json(
        audit_path,
        {
            "experiment": str(experiment),
            "get_last_checkpoint": None,
        },
        label="freshness",
    )


def _dotted_value(config: object, dotted_name: str) -> object:
    current = config
    for component in dotted_name.split("."):
        try:
            current = getattr(current, component)
        except AttributeError as exc:
            raise RuntimeError(f"recipe field is missing: {dotted_name}") from exc
    return current


def assert_recipe_contract(config: object) -> None:
    """Require the fully resolved configuration to match the pinned recipe."""
    mismatches = {}
    for field, expected in PINNED_RECIPE.items():
        try:
            actual = _dotted_value(config, field)
        except RuntimeError:
            actual = "<missing>"
        if actual != expected:
            mismatches[field] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "pinned fine-tune recipe mismatch:\n" + json.dumps(mismatches, default=repr, indent=2, sort_keys=True)
        )


def parse_preflight_only(value: str | None) -> bool:
    """Parse the sole operational switch as a strict closed boolean."""
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{PREFLIGHT_ONLY_ENV} must be exactly 0 or 1, got {value!r}")


def verify_backbone_selector(
    config_or_model: object,
    get_backbone_cls: Callable[[object], type],
) -> type:
    """Require the pinned selector to choose Qwen3Backbone."""
    model_config = getattr(config_or_model, "model", config_or_model)
    backbone_class = get_backbone_cls(model_config)
    if getattr(backbone_class, "__name__", None) != "Qwen3Backbone":
        raise RuntimeError(
            "pinned selector must return Qwen3Backbone, got "
            f"{getattr(backbone_class, '__name__', repr(backbone_class))}"
        )
    return backbone_class


def build_pinned_config(
    ft_config: object,
    *,
    get_default_config: Callable[[], object],
    embodiment_tag: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> object:
    """Reproduce the complete pinned NVIDIA fine-tune argument mapping."""
    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.extra_augmentation_config = (
        json.loads(ft_config.extra_augmentation_config) if ft_config.extra_augmentation_config else None
    )
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = COSMOS_MODEL_ID
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    apply_runtime_pins(config, cache_root=cache_root)
    return config


def install_training_arguments_audit(module: object, destination: Path) -> None:
    """Wrap pinned TrainingArguments to audit effective, secret-free values."""
    _validate_new_absolute_path(destination, label="training arguments")
    original_constructor = module.TrainingArguments

    def audited_training_arguments(*args: object, **kwargs: object) -> object:
        arguments = original_constructor(*args, **kwargs)
        effective_deepspeed = getattr(arguments, "deepspeed", "<missing>")
        if effective_deepspeed is not None:
            raise RuntimeError(f"effective TrainingArguments.deepspeed must be None, got {effective_deepspeed!r}")

        to_dict = getattr(arguments, "to_dict", None)
        if not callable(to_dict):
            raise RuntimeError("effective TrainingArguments must provide to_dict()")
        serialized = to_dict()
        if not isinstance(serialized, Mapping):
            raise RuntimeError("effective TrainingArguments.to_dict() must return a mapping")
        missing = [field for field in TRAINING_ARGUMENT_AUDIT_FIELDS if field not in serialized]
        if missing:
            raise RuntimeError("effective TrainingArguments audit fields are missing: " + ", ".join(missing))
        projection = {field: serialized[field] for field in TRAINING_ARGUMENT_AUDIT_FIELDS}
        projection["deepspeed"] = effective_deepspeed
        _atomic_write_new_json(
            destination,
            projection,
            label="training arguments",
        )
        return arguments

    module.TrainingArguments = audited_training_arguments


def _class_name(value: object | None) -> str | None:
    return None if value is None else type(value).__name__


def run_offline_preflight(
    config: object,
    pipeline_class: Callable[[object, Path], object],
    save_cfg_dir: Path,
    *,
    torch_module: object,
    gc_collect: Callable[[], object] = gc.collect,
) -> dict[str, str | None]:
    """Run official pipeline setup, record loaded classes, and release resources."""
    if config.training.use_wandb is not False:
        raise RuntimeError("offline preflight requires W&B to be disabled")
    save_cfg_dir.mkdir(parents=True, exist_ok=False)
    pipeline = pipeline_class(config, save_cfg_dir)
    try:
        pipeline.setup()
        record = {
            "model_class": _class_name(getattr(pipeline, "model", None)),
            "processor_class": _class_name(getattr(pipeline, "processor", None)),
            "train_dataset_class": _class_name(getattr(pipeline, "train_dataset", None)),
            "eval_dataset_class": _class_name(getattr(pipeline, "eval_dataset", None)),
            "data_collator_class": _class_name(getattr(pipeline, "data_collator", None)),
        }
    finally:
        for dataset_name in ("train_dataset", "eval_dataset"):
            dataset = getattr(pipeline, dataset_name, None)
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        del close, dataset
        for attribute in (
            "model",
            "processor",
            "train_dataset",
            "eval_dataset",
            "data_collator",
        ):
            if hasattr(pipeline, attribute):
                setattr(pipeline, attribute, None)
        del pipeline
        gc_collect()
        torch_module.cuda.empty_cache()
    return record


def _load_runtime_dependencies() -> SimpleNamespace:
    """Import GR00T and heavyweight runtime packages only after offline validation."""
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.finetune_config import FinetuneConfig
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment import experiment as experiment_module
    from gr00t.experiment.experiment import run
    from gr00t.experiment.launch_finetune import load_modality_config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from huggingface_hub import snapshot_download
    import torch
    from transformers.trainer_utils import get_last_checkpoint
    import tyro

    return SimpleNamespace(
        cache_root=DEFAULT_CACHE_ROOT,
        torch=torch,
        tyro=tyro,
        FinetuneConfig=FinetuneConfig,
        get_default_config=get_default_config,
        load_modality_config=load_modality_config,
        EmbodimentTag=EmbodimentTag,
        experiment_module=experiment_module,
        run=run,
        get_backbone_cls=get_backbone_cls,
        Gr00tN1d7Pipeline=Gr00tN1d7Pipeline,
        snapshot_download=snapshot_download,
        get_last_checkpoint=get_last_checkpoint,
        gc_collect=gc.collect,
    )


def _required_audit_path(environ: Mapping[str, str], variable: str) -> Path:
    value = environ.get(variable)
    if value is None or value == "":
        raise RuntimeError(f"{variable} must name an absolute nonexistent path")
    path = Path(value)
    _validate_new_absolute_path(path, label=variable)
    return path


def main(
    *,
    environ: MutableMapping[str, str] | None = None,
    dependencies: object | None = None,
) -> None:
    """Validate the immutable runtime contract and run preflight or training."""
    environment = os.environ if environ is None else environ
    assert_offline_environment(environment)
    preflight_only = parse_preflight_only(environment.get(PREFLIGHT_ONLY_ENV))
    runtime = _load_runtime_dependencies() if dependencies is None else dependencies

    environment.setdefault("LOGURU_LEVEL", "INFO")
    ft_config = runtime.tyro.cli(runtime.FinetuneConfig, description=__doc__)
    ft_config.embodiment_tag = runtime.EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value
    if ft_config.modality_config_path is not None:
        runtime.load_modality_config(ft_config.modality_config_path)

    cache_root = Path(getattr(runtime, "cache_root", DEFAULT_CACHE_ROOT))
    config = build_pinned_config(
        ft_config,
        get_default_config=runtime.get_default_config,
        embodiment_tag=embodiment_tag,
        cache_root=cache_root,
    )
    assert_recipe_contract(config)

    if preflight_only:
        if config.training.use_wandb is not False:
            raise RuntimeError("offline preflight requires W&B to be disabled")
    elif config.training.use_wandb is not True:
        raise RuntimeError("production training requires W&B to be enabled")

    snapshot = resolve_snapshot(
        cache_root,
        snapshot_download=runtime.snapshot_download,
    )
    backbone_class = verify_backbone_selector(config.model, runtime.get_backbone_cls)
    print(
        json.dumps(
            {
                "backbone_class": backbone_class.__name__,
                "cosmos_model_id": config.model.model_name,
                "cosmos_revision": config.model.model_revision,
                "cosmos_snapshot": str(snapshot),
            },
            sort_keys=True,
        )
    )

    experiment_name = config.training.experiment_name
    if not isinstance(experiment_name, str) or not experiment_name:
        raise RuntimeError("training.experiment_name must be a nonempty string")
    freshness_audit = _required_audit_path(environment, FRESHNESS_AUDIT_ENV)
    training_arguments_audit = None
    if not preflight_only:
        training_arguments_audit = _required_audit_path(environment, TRAINING_ARGS_AUDIT_ENV)

    experiment = Path(config.training.output_dir) / experiment_name
    assert_fresh_experiment(
        experiment,
        get_last_checkpoint=runtime.get_last_checkpoint,
        audit_path=freshness_audit,
    )

    if preflight_only:
        preflight_record = run_offline_preflight(
            config,
            runtime.Gr00tN1d7Pipeline,
            experiment / "experiment_cfg",
            torch_module=runtime.torch,
            gc_collect=getattr(runtime, "gc_collect", gc.collect),
        )
        print(json.dumps({"offline_preflight": preflight_record}, sort_keys=True))
        return

    install_training_arguments_audit(
        runtime.experiment_module,
        training_arguments_audit,
    )
    runtime.run(config)


if __name__ == "__main__":
    main()
