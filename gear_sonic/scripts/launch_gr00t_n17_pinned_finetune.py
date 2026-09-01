#!/usr/bin/env python3
"""Launch the pinned GR00T N1.7 fine-tune recipe without network fallback."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace, TracebackType

COSMOS_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT_RELATIVE = "models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT_CONFIG_IDENTITY: dict[str, object] = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "transformers_version": "4.57.0.dev0",
}
REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

DEFAULT_CACHE_ROOT = Path("/root/.cache/huggingface")
ALBUMENTATIONS_UPDATE_ENV = "NO_ALBUMENTATIONS_UPDATE"
ALBUMENTATIONS_UPDATE_VALUE = "1"
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
    "training.skip_weight_loading": False,
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


@contextmanager
def _albumentations_import_environment(
    environ: MutableMapping[str, str],
) -> Iterator[None]:
    """Suppress Albumentations update checks before importing GR00T."""
    injected_is_process = environ is os.environ
    injected_previous = environ.get(ALBUMENTATIONS_UPDATE_ENV)
    process_previous = os.environ.get(ALBUMENTATIONS_UPDATE_ENV)
    values = [injected_previous]
    if not injected_is_process:
        values.append(process_previous)
    if any(value is not None and value != ALBUMENTATIONS_UPDATE_VALUE for value in values):
        raise RuntimeError(
            f"{ALBUMENTATIONS_UPDATE_ENV} must be exactly {ALBUMENTATIONS_UPDATE_VALUE!r} when preconfigured"
        )

    environ[ALBUMENTATIONS_UPDATE_ENV] = ALBUMENTATIONS_UPDATE_VALUE
    os.environ[ALBUMENTATIONS_UPDATE_ENV] = ALBUMENTATIONS_UPDATE_VALUE
    try:
        yield
    finally:
        if not injected_is_process:
            if injected_previous is None:
                environ.pop(ALBUMENTATIONS_UPDATE_ENV, None)
            else:
                environ[ALBUMENTATIONS_UPDATE_ENV] = injected_previous
            if process_previous is None:
                os.environ.pop(ALBUMENTATIONS_UPDATE_ENV, None)
            else:
                os.environ[ALBUMENTATIONS_UPDATE_ENV] = process_previous


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


def _reject_duplicate_snapshot_config_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Cosmos snapshot config contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_snapshot_config(value: str) -> object:
    raise RuntimeError(f"Cosmos snapshot config contains nonstandard numeric constant: {value}")


def _validate_cosmos_snapshot_config(snapshot: Path) -> None:
    """Fail closed unless the resolved snapshot declares the pinned Qwen3 VL identity."""
    snapshot = Path(snapshot)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("exact resolved Cosmos snapshot must be a real directory")
    config_path = snapshot / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError("exact resolved Cosmos snapshot config.json must be a real file")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_snapshot_config_pairs,
                parse_constant=_reject_nonfinite_snapshot_config,
            )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("exact resolved Cosmos snapshot config.json is not valid JSON") from exc
    if type(payload) is not dict:
        raise RuntimeError("exact resolved Cosmos snapshot config.json must contain an object")
    for field, expected in COSMOS_SNAPSHOT_CONFIG_IDENTITY.items():
        actual = payload.get(field)
        if actual != expected:
            raise RuntimeError(f"Cosmos snapshot config {field} mismatch: expected {expected!r}, got {actual!r}")


@contextmanager
def _offline_cosmos_model_info_shim(
    snapshot: Path,
    huggingface_hub_module: object,
) -> Iterator[None]:
    """Classify only the pinned Cosmos repo locally during official GR00T loading."""
    _validate_cosmos_snapshot_config(snapshot)
    original_model_info = getattr(huggingface_hub_module, "model_info", None)
    if not callable(original_model_info):
        raise RuntimeError("huggingface_hub.model_info must be callable before installing offline shim")

    def pinned_model_info(repo_id: str, *args: object, **kwargs: object) -> SimpleNamespace:
        if repo_id != COSMOS_MODEL_ID:
            raise RuntimeError("offline model_info shim only supports the exact canonical Cosmos model ID")
        if args or kwargs:
            raise RuntimeError("offline Cosmos model_info shim does not accept additional arguments")
        return SimpleNamespace(tags=["qwen3_vl"])

    setattr(huggingface_hub_module, "model_info", pinned_model_info)
    try:
        yield
    finally:
        setattr(huggingface_hub_module, "model_info", original_model_info)


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


def _cleanup_failure_summary(
    failures: list[tuple[BaseException, TracebackType | None]],
) -> RuntimeError:
    details = "; ".join(f"{type(error).__name__}: {error}" for error, _traceback in failures)
    return RuntimeError(f"offline preflight cleanup failures: {details}")


def run_offline_preflight(
    config: object,
    pipeline_class: Callable[[object, Path], object],
    save_cfg_dir: Path,
    *,
    torch_module: object,
    cosmos_snapshot: Path,
    huggingface_hub_module: object,
    gc_collect: Callable[[], object] = gc.collect,
) -> dict[str, str | None]:
    """Run official pipeline setup, record loaded classes, and release resources."""
    if config.training.use_wandb is not False:
        raise RuntimeError("offline preflight requires W&B to be disabled")
    _validate_cosmos_snapshot_config(cosmos_snapshot)
    save_cfg_dir.mkdir(parents=True, exist_ok=False)
    pipeline = pipeline_class(config, save_cfg_dir)
    record = None
    setup_failure: tuple[BaseException, TracebackType | None] | None = None
    try:
        with _offline_cosmos_model_info_shim(cosmos_snapshot, huggingface_hub_module):
            pipeline.setup()
        record = {
            "model_class": _class_name(getattr(pipeline, "model", None)),
            "processor_class": _class_name(getattr(pipeline, "processor", None)),
            "train_dataset_class": _class_name(getattr(pipeline, "train_dataset", None)),
            "eval_dataset_class": _class_name(getattr(pipeline, "eval_dataset", None)),
            "data_collator_class": _class_name(getattr(pipeline, "data_collator", None)),
        }
    except BaseException as error:
        setup_failure = (error, error.__traceback__)

    cleanup_failures: list[tuple[BaseException, TracebackType | None]] = []
    for dataset_name in ("train_dataset", "eval_dataset"):
        dataset = getattr(pipeline, dataset_name, None)
        close = getattr(dataset, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                cleanup_failures.append((error, error.__traceback__))
    del close, dataset
    for attribute in (
        "model",
        "processor",
        "train_dataset",
        "eval_dataset",
        "data_collator",
    ):
        if hasattr(pipeline, attribute):
            try:
                setattr(pipeline, attribute, None)
            except BaseException as error:
                cleanup_failures.append((error, error.__traceback__))
    del pipeline
    try:
        gc_collect()
    except BaseException as error:
        cleanup_failures.append((error, error.__traceback__))
    try:
        torch_module.cuda.empty_cache()
    except BaseException as error:
        cleanup_failures.append((error, error.__traceback__))

    if setup_failure is not None:
        error, traceback = setup_failure
        if cleanup_failures:
            raise error.with_traceback(traceback) from _cleanup_failure_summary(cleanup_failures)
        raise error.with_traceback(traceback)
    if cleanup_failures:
        error, traceback = cleanup_failures[0]
        if len(cleanup_failures) > 1:
            raise error.with_traceback(traceback) from _cleanup_failure_summary(cleanup_failures[1:])
        raise error.with_traceback(traceback)
    if record is None:
        raise RuntimeError("offline preflight setup returned without a class record")
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
    import huggingface_hub
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
        snapshot_download=huggingface_hub.snapshot_download,
        huggingface_hub=huggingface_hub,
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
    dependency_loader: Callable[[], object] | None = None,
) -> None:
    """Validate the immutable runtime contract and run preflight or training."""
    environment = os.environ if environ is None else environ
    assert_offline_environment(environment)
    preflight_only = parse_preflight_only(environment.get(PREFLIGHT_ONLY_ENV))
    load_dependencies = _load_runtime_dependencies if dependency_loader is None else dependency_loader
    with _albumentations_import_environment(environment):
        runtime = load_dependencies() if dependencies is None else dependencies

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
            cosmos_snapshot=snapshot,
            huggingface_hub_module=runtime.huggingface_hub,
            gc_collect=getattr(runtime, "gc_collect", gc.collect),
        )
        print(json.dumps({"offline_preflight": preflight_record}, sort_keys=True))
        return

    install_training_arguments_audit(
        runtime.experiment_module,
        training_arguments_audit,
    )
    with _offline_cosmos_model_info_shim(snapshot, runtime.huggingface_hub):
        runtime.run(config)


if __name__ == "__main__":
    main()
