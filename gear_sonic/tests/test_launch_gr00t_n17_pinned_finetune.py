import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from gear_sonic.scripts.launch_gr00t_n17_pinned_finetune import (
    COSMOS_MODEL_ID,
    COSMOS_REVISION,
    COSMOS_SNAPSHOT_RELATIVE,
    REQUIRED_OFFLINE_ENV,
    apply_runtime_pins,
    assert_fresh_experiment,
    assert_offline_environment,
    assert_recipe_contract,
    build_pinned_config,
    install_training_arguments_audit,
    main,
    parse_preflight_only,
    resolve_snapshot,
    run_offline_preflight,
    verify_backbone_selector,
)

EXPECTED_RECIPE = {
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


def expected_cosmos_snapshot(cache_root: Path) -> Path:
    return cache_root / "hub" / "models--nvidia--Cosmos-Reason2-2B" / "snapshots" / COSMOS_REVISION


def fake_config(*, use_wandb: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        load_config_path="sentinel",
        model=SimpleNamespace(
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            tune_top_llm_layers=0,
            state_dropout_prob=0.2,
            random_rotation_angle=None,
            color_jitter_params={
                "brightness": 0.3,
                "contrast": 0.4,
                "saturation": 0.5,
                "hue": 0.08,
            },
            extra_augmentation_config=None,
            load_bf16="sentinel",
            reproject_vision="sentinel",
            model_name="sentinel",
            model_revision="sentinel",
            backbone_trainable_params_fp32="sentinel",
            use_relative_action="sentinel",
        ),
        training=SimpleNamespace(
            experiment_name="experiment",
            start_from_checkpoint="sentinel",
            optim="adamw_torch",
            global_batch_size=32,
            batch_size=None,
            dataloader_num_workers=4,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            lr_scheduler_type="cosine",
            output_dir="sentinel",
            save_steps=1000,
            save_total_limit=5,
            num_gpus=1,
            use_wandb=use_wandb,
            use_ddp=False,
            max_steps=20000,
            weight_decay=1e-5,
            warmup_ratio=0.05,
            warmup_steps=0,
            max_grad_norm=1.0,
            logging_steps=10,
            wandb_project="gr00t-n1.7-pnp-trash",
            save_only_model=False,
            skip_weight_loading=False,
            tf32=True,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            transformers_local_files_only="sentinel",
            transformers_cache_dir="sentinel",
        ),
        data=SimpleNamespace(
            datasets="sentinel",
            shard_size=1024,
            episode_sampling_rate=0.1,
            num_shards_per_epoch=100000,
            shuffle=True,
            seed=42,
        ),
    )


def fake_finetune_config(tmp_path: Path, *, use_wandb: bool) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_path="/models/gr00t/snapshot",
        dataset_path=f"/dataset/a{os.pathsep}/dataset/b",
        embodiment_tag="UNITREE_G1_SONIC",
        modality_config_path="gr00t/configs/data/embodiment_configs.py",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        state_dropout_prob=0.2,
        random_rotation_angle=None,
        color_jitter_params={
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        },
        extra_augmentation_config=None,
        experiment_name="experiment",
        global_batch_size=32,
        dataloader_num_workers=4,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        output_dir=str(tmp_path / "outputs"),
        save_steps=13,
        save_total_limit=5,
        num_gpus=1,
        use_wandb=use_wandb,
        max_steps=17,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        wandb_project="gr00t-n1.7-pnp-trash",
        save_only_model=False,
        skip_weight_loading=False,
        shard_size=1024,
        episode_sampling_rate=0.1,
        num_shards_per_epoch=100000,
    )


def _training_arguments_payload(deepspeed: object) -> dict[str, object]:
    return {
        "deepspeed": deepspeed,
        "report_to": ["wandb"],
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 1000,
        "save_total_limit": 5,
        "save_only_model": False,
        "fp16": False,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 4,
        "seed": 42,
    }


def test_module_import_does_not_import_gr00t() -> None:
    module_path = Path(__file__).parents[1] / "scripts" / "launch_gr00t_n17_pinned_finetune.py"
    script = f"""
import builtins
import runpy

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'gr00t' or name.startswith('gr00t.'):
        raise AssertionError(f'unexpected GR00T import: {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
runpy.run_path({str(module_path)!r}, run_name='import_safety_check')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_pinned_constants_are_exact() -> None:
    assert COSMOS_MODEL_ID == "nvidia/Cosmos-Reason2-2B"
    assert COSMOS_REVISION == "9ce19a195e423419c349abfc86fd07178b230561"
    assert COSMOS_SNAPSHOT_RELATIVE == (
        "models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
    )
    assert REQUIRED_OFFLINE_ENV == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_apply_runtime_pins_preserves_selector_name_and_sets_offline_fields(
    tmp_path: Path,
) -> None:
    config = fake_config()
    apply_runtime_pins(config, cache_root=tmp_path)
    assert config.model.model_name == "nvidia/Cosmos-Reason2-2B"
    assert config.model.model_revision == "9ce19a195e423419c349abfc86fd07178b230561"
    assert config.training.transformers_local_files_only is True
    assert config.training.transformers_cache_dir == str(tmp_path / "hub")


def test_assert_offline_environment_accepts_only_required_exact_values() -> None:
    assert_offline_environment({**REQUIRED_OFFLINE_ENV, "UNRELATED": "allowed"})


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {**REQUIRED_OFFLINE_ENV, "HF_HUB_OFFLINE": "0"},
        {**REQUIRED_OFFLINE_ENV, "TRANSFORMERS_OFFLINE": "true"},
    ],
)
def test_assert_offline_environment_rejects_missing_or_nonexact_values(
    environment: dict[str, str],
) -> None:
    with pytest.raises(RuntimeError, match="offline environment"):
        assert_offline_environment(environment)


def test_resolve_snapshot_uses_exact_id_revision_cache_and_local_only(tmp_path: Path) -> None:
    expected = expected_cosmos_snapshot(tmp_path)
    expected.mkdir(parents=True)
    calls = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(expected)

    resolved = resolve_snapshot(tmp_path, snapshot_download=snapshot_download)

    assert resolved == expected.resolve()
    assert calls == [
        {
            "repo_id": COSMOS_MODEL_ID,
            "revision": COSMOS_REVISION,
            "cache_dir": str(tmp_path / "hub"),
            "local_files_only": True,
        }
    ]


def test_resolve_snapshot_matches_realistic_hub_cache_layout(tmp_path: Path) -> None:
    expected = expected_cosmos_snapshot(tmp_path)
    expected.mkdir(parents=True)

    def realistic_snapshot_download(
        *,
        repo_id: str,
        revision: str,
        cache_dir: str,
        local_files_only: bool,
    ) -> str:
        assert local_files_only is True
        repo_folder_name = f"models--{repo_id.replace('/', '--')}"
        cached_snapshot = Path(cache_dir) / repo_folder_name / "snapshots" / revision
        if not cached_snapshot.is_dir():
            raise FileNotFoundError(cached_snapshot)
        return str(cached_snapshot)

    assert resolve_snapshot(tmp_path, snapshot_download=realistic_snapshot_download) == expected.resolve()


def test_resolve_snapshot_rejects_resolution_to_other_cached_revision(tmp_path: Path) -> None:
    expected = expected_cosmos_snapshot(tmp_path)
    expected.mkdir(parents=True)
    other = tmp_path / "hub" / "other"
    other.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="exact Cosmos snapshot"):
        resolve_snapshot(tmp_path, snapshot_download=lambda **_kwargs: str(other))


def test_resolve_snapshot_rejects_missing_expected_snapshot(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exact Cosmos snapshot"):
        resolve_snapshot(tmp_path, snapshot_download=lambda **_kwargs: str(tmp_path))


def test_resolve_snapshot_rejects_symlinked_snapshot_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual-snapshot"
    actual.mkdir()
    expected = expected_cosmos_snapshot(tmp_path)
    expected.parent.mkdir(parents=True)
    expected.symlink_to(actual, target_is_directory=True)

    with pytest.raises(RuntimeError, match="exact Cosmos snapshot"):
        resolve_snapshot(tmp_path, snapshot_download=lambda **_kwargs: str(expected))


def test_resolve_snapshot_rejects_alias_returned_for_exact_directory(tmp_path: Path) -> None:
    expected = expected_cosmos_snapshot(tmp_path)
    expected.mkdir(parents=True)
    alias = tmp_path / "snapshot-alias"
    alias.symlink_to(expected, target_is_directory=True)

    with pytest.raises(RuntimeError, match="exact Cosmos snapshot"):
        resolve_snapshot(tmp_path, snapshot_download=lambda **_kwargs: str(alias))


@pytest.mark.parametrize("kind", ["directory", "file", "dangling-symlink"])
def test_assert_fresh_experiment_rejects_every_preexisting_path_kind(
    tmp_path: Path,
    kind: str,
) -> None:
    experiment = tmp_path / "experiment"
    if kind == "directory":
        experiment.mkdir()
    elif kind == "file":
        experiment.write_text("occupied", encoding="utf-8")
    else:
        experiment.symlink_to(tmp_path / "missing")

    with pytest.raises(RuntimeError, match="fresh experiment"):
        assert_fresh_experiment(
            experiment,
            get_last_checkpoint=lambda _path: None,
            audit_path=(tmp_path / "freshness-runtime.json").resolve(),
        )


def test_assert_fresh_experiment_rejects_checkpoint(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoint-1"
    with pytest.raises(RuntimeError, match="checkpoint"):
        assert_fresh_experiment(
            experiment,
            get_last_checkpoint=lambda _path: str(checkpoint),
            audit_path=(tmp_path / "freshness-runtime.json").resolve(),
        )


def test_assert_fresh_experiment_creates_empty_dir_and_records_none(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    audit_path = (tmp_path / "freshness-runtime.json").resolve()

    assert_fresh_experiment(
        experiment,
        get_last_checkpoint=lambda path: None if Path(path).is_dir() else "missing",
        audit_path=audit_path,
    )

    assert experiment.is_dir()
    assert list(experiment.iterdir()) == []
    assert json.loads(audit_path.read_text(encoding="utf-8")) == {
        "experiment": str(experiment),
        "get_last_checkpoint": None,
    }


def test_assert_fresh_experiment_rejects_mutation_by_checkpoint_probe(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"

    def mutate(path: str) -> None:
        (Path(path) / "unexpected").write_text("created", encoding="utf-8")
        return None

    with pytest.raises(RuntimeError, match="empty"):
        assert_fresh_experiment(
            experiment,
            get_last_checkpoint=mutate,
            audit_path=(tmp_path / "freshness-runtime.json").resolve(),
        )


@pytest.mark.parametrize("preexisting_kind", ["file", "dangling-symlink"])
def test_assert_fresh_experiment_never_reuses_audit_path(
    tmp_path: Path,
    preexisting_kind: str,
) -> None:
    audit_path = (tmp_path / "freshness-runtime.json").resolve()
    if preexisting_kind == "file":
        audit_path.write_text("do not overwrite", encoding="utf-8")
    else:
        audit_path.symlink_to(tmp_path / "missing")

    with pytest.raises(RuntimeError, match="audit path"):
        assert_fresh_experiment(
            tmp_path / "experiment",
            get_last_checkpoint=lambda _path: None,
            audit_path=audit_path,
        )


def test_assert_fresh_experiment_requires_absolute_audit_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="absolute"):
        assert_fresh_experiment(
            tmp_path / "experiment",
            get_last_checkpoint=lambda _path: None,
            audit_path=Path("relative.json"),
        )


def test_verify_backbone_selector_requires_qwen3_backbone() -> None:
    fake_qwen3_backbone = type("Qwen3Backbone", (), {})
    config = fake_config()
    assert verify_backbone_selector(config, lambda _model: fake_qwen3_backbone) is fake_qwen3_backbone


def test_verify_backbone_selector_rejects_other_backbone() -> None:
    other_backbone = type("OtherBackbone", (), {})
    with pytest.raises(RuntimeError, match="Qwen3Backbone"):
        verify_backbone_selector(fake_config().model, lambda _model: other_backbone)


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_parse_preflight_only_accepts_closed_boolean(
    value: str | None,
    expected: bool,
) -> None:
    assert parse_preflight_only(value) is expected


@pytest.mark.parametrize("value", ["", "true", "false", "yes", "2", " 1"])
def test_parse_preflight_only_rejects_other_values(value: str) -> None:
    with pytest.raises(RuntimeError, match="GR00T_PINNED_PREFLIGHT_ONLY"):
        parse_preflight_only(value)


def test_assert_recipe_contract_accepts_complete_exact_recipe() -> None:
    assert_recipe_contract(fake_config())


def test_assert_recipe_contract_reports_every_mismatched_field() -> None:
    config = fake_config()
    config.training.optim = "adamw_hf"
    config.data.seed = 7
    config.model.tune_llm = True

    with pytest.raises(RuntimeError) as exc_info:
        assert_recipe_contract(config)

    message = str(exc_info.value)
    assert "training.optim" in message
    assert "data.seed" in message
    assert "model.tune_llm" in message


def test_build_pinned_config_applies_complete_launcher_mapping(tmp_path: Path) -> None:
    config = fake_config(use_wandb=False)
    config.data.datasets = None

    class DefaultConfig:
        def load_dict(self, payload: dict[str, object]) -> SimpleNamespace:
            config.data.datasets = payload["data"]["datasets"]
            config.data.download_cache = payload["data"]["download_cache"]
            return config

    ft_config = fake_finetune_config(tmp_path, use_wandb=True)
    built = build_pinned_config(
        ft_config,
        get_default_config=lambda: DefaultConfig(),
        embodiment_tag="UNITREE_G1_SONIC",
        cache_root=tmp_path,
    )

    assert built is config
    assert config.load_config_path is None
    assert config.data.download_cache is False
    assert config.data.datasets == [
        {
            "dataset_paths": ["/dataset/a", "/dataset/b"],
            "mix_ratio": 1.0,
            "embodiment_tag": "UNITREE_G1_SONIC",
        }
    ]
    assert config.model.load_bf16 is False
    assert config.model.reproject_vision is False
    assert config.model.backbone_trainable_params_fp32 is True
    assert config.model.use_relative_action is True
    assert config.training.experiment_name == "experiment"
    assert config.training.start_from_checkpoint == "/models/gr00t/snapshot"
    assert config.training.output_dir == str(tmp_path / "outputs")
    assert config.training.save_steps == 13
    assert config.training.max_steps == 17
    assert config.training.use_wandb is True
    assert config.training.skip_weight_loading is False
    assert_recipe_contract(config)


def test_build_pinned_config_parses_extra_augmentation_json(tmp_path: Path) -> None:
    config = fake_config()
    ft_config = fake_finetune_config(tmp_path, use_wandb=True)
    ft_config.extra_augmentation_config = '{"masked_region_transforms": []}'

    class DefaultConfig:
        def load_dict(self, _payload: dict[str, object]) -> SimpleNamespace:
            return config

    built = build_pinned_config(
        ft_config,
        get_default_config=lambda: DefaultConfig(),
        embodiment_tag="UNITREE_G1_SONIC",
        cache_root=tmp_path,
    )
    assert built.model.extra_augmentation_config == {"masked_region_transforms": []}


def test_training_arguments_audit_records_only_secret_free_projection(
    tmp_path: Path,
) -> None:
    payload = {**_training_arguments_payload(None), "hub_token": "secret-value"}
    arguments = SimpleNamespace(deepspeed=None, to_dict=lambda: payload)
    calls = []

    def constructor(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return arguments

    module = SimpleNamespace(TrainingArguments=constructor)
    destination = (tmp_path / "training-arguments.json").resolve()
    install_training_arguments_audit(module, destination)

    returned = module.TrainingArguments("positional", deepspeed=None)

    assert returned is arguments
    assert calls == [(("positional",), {"deepspeed": None})]
    recorded = json.loads(destination.read_text(encoding="utf-8"))
    assert recorded == _training_arguments_payload(None)
    assert "secret-value" not in destination.read_text(encoding="utf-8")


def test_training_arguments_audit_rejects_active_deepspeed(tmp_path: Path) -> None:
    deepspeed = {"zero_optimization": {"stage": 2}}
    arguments = SimpleNamespace(
        deepspeed=deepspeed,
        to_dict=lambda: _training_arguments_payload(deepspeed),
    )
    module = SimpleNamespace(TrainingArguments=lambda **_kwargs: arguments)
    destination = (tmp_path / "training-arguments.json").resolve()
    install_training_arguments_audit(module, destination)

    with pytest.raises(RuntimeError, match="deepspeed"):
        module.TrainingArguments(deepspeed=arguments.deepspeed)
    assert not destination.exists()


def test_training_arguments_audit_rejects_incomplete_projection(tmp_path: Path) -> None:
    payload = _training_arguments_payload(None)
    del payload["seed"]
    arguments = SimpleNamespace(deepspeed=None, to_dict=lambda: payload)
    module = SimpleNamespace(TrainingArguments=lambda **_kwargs: arguments)
    destination = (tmp_path / "training-arguments.json").resolve()
    install_training_arguments_audit(module, destination)

    with pytest.raises(RuntimeError, match="seed"):
        module.TrainingArguments(deepspeed=None)
    assert not destination.exists()


@pytest.mark.parametrize("destination_kind", ["relative", "file", "dangling-symlink"])
def test_training_arguments_audit_requires_absolute_nonexistent_destination(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    if destination_kind == "relative":
        destination = Path("relative.json")
    else:
        destination = (tmp_path / "training-arguments.json").resolve()
        if destination_kind == "file":
            destination.write_text("occupied", encoding="utf-8")
        else:
            destination.symlink_to(tmp_path / "missing")
    module = SimpleNamespace(TrainingArguments=lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="audit path"):
        install_training_arguments_audit(module, destination)


def test_run_offline_preflight_runs_official_setup_records_classes_and_cleans_up(
    tmp_path: Path,
) -> None:
    events = []

    class Model:
        pass

    class Processor:
        pass

    class TrainDataset:
        def close(self) -> None:
            events.append("train-close")

    class EvalDataset:
        def close(self) -> None:
            events.append("eval-close")

    class Collator:
        pass

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            events.append(("init", config, save_cfg_dir))

        def setup(self) -> None:
            events.append("setup")
            self.model = Model()
            self.processor = Processor()
            self.train_dataset = TrainDataset()
            self.eval_dataset = EvalDataset()
            self.data_collator = Collator()

    torch_module = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache")))
    config = fake_config(use_wandb=False)
    save_cfg_dir = tmp_path / "experiment" / "experiment_cfg"

    record = run_offline_preflight(
        config,
        Pipeline,
        save_cfg_dir,
        torch_module=torch_module,
        gc_collect=lambda: events.append("gc-collect"),
    )

    assert record == {
        "model_class": "Model",
        "processor_class": "Processor",
        "train_dataset_class": "TrainDataset",
        "eval_dataset_class": "EvalDataset",
        "data_collator_class": "Collator",
    }
    assert save_cfg_dir.is_dir()
    assert events == [
        ("init", config, save_cfg_dir),
        "setup",
        "train-close",
        "eval-close",
        "gc-collect",
        "empty-cache",
    ]


def test_run_offline_preflight_requires_wandb_disabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="W&B"):
        run_offline_preflight(
            fake_config(use_wandb=True),
            lambda *_args: pytest.fail("pipeline must not be constructed"),
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        )


def test_run_offline_preflight_releases_loaded_objects_before_collection(
    tmp_path: Path,
) -> None:
    references: list[weakref.ReferenceType[object]] = []

    class LoadedObject:
        pass

    class Pipeline:
        def __init__(self, _config: object, _save_cfg_dir: Path) -> None:
            pass

        def setup(self) -> None:
            self.model = LoadedObject()
            self.processor = LoadedObject()
            self.train_dataset = LoadedObject()
            self.eval_dataset = LoadedObject()
            self.data_collator = LoadedObject()
            references.extend(
                weakref.ref(value)
                for value in (
                    self.model,
                    self.processor,
                    self.train_dataset,
                    self.eval_dataset,
                    self.data_collator,
                )
            )

    def assert_released() -> None:
        assert all(reference() is None for reference in references)

    run_offline_preflight(
        fake_config(use_wandb=False),
        Pipeline,
        tmp_path / "experiment_cfg",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        gc_collect=assert_released,
    )


def test_main_validates_offline_environment_before_runtime_dependencies() -> None:
    class Dependencies:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"runtime dependency accessed too early: {name}")

    with pytest.raises(RuntimeError, match="offline environment"):
        main(environ={}, dependencies=Dependencies())


def _fake_runtime_dependencies(
    tmp_path: Path,
    *,
    use_wandb: bool,
    events: list[object],
) -> SimpleNamespace:
    config = fake_config(use_wandb=use_wandb)
    config.data.datasets = None
    ft_config = fake_finetune_config(tmp_path, use_wandb=use_wandb)
    snapshot = expected_cosmos_snapshot(tmp_path)
    snapshot.mkdir(parents=True)

    class DefaultConfig:
        def load_dict(self, payload: dict[str, object]) -> SimpleNamespace:
            config.data.download_cache = payload["data"]["download_cache"]
            config.data.datasets = payload["data"]["datasets"]
            return config

    class EmbodimentTag:
        @staticmethod
        def resolve(value: str) -> SimpleNamespace:
            events.append(("resolve-embodiment", value))
            return SimpleNamespace(value=value)

    class Model:
        pass

    class Processor:
        pass

    class TrainDataset:
        def close(self) -> None:
            events.append("close-train")

    class Collator:
        pass

    class Pipeline:
        def __init__(self, received_config: object, save_cfg_dir: Path) -> None:
            events.append(("pipeline-init", received_config, save_cfg_dir))

        def setup(self) -> None:
            events.append("pipeline-setup")
            self.model = Model()
            self.processor = Processor()
            self.train_dataset = TrainDataset()
            self.eval_dataset = None
            self.data_collator = Collator()

    qwen3_backbone = type("Qwen3Backbone", (), {})
    experiment_module = SimpleNamespace(TrainingArguments=lambda **_kwargs: None)

    def snapshot_download(**kwargs: object) -> str:
        events.append(("snapshot-download", kwargs))
        return str(snapshot)

    return SimpleNamespace(
        cache_root=tmp_path,
        tyro=SimpleNamespace(cli=lambda *_args, **_kwargs: ft_config),
        FinetuneConfig=object,
        get_default_config=lambda: DefaultConfig(),
        load_modality_config=lambda path: events.append(("load-modality", path)),
        EmbodimentTag=EmbodimentTag,
        get_backbone_cls=lambda _model: qwen3_backbone,
        get_last_checkpoint=lambda path: events.append(("checkpoint-probe", path)),
        snapshot_download=snapshot_download,
        Gr00tN1d7Pipeline=Pipeline,
        experiment_module=experiment_module,
        run=lambda received_config: events.append(("run", received_config)),
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache"))),
        gc_collect=lambda: events.append("gc-collect"),
    )


def test_main_preflight_runs_setup_and_never_calls_training(tmp_path: Path, capsys) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(
        tmp_path,
        use_wandb=False,
        events=events,
    )
    freshness_audit = (tmp_path / "freshness.json").resolve()
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "1",
        "GR00T_FRESHNESS_AUDIT_PATH": str(freshness_audit),
    }

    main(environ=environment, dependencies=dependencies)

    assert not any(isinstance(event, tuple) and event[0] == "run" for event in events)
    assert (tmp_path / "outputs" / "experiment" / "experiment_cfg").is_dir()
    assert json.loads(freshness_audit.read_text(encoding="utf-8"))["get_last_checkpoint"] is None
    output = capsys.readouterr().out
    assert COSMOS_MODEL_ID in output
    assert COSMOS_REVISION in output
    assert "Qwen3Backbone" in output
    assert "Model" in output
    assert "Processor" in output
    assert "TrainDataset" in output


def test_main_production_installs_audit_then_calls_training(tmp_path: Path) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(
        tmp_path,
        use_wandb=True,
        events=events,
    )
    training_arguments = SimpleNamespace(
        deepspeed=None,
        to_dict=lambda: _training_arguments_payload(None),
    )
    dependencies.experiment_module.TrainingArguments = lambda **_kwargs: training_arguments

    def run(config: object) -> None:
        events.append(("run", config))
        returned = dependencies.experiment_module.TrainingArguments(deepspeed=None)
        assert returned is training_arguments

    dependencies.run = run
    training_audit = (tmp_path / "training-arguments.json").resolve()
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "0",
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str(training_audit),
    }

    main(environ=environment, dependencies=dependencies)

    assert any(isinstance(event, tuple) and event[0] == "run" for event in events)
    assert json.loads(training_audit.read_text(encoding="utf-8"))["deepspeed"] is None
    assert not any(isinstance(event, tuple) and event[0] == "pipeline-init" for event in events)


@pytest.mark.parametrize(
    ("preflight", "use_wandb", "expected"),
    [("1", True, "disabled"), ("0", False, "enabled")],
)
def test_main_enforces_branch_specific_wandb_before_freshness_mutation(
    tmp_path: Path,
    preflight: str,
    use_wandb: bool,
    expected: str,
) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(
        tmp_path,
        use_wandb=use_wandb,
        events=events,
    )
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": preflight,
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str((tmp_path / "training-arguments.json").resolve()),
    }

    with pytest.raises(RuntimeError, match=expected):
        main(environ=environment, dependencies=dependencies)

    assert not (tmp_path / "outputs" / "experiment").exists()
