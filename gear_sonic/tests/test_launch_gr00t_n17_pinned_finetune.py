import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from gear_sonic.scripts import launch_gr00t_n17_pinned_finetune as launcher
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


def expected_cosmos_snapshot(cache_root: Path) -> Path:
    return cache_root / "hub" / "models--nvidia--Cosmos-Reason2-2B" / "snapshots" / COSMOS_REVISION


def write_cosmos_snapshot_config(snapshot: Path, **overrides: object) -> Path:
    snapshot.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "transformers_version": "4.57.0.dev0",
        **overrides,
    }
    (snapshot / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return snapshot


def write_cosmos_snapshot_config_blob_link(snapshot: Path) -> Path:
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "pinned-config-blob"
    blob.write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "transformers_version": "4.57.0.dev0",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "config.json").symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def offline_hub(events: list[object] | None = None) -> tuple[SimpleNamespace, object]:
    def network_model_info(*args: object, **kwargs: object) -> object:
        if events is not None:
            events.append(("network-model-info", args, kwargs))
        raise RuntimeError("HF_HUB_OFFLINE: model_info network access is forbidden")

    hub = SimpleNamespace(model_info=network_model_info)
    return hub, network_model_info


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


class FakeModelConfig:
    def __init__(
        self,
        *,
        model_name: str = COSMOS_MODEL_ID,
        model_revision: str | None = None,
        serialization_error: BaseException | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.serialization_error = serialization_error

    def to_filtered_json(self) -> str:
        if self.serialization_error is not None:
            raise self.serialization_error
        return json.dumps(
            {
                "model_name": self.model_name,
                "model_revision": self.model_revision,
            }
        )


def pinned_preflight_config(tmp_path: Path, *, use_wandb: bool = False) -> SimpleNamespace:
    config = fake_config(use_wandb=use_wandb)
    apply_runtime_pins(config, cache_root=tmp_path)
    return config


def initialize_fake_pipeline(
    pipeline: object,
    config: object,
    save_cfg_dir: Path,
) -> None:
    pipeline.config = config
    pipeline.save_cfg_dir = save_cfg_dir
    pipeline.transformers_loading_kwargs = {
        "revision": config.model.model_revision,
        "local_files_only": config.training.transformers_local_files_only,
        "cache_dir": config.training.transformers_cache_dir,
    }


def write_pre_repair_model_artifact(save_cfg_dir: Path, model_config: FakeModelConfig) -> None:
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    (save_cfg_dir / "final_model_config.json").write_text(
        model_config.to_filtered_json(),
        encoding="utf-8",
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


def test_assert_recipe_contract_rejects_skip_weight_loading() -> None:
    config = fake_config()
    config.training.skip_weight_loading = True

    with pytest.raises(RuntimeError, match="training.skip_weight_loading"):
        assert_recipe_contract(config)


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
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, original_model_info = offline_hub(events)

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

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
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            events.append("setup")
            model_info = hub.model_info(COSMOS_MODEL_ID)
            assert model_info.tags == ["qwen3_vl"]
            self.model = self._create_model()
            self.processor = Processor()
            self.train_dataset = TrainDataset()
            self.eval_dataset = EvalDataset()
            self.data_collator = Collator()

    torch_module = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache")))
    config = pinned_preflight_config(tmp_path)
    save_cfg_dir = tmp_path / "experiment" / "experiment_cfg"

    record = run_offline_preflight(
        config,
        Pipeline,
        save_cfg_dir,
        torch_module=torch_module,
        cosmos_snapshot=snapshot,
        huggingface_hub_module=hub,
        pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
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
    assert hub.model_info is original_model_info
    assert events == [
        ("init", config, save_cfg_dir),
        "setup",
        "train-close",
        "eval-close",
        "gc-collect",
        "empty-cache",
    ]


def test_run_offline_preflight_persists_exact_identity_on_created_model(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    created_models = []

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()
            self.processor = None
            self.train_dataset = None
            self.eval_dataset = None
            self.data_collator = None

    original_create_model = vars(Pipeline)["_create_model"]

    config = pinned_preflight_config(tmp_path)
    run_offline_preflight(
        config,
        Pipeline,
        tmp_path / "experiment_cfg",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        cosmos_snapshot=snapshot,
        huggingface_hub_module=hub,
        pipeline_create_model_descriptor=original_create_model,
    )

    assert created_models[0].config.model_name == COSMOS_MODEL_ID
    assert created_models[0].config.model_revision == COSMOS_REVISION
    persisted = json.loads((tmp_path / "experiment_cfg" / "final_model_config.json").read_text(encoding="utf-8"))
    assert persisted == {
        "model_name": COSMOS_MODEL_ID,
        "model_revision": COSMOS_REVISION,
    }
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_model_identity_hook_restores_after_setup_failure(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    created_models = []

    class SetupFailure(RuntimeError):
        pass

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()
            raise SetupFailure("setup failed after model creation")

    original_create_model = vars(Pipeline)["_create_model"]

    with pytest.raises(SetupFailure, match="setup failed after model creation"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert created_models[0].config.model_name == COSMOS_MODEL_ID
    assert created_models[0].config.model_revision == COSMOS_REVISION
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_rejects_model_config_that_does_not_persist_revision(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class Config(FakeModelConfig):
        def __init__(self) -> None:
            object.__setattr__(self, "model_name", COSMOS_MODEL_ID)
            object.__setattr__(self, "model_revision", None)
            object.__setattr__(self, "serialization_error", None)

        def __setattr__(self, name: str, value: object) -> None:
            if name != "model_revision":
                object.__setattr__(self, name, value)

    class Model:
        config = Config()

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]

    with pytest.raises(RuntimeError, match="model_revision"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_accepts_already_pinned_returned_identity(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    created_models = []

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=COSMOS_REVISION)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    run_offline_preflight(
        pinned_preflight_config(tmp_path),
        Pipeline,
        tmp_path / "experiment_cfg",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        cosmos_snapshot=snapshot,
        huggingface_hub_module=hub,
        pipeline_create_model_descriptor=original_create_model,
    )

    assert created_models[0].config.model_revision == COSMOS_REVISION
    assert vars(Pipeline)["_create_model"] is original_create_model


@pytest.mark.parametrize("missing_from", ["returned-config", "persisted-config"])
def test_run_offline_preflight_distinguishes_missing_revision_from_explicit_none(
    tmp_path: Path,
    missing_from: str,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class MissingRevisionConfig:
        model_name = COSMOS_MODEL_ID

        def to_filtered_json(self) -> str:
            return json.dumps(vars(self))

    class Model:
        def __init__(self) -> None:
            self.config = (
                MissingRevisionConfig()
                if missing_from == "returned-config"
                else FakeModelConfig(model_revision=None)
            )

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            payload = {"model_name": COSMOS_MODEL_ID}
            if missing_from != "persisted-config":
                payload["model_revision"] = None
            self.save_cfg_dir.joinpath("final_model_config.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match="model_revision.*missing"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


@pytest.mark.parametrize(
    ("model_name", "model_revision", "match"),
    [
        ("other/model", None, "model_name"),
        (COSMOS_MODEL_ID, "wrong-revision", "model_revision"),
    ],
)
def test_run_offline_preflight_rejects_wrong_returned_model_identity(
    tmp_path: Path,
    model_name: str,
    model_revision: str | None,
    match: str,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    created_models = []

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(
                model_name=model_name,
                model_revision=model_revision,
            )

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match=match):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert created_models[0].config.model_name == model_name
    assert created_models[0].config.model_revision == model_revision
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_restores_hook_after_model_config_serialization_failure(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            model.config.serialization_error = RuntimeError("serialization failed")
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match="serialization"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_restores_hook_after_artifact_rewrite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    monkeypatch.setattr(launcher.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))

    with pytest.raises(RuntimeError, match="rewrite"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


@pytest.mark.parametrize(
    ("failure_kind", "match"),
    [
        ("outer-name", "outer.*model_name"),
        ("outer-revision", "outer.*model_revision"),
        ("loading-revision", "transformers revision"),
        ("local-files", "local_files_only"),
        ("cache", "cache_dir"),
    ],
)
def test_run_offline_preflight_validates_outer_identity_before_original_model_call(
    tmp_path: Path,
    failure_kind: str,
    match: str,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    original_called = False
    config = pinned_preflight_config(tmp_path)
    if failure_kind == "outer-name":
        config.model.model_name = "other/model"
    elif failure_kind == "outer-revision":
        config.model.model_revision = "wrong-revision"

    class Pipeline:
        def __init__(self, received_config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, received_config, save_cfg_dir)
            if failure_kind == "loading-revision":
                self.transformers_loading_kwargs["revision"] = "wrong-revision"
            elif failure_kind == "local-files":
                self.transformers_loading_kwargs["local_files_only"] = False
            elif failure_kind == "cache":
                self.transformers_loading_kwargs["cache_dir"] = "/wrong/cache"

        def _create_model(self) -> object:
            nonlocal original_called
            original_called = True
            raise AssertionError("original model creation must not run")

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match=match):
        run_offline_preflight(
            config,
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert original_called is False
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_rejects_wrong_pre_repair_artifact_identity(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    created_models = []

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            self.save_cfg_dir.joinpath("final_model_config.json").write_text(
                json.dumps({"model_name": "other/model", "model_revision": None}),
                encoding="utf-8",
            )
            return model

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match="final_model_config.*model_name"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert created_models[0].config.model_revision is None
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_preserves_original_model_creation_failure(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class OriginalLoadFailure(RuntimeError):
        pass

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> object:
            raise OriginalLoadFailure("original load failed")

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(OriginalLoadFailure, match="original load failed"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_preserves_primary_when_restoration_fails_on_python310(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class OriginalLoadFailure(RuntimeError):
        pass

    class RestorationBlockingMeta(type):
        def __setattr__(cls, name: str, value: object) -> None:
            current = vars(cls).get(name)
            if (
                name == "_create_model"
                and getattr(current, "_groot_pinned_identity_hook", False)
                and not getattr(value, "_groot_pinned_identity_hook", False)
            ):
                raise RuntimeError("restoration blocked")
            super().__setattr__(name, value)

    class Pipeline(metaclass=RestorationBlockingMeta):
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> object:
            raise OriginalLoadFailure("original load failed")

        def setup(self) -> None:
            self.model = self._create_model()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(OriginalLoadFailure, match="^original load failed$") as exc_info:
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert str(exc_info.value) == "original load failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "restoration blocked" in str(exc_info.value.__cause__)


def test_run_offline_preflight_rejects_unexpected_create_model_descriptor(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class Pipeline:
        def __init__(self, _config: object, _save_cfg_dir: Path) -> None:
            pass

        def _create_model(self) -> object:
            raise AssertionError("model creation must not run")

        def setup(self) -> None:
            raise AssertionError("setup must not run")

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match="descriptor is unexpected"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=lambda _self: None,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_requires_exactly_one_model_creation_call(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> object:
            raise AssertionError("unused official model creation method")

        def setup(self) -> None:
            self.model = object()

    original_create_model = vars(Pipeline)["_create_model"]
    with pytest.raises(RuntimeError, match="must run exactly once"):
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=original_create_model,
        )

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_run_offline_preflight_requires_wandb_disabled(tmp_path: Path) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()
    with pytest.raises(RuntimeError, match="W&B"):
        run_offline_preflight(
            fake_config(use_wandb=True),
            lambda *_args: pytest.fail("pipeline must not be constructed"),
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=None,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_type": "mistral"}, "model_type"),
        ({"architectures": ["MistralForCausalLM"]}, "architectures"),
        ({"transformers_version": "4.57.1"}, "transformers_version"),
    ],
)
def test_run_offline_preflight_rejects_inexact_cosmos_snapshot_before_setup(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path), **overrides)
    hub, original_model_info = offline_hub()

    with pytest.raises(RuntimeError, match=match):
        run_offline_preflight(
            fake_config(use_wandb=False),
            lambda *_args: pytest.fail("pipeline must not be constructed"),
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=None,
        )

    assert hub.model_info is original_model_info


def test_run_offline_preflight_accepts_config_symlink_to_canonical_blob(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config_blob_link(expected_cosmos_snapshot(tmp_path))
    hub, original_model_info = offline_hub()

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> object:
            model = SimpleNamespace(config=FakeModelConfig(model_revision=None))
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            assert hub.model_info(COSMOS_MODEL_ID).tags == ["qwen3_vl"]
            self.model = self._create_model()

    record = run_offline_preflight(
        pinned_preflight_config(tmp_path),
        Pipeline,
        tmp_path / "experiment_cfg",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        cosmos_snapshot=snapshot,
        huggingface_hub_module=hub,
        pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
    )

    assert record == {
        "model_class": "SimpleNamespace",
        "processor_class": None,
        "train_dataset_class": None,
        "eval_dataset_class": None,
        "data_collator_class": None,
    }
    assert hub.model_info is original_model_info


@pytest.mark.parametrize("link_kind", ["dangling", "escaping"])
def test_run_offline_preflight_rejects_invalid_config_symlink_before_setup(
    tmp_path: Path,
    link_kind: str,
) -> None:
    snapshot = expected_cosmos_snapshot(tmp_path)
    snapshot.mkdir(parents=True)
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    config_link = snapshot / "config.json"
    if link_kind == "dangling":
        config_link.symlink_to(Path("../../blobs/missing-config-blob"))
    else:
        outside = tmp_path / "outside-config.json"
        outside.write_text(
            json.dumps(
                {
                    "model_type": "qwen3_vl",
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                    "transformers_version": "4.57.0.dev0",
                }
            ),
            encoding="utf-8",
        )
        config_link.symlink_to(outside)
    hub, original_model_info = offline_hub()

    with pytest.raises(RuntimeError, match="config.json"):
        run_offline_preflight(
            fake_config(use_wandb=False),
            lambda *_args: pytest.fail("pipeline must not be constructed"),
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=None,
        )

    assert hub.model_info is original_model_info


def test_run_offline_preflight_rejects_unexpected_model_info_repo_and_restores(
    tmp_path: Path,
) -> None:
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, original_model_info = offline_hub()

    class Pipeline:
        def __init__(self, _config: object, _save_cfg_dir: Path) -> None:
            pass

        def _create_model(self) -> object:
            return SimpleNamespace(config=SimpleNamespace(model_name=COSMOS_MODEL_ID, model_revision=None))

        def setup(self) -> None:
            hub.model_info("other/model")

    with pytest.raises(RuntimeError, match="only supports.*Cosmos"):
        run_offline_preflight(
            fake_config(use_wandb=False),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
        )

    assert hub.model_info is original_model_info


def test_run_offline_preflight_releases_loaded_objects_before_collection(
    tmp_path: Path,
) -> None:
    references: list[weakref.ReferenceType[object]] = []
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class LoadedObject:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> LoadedObject:
            model = LoadedObject()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()
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
        pinned_preflight_config(tmp_path),
        Pipeline,
        tmp_path / "experiment_cfg",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        cosmos_snapshot=snapshot,
        huggingface_hub_module=hub,
        pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
        gc_collect=assert_released,
    )


def test_run_offline_preflight_preserves_setup_error_and_completes_cleanup(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    instances = []
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, original_model_info = offline_hub()

    class SetupFailure(RuntimeError):
        pass

    class TrainCloseFailure(RuntimeError):
        pass

    class EvalCloseFailure(RuntimeError):
        pass

    class Dataset:
        def __init__(self, name: str, error: BaseException) -> None:
            self.name = name
            self.error = error

        def close(self) -> None:
            events.append(f"{self.name}-close")
            raise self.error

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            instances.append(self)
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()
            self.processor = object()
            self.train_dataset = Dataset("train", TrainCloseFailure("train close failed"))
            self.eval_dataset = Dataset("eval", EvalCloseFailure("eval close failed"))
            self.data_collator = object()
            events.append("setup")
            raise SetupFailure("setup failed")

    with pytest.raises(SetupFailure, match="setup failed") as exc_info:
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache"))),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
            gc_collect=lambda: events.append("gc-collect"),
        )

    assert hub.model_info is original_model_info
    assert events == ["setup", "train-close", "eval-close", "gc-collect", "empty-cache"]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "train close failed" in str(exc_info.value.__cause__)
    assert "eval close failed" in str(exc_info.value.__cause__)
    pipeline = instances[0]
    assert pipeline.model is None
    assert pipeline.processor is None
    assert pipeline.train_dataset is None
    assert pipeline.eval_dataset is None
    assert pipeline.data_collator is None


def test_run_offline_preflight_raises_first_cleanup_error_after_all_cleanup(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    instances = []
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, _original_model_info = offline_hub()

    class TrainCloseFailure(RuntimeError):
        pass

    class EvalCloseFailure(RuntimeError):
        pass

    class Dataset:
        def __init__(self, name: str, error: BaseException) -> None:
            self.name = name
            self.error = error

        def close(self) -> None:
            events.append(f"{self.name}-close")
            raise self.error

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            instances.append(self)
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            self.model = self._create_model()
            self.processor = object()
            self.train_dataset = Dataset("train", TrainCloseFailure("train close failed"))
            self.eval_dataset = Dataset("eval", EvalCloseFailure("eval close failed"))
            self.data_collator = object()
            events.append("setup")

    with pytest.raises(TrainCloseFailure, match="train close failed") as exc_info:
        run_offline_preflight(
            pinned_preflight_config(tmp_path),
            Pipeline,
            tmp_path / "experiment_cfg",
            torch_module=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache"))),
            cosmos_snapshot=snapshot,
            huggingface_hub_module=hub,
            pipeline_create_model_descriptor=vars(Pipeline)["_create_model"],
            gc_collect=lambda: events.append("gc-collect"),
        )

    assert events == ["setup", "train-close", "eval-close", "gc-collect", "empty-cache"]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "eval close failed" in str(exc_info.value.__cause__)
    pipeline = instances[0]
    assert pipeline.model is None
    assert pipeline.processor is None
    assert pipeline.train_dataset is None
    assert pipeline.eval_dataset is None
    assert pipeline.data_collator is None


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
    snapshot = write_cosmos_snapshot_config(expected_cosmos_snapshot(tmp_path))
    hub, original_model_info = offline_hub(events)

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
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

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
            initialize_fake_pipeline(self, received_config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

        def setup(self) -> None:
            events.append("pipeline-setup")
            model_info = hub.model_info(COSMOS_MODEL_ID)
            events.append(("pipeline-model-info-tags", model_info.tags))
            self.model = self._create_model()
            self.processor = Processor()
            self.train_dataset = TrainDataset()
            self.eval_dataset = None
            self.data_collator = Collator()

    qwen3_backbone = type("Qwen3Backbone", (), {})
    experiment_module = SimpleNamespace(TrainingArguments=lambda **_kwargs: None)

    def snapshot_download(**kwargs: object) -> str:
        events.append(("snapshot-download", kwargs))
        return str(snapshot)

    dependencies = SimpleNamespace(
        cache_root=tmp_path,
        tyro=SimpleNamespace(cli=lambda *_args, **_kwargs: ft_config),
        FinetuneConfig=object,
        get_default_config=lambda: DefaultConfig(),
        load_modality_config=lambda path: events.append(("load-modality", path)),
        EmbodimentTag=EmbodimentTag,
        get_backbone_cls=lambda _model: qwen3_backbone,
        get_last_checkpoint=lambda path: events.append(("checkpoint-probe", path)),
        snapshot_download=snapshot_download,
        huggingface_hub=hub,
        original_model_info=original_model_info,
        Gr00tN1d7Pipeline=Pipeline,
        Gr00tN1d7Pipeline_create_model=vars(Pipeline)["_create_model"],
        experiment_module=experiment_module,
        run=lambda received_config: (
            events.append(("training-model-info-tags", hub.model_info(COSMOS_MODEL_ID).tags)),
            events.append(("run", received_config)),
        ),
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty-cache"))),
        gc_collect=lambda: events.append("gc-collect"),
    )
    return dependencies


@pytest.mark.parametrize("injected_value", [None, "1"])
def test_main_sets_albumentations_no_update_before_dependency_loader_and_restores_injected_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_value: str | None,
) -> None:
    variable = "NO_ALBUMENTATIONS_UPDATE"
    monkeypatch.delenv(variable, raising=False)
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(tmp_path, use_wandb=False, events=events)
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "1",
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
    }
    if injected_value is not None:
        environment[variable] = injected_value
    observed = []

    def dependency_loader() -> SimpleNamespace:
        observed.append(os.environ.get(variable))
        return dependencies

    main(environ=environment, dependency_loader=dependency_loader)

    assert observed == ["1"]
    assert os.environ.get(variable) is None
    if injected_value is None:
        assert variable not in environment
    else:
        assert environment[variable] == injected_value


@pytest.mark.parametrize("conflict_source", ["injected", "process"])
def test_main_rejects_conflicting_albumentations_update_value_before_dependency_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflict_source: str,
) -> None:
    variable = "NO_ALBUMENTATIONS_UPDATE"
    monkeypatch.delenv(variable, raising=False)
    environment = {**REQUIRED_OFFLINE_ENV}
    if conflict_source == "injected":
        environment[variable] = "0"
    else:
        monkeypatch.setenv(variable, "0")
    loader_called = False

    def dependency_loader() -> SimpleNamespace:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("dependency loader must not run")

    with pytest.raises(RuntimeError, match="NO_ALBUMENTATIONS_UPDATE"):
        main(environ=environment, dependency_loader=dependency_loader)

    assert loader_called is False


def test_main_rejects_skip_weight_loading_before_snapshot_freshness_or_run(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(tmp_path, use_wandb=True, events=events)
    dependencies.tyro.cli().skip_weight_loading = True
    freshness_audit = (tmp_path / "freshness.json").resolve()
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "0",
        "GR00T_FRESHNESS_AUDIT_PATH": str(freshness_audit),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str((tmp_path / "training-arguments.json").resolve()),
    }

    with pytest.raises(RuntimeError, match="training.skip_weight_loading"):
        main(environ=environment, dependencies=dependencies)

    assert not any(
        isinstance(event, tuple) and event[0] in {"snapshot-download", "checkpoint-probe", "run"}
        for event in events
    )
    assert not freshness_audit.exists()
    assert not (tmp_path / "outputs" / "experiment").exists()


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
    assert ("pipeline-model-info-tags", ["qwen3_vl"]) in events
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


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
        model_info = dependencies.huggingface_hub.model_info(COSMOS_MODEL_ID)
        events.append(("training-model-info-tags", model_info.tags))
        events.append(("run", config))
        dependencies.Gr00tN1d7Pipeline(config, tmp_path / "training-config")._create_model()
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
    assert sum(isinstance(event, tuple) and event[0] == "pipeline-init" for event in events) == 1
    assert ("training-model-info-tags", ["qwen3_vl"]) in events
    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info


def test_main_production_persists_model_identity_and_restores_creation_hook(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(tmp_path, use_wandb=True, events=events)
    training_arguments = SimpleNamespace(
        deepspeed=None,
        to_dict=lambda: _training_arguments_payload(None),
    )
    dependencies.experiment_module.TrainingArguments = lambda **_kwargs: training_arguments
    created_models = []

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            created_models.append(model)
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

    original_create_model = vars(Pipeline)["_create_model"]
    dependencies.Gr00tN1d7Pipeline = Pipeline
    dependencies.Gr00tN1d7Pipeline_create_model = original_create_model

    def run(config: object) -> None:
        dependencies.huggingface_hub.model_info(COSMOS_MODEL_ID)
        pipeline = Pipeline(config, tmp_path / "training-config")
        pipeline._create_model()

    dependencies.run = run
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "0",
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str((tmp_path / "training-arguments.json").resolve()),
    }

    main(environ=environment, dependencies=dependencies)

    assert created_models[0].config.model_name == COSMOS_MODEL_ID
    assert created_models[0].config.model_revision == COSMOS_REVISION
    assert (
        json.loads((tmp_path / "training-config" / "final_model_config.json").read_text(encoding="utf-8"))[
            "model_revision"
        ]
        == COSMOS_REVISION
    )
    assert vars(Pipeline)["_create_model"] is original_create_model


def test_main_production_restores_creation_hook_after_artifact_rewrite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(tmp_path, use_wandb=True, events=events)
    training_arguments = SimpleNamespace(
        deepspeed=None,
        to_dict=lambda: _training_arguments_payload(None),
    )
    dependencies.experiment_module.TrainingArguments = lambda **_kwargs: training_arguments

    class Model:
        def __init__(self) -> None:
            self.config = FakeModelConfig(model_revision=None)

    class Pipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            initialize_fake_pipeline(self, config, save_cfg_dir)

        def _create_model(self) -> Model:
            model = Model()
            write_pre_repair_model_artifact(self.save_cfg_dir, model.config)
            return model

    original_create_model = vars(Pipeline)["_create_model"]
    dependencies.Gr00tN1d7Pipeline = Pipeline
    dependencies.Gr00tN1d7Pipeline_create_model = original_create_model

    def run(config: object) -> None:
        Pipeline(config, tmp_path / "training-config")._create_model()

    dependencies.run = run
    monkeypatch.setattr(
        launcher.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "0",
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str((tmp_path / "training-arguments.json").resolve()),
    }

    with pytest.raises(RuntimeError, match="rewrite"):
        main(environ=environment, dependencies=dependencies)

    assert vars(Pipeline)["_create_model"] is original_create_model


def test_main_production_restores_model_info_when_training_fails(tmp_path: Path) -> None:
    events: list[object] = []
    dependencies = _fake_runtime_dependencies(tmp_path, use_wandb=True, events=events)
    training_arguments = SimpleNamespace(
        deepspeed=None,
        to_dict=lambda: _training_arguments_payload(None),
    )
    dependencies.experiment_module.TrainingArguments = lambda **_kwargs: training_arguments
    original_create_model = dependencies.Gr00tN1d7Pipeline_create_model

    def fail_training(_config: object) -> None:
        dependencies.huggingface_hub.model_info(COSMOS_MODEL_ID)
        raise RuntimeError("training failed")

    dependencies.run = fail_training
    environment = {
        **REQUIRED_OFFLINE_ENV,
        "GR00T_PINNED_PREFLIGHT_ONLY": "0",
        "GR00T_FRESHNESS_AUDIT_PATH": str((tmp_path / "freshness.json").resolve()),
        "GR00T_TRAINING_ARGS_AUDIT_PATH": str((tmp_path / "training-arguments.json").resolve()),
    }

    with pytest.raises(RuntimeError, match="training failed"):
        main(environ=environment, dependencies=dependencies)

    assert dependencies.huggingface_hub.model_info is dependencies.original_model_info
    assert vars(dependencies.Gr00tN1d7Pipeline)["_create_model"] is original_create_model


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
