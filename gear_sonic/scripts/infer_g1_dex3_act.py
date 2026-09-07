"""Run one language-free ACT action chunk for a four-camera G1 observation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)


def load_observation(source: Path | io.BufferedIOBase | io.BytesIO) -> dict[str, np.ndarray]:
    with np.load(source, allow_pickle=False) as archive:
        required = ("observation.state",) + IMAGE_KEYS
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"missing observation keys: {', '.join(missing)}")
        state = np.asarray(archive["observation.state"])
        if state.shape != (28,) or state.dtype != np.float32 or not np.all(np.isfinite(state)):
            raise ValueError("observation.state must be finite real float32[28]")
        result = {"observation.state": state.astype(np.float32, copy=False)}
        for key in IMAGE_KEYS:
            image = np.asarray(archive[key])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 HWC RGB")
            result[key] = image
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class G1ActModel:
    """Loaded ACT policy and its processors, suitable for repeated inference."""

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
            import torch
        except ImportError as exc:
            raise RuntimeError("ACT inference requires torch and lerobot 0.4.4") from exc
        self.device = device
        self.checkpoint = Path(checkpoint)
        self.model_digest = _digest(self.checkpoint / "model.safetensors")
        self.torch = torch
        config = PreTrainedConfig.from_pretrained(str(self.checkpoint), local_files_only=True)
        config.device = device
        # The full checkpoint already contains the backbone. Avoid downloading
        # initialization weights that would immediately be overwritten.
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(
            str(self.checkpoint), config=config, local_files_only=True, strict=True
        ).to(device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.policy.eval()

    def infer(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(observation["observation.state"])[None].to(self.device)}
        for key in IMAGE_KEYS:
            batch[key] = (
                torch.from_numpy(observation[key].transpose(2, 0, 1))[None].float().div(255).to(self.device)
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(actions)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (1, 100, 28) or not np.all(np.isfinite(actions)):
            raise ValueError("ACT output must be finite shape [100, 28]")
        return actions[0], time.perf_counter() - started


def _group_stats(actions: np.ndarray) -> dict:
    return {
        name: {
            "min": float(actions[:, start:end].min()),
            "max": float(actions[:, start:end].max()),
            "temporal_std": float(actions[:, start:end].std(axis=0).mean()),
        }
        for name, start, end in (
            ("left_arm", 0, 7),
            ("right_arm", 7, 14),
            ("left_hand", 14, 21),
            ("right_hand", 21, 28),
        )
    }


def run_inference(checkpoint: Path, observation_path: Path, output_dir: Path, device: str = "cuda") -> dict:
    if output_dir.exists():
        raise ValueError("output directory must be new")
    observation = load_observation(observation_path)
    output_dir.mkdir(parents=True)
    model = G1ActModel(checkpoint, device)
    actions, inference_seconds = model.infer(observation)
    np.save(output_dir / "actions.npy", actions)
    summary = {
        "chunk_hz": 30,
        "chunk_length": 100,
        "action_shape": [100, 28],
        "device": device,
        "model_digest": model.model_digest,
        "observation_digest": _digest(observation_path),
        "inference_seconds": inference_seconds,
        "language_conditioned": False,
        "groups": _group_stats(actions),
        "strict_checkpoint_load": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_inference(args.checkpoint, args.observation, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
