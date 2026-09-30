"""Score recorded episodes, re-querying GR00T at every language boundary.

Runs in the Isaac-GR00T Docker environment. This measures action reconstruction
on recorded observations; it does not execute actions or measure task success.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time


def query_intervals(prompts, stride):
    """Yield chunks capped at the next prompt change and the episode end."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    start = 0
    while start < len(prompts):
        end = start + 1
        while end < min(start + stride, len(prompts)) and prompts[end] == prompts[start]:
            end += 1
        yield start, end
        start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_observation_gr00t
    from gr00t.policy.server_client import PolicyClient
    import numpy as np

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy = PolicyClient(host=args.host, port=args.port, timeout_ms=60000)
    if not policy.ping():
        raise RuntimeError("PolicyServer did not answer ping")
    modality = policy.get_modality_config()
    action_keys = modality["action"].modality_keys
    if args.stride <= 0 or args.stride > len(modality["action"].delta_indices):
        raise ValueError("stride must be between 1 and the policy action horizon")
    observation_modality = {k: v for k, v in modality.items() if k != "action"}
    loader = LeRobotEpisodeLoader(args.dataset_path, modality, video_backend="torchcodec")
    embodiment = EmbodimentTag.UNITREE_G1_SONIC
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    episodes = []
    latencies = []

    def score(name, error):
        error = np.asarray(error, dtype=np.float64)
        totals[name][0] += float(np.square(error).sum())
        totals[name][1] += float(np.abs(error).sum())
        totals[name][2] += error.size
        return {"mse": float(np.square(error).mean()), "mae": float(np.abs(error).mean())}

    for episode_id in range(len(loader)):
        trajectory = loader[episode_id]
        language_key = modality["language"].modality_keys[0]
        prompts = trajectory[f"language.{language_key}"].tolist()
        targets = {key: np.stack(trajectory[f"action.{key}"]) for key in action_keys}
        predictions = {key: np.empty_like(value) for key, value in targets.items()}
        queries = []
        for start, end in query_intervals(prompts, args.stride):
            sample = extract_step_data(trajectory, start, observation_modality, embodiment)
            if sample.text != prompts[start]:
                raise ValueError(f"Language mismatch at episode {episode_id}, frame {start}")
            observation = {f"state.{key}": value for key, value in sample.states.items()}
            observation.update({f"video.{key}": np.asarray(value) for key, value in sample.images.items()})
            observation.update({key: sample.text for key in modality["language"].modality_keys})
            before = time.monotonic()
            action, _ = policy.get_action(parse_observation_gr00t(observation, modality))
            latencies.append(time.monotonic() - before)
            for key in action_keys:
                chunk = np.asarray(action[key])[0]
                if chunk.shape[0] < end - start or chunk.shape[1:] != targets[key].shape[1:]:
                    raise ValueError(f"Unexpected {key} action shape: {chunk.shape}")
                if not np.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite {key} action at episode {episode_id}, frame {start}")
                predictions[key][start:end] = chunk[: end - start]
            queries.append({"start": start, "end": end, "prompt": sample.text})

        target = np.concatenate([targets[key] for key in action_keys], axis=-1)
        prediction = np.concatenate([predictions[key] for key in action_keys], axis=-1)
        error = prediction - target
        if not np.isfinite(error).all():
            raise ValueError(f"Nonfinite error in episode {episode_id}")
        metrics = {"all": score("all", error)}
        for key in action_keys:
            metrics[key] = score(key, predictions[key] - targets[key])
        for prompt in dict.fromkeys(prompts):
            metrics[prompt] = score(prompt, error[np.asarray(prompts) == prompt])
        row = {"episode": episode_id, "frames": len(trajectory), "queries": queries, "metrics": metrics}
        episodes.append(row)
        np.savez_compressed(
            output / f"episode_{episode_id:06d}.npz",
            target=target,
            prediction=prediction,
            prompts=np.asarray(prompts),
            query_starts=np.asarray([q["start"] for q in queries]),
            action_keys=np.asarray(action_keys),
            action_dimensions=np.asarray([targets[key].shape[-1] for key in action_keys]),
        )
        (output / f"episode_{episode_id:06d}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(
            json.dumps(
                {"episode": episode_id, "frames": len(trajectory), "queries": len(queries), "metrics": metrics}
            ),
            flush=True,
        )

    summary = {
        "dataset": args.dataset_path,
        "policy_endpoint": f"{args.host}:{args.port}",
        "stride": args.stride,
        "boundary_behavior": "truncate previous chunk; query at the first frame with the new prompt",
        "episodes": len(episodes),
        "frames": sum(row["frames"] for row in episodes),
        "queries": len(latencies),
        "query_seconds": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
        "metrics": {
            name: {"mse": sq / count, "mae": ab / count, "elements": count}
            for name, (sq, ab, count) in totals.items()
        },
        "interpretation": (
            "Unnormalized action errors on training recordings, weighted by frame and dimension. "
            "Not held-out evaluation or robot task success."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
