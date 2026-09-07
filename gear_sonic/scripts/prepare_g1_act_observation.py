"""Prepare the first recorded G1 observation for the language-free ACT demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
DATASET_REVISION = "02cb29161f5adb2981a072aa35b408d82a556d07"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DATASET_FPS = 30.0


def _read_first_row(dataset_dir: Path, requested_frame_index: int = 0) -> tuple[np.ndarray, int, int, float]:
    if isinstance(requested_frame_index, bool) or not isinstance(requested_frame_index, (int, np.integer)):
        raise ValueError("frame_index must be an integer >= 0")
    requested_frame_index = int(requested_frame_index)
    if requested_frame_index < 0:
        raise ValueError("frame_index must be an integer >= 0")
    parquet_path = dataset_dir / "data/chunk-000/file-000.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the dataset parquet file") from exc

    columns = ["observation.state", "timestamp", "frame_index", "episode_index"]
    frame = pd.read_parquet(parquet_path, columns=columns)
    if frame.empty:
        raise ValueError("dataset parquet contains no rows")
    first_row = frame.iloc[0]
    first_episode = int(first_row["episode_index"])
    if int(first_row["frame_index"]) != 0 or not np.isclose(
        float(first_row["timestamp"]), 0.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("first parquet row must be episode frame 0 at timestamp approximately 0")
    candidates = frame[
        (frame["episode_index"] == first_episode) & (frame["frame_index"] == requested_frame_index)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"frame_index {requested_frame_index} must identify exactly one row in first episode "
            f"{first_episode}; found {len(candidates)}"
        )
    row = candidates.iloc[0]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    if state.shape != (28,) or not np.all(np.isfinite(state)):
        raise ValueError("first observation.state must be finite float32[28]")
    frame_index = int(row["frame_index"])
    episode_index = int(row["episode_index"])
    timestamp = float(row["timestamp"])
    expected_timestamp = frame_index / DATASET_FPS
    if not np.isclose(timestamp, expected_timestamp, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"frame_index {frame_index} has timestamp {timestamp}, expected approximately {expected_timestamp}"
        )
    return state, episode_index, frame_index, timestamp


def _read_first_frame(video_path: Path, frame_index: int = 0) -> np.ndarray:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected_bytes = FRAME_WIDTH * FRAME_HEIGHT * 3
    if len(completed.stdout) != expected_bytes:
        raise ValueError(f"{video_path} produced {len(completed.stdout)} bytes, expected {expected_bytes}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3).copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_observation(dataset_dir: Path, output: Path, frame_index: int = 0) -> Path:
    if output.exists():
        raise FileExistsError(f"output must be a new path: {output}")
    state, episode_index, frame_index, timestamp = _read_first_row(dataset_dir, frame_index)
    arrays = {"observation.state": state}
    video_files = {}
    for camera in CAMERAS:
        filename = dataset_dir / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        video_files[camera] = filename
        arrays[f"observation.images.{camera}"] = _read_first_frame(filename, frame_index)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    # Import the sibling contract validator without depending on the caller's cwd.
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    try:
        from infer_g1_dex3_act import load_observation

        load_observation(output)
    finally:
        sys.path.pop(0)

    provenance = {
        "dataset_revision": DATASET_REVISION,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "video_frame_selection": "decoded_frame_index",
        "parquet": "data/chunk-000/file-000.parquet",
        "videos": {camera: str(path.relative_to(dataset_dir)) for camera, path in video_files.items()},
        "npz_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-index", default=0, type=int)
    args = parser.parse_args(argv)
    prepare_observation(args.dataset_dir, args.output, args.frame_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
