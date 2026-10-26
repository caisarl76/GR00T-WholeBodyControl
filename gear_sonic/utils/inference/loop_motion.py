"""Ready-pose asset for the autonomous handover loop."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

READY_POSE_PATH = Path(__file__).resolve().parents[2] / "data/handover/ready_pose_261020_ep76.npz"
TOKEN_ABS_MAX = 1.25
_SHAPES = dict(
    token=(64,),
    hands=(14,),
    state=(28,),
    raise_tokens=(83, 64),
    raise_hands=(83, 14),
    lower_tokens=(83, 64),
    lower_hands=(83, 14),
)


@dataclass(frozen=True)
class ReadyPose:
    token: np.ndarray
    hands: np.ndarray
    state: np.ndarray
    raise_tokens: np.ndarray
    raise_hands: np.ndarray
    lower_tokens: np.ndarray
    lower_hands: np.ndarray


def load_ready_pose(path: Path = READY_POSE_PATH) -> ReadyPose:
    """Load and validate the ready pose; extra npz keys (meta, ...) are ignored."""
    with np.load(path, allow_pickle=False) as npz:
        missing = [k for k in _SHAPES if k not in npz.files]
        if missing:
            raise ValueError(f"ready pose {path} missing keys: {missing}")
        arrays = {k: np.asarray(npz[k], dtype=np.float32) for k in _SHAPES}
    for name, shape in _SHAPES.items():
        if arrays[name].shape != shape:
            raise ValueError(f"ready pose {name} shape {arrays[name].shape} != {shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"ready pose {name} is not finite")
    if np.abs(arrays["token"]).max() > TOKEN_ABS_MAX:
        raise ValueError(f"ready pose |token| exceeds {TOKEN_ABS_MAX}")
    return ReadyPose(**arrays)
