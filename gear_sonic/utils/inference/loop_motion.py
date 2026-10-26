"""Ready-pose asset for the autonomous handover loop."""

from collections import deque
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


class _Segment:
    """One queued motion. ``i`` is its own tick index; ``origin`` is captured on its first tick."""

    def __init__(self, kind, label, tokens=None, hands=None, ticks=0, stream=None):
        self.kind, self.label, self.tokens, self.hands, self.ticks, self.stream = (
            kind, label, tokens, hands, ticks, stream)
        self.i, self.origin = 0, None


class PoseCommander:
    """Single source of the 64-D token and 14 Dex3 hand targets sent on every 50 Hz tick."""

    def __init__(self, max_token_step: float):
        self.max_token_step = float(max_token_step)
        self.last_token = np.zeros(64, np.float32)
        self.last_hands = np.zeros(14, np.float32)
        self.recent_tokens = deque(maxlen=25)
        self._queue: list[_Segment] = []
        self.frozen = False

    @property
    def busy(self) -> bool:
        return bool(self._queue)

    def start(self, token, hands) -> None:
        self.last_token = np.asarray(token, np.float32).reshape(64).copy()
        self.last_hands = np.asarray(hands, np.float32).reshape(14).copy()
        self.hold()

    def hold(self) -> None:
        self._queue.clear()
        self.frozen = False

    def _add(self, seg: _Segment) -> None:
        if self.frozen:
            raise RuntimeError("PoseCommander is frozen; resume, hold or start first")
        # a follow never completes by itself; new motion ends every queued follow
        if seg.kind != "follow":
            self._queue = [s for s in self._queue if s.kind != "follow"]
        elif self._queue and self._queue[0].kind == "follow":
            self._queue.pop(0)
        self._queue.append(seg)

    def blend_to(self, token, hands, ticks: int = 50, label: str = "blend") -> None:
        if ticks < 1:
            raise ValueError(f"blend ticks must be >= 1, got {ticks}")
        self._add(_Segment("blend", label, np.asarray(token, np.float32).reshape(64),
                           np.asarray(hands, np.float32).reshape(14), ticks))

    def play(self, tokens, hands, label: str) -> None:
        tokens, hands = np.array(tokens, np.float32), np.array(hands, np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != 64 or hands.shape != (len(tokens), 14) or len(tokens) == 0:
            raise ValueError(f"play needs [T, 64] tokens and [T, 14] hands, got {tokens.shape}, {hands.shape}")
        self._add(_Segment("play", label, tokens, hands, len(tokens)))

    def follow(self, stream, label: str = "episode") -> None:
        self._add(_Segment("follow", label, stream=stream))

    def freeze(self) -> None:
        self.frozen = True

    def resume(self) -> None:
        self.frozen = False

    def _slew(self, target):
        step = self.max_token_step
        delta = target - self.last_token
        if step <= 0 or np.abs(delta).max() <= step:
            return np.asarray(target, np.float32)
        return (self.last_token + np.clip(delta, -step, step)).astype(np.float32)

    def tick(self, now: float):
        if self.frozen or not self._queue:
            label = "frozen" if self.frozen else "hold"
            token, hands = self.last_token.copy(), self.last_hands.copy()
        else:
            seg = self._queue[0]
            if seg.kind == "follow":
                if seg.i == 0:
                    seg.stream.last_token = self.last_token.copy()
                token, hands = seg.stream.at(now)
            else:
                if seg.origin is None:
                    seg.origin = (self.last_token.copy(), self.last_hands.copy())
                if seg.kind == "blend":
                    w = (seg.i + 1) / seg.ticks
                    target = np.asarray((1 - w) * seg.origin[0] + w * seg.tokens, np.float32)
                    hands = np.asarray((1 - w) * seg.origin[1] + w * seg.hands, np.float32)
                else:
                    target, hands = seg.tokens[seg.i], seg.hands[seg.i]
                token = self._slew(target)
            label = seg.label
            seg.i += 1
            if seg.kind != "follow" and seg.i >= seg.ticks:
                self._queue.pop(0)
        self.last_token, self.last_hands = np.asarray(token, np.float32), np.asarray(hands, np.float32)
        self.recent_tokens.append(self.last_token.copy())
        return self.last_token.copy(), self.last_hands.copy(), label
