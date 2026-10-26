"""Trial recorder for the autonomous handover loop, in the layout of lerobot a6e267c6 eval_session.py
(`_save_npz`, `write_trial`): trial_NNN.npz (50 Hz log) and trial_NNN_frames.npz (camera JPEGs).

tick/frame only append to lists; close() hands them to one background writer thread and returns at once, so the
50 Hz control loop never waits on disk. This module never writes results.jsonl (the scorer does, on the verdict).
"""

import json
import logging
import math
import os
from pathlib import Path
import queue
import threading

import numpy as np

logger = logging.getLogger(__name__)

_LOG_KEYS = ("wall", "t_ep", "phase", "frame", "token", "hands", "chunk_t0", "state")
_DTYPES = {
    "wall": np.float64,
    "t_ep": np.float64,
    "frame": np.int64,
    "token": np.float32,
    "hands": np.float32,
    "chunk_t0": np.float64,
    "state": np.float32,
}


def _save_npz(path: Path, compress: bool = True, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compress else np.savez)(f, **arrays)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written trial


def _write_trial(session_dir: Path, trial: int, log: dict, frames: list, extra: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    wall = np.asarray(log["wall"], np.float64)
    episode = wall[np.asarray(log["phase"]) == "episode"]
    wall_start, wall_stop = (float(episode[0]), float(episode[-1])) if len(episode) else (math.nan, math.nan)
    n = len(wall)
    arrays = {k: np.asarray(log[k], _DTYPES.get(k)) for k in _LOG_KEYS}
    if not n:  # an empty log would lose the column widths
        arrays["token"], arrays["hands"], arrays["state"] = (np.zeros((0, w), np.float32) for w in (64, 14, 28))
    base = f"trial_{trial:03d}"
    _save_npz(
        session_dir / f"{base}.npz",
        **arrays,
        joint_ref=np.full((n, 31), np.nan, np.float32),
        **extra,
        wall_start=wall_start,
        wall_stop=wall_stop,
    )
    blobs = [b for _, _, b in frames]
    _save_npz(
        session_dir / f"{base}_frames.npz",
        compress=False,  # JPEG bytes are already compressed
        jpeg=np.frombuffer(b"".join(blobs), np.uint8),
        offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
        t_recv=np.array([t for t, _, _ in frames], np.float64),
        names=np.array([nm for _, nm, _ in frames], dtype=str),
    )


class TrialWriter:
    """One lazily started daemon thread writing queued trials. A failed write is logged, kept in `errors`, and
    does not stop the thread."""

    errors: list[BaseException] = []
    _queue: "queue.Queue" = queue.Queue()
    _cond = threading.Condition()
    _outstanding = 0
    _pending: set[tuple[Path, int]] = set()
    _thread: threading.Thread | None = None

    @classmethod
    def submit(cls, session_dir: Path, trial: int, log: dict, frames: list, extra: dict) -> None:
        with cls._cond:
            cls._outstanding += 1
            cls._pending.add((session_dir, trial))
            if cls._thread is None:
                cls._thread = threading.Thread(target=cls._run, name="trial-writer", daemon=True)
                cls._thread.start()
        cls._queue.put((session_dir, trial, log, frames, extra))

    @classmethod
    def _run(cls) -> None:
        while True:
            session_dir, trial, log, frames, extra = cls._queue.get()
            try:
                _write_trial(session_dir, trial, log, frames, extra)
            except BaseException as e:  # noqa: BLE001 - the thread must outlive any one bad write
                logger.exception("trial %03d write to %s failed", trial, session_dir)
                cls.errors.append(e)
            with cls._cond:
                cls._pending.discard((session_dir, trial))
                cls._outstanding -= 1
                cls._cond.notify_all()

    @classmethod
    def flush(cls, timeout_s: float) -> bool:
        """True once every queued write is done (or has failed), False on timeout."""
        with cls._cond:
            return cls._cond.wait_for(lambda: cls._outstanding == 0, timeout_s)

    @classmethod
    def queued(cls, session_dir: Path) -> int:
        """Queued trials for `session_dir` whose frames file is not on disk yet."""
        with cls._cond:
            return sum(
                1 for d, t in cls._pending if d == session_dir and not (d / f"trial_{t:03d}_frames.npz").exists()
            )


def next_trial(session_dir: Path) -> int:
    session_dir = Path(session_dir)
    return len(list(session_dir.glob("trial_*_frames.npz"))) + TrialWriter.queued(session_dir)


class TrialRecorder:
    def __init__(self, session_dir: Path, trial: int, args: dict, model: str):
        self.session_dir, self.trial, self.args, self.model = Path(session_dir), trial, args, model
        self._log: dict[str, list] | None = {k: [] for k in _LOG_KEYS}
        self._frames: list[tuple[float, str, bytes]] = []

    def tick(self, wall, t_ep, phase, frame, token, hands, chunk_t0, state28) -> None:
        # copy the arrays: the caller may reuse its buffers
        row = (wall, t_ep, phase, frame, np.array(token), np.array(hands), chunk_t0, np.array(state28))
        for k, v in zip(_LOG_KEYS, row, strict=True):
            self._log[k].append(v)

    def frame(self, t_recv: float, name: str, jpeg: bytes) -> None:
        self._frames.append((t_recv, name, jpeg))

    def close(self, episode_end: str) -> None:
        if self._log is None:
            raise RuntimeError(f"trial {self.trial} already closed")
        log, frames, self._log, self._frames = self._log, self._frames, None, []
        extra = {"episode_end": episode_end, "args": json.dumps(self.args), "model": self.model}
        TrialWriter.submit(self.session_dir, self.trial, log, frames, extra)
