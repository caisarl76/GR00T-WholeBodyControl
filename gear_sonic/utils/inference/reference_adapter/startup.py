"""Measured startup gate for the recorded G1 ACT integration experiment."""

from __future__ import annotations

import numpy as np


def settled_observation(records: list[dict], *, reset_ns: int) -> dict | None:
    """Return a grounded baseline only after 0.5 s of measured settling.

    This checks standing under native SONIC control, before ACT playback.
    It is a startup condition, not a general balance or task-success metric.
    """
    rows = [row for row in records if row.get("monotonic_ns", 0) > reset_ns and "body_q" in row]
    if len(rows) < 2:
        return None
    cutoff = rows[-1]["monotonic_ns"] - 500_000_000
    window = [row for row in rows if row["monotonic_ns"] >= cutoff - 20_000_000]
    stamps = np.asarray([row["monotonic_ns"] for row in window], dtype=np.int64)
    if stamps[-1] - stamps[0] < 500_000_000 or np.any(np.diff(stamps) > 100_000_000):
        return None
    for row in window:
        if row.get("foot_contacts") != {"left": True, "right": True}:
            return None
        q = np.asarray(row["body_q"], dtype=float)
        dq = np.asarray(row["body_dq"], dtype=float)
        pose = np.asarray(row["floating_base_pose"], dtype=float)
        velocity = np.asarray(row.get("floating_base_vel", []), dtype=float)
        if q.shape != (29,) or dq.shape != (29,) or pose.shape != (7,) or velocity.shape != (6,):
            return None
        if not np.all(np.isfinite(np.r_[q, dq, pose, velocity])):
            return None
        norm = np.linalg.norm(pose[3:7])
        if not 0.999 <= norm <= 1.001:
            return None
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.sum(pose[4:6] ** 2) / norm**2, -1, 1)))
        if (
            not 0.65 <= pose[2] <= 0.9
            or tilt > 10
            or np.max(np.abs(q[12:15])) > 0.15
            or np.max(np.abs(dq)) > 0.5
            or np.linalg.norm(velocity[:3]) > 0.1
            or np.linalg.norm(velocity[3:]) > 0.3
        ):
            return None
    return rows[-1]
