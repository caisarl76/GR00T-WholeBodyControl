"""Convert world-space keyboard/standing intent to SONIC's current motion frame."""

from dataclasses import replace
import math

import numpy as np


def planner_command_in_reference_frame(command, state):
    quat = np.asarray(state.get("reference_heading_quat"), dtype=float).reshape(-1)
    active = np.asarray(state.get("planner_reference_active"), dtype=float).reshape(-1)
    if (
        quat.shape != (4,)
        or not np.all(np.isfinite(quat))
        or np.linalg.norm(quat) < 1e-9
        or active.shape != (1,)
        or active[0] not in (0, 1)
    ):
        raise ValueError("Missing valid planner reference telemetry; rebuild and restart gear_sonic_deploy")
    if active[0] == 0:
        # The fresh planner initializes at reference yaw zero. Old POSE-frame
        # angles must not be applied during its initialization.
        return replace(command, facing=(1.0, 0.0, 0.0), movement=(0.0, 0.0, 0.0), mode=0, speed=0.0)
    w, x, y, z = quat / np.linalg.norm(quat)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)

    def inverse_rotate(vector):
        vx, vy, vz = vector
        return (c * vx + s * vy, -s * vx + c * vy, vz)

    return replace(command, facing=inverse_rotate(command.facing), movement=inverse_rotate(command.movement))
