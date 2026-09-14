import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    out = Path(sys.argv[1])
    cfg = json.loads(Path(sys.argv[2]).read_text())
    sys.path.insert(0, cfg["source_root"])
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator

    config = SimLoopConfig(interface="lo", enable_onscreen=False, enable_offscreen=False).load_wbc_yaml()
    config.update(ROBOT_SCENE=cfg["robot_scene"], PRINT_SCENE_INFORMATION=False, USE_JOYSTICK=0)
    sim = BaseSimulator(config, onscreen=False, offscreen=False)
    out.joinpath("sim.ready").touch()
    start = time.monotonic()
    with out.joinpath("physics.csv").open("w", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(["wall", "sim_time", "z", "tilt_deg", "yaw_deg", "band", "command_age_ms"])
        tick = 0
        while time.monotonic() - start < 100 and not out.joinpath("stop").exists():
            t = time.monotonic()
            if out.joinpath("release").exists():
                sim.sim_env.elastic_band.enable = False
            sim.sim_env.sim_step()
            if tick % 4 == 0:
                q = sim.sim_env.mj_data.qpos
                w, x, y, z = q[3:7]
                tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
                yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                writer.writerow(
                    [t, sim.sim_env.mj_data.time, q[2], tilt, yaw, int(sim.sim_env.elastic_band.enable), -1]
                )
            tick += 1
            time.sleep(max(0, 0.005 - (time.monotonic() - t)))
    sim.close()
    os._exit(0)


if __name__ == "__main__":
    main()
