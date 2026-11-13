"""G2: latency benchmark of the arm-only IK (PRD 6.2). Isolated; nothing imports this.

Run: PYTHONPATH=$PWD python gear_sonic/scripts/gates/g2_ik_bench.py
Target source: random reachable targets only (FK of random in-limit q, linear q-sweeps).
Recorded-episode replay (teleop.smpl_joints) is not implemented: none exist on this host.
Per tick (timed): anchor-delta target transforms (PRD 6.2) from controller poses in W and the
IK solve; the in-tick fk_wrist_in_torso calls (held solution, both arms) are included as
conservative overhead.
"""

import time

import numpy as np
import pinocchio as pin

from gear_sonic.utils.teleop.manipulation.arm_ik import DT, NUM_STEPS, ArmIK

N_TRAJ, TICKS_PER_TRAJ, WARMUP, TIMED, K = 1010, 10, 100, 10_000, 1.0
SIDES = ("left", "right")


def rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def anchor_delta_target(p_c, R_c, a):
    """PRD 6.2 per-tick mapping: controller pose in W -> wrist target in torso frame."""
    Rz0 = a["Rz0"]
    dp = Rz0.T @ (p_c - a["p_c0"])
    dR = Rz0.T @ (R_c @ a["R_c0"].T) @ Rz0
    return a["p_w0"] + K * dp, dR @ a["R_w0"]


def build_ticks(ik, rng):
    """List of (anchor_by_side, q_posture, controller_poses_by_side) per tick, grouped by trajectory."""
    ticks = []
    for _ in range(N_TRAJ):
        qa, qb = rng.uniform(ik.lower, ik.upper, (2, 14))
        psi0 = rng.uniform(-np.pi, np.pi)
        anchors = {}
        for s in SIDES:
            p_w0, R_w0 = ik.fk_wrist_in_torso(qa, s)
            R_c0 = pin.exp3(rng.normal(size=3))
            anchors[s] = dict(Rz0=rz(psi0), p_c0=rng.normal(size=3), R_c0=R_c0, p_w0=p_w0, R_w0=R_w0)
        for i in range(TICKS_PER_TRAJ):
            q = qa + (qb - qa) * (i + 1) / TICKS_PER_TRAJ
            ctrl, want = {}, {}
            for s in SIDES:
                a = anchors[s]
                p_t, R_t = ik.fk_wrist_in_torso(q, s)
                want[s] = (p_t, R_t)
                # invert the 6.2 mapping so the mapped target equals the FK target
                ctrl[s] = (
                    a["p_c0"] + a["Rz0"] @ ((p_t - a["p_w0"]) / K),
                    a["Rz0"] @ (R_t @ a["R_w0"].T) @ a["Rz0"].T @ a["R_c0"],
                )
            ticks.append((anchors, qa, ctrl, want, i == 0))
    return ticks


def main():
    cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    print(f"CPU: {cpu}")
    print("no recorded episodes found; random FK targets only")
    ik = ArmIK()
    print(f"solver settings: dt={DT} num_steps={NUM_STEPS} (pink/quadprog), k={K}")
    ticks = build_ticks(ik, np.random.default_rng(0))
    assert len(ticks) == WARMUP + TIMED and N_TRAJ >= 1000

    q_prev = ticks[0][1]
    lat, perr, aerr = [], [], []
    for n, (anchors, q_post, ctrl, want, first) in enumerate(ticks):
        if first:
            q_prev = q_post
        t0 = time.perf_counter_ns()
        for s in SIDES:  # torso-frame FK of the held solution (what the streamer re-anchors from)
            ik.fk_wrist_in_torso(q_prev, s)
        targets = {s: anchor_delta_target(*ctrl[s], anchors[s]) for s in SIDES}
        q_prev, ok = ik.solve(targets, q_prev, q_post)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        if n >= WARMUP:
            for s in SIDES:  # 6.2 transform self-check (untimed)
                assert np.allclose(targets[s][0], want[s][0]) and np.allclose(targets[s][1], want[s][1])
            lat.append(dt_ms)
            for s in SIDES:  # untimed accuracy (mapped target vs sweep FK target)
                p, R = ik.fk_wrist_in_torso(q_prev, s)
                perr.append(np.linalg.norm(p - want[s][0]))
                aerr.append(np.degrees(np.linalg.norm(pin.log3(R.T @ want[s][1]))))
    lat = np.array(lat)
    p50, p99, mx = np.percentile(lat, 50), np.percentile(lat, 99), lat.max()
    print(f"ticks timed={len(lat)} trajectories={TIMED // TICKS_PER_TRAJ}")
    print(f"latency ms: p50={p50:.3f} p99={p99:.3f} max={mx:.3f}")
    print(f"tracking (sweep, {TICKS_PER_TRAJ} ticks/traj): pos err median={np.median(perr)*100:.2f} cm p99={np.percentile(perr,99)*100:.2f} cm; "
          f"ori err median={np.median(aerr):.2f} deg p99={np.percentile(aerr,99):.2f} deg")
    print(f"G2 {'PASS' if p99 < 10 else 'FAIL'} (p99 < 10 ms)")


if __name__ == "__main__":
    main()
