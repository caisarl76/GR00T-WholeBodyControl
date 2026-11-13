"""G0 analysis tests on synthetic npz files in the Task 1 recorder save format."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.scripts.gates.g0_analyze import analyze_twist, chest_twist_joints, spine_twist

NS = 1_000_000_000
T0 = 1000.0
TB, TC, ST2, RCH = T0 + 4, T0 + 16, T0 + 28, T0 + 32  # turn_body, twist_chest, still, reach markers
REACH_T = [RCH + 4 + 8 * k for k in range(5)]
END = REACH_T[-1] + 8
REACH_LEN = 3.0


def _bump(t, start, amp, length=REACH_LEN):
    u = (t - start) / length
    return amp * np.sin(np.pi * np.clip(u, 0, 1)) ** 2 * ((u >= 0) & (u <= 1))


def _wave(t, start, amp):
    u = t - start
    return amp * np.sin(2 * np.pi * u / 4.0) * ((u >= 0) & (u < 12))


def _quat_yaw(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, 0.0, np.sin(h)]


def _joints_with_chest_twist(twist_deg, bend_deg=20.0):
    """Root-local (z-up) joints: hips along y; shoulders bent forward (about y) then twisted about up."""
    j = np.zeros((24, 3))
    j[1], j[2] = [0, 0.1, 0], [0, -0.1, 0]
    rot = R.from_euler("z", twist_deg, degrees=True) * R.from_euler("y", bend_deg, degrees=True)
    j[16], j[17] = rot.apply([0, 0.2, 0.3]), rot.apply([0, -0.2, 0.3])
    return j


def build(tmp_path, *, reach_ref_pelvis=0.0, reach_ref_spine=0.0, extra_b=0.0, extra_c=0.0, extra_bc=0.0,
          waist_gain=1.0, lag_b=0.0, per_reach=None, n_reaches=5, op=None, sim_extra=0.0, drop=None,
          placeholder_target=False, flip_waist=False, spine_pose_garbage=False, skip_marker=None, sign_pelvis=1.0):
    """Synthetic recording. ``per_reach``: {k: dict(overrides)} for reach k. ``op``: None | 'flat' | 'follow'."""
    per_reach = per_reach or {}
    start_reach = REACH_T[:n_reaches]

    def reach_sum(t, key, default):
        out = np.zeros_like(t)
        for k, s in enumerate(start_reach):
            out = out + _bump(t, s, per_reach.get(k, {}).get(key, default))
        return out

    def ref_pelvis(t):
        return _wave(t, TB, 30.0) + reach_sum(t, "reach_ref_pelvis", reach_ref_pelvis)

    def ref_spine(t):
        return _wave(t, TC, 20.0) + reach_sum(t, "reach_ref_spine", reach_ref_spine)

    def tgt(t):
        return ref_pelvis(t - lag_b) + reach_sum(t, "extra_b", extra_b)

    def meas_p(t):
        return tgt(t) + reach_sum(t, "extra_c", extra_c)

    def waist(t):
        return waist_gain * ref_spine(t) * (-1 if flip_waist else 1) + reach_sum(t, "extra_bc", extra_bc)

    def op_yaw(t):
        cal = ref_pelvis(t) * ((t < RCH)) + ref_spine(t) * (t < RCH)
        if op == "flat":
            return cal
        return ref_pelvis(t) + ref_spine(t)

    d = {}
    marks = [(T0 + 1, "still"), (TB, "turn_body"), (TC, "twist_chest"), (ST2, "still"), (RCH, "reach"), (END - 1, "end")]
    marks = [m for m in marks if m[1] != skip_marker]
    d["markers_src_ns"] = np.array([int(t * NS) for t, _ in marks], dtype=np.int64)
    d["markers_recv_ns"] = d["markers_src_ns"] + 1_000_000
    raw = np.empty(len(marks), dtype=object)
    raw[:] = [{"name": n} for _, n in marks]
    d["markers_raw"] = raw

    def stream(name, t, make, extra=None):
        if drop and name == drop[0]:
            t = t[(t < drop[1]) | (t > drop[2])]
        d[f"{name}_src_ns"] = np.round(t * NS).astype(np.int64)
        d[f"{name}_recv_ns"] = d[f"{name}_src_ns"] + 2_000_000
        r = np.empty(len(t), dtype=object)
        r[:] = [make(x) for x in t]
        d[f"{name}_raw"] = r

    t50 = np.arange(T0, END, 0.02)
    t200 = np.arange(T0, END, 0.005)
    wrist_rest = np.array([0.2, -0.2, 0.0])

    def pose(t):
        pose_aa = np.zeros((1, 21, 3))
        pose_aa[0, 5] = [0, 0.0 if spine_pose_garbage else np.radians(ref_spine(t)), 0]
        joints = _joints_with_chest_twist(ref_spine(t))[None]
        joints[0, 21] = wrist_rest + [reach_sum(np.array(t), "w", 0.3), 0, 0]
        return {
            "body_quat_w": np.array([_quat_yaw(sign_pelvis * ref_pelvis(t))] * 1),
            "smpl_pose": pose_aa,
            "smpl_joints": joints,
            "timestamp_monotonic": np.array([t]),
        }

    def g1(t):
        qm = np.zeros(29)
        qm[12] = np.radians(waist(t))
        qt = np.zeros(29)
        if not placeholder_target:
            qt[12] = np.radians(waist(t))
        qtgt = _quat_yaw(0.0) if placeholder_target else _quat_yaw(tgt(t))
        return {"t_mono_ns": int(round(t * NS)), "base_quat_target": qtgt, "body_q_target": qt,
                "base_quat_measured": _quat_yaw(meas_p(t)), "body_q_measured": qm}

    def sim(t):
        return {"t_mono_ns": int(round(t * NS)), "t_sim": t - T0,
                "pelvis_quat_wxyz": _quat_yaw(meas_p(t) + reach_sum(np.array(t), "sim_extra", sim_extra))}

    def opmsg(t):
        a = np.radians(op_yaw(t))
        trk = np.zeros((3, 7))
        trk[1, 3:7] = [0, np.sin(a / 2), 0, np.cos(a / 2)]
        return {"t_mono_ns": int(round(t * NS)), "trackers": trk, "idx": 1}

    def reach_w(t):  # patch wrist displacement
        return reach_sum(np.array(t), "w", 0.3)

    # wrist reach displacement uses 'w' key default 0.3 for every reach present
    stream("pose", t50, pose)
    stream("g1_debug", t50, g1)
    stream("sim_gt", t200, sim)
    d["sim_gt_t_sim"] = np.array([x - T0 for x in d["sim_gt_src_ns"] / NS])
    if op:
        stream("operator_ref", np.arange(T0, END, 0.01), opmsg)
    path = tmp_path / "g0.npz"
    np.savez(path, **d)
    return str(path)


def test_body_quat_w_yaw_not_joint_positions(tmp_path):
    # root yaw turns 10 deg during reaches; root-local smpl_joints only move by the reach itself
    res = analyze_twist(build(tmp_path, reach_ref_pelvis=10.0))
    assert len(res["reaches"]) == 5
    assert all(abs(r["ref_twist"]["pelvis"] - 10.0) < 0.3 for r in res["reaches"])


def test_spine_twist_swing_twist():
    p = np.zeros((21, 3))
    p[2] = R.from_euler("x", 20, degrees=True).as_rotvec()  # forward bend in spine1
    p[8] = R.from_euler("y", 8, degrees=True).as_rotvec()  # twist in spine3
    assert abs(np.degrees(spine_twist(p)) - 8.0) < 0.1


def test_chest_twist_joints_with_forward_bend():
    for bend in (0.0, 20.0):
        got = np.degrees(chest_twist_joints(_joints_with_chest_twist(8.0, bend)))
        assert abs(got - 8.0) < 0.2


def test_robot_side_ignores_ref_spine_twist(tmp_path):
    res = analyze_twist(build(tmp_path, spine_pose_garbage=True, extra_bc=6.0))
    assert res["robot_side"] == "chest_residual"
    assert res["calibration"]["BC"]["valid"] and not res["calibration"]["BC_spine_diag"]["valid"]


def test_calibration_valid_and_gain(tmp_path):
    res = analyze_twist(build(tmp_path, waist_gain=0.9))
    c = res["calibration"]["BC"]
    assert c["valid"] and abs(c["gain"] - 0.9) < 0.02 and res["robot_side"] == "none"
    res = analyze_twist(build(tmp_path, waist_gain=0.5))
    assert res["calibration"]["BC"]["status"] == "unmapped"
    assert res["robot_side"] == "inconclusive"


def test_placeholder_target_unobservable(tmp_path):
    res = analyze_twist(build(tmp_path, placeholder_target=True))
    assert res["calibration"]["B"]["status"] == "target_unobservable"
    assert res["calibration"]["C"]["status"] == "target_unobservable"
    assert "tgt_pelvis_yaw" in res["target_unobservable"]
    assert res["robot_side"] == "inconclusive"


def test_calibration_sign_flip_invalid(tmp_path):
    res = analyze_twist(build(tmp_path, flip_waist=True))
    assert not res["calibration"]["BC"]["valid"] and res["robot_side"] == "inconclusive"


def test_robot_side_pipeline(tmp_path):
    assert analyze_twist(build(tmp_path, extra_b=6.0))["robot_side"] == "pipeline"


def test_robot_side_sonic(tmp_path):
    assert analyze_twist(build(tmp_path, extra_c=8.0))["robot_side"] == "sonic"


def test_robot_side_chest_residual(tmp_path):
    res = analyze_twist(build(tmp_path, extra_bc=6.0))
    assert res["robot_side"] == "chest_residual"


def test_robot_side_label_order(tmp_path):
    res = analyze_twist(build(tmp_path, extra_b=6.0, extra_bc=6.0))
    assert res["robot_side"] == "pipeline+chest_residual"


def test_robot_side_none_when_target_tracks_reference_with_lag(tmp_path):
    res = analyze_twist(build(tmp_path, lag_b=0.15, reach_ref_pelvis=8.0))
    assert res["robot_side"] == "none"
    assert all(abs(r["L"]["B"] - 0.15) < 0.02 for r in res["reaches"])


def test_reference_unverified_without_operator_ref(tmp_path):
    res = analyze_twist(build(tmp_path, reach_ref_pelvis=10.0))
    assert res["reference_side"] == "twist_in_reference_operator_unverified"
    assert not res["operator_ref"]


def test_reference_none(tmp_path):
    assert analyze_twist(build(tmp_path))["reference_side"] == "none"


def test_reference_pico_with_still_operator(tmp_path):
    res = analyze_twist(build(tmp_path, reach_ref_pelvis=10.0, op="flat"))
    assert res["operator_ref"] and res["calibration"]["A"]["valid"]
    assert res["reference_side"] == "pico"


def test_reference_operator_moved(tmp_path):
    res = analyze_twist(build(tmp_path, reach_ref_pelvis=6.0, op="follow"))
    assert res["reference_side"] == "operator_moved"


def test_inconclusive_on_gap(tmp_path):
    res = analyze_twist(build(tmp_path, drop=("g1_debug", REACH_T[2] + 0.5, REACH_T[2] + 1.0)))
    assert res["robot_side"] == "inconclusive" and any("gap" in r for r in res["reasons"])


def test_inconclusive_on_disagreeing_reaches(tmp_path):
    res = analyze_twist(build(tmp_path, per_reach={2: {"extra_b": 6.0}}))
    assert res["robot_side"] == "inconclusive" and any("disagree" in r for r in res["reasons"])


def test_inconclusive_on_ambiguous_band(tmp_path):
    res = analyze_twist(build(tmp_path, extra_b=2.2))
    assert res["robot_side"] == "inconclusive" and any("ambiguous" in r for r in res["reasons"])


def test_inconclusive_on_imu_sim_mismatch(tmp_path):
    res = analyze_twist(build(tmp_path, sim_extra=3.0))
    assert res["robot_side"] == "inconclusive" and any("imu_vs_sim" in r for r in res["reasons"])


def test_inconclusive_on_missing_stream(tmp_path):
    p = build(tmp_path)
    d = dict(np.load(p, allow_pickle=True))
    for k in [k for k in d if k.startswith("g1_debug")]:
        del d[k]
    np.savez(p, **d)
    res = analyze_twist(p)
    assert res["robot_side"] == "inconclusive" and any("missing stream" in r for r in res["reasons"])


def test_inconclusive_on_four_reaches(tmp_path):
    res = analyze_twist(build(tmp_path, n_reaches=4))
    assert res["robot_side"] == "inconclusive" and any("reaches detected" in r for r in res["reasons"])


def test_inconclusive_on_missing_calibration_segment(tmp_path):
    res = analyze_twist(build(tmp_path, skip_marker="twist_chest"))
    assert res["robot_side"] == "inconclusive" and any("calibration segment" in r for r in res["reasons"])


def test_calibration_lag_compensated(tmp_path):
    res = analyze_twist(build(tmp_path, lag_b=0.25))
    c = res["calibration"]["B"]
    assert c["valid"] and abs(c["gain"] - 1.0) < 0.05 and abs(c["L"] - 0.25) < 0.02


def test_calibration_lag_outside_range_unmapped(tmp_path):
    # period 4 s: a 0.5 s lag leaves 0.2 s after the 0.3 s search, corr ~0.95 (still valid), so use 0.8 s
    res = analyze_twist(build(tmp_path, lag_b=0.8))
    assert res["calibration"]["B"]["status"] == "unmapped" and res["robot_side"] == "inconclusive"


def test_nan_in_reach_baseline_inconclusive(tmp_path):
    p = REACH_T[2]
    res = analyze_twist(build(tmp_path, drop=("g1_debug", p - 0.9, p - 0.5)))
    assert res["robot_side"] == "inconclusive" and any("NaN in window" in r for r in res["reasons"])
