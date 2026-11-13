"""Tests for the G1b still-standing threshold derivation (synthetic data)."""

import inspect

import numpy as np

from gear_sonic.scripts.gates import g1b_still_thresholds as S

P = 0.02  # nominal feedback period


def _samples(n_s=30.0, dq=lambda t: 0.01, gyro=lambda t: 0.01, drop=()):
    r = np.arange(0, n_s, P)
    r = np.array([x for x in r if not any(a <= x < b for a, b in drop)])
    leg = np.stack([np.array([dq(x)] * 12) for x in r])
    gv = np.stack([np.array([gyro(x), 0.0, 0.0]) for x in r])
    return S.make_samples(r, leg, gv, period_s=P)


def test_feedback_still_uses_only_received_samples():
    r = np.sort(np.append(np.arange(0, 30, P), 10.005))  # one sample arrives just after t = 10.0
    dq = np.where(np.isclose(r, 10.005), 9.0, 0.01)  # ... carrying a huge reading
    s = S.make_samples(r, np.stack([dq] * 12, axis=1), np.zeros((len(r), 3)), period_s=P)
    assert S.feedback_still(s, 10.0, 0.05, 0.05)[0] is True  # not visible at t
    assert S.feedback_still(s, 10.01, 0.05, 0.05)[0] is False  # visible once received


def test_feedback_still_stale_and_gap():
    s = _samples(drop=((10.0, 10.5),))
    ok, why = S.feedback_still(s, 10.3, 0.05, 0.05)
    assert not ok and "stale" in why  # newest sample 0.3 s old
    ok, why = S.feedback_still(s, 10.9, 0.05, 0.05)
    assert not ok and why == "gap in feedback"  # a 0.5 s gap inside the window
    assert S.feedback_still(s, 12.0, 0.05, 0.05)[0] is True
    ok, why = S.feedback_still(s, 0.5, 0.05, 0.05)
    assert not ok and "cover" in why  # < 1 s of history
    assert S.feedback_still(_samples(dq=lambda t: 0.06), 10.0, 0.05, 0.05)[1].startswith("leg dq")
    assert S.feedback_still(_samples(gyro=lambda t: 0.06), 10.0, 0.05, 0.05)[1].startswith("gyro")


def _gt(z_off, i=300):
    t = np.arange(0, 6, 0.01)
    z = np.full(len(t), 0.787 + z_off)
    xy, yaw = np.zeros((len(t), 2)), np.zeros(len(t))
    return S.gt_still(t, z, xy, yaw, 0.787, i)


def test_gt_still_drift_limit():
    assert _gt(0.002) is True
    assert _gt(0.004) is False  # stationary but 4 mm from z_final
    assert S.gt_still(np.arange(0, 6, 0.01), np.full(600, 0.787), np.zeros((600, 2)), np.zeros(600), None, 300) is False  # not static


def _fake_run():
    """20 s: still, 5 s moving only in xy (slow, small feedback), still. Feedback reads 0.05 while sliding."""
    t = np.arange(0, 20, 0.01)
    xy = np.stack([np.where(t > 10, np.minimum(t - 10, 5) * 0.05, 0.0), np.zeros(len(t))], axis=1)  # 5 cm/s in x only
    z, yaw = np.full(len(t), 0.787), np.zeros(len(t))
    gt = [S.gt_still(t, z, xy, yaw, 0.787, i) for i in range(len(t))]
    slide = lambda x: 10 <= x < 15.5
    s = _samples(20.0, dq=lambda x: 0.05 if slide(x) else 0.01, gyro=lambda x: 0.02 if slide(x) else 0.005)
    st = [S.window_stats(s, x) for x in t]
    r = S.Run.__new__(S.Run)
    r.t, r.gt, r.samples, r.unk = t, np.array([bool(g) for g in gt]), s, np.array([g is None for g in gt])
    r.ok, r.dq, r.gy = (np.array([a[0] is None for a in st]), np.array([a[1] for a in st]), np.array([a[2] for a in st]))
    return r


def test_still_thresholds_zero_false_still():
    r = _fake_run()
    assert not r.gt[1300]
    sel = S.select_thresholds(r)
    assert sel is not None
    a = r.accept(sel["still_dq_max"], sel["still_gyro_max"])
    assert not (a & ~r.gt).any() and (a & r.gt).sum() > 0
    assert sel["still_dq_max"] <= 0.05  # a looser dq limit would accept the xy-only slide
    assert r.accept(0.1, 0.1)[1300]  # such a pair would falsely accept it: selection excludes it


def test_selection_never_sees_validation():
    params = inspect.signature(S.select_thresholds).parameters
    assert list(params) == ["run"]
    src = inspect.getsource(S.main)
    assert src.index("select_thresholds(A)") > 0 and "select_thresholds(B)" not in src


def test_missing_gt_is_unknown_not_moving():
    t = np.arange(0, 6, 0.01)
    z, xy, yaw = np.full(len(t), 0.787), np.zeros((len(t), 2)), np.zeros(len(t))
    z[300] = np.nan  # one 10 ms sim_gt hole inside the window of instants 300..400
    assert S.gt_still(t, z, xy, yaw, 0.787, 350) is True  # skipped like g1_analyze zwin, not "moving"
    z[250:270] = np.nan  # 200 ms hole > sync contract
    assert S.gt_still(t, z, xy, yaw, 0.787, 350) is None  # unknown, neither still nor moving
    r = S.Run.__new__(S.Run)  # unknown instants are excluded from false acceptance and acceptance counts
    r.t, r.gt, r.unk = t[:4], np.array([False, False, True, False]), np.array([False, True, False, False])
    r.ok, r.dq, r.gy = np.ones(4, bool), np.zeros(4), np.zeros(4)
    r.samples, r.cls, r.holds = {"period_s": P}, list("cccc"), {"hold": [], "iso": []}
    e = S.evaluate(r, 0.1, 0.1)
    assert e["false_acceptances"] == 2 and e["unknown_gt_instants"] == 1 and e["gt_still_instants"] == 1
