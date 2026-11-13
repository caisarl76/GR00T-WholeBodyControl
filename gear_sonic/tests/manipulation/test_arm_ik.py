"""G2 arm-only IK prototype tests (reduced 14-joint model, torso-frame targets)."""

import numpy as np
import pinocchio as pin
import pytest

from gear_sonic.utils.teleop.manipulation.arm_ik import ArmIK


@pytest.fixture(scope="module")
def ik():
    return ArmIK()


def _rand_q(ik, rng, n=1):
    return rng.uniform(ik.lower, ik.upper, size=(n, 14))


def test_only_arm_joints_in_model(ik):
    assert ik.model.nq == 14
    assert ik.model.nv == 14


def test_fk_ik_roundtrip(ik):
    rng = np.random.default_rng(0)
    for q in _rand_q(ik, rng, 100):
        targets = {s: ik.fk_wrist_in_torso(q, s) for s in ("left", "right")}
        q_init = np.clip(q + rng.normal(0, 0.15, 14), ik.lower, ik.upper)
        sol, ok = ik.solve(targets, q_init, q_init)
        assert ok
        for s in ("left", "right"):
            p, R = ik.fk_wrist_in_torso(sol, s)
            tp, tR = targets[s]
            assert np.linalg.norm(p - tp) < 0.01
            ang = np.degrees(np.linalg.norm(pin.log3(R.T @ tR)))
            assert ang < 5.0


def test_unreachable_target_returns_limited_solution(ik):
    q = np.zeros(14)
    far = (np.array([2.0, 0.0, 0.0]), np.eye(3))
    sol, ok = ik.solve({"left": far, "right": far}, q, q)
    assert ok is True
    assert np.all(sol >= ik.lower - 1e-9) and np.all(sol <= ik.upper + 1e-9)


def test_solution_within_joint_limits(ik):
    rng = np.random.default_rng(1)
    for q in _rand_q(ik, rng, 20):
        t = {s: ik.fk_wrist_in_torso(q, s) for s in ("left", "right")}
        t["left"] = (t["left"][0] + 0.5, t["left"][1])
        sol, _ = ik.solve(t, np.zeros(14), np.zeros(14))
        assert np.all(sol >= ik.lower - 1e-9) and np.all(sol <= ik.upper + 1e-9)


def test_qp_failure_returns_not_ok(ik, monkeypatch):
    import gear_sonic.utils.teleop.manipulation.arm_ik as m

    import qpsolvers

    monkeypatch.setattr(
        m.qpsolvers, "solve_problem", lambda problem, **k: qpsolvers.Solution(problem, found=False, x=None)
    )
    q = np.full(14, 10.0)  # out of limits on purpose
    sol, ok = ik.solve({"left": ik.fk_wrist_in_torso(np.zeros(14), "left")}, q, np.zeros(14))
    assert ok is False
    assert np.array_equal(sol, np.clip(q, ik.lower, ik.upper))
