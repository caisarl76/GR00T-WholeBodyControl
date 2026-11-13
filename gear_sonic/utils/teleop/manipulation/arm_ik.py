"""Arm-only IK on a reduced pinocchio model (G2 prototype; not wired into streamer/deploy).

Only the 14 arm joints are free; waist, legs and hands are locked at q0, so the solver
cannot move anything else. Targets and FK are expressed in the torso_link frame.
Arm joint order = entries 3..16 of the 17-joint upper-body list: left/right interleaved
(L shoulder_pitch, R shoulder_pitch, L shoulder_roll, ...).
Joint limits are robot-model limits incl. g1_supplemental_info shoulder_roll narrowing
(arms cannot cross midline), same as BodyIKSolver; not the raw URDF limits.
solve() mirrors pink 3.1.0 solve_ik (check_limits, build_ik, qpsolvers.solve_problem) but checks
result.found / x instead of asserting; ok=False only if the QP failed, then the clipped init
is returned.
"""

import numpy as np
import pink
from pink.tasks import FrameTask
import pinocchio as pin
import qpsolvers

from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.body.body_ik_solver import WeightedPostureTask
from decoupled_wbc.control.teleop.solver.body.body_ik_solver_settings import BodyIKSolverSettings

_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
ARM_JOINTS = [f"{s}_{j}_joint" for j in _ARM for s in ("left", "right")]  # interleaved L/R
_POSTURE_KEYS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw"]

# Per-solve settings (shared by tests and benchmark). dt/steps differ from BodyIKSolverSettings
# (dt 0.05, 3 steps) to reach <1 cm / <5 deg round-trip accuracy from a cold-ish start.
DT = 0.05
NUM_STEPS = 10


class ArmIK:
    def __init__(self, robot_model=None, dt: float = DT, num_steps: int = NUM_STEPS):
        rm = robot_model or instantiate_g1_robot_model()
        full = rm.pinocchio_wrapper.model
        arm_ids = {full.getJointId(n) for n in ARM_JOINTS}
        lock = [i for i in range(1, full.njoints) if i not in arm_ids]
        q0 = rm.pinocchio_wrapper.q0.copy()
        self.model = pin.buildReducedModel(full, lock, q0)
        self.data = self.model.createData()
        self.dt, self.num_steps = dt, num_steps

        # reduced q order follows joint id order; map to our interleaved order
        idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in ARM_JOINTS]
        self._perm = np.array(idx)  # our[i] -> reduced q index
        for n, lo, hi in ((n, *self._lim(rm, n)) for n in ARM_JOINTS):
            k = self.model.joints[self.model.getJointId(n)].idx_q
            self.model.lowerPositionLimit[k], self.model.upperPositionLimit[k] = lo, hi
        self.lower = self.model.lowerPositionLimit[self._perm].copy()
        self.upper = self.model.upperPositionLimit[self._perm].copy()

        # torso frame expressed in the (pelvis) base frame of the reduced model, at locked q0
        d0 = self.model.createData()
        pin.framesForwardKinematics(self.model, d0, pin.neutral(self.model))
        fid = self.model.getFrameId("torso_link")
        self._T_base_torso = d0.oMf[fid].copy()
        self._frames = {s: self.model.getFrameId(f"{s}_wrist_yaw_link") for s in ("left", "right")}

        st = BodyIKSolverSettings()
        w = np.ones(14)
        for i, n in enumerate(ARM_JOINTS):
            side_key = _POSTURE_KEYS[i // 2]
            w[self._perm[i]] = st.posture_weight.get(side_key, 1.0)  # wrist_roll unset in settings -> 1.0 (as BodyIKSolver)
        self._tasks = {
            s: FrameTask(f"{s}_wrist_yaw_link", **st.link_costs["hand"]) for s in ("left", "right")
        }
        self._posture = WeightedPostureTask(st.posture_cost, w, st.posture_lm_damping)
        self._solver = "quadprog"

    @staticmethod
    def _lim(rm, name):
        k = rm.joint_to_dof_index[name]
        return rm.lower_joint_limits[k], rm.upper_joint_limits[k]

    def _to_red(self, q14):
        q = np.zeros(14)
        q[self._perm] = q14
        return q

    def fk_wrist_in_torso(self, arm_q14, side):
        pin.framesForwardKinematics(self.model, self.data, self._to_red(arm_q14))
        T = self._T_base_torso.actInv(self.data.oMf[self._frames[side]])
        return T.translation.copy(), T.rotation.copy()

    def solve(self, targets, q_init14, q_posture14):
        cfg = pink.Configuration(self.model, self.data, self._to_red(np.clip(q_init14, self.lower, self.upper)))
        self._posture.set_target(self._to_red(np.clip(q_posture14, self.lower, self.upper)))
        tasks = [self._posture]
        for side, (p, R) in targets.items():
            self._tasks[side].set_target(self._T_base_torso * pin.SE3(R, p))
            tasks.append(self._tasks[side])
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        q_init = cfg.q[self._perm].copy()
        for _ in range(self.num_steps):
            cfg.check_limits()  # as solve_ik does (safety_break=True); q is clipped every step
            res = qpsolvers.solve_problem(pink.build_ik(cfg, tasks, self.dt), solver=self._solver)
            if not res.found or res.x is None:
                return q_init, False
            cfg.q = np.clip(cfg.q + res.x, lo, hi)  # res.x is dq = v * dt
            cfg.update()
        return cfg.q[self._perm].copy(), True
