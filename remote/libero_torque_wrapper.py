"""Wrap LIBERO/robosuite's OSC controller with joint-space torque controllers.

The VLA still emits the same 7-D end-effector actions; the OSC `set_goal`
converts them to a Cartesian goal.  This wrapper keeps that goal, solves a
damped-least-squares IK for a joint target, and replaces the OSC torque law
with CTC / SMC / RBF-SMC from the course project.
"""
import os
import math
import numpy as np
import mujoco
from robosuite.utils.control_utils import orientation_error

HOME = np.array([0., -.25, 0., -1.7, 0., 1.5, -.7853])


class TorqueControllerWrapper:
    def __init__(self, osc, method="CTC", config=None):
        self.osc = osc
        self.method = method
        self.c = {} if config is None else config
        self.sim = osc.sim
        self.m = self.sim.model._model
        self.qidx = np.asarray(osc.qpos_index)
        self.vidx = np.asarray(osc.qvel_index)
        self.site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, osc.eef_name)
        self.W = np.zeros((7, 33))
        self.centers = np.random.default_rng(20261031).uniform(-1, 1, (33, 21))
        self.Ks = np.array([20, 20, 15, 15, 2, 2, 1.]) * self.c.get("ks_scale", 1.)
        self.rho = np.array([3, 3, 2, 2, .5, .5, .2]) * self.c.get("rho_scale", 1.)
        self.last_tau = np.zeros(7)
        self._steps = 0
        self._qtar = None
        self._goal_key = None
        self._qref = None
        self.debug = bool(self.c.get("debug", False)) or os.environ.get("TORQUE_DEBUG") == "1"

    def __getattr__(self, name):
        return getattr(self.osc, name)

    def set_goal(self, action, **kwargs):
        self._qtar = None
        self._goal_key = None
        if self.debug:
            print(f"[torque-goal] action={np.round(action, 4)}", flush=True)
        return self.osc.set_goal(action, **kwargs)

    def reset_goal(self):
        self.W[:] = 0.
        self._steps = 0
        self._qtar = None
        self._goal_key = None
        self._qref = None
        return self.osc.reset_goal()

    def _forward(self, qpos, qvel):
        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(self.m, d)
        return d

    def _bias(self, qpos, qvel):
        d = self._forward(qpos, qvel)
        return d.qfrc_bias.copy(), d.qfrc_passive.copy()

    def _ik_once(self, q0, target_pos, target_ori, iters, ori_weight):
        q = q0.copy()
        lo = self.m.jnt_range[self.qidx, 0] + .01
        hi = self.m.jnt_range[self.qidx, 1] - .01
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.sim.data._data.qpos
        d.qvel[:] = 0
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        dp = np.zeros(3)
        do = np.zeros(3)
        for _ in range(iters):
            d.qpos[self.qidx] = q
            mujoco.mj_forward(self.m, d)
            dp = np.array(target_pos) - d.site_xpos[self.site_id]
            do = orientation_error(np.array(target_ori), d.site_xmat[self.site_id].reshape(3, 3))
            err = np.r_[dp, ori_weight * do]
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.m, d, jp, jr, self.site_id)
            J = np.vstack([jp[:, self.vidx], jr[:, self.vidx]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            q = np.clip(q + .5 * dq, lo, hi)
        # Re-evaluate at the returned q so the residual is not the one from
        # before the last update.
        d.qpos[self.qidx] = q
        mujoco.mj_forward(self.m, d)
        dp = np.array(target_pos) - d.site_xpos[self.site_id]
        do = orientation_error(np.array(target_ori), d.site_xmat[self.site_id].reshape(3, 3))
        return q, dp, do

    def _ik(self, q0, target_pos, target_ori):
        # Full 6-D IK first.  If the position residual is large while the
        # orientation error is small, the target orientation is likely
        # infeasible at that point; retry with position priority and only use
        # the fallback if it substantially improves position without a large
        # orientation error.
        iters = int(self.c.get("ik_iters", 60))
        weight = float(self.c.get("ik_ori_weight", 1.0))
        q, dp, do = self._ik_once(q0, target_pos, target_ori, iters, weight)
        if weight > 0 and np.linalg.norm(dp) > 5e-3:
            q2, dp2, do2 = self._ik_once(q0, target_pos, target_ori, 120, 0.)
            if (np.linalg.norm(dp2) < .5 * np.linalg.norm(dp)
                    and np.linalg.norm(do2) < .2):
                q, dp, do = q2, dp2, do2
        if self.debug and self._steps % 500 == 1:
            d = mujoco.MjData(self.m)
            d.qpos[:] = self.sim.data._data.qpos
            d.qpos[self.qidx] = q
            mujoco.mj_forward(self.m, d)
            print(
                "[torque-ik] step={} |dp|={:.5f} |do|={:.5f} target={} site={}".format(
                    self._steps, float(np.linalg.norm(dp)), float(np.linalg.norm(do)),
                    np.round(target_pos, 4), np.round(d.site_xpos[self.site_id], 4)
                ),
                flush=True,
            )
        return q

    def run_controller(self):
        o = self.osc
        o.update()
        self._steps += 1
        d = self.sim.data
        qfull = d.qpos.copy()
        vfull = d.qvel.copy()
        q = qfull[self.qidx].copy()
        v = vfull[self.vidx].copy()
        goal_key = (np.asarray(o.goal_pos).tobytes(), np.asarray(o.goal_ori).tobytes())
        if self._qtar is None or goal_key != self._goal_key:
            self._qtar = self._ik(q, o.goal_pos, o.goal_ori)
            self._goal_key = goal_key
        qtar = self._qtar
        dt = float(self.m.opt.timestep)
        if self._qref is None:
            self._qref = q.copy()
        qref_prev = self._qref.copy()
        # First-order reference filter: the IK target is a step, but feeding
        # the step directly into a high-gain torque law saturates the joints.
        # Smooth it at the simulator substep rate, then use its derivative as
        # the reference velocity for CTC/SMC.
        ref_tau = float(self.c.get("ref_tau", .1))
        alpha = 1. - math.exp(-dt / max(ref_tau, 1e-6))
        self._qref += alpha * (qtar - self._qref)
        qd = self._qref.copy()
        dqd = np.clip((qd - qref_prev) / dt, -2., 2.)
        ddqd = np.zeros(7)
        e = q - qd
        de = v - dqd
        lam = self.c.get("lambda", 8.)
        s = de + lam * e
        vr = np.zeros(self.m.nv); ar = np.zeros(self.m.nv)
        vr[self.vidx] = dqd - lam * e
        ar[self.vidx] = ddqd - lam * de
        h, passive = self._bias(qfull, vfull)
        h = h[self.vidx]; passive = passive[self.vidx]
        Mfull = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, Mfull, d.qM)
        M = Mfull[np.ix_(self.vidx, self.vidx)]
        g, _ = self._bias(qfull, np.zeros_like(vfull))
        g = g[self.vidx]
        if self.method == "CTC":
            a = ddqd - self.c.get("ctc_kd", 16.) * de - self.c.get("ctc_kp", 64.) * e
            tau = M @ a + h - passive
        else:
            cvr = .5 * (self._bias(qfull, vfull + vr)[0][self.vidx] - h
                        - self._bias(qfull, vr)[0][self.vidx] + g)
            tau = M @ ar + cvr + g - passive - self.Ks * s
            if self.method != "RBF_NO_ROBUST":
                tau -= self.rho * np.clip(s / self.c.get("phi", .15), -1, 1)
            z = np.r_[(q - HOME) / .4, v, ar[self.vidx] / 5]
            phi = np.exp(-np.sum((self.centers - z) ** 2, axis=1) / 32)
            phi /= max(np.linalg.norm(phi), 1e-12)
            if self.method.startswith("RBF"):
                tau = tau + self.W @ phi
                # 20 Hz control: one update per control step.
                if self.method != "RBF_FROZEN" or self._steps * dt < 3.0:
                    self.W += dt * (-self.c.get("gamma", 8.) * np.outer(s, phi) - .02 * self.W)
                    self.W *= min(1, 50 / max(np.linalg.norm(self.W), 1e-12))
        self.last_tau = tau
        if self.debug and self._steps % 500 == 1:
            print(
                "[torque-debug] step={} q_err={:.4f} ee_err={:.4f} "
                "|tau|={:.2f} max|tau|={:.2f} q={} qtar={} ee={} goal={}".format(
                    self._steps,
                    float(np.linalg.norm(e)),
                    float(np.linalg.norm(o.ee_pos - o.goal_pos)),
                    float(np.linalg.norm(tau)),
                    float(np.max(np.abs(tau))),
                    np.round(q, 3),
                    np.round(qtar, 3),
                    np.round(o.ee_pos, 3),
                    np.round(o.goal_pos, 3),
                ),
                flush=True,
            )
        o.new_update = True
        return tau


def install(env, method, config):
    """Install the torque wrapper and keep it installed across env.reset().

    robosuite's ``env.reset()`` rebuilds the robot (and its
    OperationalSpaceController) rather than only resetting the existing
    instance.  Wrapping ``env.robots[0].controller`` once is therefore not
    enough: the wrapper is silently discarded before the first control step of
    every episode.  Wrap ``env.reset`` so the current robot's newly created OSC
    controller is wrapped immediately after every reset.
    """
    if not getattr(env, "_torque_wrapper_installed", False):
        original_reset = env.reset

        def reset_and_wrap(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            robot = env.robots[0]
            robot.controller = TorqueControllerWrapper(robot.controller, method, config)
            return result

        env.reset = reset_and_wrap
        env._torque_wrapper_installed = True

    robot = env.robots[0]
    if not isinstance(robot.controller, TorqueControllerWrapper):
        robot.controller = TorqueControllerWrapper(robot.controller, method, config)
    return env
