"""Wrap LIBERO/robosuite's OSC controller with joint-space torque controllers.

The VLA still emits the same 7-D end-effector actions; the OSC `set_goal`
converts them to a Cartesian goal.  This wrapper keeps that goal, solves a
damped-least-squares IK for a joint target, and replaces the OSC torque law
with CTC / SMC / RBF-SMC from the course project.
"""
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

    def __getattr__(self, name):
        return getattr(self.osc, name)

    def set_goal(self, action, **kwargs):
        return self.osc.set_goal(action, **kwargs)

    def reset_goal(self):
        self.W[:] = 0.
        self._steps = 0
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

    def _ik(self, q0, target_pos, target_ori):
        q = q0.copy()
        lo = self.m.jnt_range[self.qidx, 0] + .01
        hi = self.m.jnt_range[self.qidx, 1] - .01
        for _ in range(80):
            d = mujoco.MjData(self.m)
            d.qpos[:] = self.sim.data._data.qpos
            d.qpos[self.qidx] = q
            d.qvel[:] = 0
            mujoco.mj_forward(self.m, d)
            dp = np.array(target_pos) - d.site_xpos[self.site_id]
            do = orientation_error(np.array(target_ori), d.site_xmat[self.site_id].reshape(3, 3))
            err = np.r_[dp, do]
            if np.linalg.norm(err) < 1e-4:
                break
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, d, jp, jr, self.site_id)
            J = np.vstack([jp[:, self.vidx], jr[:, self.vidx]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            q = np.clip(q + .5 * dq, lo, hi)
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
        qtar = self._ik(q, o.goal_pos, o.goal_ori)
        dt = .05
        dqd = (qtar - q) / dt
        dqd = np.clip(dqd, -2.0, 2.0)
        qd = qtar
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
        o.new_update = True
        return tau


def install(env, method, config):
    env.robots[0].controller = TorqueControllerWrapper(env.robots[0].controller, method, config)
    return env
