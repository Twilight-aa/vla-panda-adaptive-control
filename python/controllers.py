"""Shared torque controllers for free-space tracking and contact tasks.

The same CTC / SMC / RBF-SMC equations are used by run_panda_local.py and
grasp_transport.py so the contact experiment no longer duplicates controller
logic.  The nominal model is independent of the true simulated plant.
"""
import numpy as np
import mujoco as mj
from run_panda_local import HOME, LIMIT, setup, bias


class TorqueController:
    def __init__(self, method, config=None, nominal=None, dt=.001):
        self.method = method
        self.c = {} if config is None else config
        self.m, self.d = setup() if nominal is None else nominal
        self.dt = dt
        self.W = np.zeros((7, 33))
        self.centers = np.random.default_rng(20261031).uniform(-1, 1, (33, 21))
        self.Ks = np.array([20, 20, 15, 15, 2, 2, 1.]) * self.c.get('ks_scale', 1.)
        self.rho = np.array([3, 3, 2, 2, .5, .5, .2]) * self.c.get('rho_scale', 1.)

    def step(self, q, v, qd, dqd, ddqd, limits=None, t=None):
        """Return (torque, saturated, rbf_compensation)."""
        m, d, c = self.m, self.d, self.c
        e = q[:7] - qd
        de = v[:7] - dqd
        lam = c.get('lambda', 8.)
        s = de + lam * e
        vr = np.zeros(m.nv)
        ar = np.zeros(m.nv)
        vr[:7] = dqd - lam * e
        ar[:7] = ddqd - lam * de
        h = bias(m, d, q, v)
        passive = d.qfrc_passive.copy()
        M = np.zeros((m.nv, m.nv))
        mj.mj_fullM(m, d, M)
        g = bias(m, d, q, np.zeros(m.nv))
        comp = np.zeros(7)
        if self.method == 'CTC':
            a = np.zeros(m.nv)
            a[:7] = ddqd - c.get('ctc_kd', 16.) * de - c.get('ctc_kp', 64.) * e
            tau = (M @ a + h - passive)[:7]
        else:
            cvr = .5 * (bias(m, d, q, v + vr) - h - bias(m, d, q, vr) + g)
            tau = (M @ ar + cvr + g - passive)[:7] - self.Ks * s
            if self.method != 'RBF_NO_ROBUST':
                tau -= self.rho * np.clip(s / c.get('phi', .15), -1, 1)
            z = np.r_[(q[:7] - HOME) / .4, v[:7], ar[:7] / 5]
            phi = np.exp(-np.sum((self.centers - z) ** 2, axis=1) / (2 * 4 ** 2))
            phi /= max(np.linalg.norm(phi), 1e-12)
            comp = self.W @ phi
            if self.method.startswith('RBF'):
                tau += comp
                if self.method != 'RBF_FROZEN' or (t is not None and t < 3.):
                    self.W += self.dt * (-c.get('gamma', 8.) * np.outer(s, phi) - .02 * self.W)
                    self.W *= min(1, 50 / max(np.linalg.norm(self.W), 1e-12))
        lim = LIMIT if limits is None else limits
        u = np.clip(tau, -lim, lim)
        return u, bool(np.any(np.abs(tau) > lim)), comp
