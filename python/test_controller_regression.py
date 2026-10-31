"""Regression checks for the shared controller and contact audit.

Run from the project directory:
    .venv/bin/python python/test_controller_regression.py

1. Shared TorqueController matches the archived inline equations on synthetic
   states (CTC, SMC, RBF-SMC).
2. Free-tracking replays match results/fair_tuning_v2/heldout.csv.
3. Grasp replays match results/grasp_transport/summary.csv.
4. The fixed setup reproduces zero contacts for the former seed=23 anomaly.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import mujoco as mj
import run_panda_local as sim
from controllers import TorqueController

ROOT = sim.ROOT
sel = json.loads((ROOT / 'results/fair_tuning_v2/selected.json').read_text())


def ref2(t):
    w = 2 * np.pi / 4.7
    a = sim.AMP * 1.25
    ph = np.arange(7) * .2
    return sim.HOME + a * np.sin(w * t + ph), a * w * np.cos(w * t + ph), -a * w * w * np.sin(w * t + ph)


def old_torque(method, c, q, v, qd, dqd, ddqd, limits):
    nom, nd = sim.setup()
    lam = c.get('lambda', 8.)
    e = q[:7] - qd; de = v[:7] - dqd; s = de + lam * e
    vr = np.zeros(9); ar = np.zeros(9)
    vr[:7] = dqd - lam * e; ar[:7] = ddqd - lam * de
    h = sim.bias(nom, nd, q, v); passive = nd.qfrc_passive.copy()
    M = np.zeros((9, 9)); mj.mj_fullM(nom, nd, M)
    g = sim.bias(nom, nd, q, np.zeros(9))
    centers = np.random.default_rng(20261031).uniform(-1, 1, (33, 21))
    W = np.zeros((7, 33))
    if method == 'CTC':
        a = np.zeros(9); a[:7] = ddqd - c.get('ctc_kd', 16.) * de - c.get('ctc_kp', 64.) * e
        tau = (M @ a + h - passive)[:7]
    else:
        cvr = .5 * (sim.bias(nom, nd, q, v + vr) - h - sim.bias(nom, nd, q, vr) + g)
        Ks = np.array([20, 20, 15, 15, 2, 2, 1.]) * c.get('ks_scale', 1.)
        rho = np.array([3, 3, 2, 2, .5, .5, .2]) * c.get('rho_scale', 1.)
        tau = (M @ ar + cvr + g - passive)[:7] - Ks * s - rho * np.clip(s / c.get('phi', .15), -1, 1)
        z = np.r_[(q[:7] - sim.HOME) / .4, v[:7], ar[:7] / 5]
        phi = np.exp(-np.sum((centers - z) ** 2, axis=1) / (2 * 4 ** 2))
        phi /= max(np.linalg.norm(phi), 1e-12)
        if method == 'RBF_SMC':
            tau += W @ phi
    return np.clip(tau, -limits, limits)


def check_unit():
    rng = np.random.default_rng(7)
    qd, dqd, ddqd = sim.ref(.4)
    worst = 0.
    for method in ['CTC', 'SMC', 'RBF_SMC']:
        q = sim.HOME + rng.normal(0, .05, 7)
        v = rng.normal(0, .05, 7)
        qq = np.r_[q, .04, .04]; vv = np.r_[v, 0., 0.]
        c = sel[method]
        old = old_torque(method, c, qq, vv, qd, dqd, ddqd, sim.LIMIT)
        new, _, _ = TorqueController(method, c, dt=.001).step(qq, vv, qd, dqd, ddqd, limits=sim.LIMIT)
        worst = max(worst, float(np.max(np.abs(old - new))))
    assert worst < 1e-10, f'unit formula mismatch {worst}'
    print(f'unit controller equivalence: max |diff| = {worst:.3e}')


def check_free():
    rows = list(csv.DictReader((ROOT / 'results/fair_tuning_v2/heldout.csv').open()))
    cases = [('nominal', 'CTC', 21, 0), ('payload', 'SMC', 21, 0),
             ('combined', 'RBF_SMC', 21, 0), ('payload', 'CTC', 23, 1),
             ('combined', 'RBF_SMC', 23, 1)]
    original_ref = sim.ref
    worst = 0.; worst_contact = 0
    try:
        for sc, method, seed, traj in cases:
            old = next(r for r in rows if r['scenario'] == sc and r['controller'] == method
                       and int(r['seed']) == seed and int(r['trajectory']) == traj)
            sim.ref = original_ref if traj == 0 else ref2
            r = sim.run(method, sc, seed, Path('/tmp/controller_regression'),
                        options={**sel[method], 'initial_std': .02, 'save_traces': False})
            worst = max(worst, abs(r['ee_rmse_m'] - float(old['ee_rmse_m'])),
                        abs(r['torque_rms_Nm'] - float(old['torque_rms_Nm'])))
            worst_contact = max(worst_contact, r['contact_steps'])
    finally:
        sim.ref = original_ref
    assert worst < 1e-8, f'free replay mismatch {worst}'
    assert worst_contact == 0, 'unexpected contact in free replay'
    print(f'free-tracking replay: max |diff| = {worst:.3e}, contacts = 0')


def check_grasp():
    import grasp_transport as gt
    rows = list(csv.DictReader((ROOT / 'results/grasp_transport/summary.csv').open()))
    cases = [('nominal', 'CTC'), ('heavy', 'CTC'), ('pulse', 'SMC'), ('low_friction', 'RBF_SMC')]
    worst = 0.
    for sc, method in cases:
        old = next(r for r in rows if r['scenario'] == sc and r['method'] == method and r['seed'] == '31')
        r = gt.run(method, 31, sc, config=sel[method], outdir='/tmp/controller_regression_grasp')
        for key in ['success', 'lifted', 'max_object_height_m', 'final_xy_error_m',
                    'transport_relative_motion_m']:
            a, b = r[key], old[key]
            if isinstance(a, (bool, np.bool_)):
                assert bool(a) == (b == 'True'), f'grasp bool mismatch {sc}/{method}/{key}'
            else:
                worst = max(worst, abs(float(a) - float(b)))
    assert worst < 1e-8, f'grasp replay mismatch {worst}'
    print(f'grasp replay: max |diff| = {worst:.3e}')


def check_contact_audit():
    r = sim.run('CTC', 'payload', 23, Path('/tmp/controller_regression'),
                options={**sel['CTC'], 'initial_std': .02, 'save_traces': False})
    assert r['contact_steps'] == 0 and r['initial_contacts'] == 0
    print('contact audit: former seed=23 payload replay has zero contacts')


if __name__ == '__main__':
    check_unit()
    check_free()
    check_grasp()
    check_contact_audit()
    print('ALL REGRESSION CHECKS PASSED')
