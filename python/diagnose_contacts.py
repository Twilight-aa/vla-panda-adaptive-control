"""Reproduce a free-motion episode and log every contact pair.

Diagnostic only: it replays the exact controller/initial-state logic from
run_panda_local.run so the reported contact pairs can be attributed.
"""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco as mj
import run_panda_local as sim
from controllers import TorqueController

OUT = sim.ROOT / 'results/contact_audit'


def ref2(t):
    w = 2 * np.pi / 4.7
    a = sim.AMP * 1.25
    phase = np.arange(7) * .2
    return (sim.HOME + a * np.sin(w * t + phase),
            a * w * np.cos(w * t + phase),
            -a * w * w * np.sin(w * t + phase))


def geom_name(m, gid):
    name = m.geom(gid).name
    return name if name else f'{m.body(m.geom_bodyid[gid]).name}:geom{gid}'


def run(method, scenario, seed, trajectory, config, outdir):
    mass = .5 if scenario in ['payload', 'combined', 'low_torque'] else (
        .8 if scenario == 'unseen_payload' else 0.)
    m, d = sim.setup(mass, 3 if scenario in ['combined', 'friction'] else 1)
    nom, nd = sim.setup()
    rng = np.random.default_rng(seed)
    d.qpos[:7] = sim.HOME + rng.normal(0, .02, 7)
    d.qvel[:7] = sim.ref(0)[1]
    d.ctrl[7] = 255
    mj.mj_forward(m, d)
    hand = m.body('hand').id
    dt = m.opt.timestep
    limits = sim.LIMIT * (.45 if scenario == 'low_torque' else 1.)
    ctrl = TorqueController(method, config, nominal=(nom, nd), dt=dt)
    reference = sim.ref if trajectory == 0 else ref2
    records = []
    for k in range(6000):
        t = k * dt
        qd, dqd, ddqd = reference(t)
        q = d.qpos.copy(); v = d.qvel.copy()
        u, _, _ = ctrl.step(q, v, qd, dqd, ddqd, limits=limits, t=t)
        d.ctrl[:7] = u
        d.xfrc_applied[:] = 0
        if scenario in ['pulse', 'combined'] and 3 <= t < 3.2:
            d.xfrc_applied[hand, :3] = [8, -4, 0]
        mj.mj_step(m, d)
        for ci in range(d.ncon):
            c = d.contact[ci]
            f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
            records.append(dict(
                t=float(t), step=int(k), pair=(geom_name(m, c.geom1), geom_name(m, c.geom2)),
                bodies=(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name),
                dist=float(c.dist), pos=c.pos.round(6).tolist(),
                force_norm=float(np.linalg.norm(f[:3])),
                qpos=d.qpos.copy().round(6).tolist(), ctrl=d.ctrl[:].copy().tolist()))
    return records


def summarize(records):
    if not records:
        return {'n_contact_steps': 0, 'pairs': {}}
    groups = {}
    for r in records:
        key = ' / '.join(r['pair'])
        g = groups.setdefault(key, {'count': 0, 't_min': r['t'], 't_max': r['t'],
                                    'force_max': 0., 'dist_min': 1e9,
                                    'bodies': r['bodies']})
        g['count'] += 1; g['t_min'] = min(g['t_min'], r['t']); g['t_max'] = max(g['t_max'], r['t'])
        g['force_max'] = max(g['force_max'], r['force_norm']); g['dist_min'] = min(g['dist_min'], r['dist'])
    return {'n_contact_steps': len(records), 'pairs': groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='CTC')
    ap.add_argument('--scenario', default='payload')
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--trajectory', type=int, default=0)
    ap.add_argument('--config', default=None)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    selected = json.loads((sim.ROOT / 'results/fair_tuning/selected.json').read_text())
    config = json.loads(args.config) if args.config else selected[args.method]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f'{args.method}_{args.scenario}_s{args.seed}_t{args.trajectory}'
    records = run(args.method, args.scenario, args.seed, args.trajectory, config, OUT)
    summary = summarize(records)
    (OUT / f'{tag}.json').write_text(json.dumps(
        {'method': args.method, 'scenario': args.scenario, 'seed': args.seed,
         'trajectory': args.trajectory, 'config': config, 'summary': summary,
         'contacts': records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
