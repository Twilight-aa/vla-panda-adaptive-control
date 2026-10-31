"""Equal episode-budget development search, then locked held-out evaluation.

Usage:
  python tune_controllers.py                 # writes results/fair_tuning
  python tune_controllers.py --out results/fair_tuning_v2
"""
import argparse, csv, json, hashlib, platform
from pathlib import Path
import numpy as np
import run_panda_local as sim

METHODS = ['CTC', 'SMC', 'RBF_SMC']


def build_candidates():
    rng = np.random.default_rng(271828)
    candidates = {}
    candidates['CTC'] = [{'ctc_kp': 64., 'ctc_kd': 16.}] + [
        {'ctc_kp': float(w * w), 'ctc_kd': float(2 * z * w)}
        for w, z in zip([6, 10, 14, 18, 22, 26, 30], [1, 1, .9, 1.1, 1, .9, 1.1])]
    for method in METHODS[1:]:
        candidates[method] = [{}]
        for _ in range(7):
            c = {'ks_scale': float(rng.uniform(.7, 2.5)), 'rho_scale': float(rng.uniform(.5, 2)),
                 'phi': float(rng.uniform(.10, .3)), 'lambda': float(rng.uniform(6, 14))}
            if method == 'RBF_SMC':
                c['gamma'] = float(rng.uniform(4, 24))
            candidates[method].append(c)
    return candidates


def write(name, rows, out):
    with (out / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ref2(t):
    w = 2 * np.pi / 4.7
    a = sim.AMP * 1.25
    phase = np.arange(7) * .2
    return sim.HOME + a * np.sin(w * t + phase), a * w * np.cos(w * t + phase), -a * w * w * np.sin(w * t + phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/fair_tuning')
    ap.add_argument('--development-only', action='store_true')
    args = ap.parse_args()
    out = (sim.ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    protocol = {
        'development_seeds': [1, 2], 'test_seeds': [21, 22, 23],
        'development_scenarios': ['nominal', 'combined'],
        'test_scenarios': ['nominal', 'payload', 'friction', 'pulse', 'combined', 'unseen_payload', 'low_torque'],
        'candidates_per_method': 8, 'dev_episodes_per_method': 32,
        'shared_torque_limits': sim.LIMIT.tolist(),
        'objective': 'mean ee_rmse_m*1000 + 0.1*torque_rms_Nm + 100*saturation_fraction',
        'candidate_grid': candidates,
        'note': 'Equal episode budget, not exhaustive equal-dimensional search. Test scores cannot select gains.',
        'initial_state_note': 'setup() restores the home keyframe after mj_setConst so the gripper starts open; contact audit columns are recorded.',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []; chosen = {}
    for method in METHODS:
        scores = []
        for i, c in enumerate(candidates[method]):
            results = []
            for seed in [1, 2]:
                for sc in ['nominal', 'combined']:
                    result = sim.run(method, sc, seed, out, options={**c, 'save_traces': False})
                    score = result['ee_rmse_m'] * 1000 + .1 * result['torque_rms_Nm'] + 100 * result['saturation_fraction']
                    results.append(score)
                    rows.append(dict(candidate=i, objective=score, **result))
            scores.append(float(np.mean(results)))
            print(method, i, scores[-1], flush=True)
        chosen[method] = candidates[method][int(np.argmin(scores))]
        write('development.csv', rows, out)
    (out / 'selected.json').write_text(json.dumps(chosen, indent=2))
    if args.development_only:
        return
    # Trajectory 0 is original; trajectory 1 changes frequency, amplitudes and offset analytically.
    original_ref = sim.ref
    rows = []
    try:
        for trajectory, reference in enumerate([original_ref, ref2]):
            sim.ref = reference
            for seed in protocol['test_seeds']:
                for sc in protocol['test_scenarios']:
                    for method in METHODS:
                        result = sim.run(method, sc, seed, out,
                                         options={**chosen[method], 'initial_std': .02, 'save_traces': False})
                        rows.append(dict(trajectory=trajectory, **result))
                write('heldout.csv', rows, out)
                print('test', trajectory, seed, len(rows), flush=True)
    finally:
        sim.ref = original_ref
    files = [Path(__file__), Path(sim.__file__), Path(__file__).with_name('test_observation_noise.py'),
             Path(__file__).with_name('controllers.py')]
    (out / 'source_hashes.json').write_text(json.dumps(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))


if __name__ == '__main__':
    main()
