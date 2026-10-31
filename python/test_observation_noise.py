"""Observation-noise check with locked gains; no retuning."""
import argparse, csv, json
from pathlib import Path
import run_panda_local as sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/fair_tuning')
    ap.add_argument('--selected', default=None,
                    help='selected.json path; defaults to <out>/selected.json')
    args = ap.parse_args()
    out = (sim.ROOT / args.out).resolve()
    selected_path = Path(args.selected) if args.selected else out / 'selected.json'
    selected = json.loads(selected_path.read_text())
    rows = []
    for std in [0, .001, .005]:
        for seed in [41, 42, 43]:
            for method in ['CTC', 'SMC', 'RBF_SMC']:
                r = sim.run(method, 'combined', seed, out, options={
                    **selected[method], 'noise_std': std, 'save_traces': False, 'initial_std': .02})
                rows.append(dict(noise_std=std, **r))
        with (out / 'noise_test.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(std, flush=True)


if __name__ == '__main__':
    main()
