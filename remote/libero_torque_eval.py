"""Run LIBERO evaluation with a custom joint-torque controller.

Usage:
  python libero_torque_eval.py --method CTC --trials 1
The VLA policy and action interface are unchanged; only the low-level
robosuite controller is replaced after each environment is created.
"""
import argparse, json, sys
from pathlib import Path

import run_libero_eval as rle
from libero_torque_wrapper import install


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="CTC", choices=["OSC", "CTC", "SMC", "RBF_SMC", "RBF_NO_ROBUST", "RBF_FROZEN"])
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--checkpoint", default="/root/models/openvla-oft-libero-spatial")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--task-limit", type=int, default=None, help="debug: only run the first N tasks")
    ap.add_argument("--config", default="/root/selected.json")
    args = ap.parse_args()
    configs = json.loads(Path(args.config).read_text())
    config = configs.get(args.method, configs["RBF_SMC"])

    cfg = rle.GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.trials,
        use_wandb=False,
        center_crop=True,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    original = rle.get_libero_env

    def patched(task, model_family, resolution=256):
        env, description = original(task, model_family, resolution=resolution)
        if args.method != "OSC":
            install(env, args.method, config)
        return env, description

    rle.get_libero_env = patched
    if args.task_limit is not None:
        original_benchmark = rle.benchmark.get_benchmark_dict

        def limited_benchmark():
            suites = original_benchmark()
            suite_cls = suites[args.task_suite]

            class LimitedSuite(suite_cls):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.n_tasks = min(self.n_tasks, args.task_limit)

            suites[args.task_suite] = LimitedSuite
            return suites

        rle.benchmark.get_benchmark_dict = limited_benchmark
    print(f"Running {args.method} with config {config} task_limit={args.task_limit}", flush=True)
    return rle.eval_libero.__wrapped__(cfg)


if __name__ == "__main__":
    main()
