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
    ap.add_argument("--method", default="CTC", choices=["CTC", "SMC", "RBF_SMC", "RBF_NO_ROBUST", "RBF_FROZEN"])
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--checkpoint", default="/root/models/openvla-oft-libero-spatial")
    ap.add_argument("--trials", type=int, default=1)
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
        install(env, args.method, config)
        return env, description

    rle.get_libero_env = patched
    print(f"Running {args.method} with config {config}", flush=True)
    return rle.eval_libero.__wrapped__(cfg)


if __name__ == "__main__":
    main()
