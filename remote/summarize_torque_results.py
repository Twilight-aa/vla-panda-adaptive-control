#!/usr/bin/env python3
"""Summarize LIBERO torque-controller evaluation logs.

The evaluator writes rich-formatted logs whose "Final results" block is split
across lines and wrapped with ANSI/padding.  Parse line by line instead of
matching one large regex; the old single-regex version silently produced
{"error": "no final result"} for every method.
"""
import argparse
import json
import math
import re
from pathlib import Path

METHODS = ["CTC", "SMC", "RBF_SMC", "RBF_NO_ROBUST", "RBF_FROZEN"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def parse_log(path):
    if not path.exists():
        return {"error": "missing log", "path": str(path)}
    lines = path.read_text(errors="replace").splitlines()
    episodes = successes = None
    overall_rate = None
    per_task = []
    for i, line in enumerate(lines):
        m = re.search(r">> Total episodes:\s*(\d+)", line)
        if m:
            episodes = int(m.group(1))
        m = re.search(r">> Total successes:\s*(\d+)", line)
        if m:
            successes = int(m.group(1))
        m = re.search(r">> Overall success rate:\s*([0-9.]+)", line)
        if m:
            overall_rate = float(m.group(1))
        if "Current task success rate" in line:
            for nxt in lines[i + 1 : i + 6]:
                # The value is on its own padded line: "1.0" or "0.96 (96.0%)".
                m = re.match(r"^\s*([01](?:\.\d+)?)\s*(?:\(|$)", nxt)
                if m:
                    per_task.append(float(m.group(1)))
                    break
    m = re.search(r"EXIT:(\d+)", "\n".join(lines))
    exit_code = int(m.group(1)) if m else None
    rate = successes / episodes if episodes and successes is not None else overall_rate
    return {
        "episodes": episodes,
        "successes": successes,
        "rate": rate,
        "wilson95": wilson(successes, episodes) if episodes and successes is not None else None,
        "per_task": per_task,
        "exit_code": exit_code,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--prefix", default="eval_torque_")
    ap.add_argument("--suffix", default="_t50.log")
    ap.add_argument("--baseline", default=None, help="optional path to the original OSC log")
    ap.add_argument("--out", default="torque_core_summary.json")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out = {}
    for method in METHODS:
        out[method] = parse_log(log_dir / f"{args.prefix}{method}{args.suffix}")
    if args.baseline:
        out["OSC"] = parse_log(Path(args.baseline))

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
