# -*- coding: UTF-8 -*-
"""Summarize the A2 experiment grid: final performance per env x method across seeds.

Reads results/<GymEnv>-<method>-seed=<s>/scores.npy for method in
{baseline, no_baseline, ppo} and prints (and writes) a table of the mean final
reward +/- std across seeds. "Final reward" = mean of the last `--tail` iterations
of each run (less noisy than the single last point).

Usage:
    python analyze.py --seeds 1,2,3
    python analyze.py --envs cartpole,pendulum,cheetah --seeds 1,2,3 --tail 20
"""
import argparse
import os

import numpy as np

ENV_MAP = {
    "cartpole": "CartPoleBulletEnv-v1",
    "pendulum": "InvertedPendulumBulletEnv-v0",
    "cheetah": "HalfCheetahBulletEnv-v0",
}
METHODS = ["baseline", "no_baseline", "ppo"]
LABEL = {"baseline": "PG + baseline", "no_baseline": "PG no-baseline", "ppo": "PPO"}


def final_reward(path, tail):
    scores = np.load(path)
    return float(np.mean(scores[-tail:])) if len(scores) >= tail else float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", default="results")
    ap.add_argument("--envs", default="cartpole,pendulum,cheetah")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--tail", type=int, default=20, help="avg over last N iterations")
    ap.add_argument("--out", default="results/analysis.txt")
    args = ap.parse_args()

    envs = args.envs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    lines = []
    header = "Final reward (mean of last {} iters), averaged over seeds {}".format(
        args.tail, seeds)
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("{:10s} {:16s} {:>16s}  {}".format("env", "method", "mean +/- std", "n_seeds"))
    lines.append("-" * 60)

    for env in envs:
        gym_name = ENV_MAP[env]
        for method in METHODS:
            vals = []
            for s in seeds:
                p = os.path.join(args.directory, "{}-{}-seed={}".format(gym_name, method, s), "scores.npy")
                if os.path.exists(p):
                    vals.append(final_reward(p, args.tail))
            if vals:
                lines.append("{:10s} {:16s} {:>8.1f} +/- {:5.1f}  {}".format(
                    env, LABEL[method], float(np.mean(vals)), float(np.std(vals)), len(vals)))
            else:
                lines.append("{:10s} {:16s} {:>16s}  0  (no runs found)".format(env, LABEL[method], "--"))
        lines.append("-" * 60)

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print("\nWrote {}".format(args.out))


if __name__ == "__main__":
    main()
