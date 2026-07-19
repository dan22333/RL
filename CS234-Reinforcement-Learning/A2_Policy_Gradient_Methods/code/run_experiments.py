# -*- coding: UTF-8 -*-
"""Run the full A2 experiment grid in parallel across CPU cores.

Grid: {cartpole, pendulum, cheetah} x {baseline, no_baseline, ppo} x seeds.
Each run is pinned to a single BLAS/OMP thread (these tiny MLPs are bound by
serial PyBullet env-stepping, not matrix math), so N single-thread processes on
N cores give ~N x throughput. Cheetah jobs are scheduled first (longest pole).

Usage:
    python run_experiments.py --seeds 1,2,3 --jobs 6
    python run_experiments.py --envs cartpole,pendulum --seeds 1,2,3 --jobs 6
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable  # the same interpreter that launched this (cs234a2)

# method -> extra main.py flags
METHOD_FLAGS = {
    "baseline": ["--baseline"],
    "no_baseline": ["--no-baseline"],
    "ppo": ["--ppo"],
}
# rough relative cost, to schedule the slow ones first
ENV_COST = {"cheetah": 3, "pendulum": 2, "cartpole": 1}


def build_jobs(envs, methods, seeds):
    jobs = []
    for env in envs:
        for method in methods:
            for seed in seeds:
                jobs.append((env, method, seed))
    # slowest env first so long jobs start early and pack well
    jobs.sort(key=lambda j: -ENV_COST[j[0]])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="cartpole,pendulum,cheetah")
    ap.add_argument("--methods", default="baseline,no_baseline,ppo")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--jobs", type=int, default=6, help="max concurrent processes")
    ap.add_argument("--logdir", default="results/_runlogs")
    args = ap.parse_args()

    envs = args.envs.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = build_jobs(envs, methods, seeds)
    os.makedirs(args.logdir, exist_ok=True)

    single_thread = dict(os.environ)
    single_thread.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    )

    print("Total jobs: {} | concurrency: {}".format(len(jobs), args.jobs), flush=True)
    running = []   # list of (proc, tag, logfile_handle, start_time)
    done, failed = [], []
    queue = list(jobs)

    def launch(job):
        env, method, seed = job
        tag = "{}-{}-seed={}".format(env, method, seed)
        cmd = [PY, "main.py", "--env-name", env, "--seed", str(seed)] + METHOD_FLAGS[method]
        lf = open(os.path.join(args.logdir, tag + ".log"), "w")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=single_thread)
        print("  START {}".format(tag), flush=True)
        return (proc, tag, lf, time.time())

    while queue or running:
        while queue and len(running) < args.jobs:
            running.append(launch(queue.pop(0)))
        still = []
        for proc, tag, lf, t0 in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, tag, lf, t0))
            else:
                lf.close()
                mins = (time.time() - t0) / 60.0
                if rc == 0:
                    done.append(tag)
                    print("  DONE  {} ({:.1f} min)  [{}/{}]".format(tag, mins, len(done), len(jobs)), flush=True)
                else:
                    failed.append(tag)
                    print("  FAIL  {} (rc={})".format(tag, rc), flush=True)
        running = still
        time.sleep(2)

    print("\nAll finished. ok={} failed={}".format(len(done), len(failed)), flush=True)
    if failed:
        print("FAILED:", ", ".join(failed), flush=True)

    # Auto-generate comparison plots + analysis table (best-effort).
    seeds_arg = ",".join(str(s) for s in seeds)
    if set(methods) == set(METHOD_FLAGS):  # plot.py needs all three methods
        for env in envs:
            try:
                subprocess.run([PY, "plot.py", "-d", "results", "--env-name", env,
                                "--seeds", seeds_arg], check=True)
                print("  plotted results/results-{}.png".format(env), flush=True)
            except Exception as e:
                print("  plot failed for {}: {}".format(env, e), flush=True)
    try:
        subprocess.run([PY, "analyze.py", "--envs", ",".join(envs),
                        "--seeds", seeds_arg], check=True)
    except Exception as e:
        print("  analyze failed: {}".format(e), flush=True)


if __name__ == "__main__":
    main()
