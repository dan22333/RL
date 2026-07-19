# RL

A personal reinforcement-learning workspace for **learning RL by reading the papers and
implementing them** — policy gradients and baselines, PPO, Hindsight Experience Replay (HER),
DPO/RLHF, and value-based methods — and then **trying to make them run better** (faster grids,
stabler training, cleaner rollout/render tooling).

The goal isn't to ship a library; it's to build intuition end-to-end: read the algorithm, code it,
run it across seeds and environments, plot what happened, and iterate.

## Themes

- **Policy gradients, from the ground up.** REINFORCE, advantage baselines, and PPO's clipped
  objective — implemented and compared head-to-head on the same environments.
- **Sparse-reward learning.** SAC + Hindsight Experience Replay on a pointmass task, with
  relabeling visualized so the mechanics are legible.
- **Preference-based learning (RLHF / DPO).** Reward modeling and Direct Preference Optimization
  on a continuous-control task.
- **Value-based methods.** The full Rainbow DQN stack, worked through one component at a time.
- **Optimization & tooling.** Parallel experiment grids, model save/replay, device fixes, and
  rendering — the unglamorous work that makes experiments reproducible and fast.

## Contents

### My own work
- **`Her.py`** — SAC + HER on a pointmass / sparse-reward task. Result figures and the relabeling
  animation live at the repo root (`her_*.png`, `her_relabel_animation.gif`, `reward_field.png`).
- **`race_train.py`**, **`make_maps.py`** — training and procedural map generation for a custom
  racing environment; snapshots, rendered frames, dashboards, and animations under **`raceviz/`**.

### Papers implemented & experimented with (in `CS234-Reinforcement-Learning/`)
- **A2 — Policy Gradient Methods.** REINFORCE, advantage-baseline PG, and PPO on
  `{cartpole, pendulum, cheetah}` (PyBullet). Includes a parallel experiment runner
  (`run_experiments.py`), analysis/plotting (`analyze.py`, `plot.py`), and trained-policy
  replay/render (`play.py`). Grid design: 3 envs × 3 methods × 3 seeds = 27 runs.
- **A3 — RLHF & DPO.** PPO on Hopper (`ppo_hopper.py`), reward-model RLHF (`run_rlhf.py`), and
  Direct Preference Optimization (`run_dpo.py`), with results under `results*/`.
- **A1 — Tabular MDPs.** RiverSwim / exact-solution warm-up.
- **`STUDY_GUIDE.html` / `CS234_Study_Guide.html`** — a full walkthrough of the assignments
  (questions, proofs, and code) used as a study companion.

### Value-based methods, one piece at a time (`rainbow-is-all-you-need/`)
DQN → Double DQN → Prioritized Experience Replay → Dueling → Noisy Nets → Categorical (C51) →
n-step → Rainbow → Rainbow-IQN, each as a standalone, runnable script.

## Attribution

The `CS234-Reinforcement-Learning/` and `rainbow-is-all-you-need/` directories are third-party
repositories, included here as the scaffolding I read, implemented against, and experimented on.
**Original credit belongs to their authors** —
[Rhyme0730/CS234-Reinforcement-Learning](https://github.com/Rhyme0730/CS234-Reinforcement-Learning)
and [Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need) —
see any `LICENSE`/`README` files within those directories for their terms. Everything at the
repository root and under `raceviz/` is my own work.

## Setup

A local virtual environment (`.venv/`) is intentionally excluded from version control. Recreate it
with your preferred tool and install the dependencies referenced by the scripts (PyTorch, NumPy,
Matplotlib, PyBullet, etc.). Some assignment folders ship their own `requirements.txt`.
