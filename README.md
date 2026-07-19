# RL

My reinforcement-learning workspace — where I **learn RL by reading the papers, implementing the
algorithms, and running my own experiments**: policy gradients and baselines, PPO, Hindsight
Experience Replay (HER), DPO/RLHF, and value-based methods — then **pushing on performance**
(parallel experiment grids, stabler training, cleaner rollout/render tooling).

The goal isn't to ship a library; it's to build intuition end-to-end: read the algorithm, code it,
run it across seeds and environments, plot what happened, and iterate.

## What I built

- **SAC + Hindsight Experience Replay** (`Her.py`) — Soft Actor-Critic with HER on a pointmass /
  sparse-reward task, including a visualization of the relabeling mechanics
  (`her_*.png`, `her_relabel_animation.gif`, `reward_field.png`).
- **A custom racing environment** (`race_train.py`, `make_maps.py`) — procedural map generation and
  training, with rendered frames, dashboards, and animations under **`raceviz/`**.
- **Policy gradient study (`CS234-Reinforcement-Learning/A2_Policy_Gradient_Methods/`)** — my
  implementations of REINFORCE, advantage-baseline PG, and PPO, compared head-to-head on
  `{cartpole, pendulum, cheetah}` (PyBullet). I added a **parallel experiment runner**
  (`run_experiments.py`, a 3×3×3 = 27-run grid), analysis + plotting (`analyze.py`, `plot.py`),
  model save/replay, and rendering (`play.py`).
- **RLHF & DPO (`.../A3_RLHF_DPO/`)** — PPO on Hopper (`ppo_hopper.py`), reward-model RLHF
  (`run_rlhf.py`), and Direct Preference Optimization (`run_dpo.py`), with results under `results*/`.
- **Value-based methods** — additional experiments on the full Rainbow DQN stack
  (`rainbow-is-all-you-need/`): DQN → Double DQN → Prioritized Replay → Dueling → Noisy Nets →
  Categorical (C51) → n-step → Rainbow → Rainbow-IQN, worked through one component at a time.
- **Study material** — `STUDY_GUIDE.html` / `CS234_Study_Guide.html`, my full walkthrough of the
  assignments (questions, proofs, and code).

## Themes

- **Policy gradients, from the ground up** — REINFORCE, advantage baselines, and PPO's clipped
  objective, implemented and compared on the same environments.
- **Sparse-reward learning** — SAC + HER, with relabeling made legible.
- **Preference-based learning (RLHF / DPO)** — reward modeling and DPO on continuous control.
- **Value-based methods** — the Rainbow DQN components, isolated and run individually.
- **Optimization & tooling** — parallel experiment grids, model save/replay, device fixes, and
  rendering: the unglamorous work that makes experiments reproducible and fast.

## Setup

A local virtual environment (`.venv/`) is intentionally excluded from version control. Recreate it
with your preferred tool and install the dependencies referenced by the scripts (PyTorch, NumPy,
Matplotlib, PyBullet, etc.). Some folders ship their own `requirements.txt`.

## Credits

Built on two open-source bases I picked up and ran my own experiments on: the CS234 assignment
scaffolding from [Rhyme0730/CS234-Reinforcement-Learning](https://github.com/Rhyme0730/CS234-Reinforcement-Learning)
and the Rainbow DQN implementations from [Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need)
(MIT). See the `LICENSE`/`README` files within those directories for their terms.
