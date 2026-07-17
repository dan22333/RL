# RL

A personal reinforcement-learning workspace: experiments with Soft Actor-Critic (SAC) and
Hindsight Experience Replay (HER), a racing environment, and study materials for
Stanford's CS234.

## Contents

### My own work
- **`Her.py`** — SAC + HER on a pointmass / sparse-reward task.
- **`race_train.py`**, **`make_maps.py`** — training and map-generation for a racing environment.
- **`raceviz/`** — training snapshots, rendered frames, dashboards, and animations from the racing experiments.
- Top-level result figures and animations (`her_*.png`, `her_relabel_animation.gif`, `reward_field.png`, …).

### Third-party material (included for reference/study — see Attribution)
- **`CS234-Reinforcement-Learning/`** — Stanford CS234 (Spring 2024) assignment code, cloned from
  [Rhyme0730/CS234-Reinforcement-Learning](https://github.com/Rhyme0730/CS234-Reinforcement-Learning).
  Also contains **`STUDY_GUIDE.html`**, a full walkthrough of Assignments 1–3 (questions, proofs, and code).
- **`rainbow-is-all-you-need/`** — a Rainbow DQN tutorial, cloned from
  [Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need).

## Attribution

The `CS234-Reinforcement-Learning/` and `rainbow-is-all-you-need/` directories are third-party
repositories, included here for personal reference and study. **All credit for that code belongs to
its original authors** (linked above); please refer to any `LICENSE`/`README` files within those
directories for their terms. Everything at the repository root and under `raceviz/` is my own work.

## Setup

A local virtual environment (`.venv/`) is intentionally excluded from version control. Recreate it with
your preferred tool and install the dependencies referenced by the scripts (PyTorch, NumPy, Matplotlib, etc.).
