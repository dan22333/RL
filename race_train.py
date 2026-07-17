"""
Live 3-agent "race": train Dense SAC / Sparse SAC / Sparse SAC+HER in parallel
processes and watch them try to reach the SAME goals as training progresses.

- 3 worker processes each train one agent (loop reuses Her.py's env/agent/buffer).
- Every `eval_every` episodes each worker rolls out a fixed shared set of start/goal
  pairs (deterministic, identical across agents -> a fair race) and dumps the paths
  plus training metrics (q1_loss, actor_loss, policy entropy, HER relabel-success)
  to a shared folder as an atomic pickle.
- The main process composites a 2-row x 3-col dashboard per checkpoint:
      top row    = rollout paths (dot start o, goal star, success disc)
      bottom row = q1_loss / actor_loss / entropy (+ HER relabel-success on HER panel)
  It writes dashboard_latest.png live, and stitches all frames into training_race.gif.

Run:
    python race_train.py                      # full (350 eps)
    python race_train.py --episodes 30 --eval-every 10   # fast smoke test
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import Her

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raceviz")
N_PAIRS = 5          # start/goal pairs shown per panel
PAIR_SEED = 2024     # deterministic shared scenarios


# ---------------------------------------------------------------------------
# Shared fixed eval scenarios (identical for all three agents -> fair race)
# ---------------------------------------------------------------------------

def make_pairs(n, seed, bound=1.0):
    rng = np.random.RandomState(seed)
    starts = rng.uniform(-bound, bound, size=(n, 2)).astype(np.float32)
    goals = rng.uniform(-bound, bound, size=(n, 2)).astype(np.float32)
    return [(starts[i], goals[i]) for i in range(n)]


def rollout_fixed(agent, max_steps, pairs):
    """Deterministic rollouts from preset start/goal pairs; return (path, goal, success)."""
    env = Her.PointGoalEnv(reward_type="sparse", max_steps=max_steps)  # reward unused here
    out = []
    for start, goal in pairs:
        env.t = 0
        env.pos = start.copy()
        env.goal = goal.copy()
        obs = env._obs()
        path = [env.pos.copy()]
        success = False
        for _ in range(max_steps):
            action = agent.act(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            path.append(env.pos.copy())
            success = success or info["is_success"]
            if terminated or truncated:
                break
        out.append((np.array(path), goal.copy(), bool(success)))
    return out


def atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker: train one agent, checkpoint rollouts + metrics
# ---------------------------------------------------------------------------

def train_worker(agent_name, reward_type, use_her, cfg, pairs, out_dir, seed):
    import torch
    torch.set_num_threads(1)
    Her.set_seed(seed)

    env = Her.PointGoalEnv(reward_type=reward_type, max_steps=cfg["max_steps"])
    agent = Her.SACAgent(
        obs_goal_dim=4, action_dim=2, hidden_dim=cfg["hidden_dim"], lr=cfg["lr"],
        gamma=cfg["gamma"], tau=cfg["tau"], alpha=cfg["alpha"], device="cpu",
    )
    buffer = Her.EpisodeReplayBuffer(
        capacity_episodes=cfg["buffer_episodes"], env=env, use_her=use_her,
        n_sampled_goal=cfg["n_sampled_goal"], device="cpu",
    )

    hist = {"episodes": [], "q1_loss": [], "actor_loss": [], "entropy": [], "her_success": []}
    total_steps = 0
    checkpoints = [1] + list(range(cfg["eval_every"], cfg["episodes"] + 1, cfg["eval_every"]))

    for ep in range(1, cfg["episodes"] + 1):
        obs = env.reset()
        episode = []
        for _t in range(env.max_steps):
            if total_steps < cfg["warmup_steps"]:
                action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append(Her.Transition(
                obs=obs["observation"], achieved_goal=obs["achieved_goal"],
                desired_goal=obs["desired_goal"], action=action.astype(np.float32),
                reward=float(reward), next_obs=next_obs["observation"],
                next_achieved_goal=next_obs["achieved_goal"], done=float(terminated),
            ))
            obs = next_obs
            total_steps += 1
            if terminated or truncated:
                break
        buffer.add_episode(episode)

        ep_q1, ep_actor, ep_ent, ep_her = [], [], [], []
        if len(buffer) >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_episode"]):
                batch = buffer.sample(cfg["batch_size"])
                stats = agent.update(batch)
                ep_q1.append(stats["q1_loss"])
                ep_actor.append(stats["actor_loss"])
                with __import__("torch").no_grad():
                    _, logp = agent.actor.sample(batch["obs"])
                    ep_ent.append(float((-logp.mean()).item()))
                if use_her:
                    ep_her.append(float(batch["her_success_fraction"].item()))

        if ep_q1:
            hist["episodes"].append(ep)
            hist["q1_loss"].append(float(np.mean(ep_q1)))
            hist["actor_loss"].append(float(np.mean(ep_actor)))
            hist["entropy"].append(float(np.mean(ep_ent)))
            hist["her_success"].append(float(np.mean(ep_her)) if ep_her else float("nan"))

        if ep in checkpoints:
            rollouts = rollout_fixed(agent, env.max_steps, pairs)
            n_succ = sum(int(s) for _, _, s in rollouts)
            payload = {
                "agent": agent_name, "episode": ep, "rollouts": rollouts,
                "n_succ": n_succ, "n_pairs": len(pairs),
                "hist": {k: list(v) for k, v in hist.items()},
            }
            atomic_dump(payload, os.path.join(out_dir, f"{agent_name}_ep{ep:04d}.pkl"))
            print(f"[{agent_name}] episode={ep:4d} reached {n_succ}/{len(pairs)}", flush=True)


# ---------------------------------------------------------------------------
# Compositor (runs in main process): render a dashboard per checkpoint
# ---------------------------------------------------------------------------

def render_frame(ep, panels, pairs, out_dir, thr=0.06):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["Dense SAC", "Sparse SAC", "Sparse SAC + HER"]
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(pairs)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [1.35, 1.0]})

    for col, name in enumerate(order):
        p = panels[name]
        # ---- top: rollout paths ----
        ax = axes[0][col]
        for i, (path, goal, success) in enumerate(p["rollouts"]):
            c = colors[i]
            ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.8, alpha=0.9, zorder=2)
            ax.plot(path[0, 0], path[0, 1], "o", color=c, ms=8, zorder=3)
            ax.plot(goal[0], goal[1], "*", color=c, ms=15, mec="k", mew=0.5, zorder=4)
            ax.add_patch(plt.Circle((goal[0], goal[1]), thr, color=c, alpha=0.18, zorder=1))
            ax.plot(path[-1, 0], path[-1, 1], "x", color=c, ms=8, mew=2, zorder=3)
        ax.set_title(f"{name}\nep {ep} — reached {p['n_succ']}/{p['n_pairs']}", fontsize=12)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

        # ---- bottom: training internals ----
        bx = axes[1][col]
        h = p["hist"]
        eps = h["episodes"]
        if eps:
            bx.plot(eps, h["q1_loss"], color="tab:red", lw=1.5, label="q1_loss")
            bx.plot(eps, h["actor_loss"], color="tab:blue", lw=1.5, label="actor_loss")
            bx.set_xlabel("episode"); bx.grid(True, alpha=0.3)
            bx2 = bx.twinx()
            bx2.plot(eps, h["entropy"], color="tab:green", lw=1.3, ls="--", label="entropy")
            if name == "Sparse SAC + HER":
                bx2.plot(eps, h["her_success"], color="tab:purple", lw=1.3, ls=":",
                         label="HER relabel-success")
            bx2.set_ylim(-2.5, 1.2)
            l1, la = bx.get_legend_handles_labels()
            l2, lb = bx2.get_legend_handles_labels()
            bx.legend(l1 + l2, la + lb, fontsize=8, loc="upper right")
        bx.set_xlim(0, max(1, ep))

    fig.suptitle(f"3-agent race — same {len(pairs)} start/goal pairs — episode {ep}",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    frame_path = os.path.join(out_dir, "frames", f"frame_{ep:04d}.png")
    fig.savefig(frame_path, dpi=110)
    fig.savefig(os.path.join(out_dir, "dashboard_latest.png"), dpi=110)
    plt.close(fig)
    return frame_path


def wait_for(paths, procs, timeout=180.0):
    """Wait until every path exists; abort if all workers died first."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(os.path.exists(p) for p in paths):
            return True
        if all(not pr.is_alive() for pr in procs) and not all(os.path.exists(p) for p in paths):
            # workers finished/crashed without producing this checkpoint
            return all(os.path.exists(p) for p in paths)
        time.sleep(0.3)
    return all(os.path.exists(p) for p in paths)


def build_gif(out_dir, checkpoints, fps=2):
    from PIL import Image
    files = [os.path.join(out_dir, "frames", f"frame_{ep:04d}.png") for ep in checkpoints]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("No frames to stitch.")
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    gif_path = os.path.join(out_dir, "training_race.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"\nSaved GIF: {gif_path}")


def main():
    args = Her.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)
    # clear stale checkpoints
    for f in os.listdir(OUT_DIR):
        if f.endswith(".pkl"):
            os.remove(os.path.join(OUT_DIR, f))

    pairs = make_pairs(N_PAIRS, PAIR_SEED)
    cfg = dict(
        episodes=args.episodes, max_steps=args.max_steps, eval_every=args.eval_every,
        updates_per_episode=args.updates_per_episode, batch_size=args.batch_size,
        buffer_episodes=args.buffer_episodes, warmup_steps=args.warmup_steps,
        n_sampled_goal=args.n_sampled_goal, hidden_dim=args.hidden_dim, lr=args.lr,
        gamma=args.gamma, tau=args.tau, alpha=args.alpha,
    )
    conditions = [
        ("Dense SAC", "dense", False),
        ("Sparse SAC", "sparse", False),
        ("Sparse SAC + HER", "sparse", True),
    ]

    ctx = mp.get_context("spawn")
    procs = []
    for idx, (name, reward_type, use_her) in enumerate(conditions):
        pr = ctx.Process(target=train_worker,
                         args=(name, reward_type, use_her, cfg, pairs, OUT_DIR,
                               args.seed + idx * 100))
        pr.start()
        procs.append(pr)

    checkpoints = [1] + list(range(args.eval_every, args.episodes + 1, args.eval_every))
    order = ["Dense SAC", "Sparse SAC", "Sparse SAC + HER"]
    done = []
    for ep in checkpoints:
        paths = [os.path.join(OUT_DIR, f"{n}_ep{ep:04d}.pkl") for n in order]
        if not wait_for(paths, procs):
            print(f"Timed out / workers gone before episode {ep}; stopping compositor.")
            break
        panels = {}
        for n, p in zip(order, paths):
            with open(p, "rb") as f:
                panels[n] = pickle.load(f)
        render_frame(ep, panels, pairs, OUT_DIR)
        done.append(ep)
        print(f"composited frame ep={ep}", flush=True)

    for pr in procs:
        pr.join()
    build_gif(OUT_DIR, done)
    print(f"\nLive dashboard: {os.path.join(OUT_DIR, 'dashboard_latest.png')}")


if __name__ == "__main__":
    main()
