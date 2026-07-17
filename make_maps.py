"""
Generate 2D trajectory maps for the three agents from Her.py.

Trains Dense SAC / Sparse SAC / Sparse SAC + HER (reusing Her.py's env, agent, and
replay buffer, with the same hyperparameters/seeds as Her.main()), then rolls out each
deterministic policy on a shared set of eval start/goal pairs and plots the point-mass
paths (start -> trajectory -> goal) in one 3-panel figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import Her


def train_agent(name, reward_type, use_her, args, device):
    """Same loop as Her.train_condition, but returns the trained agent."""
    print(f"\n=== Training {name} ===")
    env = Her.PointGoalEnv(reward_type=reward_type, max_steps=args.max_steps)
    agent = Her.SACAgent(
        obs_goal_dim=4, action_dim=2, hidden_dim=args.hidden_dim, lr=args.lr,
        gamma=args.gamma, tau=args.tau, alpha=args.alpha, device=device,
    )
    buffer = Her.EpisodeReplayBuffer(
        capacity_episodes=args.buffer_episodes, env=env, use_her=use_her,
        n_sampled_goal=args.n_sampled_goal, device=device,
    )
    total_steps = 0
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        episode = []
        for _t in range(env.max_steps):
            if total_steps < args.warmup_steps:
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
        if len(buffer) >= args.batch_size:
            for _ in range(args.updates_per_episode):
                agent.update(buffer.sample(args.batch_size))
        if ep % args.eval_every == 0 or ep == 1:
            sr, ar = Her.evaluate(agent, reward_type=reward_type, episodes=args.eval_episodes)
            print(f"{name:16s} episode={ep:4d} eval_success={sr:.2f} avg_return={ar:.1f}")
    return agent


def rollout_paths(agent, reward_type, n_episodes, max_steps, seed):
    """Roll out the deterministic policy; return list of (path, goal, success)."""
    env = Her.PointGoalEnv(reward_type=reward_type, max_steps=max_steps)
    old_state = np.random.get_state()
    np.random.seed(seed)  # identical start/goal pairs across all three agents
    episodes = []
    for _ in range(n_episodes):
        obs = env.reset()
        path = [env.pos.copy()]
        goal = env.goal.copy()
        success = False
        for _t in range(env.max_steps):
            action = agent.act(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            path.append(env.pos.copy())
            success = success or info["is_success"]
            if terminated or truncated:
                break
        episodes.append((np.array(path), goal, success))
    np.random.set_state(old_state)
    return episodes


def main():
    args = Her.parse_args()
    n_maps = 6
    map_seed = 2024

    Her.torch.set_num_threads(1)
    device = "cpu"
    print(f"Using device: {device}")

    conditions = [
        ("Dense SAC", "dense", False),
        ("Sparse SAC", "sparse", False),
        ("Sparse SAC + HER", "sparse", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, n_maps))
    thr = 0.06  # env success_threshold

    for idx, (name, reward_type, use_her) in enumerate(conditions):
        Her.set_seed(args.seed + idx * 100)
        agent = train_agent(name, reward_type, use_her, args, device)
        eps = rollout_paths(agent, reward_type, n_maps, args.max_steps, map_seed)

        ax = axes[idx]
        n_succ = 0
        for i, (path, goal, success) in enumerate(eps):
            c = colors[i]
            ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.8, alpha=0.9, zorder=2)
            ax.plot(path[0, 0], path[0, 1], "o", color=c, ms=8, zorder=3)      # start
            ax.plot(goal[0], goal[1], "*", color=c, ms=15, mec="k", mew=0.5, zorder=4)  # goal
            ax.add_patch(plt.Circle((goal[0], goal[1]), thr, color=c, alpha=0.18, zorder=1))
            ax.plot(path[-1, 0], path[-1, 1], "x", color=c, ms=8, mew=2, zorder=3)  # end
            n_succ += int(success)

        ax.set_title(f"{name}\n{n_succ}/{n_maps} reached goal", fontsize=12)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("y")

    # Shared legend for the marker semantics.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="gray", ls="", ms=8, label="start"),
        Line2D([0], [0], marker="*", color="gray", ls="", ms=13, mec="k", label="goal"),
        Line2D([0], [0], marker="x", color="gray", ls="", ms=8, mew=2, label="final pos"),
        Line2D([0], [0], color="gray", lw=2, label="trajectory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Point-mass rollouts (same 6 start/goal pairs per agent)", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = "her_sac_pointmass_maps.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved map: {out}")


if __name__ == "__main__":
    main()
