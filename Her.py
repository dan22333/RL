"""
SAC + HER point-mass demo.

This script trains three agents on the same 2D goal-reaching task:

1. SAC with dense reward: reward = -distance_to_goal.
2. SAC with sparse reward: reward = 0 if success else -1.
3. SAC with sparse reward + HER: same sparse reward, but replay samples are relabeled
   with hindsight goals reached later in the same episode.

Run:
    python her_sac_pointmass_demo.py

Fast smoke test:
    python her_sac_pointmass_demo.py --episodes 30 --eval-every 10 --updates-per-episode 10

Outputs:
    /current/folder/her_sac_pointmass_results.png

Dependencies:
    pip install torch numpy matplotlib
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# -----------------------------
# 1) Reproducibility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    """Make random number generation repeatable enough for a toy experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# 2) Tiny goal-conditioned env
# -----------------------------

class PointGoalEnv:
    """
    A minimal 2D continuous-control goal-reaching environment.

    State/observation:
        observation   = current 2D position [x, y]
        achieved_goal = current 2D position [x, y]
        desired_goal  = target 2D position [gx, gy]

    Action:
        2D velocity command in [-1, 1]^2.
        The env applies: position <- position + action_scale * action

    Reward modes:
        dense:  reward = -distance(position, goal)
        sparse: reward = 0 if distance < success_threshold else -1

    This env is intentionally simple so HER mechanics are visible.
    """

    def __init__(
        self,
        reward_type: str = "sparse",
        max_steps: int = 30,
        action_scale: float = 0.12,
        success_threshold: float = 0.06,
        position_bound: float = 1.0,
    ) -> None:
        assert reward_type in {"dense", "sparse"}
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.success_threshold = success_threshold
        self.position_bound = position_bound
        self.t = 0
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        """Start at a random 2D point and sample a random 2D target."""
        self.t = 0
        self.pos = np.random.uniform(-self.position_bound, self.position_bound, size=2).astype(np.float32)
        self.goal = np.random.uniform(-self.position_bound, self.position_bound, size=2).astype(np.float32)
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Apply one action.

        terminated=True only means real success.
        truncated=True means time limit. We do NOT treat truncation as terminal for bootstrapping.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.pos = self.pos + self.action_scale * action
        self.pos = np.clip(self.pos, -1.25 * self.position_bound, 1.25 * self.position_bound).astype(np.float32)
        self.t += 1

        obs = self._obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"]))
        success = bool(self.is_success(obs["achieved_goal"], obs["desired_goal"]))
        terminated = success
        truncated = self.t >= self.max_steps
        info = {"is_success": success}
        return obs, reward, terminated, truncated, info

    def _obs(self) -> Dict[str, np.ndarray]:
        """Return the HER-style dict observation."""
        return {
            "observation": self.pos.copy().astype(np.float32),
            "achieved_goal": self.pos.copy().astype(np.float32),
            "desired_goal": self.goal.copy().astype(np.float32),
        }

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """
        Vectorized reward recomputation.

        This is the function HER needs. It can compute reward for the original goal
        or for a hindsight goal that we invent later.
        """
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return -distance.astype(np.float32)
        return -(distance > self.success_threshold).astype(np.float32)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return distance < self.success_threshold


# -----------------------------
# 3) Replay buffers
# -----------------------------

@dataclass
class Transition:
    obs: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_achieved_goal: np.ndarray
    done: float


class EpisodeReplayBuffer:
    """
    Replay buffer that stores complete episodes.

    Why complete episodes?
        HER needs to choose a future achieved goal from the same trajectory.
        If we stored isolated transitions only, we would lose the temporal structure.
    """

    def __init__(
        self,
        capacity_episodes: int,
        env: PointGoalEnv,
        use_her: bool = False,
        n_sampled_goal: int = 4,
        device: str = "cpu",
    ) -> None:
        self.capacity_episodes = capacity_episodes
        self.env = env
        self.use_her = use_her
        self.n_sampled_goal = n_sampled_goal
        self.device = device
        self.episodes: List[List[Transition]] = []

    def add_episode(self, episode: List[Transition]) -> None:
        """Append one real episode. No fake data is added here."""
        if not episode:
            return
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity_episodes:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a minibatch.

        If use_her=True, about n_sampled_goal / (1 + n_sampled_goal) samples
        are relabeled with a future achieved goal. With n_sampled_goal=4, this is 80% HER.
        """
        assert self.episodes, "Cannot sample before at least one episode is stored."
        her_probability = self.n_sampled_goal / (1.0 + self.n_sampled_goal)

        obs_batch, goal_batch, action_batch = [], [], []
        reward_batch, next_obs_batch, done_batch = [], [], []
        relabeled_successes = 0
        relabeled_count = 0

        for _ in range(batch_size):
            # Pick a real episode and a real time index from that episode.
            episode = random.choice(self.episodes)
            t = random.randrange(len(episode))
            tr = episode[t]

            # Start with the original desired goal from data collection.
            goal = tr.desired_goal.copy()
            reward = tr.reward

            # HER relabeling: replace the goal with a goal achieved later in the same episode.
            if self.use_her and random.random() < her_probability:
                # FUTURE strategy: choose a transition at or after t.
                future_t = random.randrange(t, len(episode))
                goal = episode[future_t].next_achieved_goal.copy()

                # Recompute reward under the hindsight goal.
                reward = float(self.env.compute_reward(tr.next_achieved_goal, goal))

                relabeled_count += 1
                if reward == 0.0:  # sparse success convention: 0 success, -1 failure
                    relabeled_successes += 1

            # The policy/critic input is concat(observation, desired_goal).
            obs_batch.append(np.concatenate([tr.obs, goal], axis=0))
            next_obs_batch.append(np.concatenate([tr.next_obs, goal], axis=0))
            action_batch.append(tr.action)
            reward_batch.append([reward])
            done_batch.append([tr.done])
            goal_batch.append(goal)

        batch = {
            "obs": torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(np.asarray(next_obs_batch), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(np.asarray(action_batch), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(np.asarray(reward_batch), dtype=torch.float32, device=self.device),
            "dones": torch.tensor(np.asarray(done_batch), dtype=torch.float32, device=self.device),
            # These two fields are just diagnostics so you can verify HER is creating successes.
            "her_relabel_fraction": torch.tensor([relabeled_count / max(1, batch_size)], dtype=torch.float32),
            "her_success_fraction": torch.tensor([relabeled_successes / max(1, relabeled_count)], dtype=torch.float32),
        }
        return batch


# -----------------------------
# 4) SAC neural networks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy used by SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and log-probability using reparameterization.

        raw_action ~ Normal(mean, std)
        action = tanh(raw_action), therefore action is in [-1, 1].
        """
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)

        # Tanh correction term for the log-probability.
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class QCritic(nn.Module):
    """Q(s, g, a). Here s and g are already concatenated into obs_goal."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q = MLP(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class SACAgent:
    """Minimal SAC implementation for continuous actions."""

    def __init__(
        self,
        obs_goal_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = GaussianActor(obs_goal_dim, action_dim, hidden_dim).to(device)
        self.q1 = QCritic(obs_goal_dim, action_dim, hidden_dim).to(device)
        self.q2 = QCritic(obs_goal_dim, action_dim, hidden_dim).to(device)
        self.target_q1 = copy.deepcopy(self.q1).to(device)
        self.target_q2 = copy.deepcopy(self.q2).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        """Convert dict obs into concat(obs, goal), then sample an action."""
        obs_goal = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]], axis=0)
        x = torch.tensor(obs_goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(x)
        else:
            action, _ = self.actor.sample(x)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One SAC gradient update from one minibatch."""
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # ---- Critic target ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            target_q1 = self.target_q1(next_obs, next_actions)
            target_q2 = self.target_q2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            y = rewards + self.gamma * (1.0 - dones) * target_q

        # ---- Critic losses ----
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # ---- Actor loss ----
        new_actions, log_probs = self.actor.sample(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Target network soft update ----
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)


# -----------------------------
# 5) Training and evaluation
# -----------------------------

@torch.no_grad()
def evaluate(agent: SACAgent, reward_type: str, episodes: int = 30, seed_offset: int = 10_000) -> Tuple[float, float]:
    """Evaluate deterministic policy and return success rate + average return."""
    # Use a separate env; do not mutate training env.
    env = PointGoalEnv(reward_type=reward_type)
    successes = []
    returns = []
    # Make eval repeatable but not identical to training.
    old_state = np.random.get_state()
    np.random.seed(seed_offset)
    for _ in range(episodes):
        obs = env.reset()
        ep_return = 0.0
        success = False
        for _t in range(env.max_steps):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            success = success or info["is_success"]
            if terminated or truncated:
                break
        successes.append(float(success))
        returns.append(ep_return)
    np.random.set_state(old_state)
    return float(np.mean(successes)), float(np.mean(returns))


def train_condition(
    name: str,
    reward_type: str,
    use_her: bool,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, List[float]]:
    """Train one condition: dense SAC, sparse SAC, or sparse SAC+HER."""
    print(f"\n=== Training {name} ===")
    env = PointGoalEnv(reward_type=reward_type, max_steps=args.max_steps)
    agent = SACAgent(
        obs_goal_dim=4,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        device=device,
    )
    buffer = EpisodeReplayBuffer(
        capacity_episodes=args.buffer_episodes,
        env=env,
        use_her=use_her,
        n_sampled_goal=args.n_sampled_goal,
        device=device,
    )

    eval_episodes = []
    success_rates = []
    avg_returns = []
    her_success_fracs = []

    total_steps = 0

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        episode: List[Transition] = []
        ep_return = 0.0

        for _t in range(env.max_steps):
            # Random warmup creates diverse early experience before the actor is useful.
            if total_steps < args.warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)

            # done is true only for real terminal success, not for time-limit truncation.
            done_for_bootstrap = float(terminated)

            episode.append(
                Transition(
                    obs=obs["observation"],
                    achieved_goal=obs["achieved_goal"],
                    desired_goal=obs["desired_goal"],
                    action=action.astype(np.float32),
                    reward=float(reward),
                    next_obs=next_obs["observation"],
                    next_achieved_goal=next_obs["achieved_goal"],
                    done=done_for_bootstrap,
                )
            )

            obs = next_obs
            ep_return += float(reward)
            total_steps += 1

            if terminated or truncated:
                break

        buffer.add_episode(episode)

        # After each episode, run several SAC updates from replay.
        if len(buffer) >= args.batch_size:
            recent_her_success = []
            for _ in range(args.updates_per_episode):
                batch = buffer.sample(args.batch_size)
                agent.update(batch)
                if use_her:
                    recent_her_success.append(float(batch["her_success_fraction"].item()))
            if recent_her_success:
                her_success_fracs.append(float(np.mean(recent_her_success)))

        if ep % args.eval_every == 0 or ep == 1:
            success_rate, avg_return = evaluate(agent, reward_type=reward_type, episodes=args.eval_episodes)
            eval_episodes.append(ep)
            success_rates.append(success_rate)
            avg_returns.append(avg_return)
            if use_her and her_success_fracs:
                her_diag = f", HER relabeled-success≈{her_success_fracs[-1]:.2f}"
            else:
                her_diag = ""
            print(f"{name:16s} episode={ep:4d} eval_success={success_rate:.2f} avg_return={avg_return:.1f}{her_diag}")

    return {
        "eval_episodes": eval_episodes,
        "success_rates": success_rates,
        "avg_returns": avg_returns,
        "her_success_fracs": her_success_fracs,
    }


def plot_results(results: Dict[str, Dict[str, List[float]]], output_path: str) -> None:
    """Create one figure with success curves and average return curves."""
    plt.figure(figsize=(10, 5))
    for name, history in results.items():
        plt.plot(history["eval_episodes"], history["success_rates"], label=name)
    plt.xlabel("Training episodes")
    plt.ylabel("Evaluation success rate")
    plt.title("Goal-reaching success rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    success_path = output_path.replace(".png", "_success.png")
    plt.savefig(success_path, dpi=160)

    plt.figure(figsize=(10, 5))
    for name, history in results.items():
        plt.plot(history["eval_episodes"], history["avg_returns"], label=name)
    plt.xlabel("Training episodes")
    plt.ylabel("Evaluation average return")
    plt.title("Goal-reaching average return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return_path = output_path.replace(".png", "_return.png")
    plt.savefig(return_path, dpi=160)

    print(f"\nSaved graphs:\n  {success_path}\n  {return_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=350)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--updates-per-episode", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-episodes", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="her_sac_pointmass_results.png")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    # Keep this toy demo fast and predictable on laptops/CPU notebooks.
    torch.set_num_threads(1)
    args = parse_args()
    set_seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")

    conditions = [
        ("Dense SAC", "dense", False),
        ("Sparse SAC", "sparse", False),
        ("Sparse SAC + HER", "sparse", True),
    ]

    results: Dict[str, Dict[str, List[float]]] = {}
    for name, reward_type, use_her in conditions:
        # Change seed between conditions but keep them deterministic.
        set_seed(args.seed + len(results) * 100)
        results[name] = train_condition(name, reward_type, use_her, args, device)

    output_path = os.path.abspath(args.output)
    plot_results(results, output_path)


if __name__ == "__main__":
    main()
