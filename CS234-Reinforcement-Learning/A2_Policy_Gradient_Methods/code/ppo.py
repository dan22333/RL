"""Proximal Policy Optimization (PPO) — a more stable policy-gradient algorithm.

PPO is a small but powerful modification of vanilla policy gradient (see
`policy_gradient.py`). Plain policy gradient uses each collected batch for exactly one
update and can take destructively large steps. PPO fixes both:

  1. It caches the log-probability of each action under the policy that COLLECTED the
     data (`old_logprobs`). During the update it forms the importance ratio
     r = π_new(a|s) / π_old(a|s) = exp(logπ_new - logπ_old).
  2. It CLIPS that ratio to [1-ε, 1+ε] in the objective, so a single update can't move
     the policy too far from the data-collecting policy — the "proximal" in PPO.

Because the clipped objective keeps updates safe, PPO can reuse the same batch for
several gradient steps (`config.update_freq`), squeezing more learning out of each
(expensive) round of environment interaction.

`PPO` subclasses `PolicyGradient` and overrides only what differs: `update_policy`
(clipped objective), `sample_path` (also caches `old_logprobs`), and `train` (runs
`update_freq` inner steps per batch).
"""

import time

import numpy as np
import pybullet
import torch

from policy_gradient import PolicyGradient
from network_utils import np2torch
from general import export_plot


class PPO(PolicyGradient):
    """Policy-gradient agent using PPO's clipped surrogate objective."""

    def __init__(self, env, config, seed, logger=None):
        """Initialize like PolicyGradient, but force a baseline and read the clip range.

        PPO always uses a value baseline, so we set `config.use_baseline = True` before
        delegating to the parent constructor. `eps_clip` is the ε that bounds how far
        each update may move the policy.
        """
        config.use_baseline = True
        super(PPO, self).__init__(env, config, seed, logger)
        self.eps_clip = self.config.eps_clip

    def update_policy(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logprobs: np.ndarray,
    ) -> None:
        """Run one gradient step on PPO's clipped surrogate objective.

        The clipped objective for each timestep is

            L = min( r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t )

        where r_t = exp(logπ_new(a_t|s_t) - logπ_old(a_t|s_t)) is the importance ratio.
        The clip removes the incentive to push r_t far from 1, keeping the new policy
        close to the one that gathered the data. We negate and average it because
        PyTorch optimizers MINIMIZE, whereas we want to MAXIMIZE expected advantage.

        Args:
            observations: Observations, shape [batch, observation_dim].
            actions: Actions taken, shape [batch] (discrete) or [batch, action_dim].
            advantages: Advantage estimates A_t, shape [batch].
            old_logprobs: log π_old(a_t|s_t) cached at collection time, shape [batch].
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        self.optimizer.zero_grad()

        # log-probs of the SAME actions under the CURRENT policy.
        dist = self.policy.action_distribution(observations)
        log_probs = dist.log_prob(actions)

        # Importance ratio r_t and its clipped version.
        ratio = torch.exp(log_probs - old_logprobs)
        clipped_ratio = torch.clip(ratio, min=1 - self.eps_clip, max=1 + self.eps_clip)

        # Clipped surrogate; take the pessimistic (min) branch, negate to maximize.
        num_timesteps = observations.shape[0]
        loss = -torch.sum(torch.minimum(ratio * advantages, clipped_ratio * advantages)) / num_timesteps

        loss.backward()
        self.optimizer.step()

    def _follow_camera(self, env):
        """Point the GUI debug camera at the robot from the configured side angle.

        The cheetah runs along +x and would leave a static frame, so during
        --render-live we re-aim the camera at its body each step, keeping the
        yaw/pitch/distance from the CLI (defaults give a side profile). No-op if
        anything about the GUI/robot isn't available (e.g. headless fallback).
        """
        try:
            u = env.unwrapped
            p = u._p
            try:
                x, y, z = u.robot.body_xyz                    # locomotion robots
            except Exception:
                x, y, z = p.getBasePositionAndOrientation(u.cartpole)[0]  # cartpole
            p.resetDebugVisualizerCamera(
                cameraDistance=getattr(self.config, "render_dist", 2.5),
                cameraYaw=getattr(self.config, "render_yaw", 90.0),
                cameraPitch=getattr(self.config, "render_pitch", -5.0),
                cameraTargetPosition=[x, y, z],
            )
        except Exception:
            pass

    def train(self):
        """Main PPO training loop.

        Each iteration: collect a batch of trajectories, compute returns and advantages
        once, then run `update_freq` inner updates that reuse that batch (the safe reuse
        PPO's clipping enables). Also handles logging, reward statistics, and recording.
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = []  # returns of every episode sampled (for running stats)
        averaged_total_rewards = []  # per-iteration average return (saved/plotted)

        # --render-live: open a single GUI window on the training env itself, so the
        # rollouts we collect are drawn continuously (must be before the first reset).
        if getattr(self.config, "render_live", False):
            self.env.render(mode="human")
            # Strip GUI eye-candy that slows every frame: side panels + shadows.
            try:
                p = self.env.unwrapped._p
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            except Exception:
                pass

        for t in range(self.config.num_batches):

            # make the current iteration visible to sample_path's live overlay
            self.render_iteration = t
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute discounted returns (Q-value estimates) and advantages for the batch
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # PPO reuses the same batch for `update_freq` inner updates
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, old_logprobs)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            # offscreen video snapshot every render_freq iterations (no display needed)
            if getattr(self.config, "record_video", False) and (
                t % getattr(self.config, "render_freq", 10) == 0
            ):
                self.record_video_episode(iteration=t)

            # GUI replay every render_freq iterations (only if a display is available;
            # skipped in --render-live mode, which draws the rollouts continuously)
            if (
                getattr(self.config, "render", False)
                and not getattr(self.config, "render_live", False)
                and (t % getattr(self.config, "render_freq", 10) == 0)
            ):
                self.render_policy(iteration=t, avg_reward=avg_reward, num_episodes=1)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def sample_path(self, env, num_episodes=None):
        """Collect trajectories from the environment, caching PPO's `old_logprobs`.

        This mirrors `PolicyGradient.sample_path` but, at each step, also records the
        log-probability of the sampled action under the CURRENT (data-collecting) policy.
        PPO needs those stored log-probs later to form the importance ratio — the policy
        will have changed by the time the update runs, so they must be captured now.

        Args:
            env: The Gym environment to sample from.
            num_episodes: If set, sample this many episodes; otherwise fill one batch
                of `config.batch_size` timesteps.

        Returns:
            paths: list of dicts, each with numpy arrays under keys "observation",
                "action", "reward", and "old_logprobs".
            episode_rewards: list of per-episode total rewards.
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        render_live = getattr(self.config, "render_live", False)
        render_fps = getattr(self.config, "render_fps", 0)

        while num_episodes or t < self.config.batch_size:
            try:
                state = env.reset()
            except pybullet.error:
                # live GUI window was closed -> fall back to headless and retry
                env = self._rebuild_headless_env()
                render_live = False
                state = env.reset()
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0

            # --render-live: overlay the current iteration + episode number in the
            # 3D window, refreshed each episode (clear the previous text first).
            if render_live:
                try:
                    p = env.unwrapped._p
                    p.removeAllUserDebugItems()
                    p.addUserDebugText(
                        "iteration {} | episode {}".format(
                            getattr(self, "render_iteration", 0), episode + 1
                        ),
                        [0, 0, 1.4],
                        textColorRGB=[1, 0, 0],
                        textSize=1.6,
                    )
                except Exception:
                    pass

            aborted = False
            for step in range(self.config.max_ep_len):
                states.append(state)
                # Unlike PolicyGradient, also grab the action's log-prob under π_old.
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob=True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]
                try:
                    state, reward, done, info = env.step(action)
                except pybullet.error:
                    # live GUI window closed mid-episode: drop to headless,
                    # discard this partial episode, keep training
                    env = self._rebuild_headless_env()
                    render_live = False
                    aborted = True
                    break
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if render_live:
                    self._follow_camera(env)  # keep the side view tracking the cheetah
                    if render_fps:
                        time.sleep(1.0 / render_fps)  # optional speed cap
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            if aborted:
                continue  # skip the partial path; the while-loop carries on headless

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs),
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards
