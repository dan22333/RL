import numpy as np
import torch
import gym
import os
import time
import pybullet
from general import get_logger, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, np2torch
from policy import CategoricalPolicy, GaussianPolicy


class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, env, config, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module

        You do not need to implement anything in this function. However,
        you will need to use self.discrete, self.observation_dim,
        self.action_dim, and self.lr in other methods.
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.env.seed(self.seed)

        # single persistent GUI env, created lazily on first --render replay
        self.render_env = None
        # offscreen env for --record-video (renders to file, needs no display)
        self.record_env = None

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.lr = self.config.learning_rate

        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(env, config)

    def init_policy(self) -> None:
        """Build the policy network, its distribution head, and the optimizer.

        The body network maps an observation to `action_dim` outputs — action logits
        if the action space is discrete, or action means if it is continuous. We wrap
        it in the matching distribution head (`CategoricalPolicy` / `GaussianPolicy`),
        then hand ALL trainable parameters to Adam. For the Gaussian policy that
        includes `log_std` (the learnable exploration spread) alongside the weights,
        because `self.policy.parameters()` enumerates the whole module.
        """
        self.network = build_mlp(
            self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size
        )
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        """
        You don't have to change or use anything here.
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
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
            states, actions, rewards = [], [], []
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
                action = self.policy.act(states[-1][None])[0]
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
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if render_live and render_fps:
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
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path

        Uses self.config.gamma as the discount factor.
        """
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            # Discounted return G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
            # Computed right-to-left with the recurrence G_t = r_t + γ G_{t+1}
            # (O(T) instead of O(T^2)). Cast to float first so the in-place
            # γ-multiply can't be truncated by an integer reward dtype.
            returns = rewards.astype(np.float64)
            for i in range(len(returns) - 2, -1, -1):
                returns[i] += self.config.gamma * returns[i + 1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]

        TODO:
        Normalize the advantages so that they have a mean of 0 and standard
        deviation of 1. Put the result in a variable called
        normalized_advantages (which will be returned).

        Whitening the advantages (mean 0, std 1) keeps the gradient scale consistent
        from batch to batch, which stabilizes learning. The small epsilon guards
        against divide-by-zero when every advantage in the batch is identical.

        This function is called only if self.config.normalize_advantage is True.
        """
        normalized_advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data.
        To compute the loss, you will need the log probabilities of the actions
        given the observations. Note that the policy's action_distribution
        method returns an instance of a subclass of
        torch.distributions.Distribution, and that object can be used to
        compute log probabilities.
        See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
        PyTorch optimizers MINIMIZE the loss, but we want to MAXIMIZE expected
        return — hence the leading minus sign on the objective below.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)

        self.optimizer.zero_grad()

        # log π(a_t|s_t) under the current policy, for the actions actually taken.
        dist = self.policy.action_distribution(observations)
        log_prob = dist.log_prob(actions)

        # Vanilla policy-gradient objective: maximize E[log π(a|s) * A]. Negate to
        # turn it into a loss, and average over timesteps for a scale-stable gradient.
        num_timesteps = observations.shape[0]
        loss = -torch.sum(log_prob * advantages) / num_timesteps
        loss.backward()
        self.optimizer.step()

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        # --render-live: open a single GUI window on the training env itself, so
        # the rollouts we collect are drawn continuously (must be before reset).
        if getattr(self.config, "render_live", False):
            self.env.render(mode="human")

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # make the current iteration visible to sample_path's live overlay
            self.render_iteration = t
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

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

            # snapshot GUI replay every render_freq iterations (not in live mode,
            # where the training rollouts are already being drawn continuously)
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

    def record_video_episode(self, iteration=None, max_frames=200):
        """
        Render one episode of the current policy OFFSCREEN and save it as an
        animated GIF. Uses render(mode="rgb_array"), which needs no display /
        GUI window, so it works headless and from a background shell. This is
        the reliable way to watch the agent when a live GUI can't be shown.
        """
        from PIL import Image

        if self.record_env is None:
            self.record_env = gym.make(self.config.env_name)
            self.record_env.seed(self.seed)
        env = self.record_env
        u = env.unwrapped

        # Camera controls, matching the live view: yaw/pitch/distance from the CLI.
        yaw = getattr(self.config, "render_yaw", 90.0)
        pitch = getattr(self.config, "render_pitch", -5.0)
        dist = getattr(self.config, "render_dist", 2.5)
        W = getattr(u, "_render_width", 480) or 480
        H = getattr(u, "_render_height", 360) or 360

        def capture():
            """One RGB frame from a yaw/pitch/dist camera that tracks the agent.

            Uses our own view matrix (via getCameraImage) instead of the env's
            built-in render camera, so the recorded angle matches --render-yaw
            for BOTH envs (CartPole ignores _cam_* attributes). CPU/TinyRenderer
            so it stays headless.
            """
            p = u._p
            try:
                target = list(u.robot.body_xyz)              # locomotion robots
            except Exception:
                try:
                    target = list(p.getBasePositionAndOrientation(u.cartpole)[0])
                except Exception:
                    target = [0, 0, 0]
            view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
            proj = p.computeProjectionMatrixFOV(60.0, W / float(H), 0.1, 100.0)
            img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
            return np.reshape(np.asarray(img[2], dtype=np.uint8), (H, W, 4))[:, :, :3]

        state = env.reset()
        frames = []
        for step in range(self.config.max_ep_len):
            try:
                frame = capture()
            except Exception:
                frame = np.asarray(env.render(mode="rgb_array"), dtype=np.uint8)
            frames.append(Image.fromarray(frame))
            action = self.policy.act(state[None])[0]
            state, reward, done, info = env.step(action)
            if done or len(frames) >= max_frames:
                break

        tag = "iter{:03d}".format(iteration) if iteration is not None else "final"
        out = os.path.join(self.config.output_path, "video_{}.gif".format(tag))
        frames[0].save(
            out, save_all=True, append_images=frames[1:], duration=50, loop=0
        )
        self.logger.info("Saved video: {}  ({} frames)".format(out, len(frames)))
        return out

    def _rebuild_headless_env(self):
        """
        Recover from a lost GUI physics server (e.g. the user closed the live
        window mid-run): disable rendering and swap in a fresh headless env so
        training continues uninterrupted. The visualization must never be able
        to kill a run.
        """
        self.logger.info(
            "GUI window closed / physics server lost -- continuing training WITHOUT rendering."
        )
        self.config.render_live = False
        self.env = gym.make(self.config.env_name)
        self.env.seed(self.seed)
        return self.env

    def render_policy(self, iteration=None, avg_reward=None, num_episodes=1, fps=60):
        """
        Play `num_episodes` episodes of the current policy in a live PyBullet
        GUI window so you can watch the agent behave (and improve) as it trains.

        Purely for visualization: it does NOT collect data or affect training.

        Uses ONE persistent GUI window for the whole run (created on first call
        and reused), so replays don't spawn a new window each time. For PyBullet
        envs, render(mode="human") must be called before the first reset() to
        open the window. Note: while training collects data between replays, the
        window may briefly appear frozen since nothing is stepping it.
        """
        label = "iteration {}".format(iteration) if iteration is not None else "policy"
        if avg_reward is not None:
            label += "  |  avg train reward {:.1f}".format(avg_reward)
        print("\n>>> [GUI] replaying {} ...".format(label), flush=True)

        if self.render_env is None:
            self.render_env = gym.make(self.config.env_name)
            self.render_env.render(mode="human")  # open GUI window (once)
        env = self.render_env

        for _ in range(num_episodes):
            state = env.reset()
            # overlay the iteration/reward text inside the 3D window; clear the
            # previous overlay first so labels don't stack up across replays
            try:
                p = env.unwrapped._p
                p.removeAllUserDebugItems()
                p.addUserDebugText(
                    label, [0, 0, 1.4], textColorRGB=[1, 0, 0], textSize=1.6
                )
            except Exception:
                pass
            ep_reward = 0.0
            for step in range(self.config.max_ep_len):
                action = self.policy.act(state[None])[0]
                state, reward, done, info = env.step(action)
                ep_reward += reward
                if fps:
                    time.sleep(1.0 / fps)  # pace to real time so it's watchable
                if done:
                    break
            print(
                ">>> [GUI]   episode reward: {:.0f} ({} steps)".format(
                    ep_reward, step + 1
                ),
                flush=True,
            )

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env is None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def save_model(self):
        """Persist trained weights so the agent can be reloaded and replayed later.

        Saves the policy (and, if used, the baseline) state_dicts into
        `config.model_output`. Reload with `play.py`, which rebuilds the same
        architecture from the config and loads these files.
        """
        os.makedirs(self.config.model_output, exist_ok=True)
        torch.save(
            self.policy.state_dict(),
            os.path.join(self.config.model_output, "policy.pt"),
        )
        if self.config.use_baseline:
            torch.save(
                self.baseline_network.state_dict(),
                os.path.join(self.config.model_output, "baseline.pt"),
            )
        self.logger.info("Saved model weights to {}".format(self.config.model_output))

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # persist the trained weights for later replay
        self.save_model()
        # record one game at the end
        if self.config.record:
            self.record()
