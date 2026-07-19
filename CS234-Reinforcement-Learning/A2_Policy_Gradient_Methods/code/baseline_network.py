"""Baseline (value) network for variance reduction in policy-gradient training.

Policy-gradient estimates are notoriously noisy. The trick this file implements is
the *baseline*: instead of scaling each action's gradient by its raw return G_t, we
subtract a state-dependent baseline V(s_t) and scale by the ADVANTAGE

    A_t = G_t - V(s_t)          ("was this action better or worse than average
                                 from this state?")

Subtracting a state-only baseline leaves the gradient's expected value unchanged
(so we still optimize the right thing) but dramatically lowers its variance, which
makes training far more stable. Here V is a small neural net trained by regression
to predict the observed returns.
"""

import numpy as np
import torch
import torch.nn as nn

from network_utils import build_mlp, np2torch


class BaselineNetwork(nn.Module):
    """A learned state-value function V(s), used as the policy-gradient baseline.

    It is an `nn.Module` wrapping an MLP that maps an observation to a single scalar
    (the estimated value of that state), plus its own optimizer so it can be trained
    independently of the policy.
    """

    def __init__(self, env, config):
        """Build the value network and its optimizer from the config.

        Args:
            env: The Gym environment (used only to read the observation dimension).
            config: Hyperparameter bundle (layer count/size, learning rate, ...).
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate

        observation_dim = self.env.observation_space.shape[0]
        # Output size 1: the network predicts a single scalar value V(s) per state.
        self.network = build_mlp(observation_dim, 1, self.config.n_layers, self.config.layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict V(s) for a batch of observations.

        The underlying network emits shape [batch, 1]; we flatten it to [batch] so it
        lines up elementwise with the 1-D returns vector everywhere else.

        Note: prefer calling the module (`self(obs)`) over `self.network(obs)` so this
        flattening/shape guarantee is always applied.

        Args:
            observations: Tensor of shape [batch, observation_dim].

        Returns:
            Tensor of shape [batch] with the predicted value of each state.
        """
        output = torch.flatten(self.network(observations))
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Compute advantages A_t = G_t - V(s_t).

        Subtracting the baseline's value estimate from each observed return yields the
        advantage: how much better the taken trajectory did than the baseline expected.
        This is the variance-reduced signal the policy update actually uses.

        The value network is only *evaluated* here (not trained — that happens in
        `update_baseline`), so we `.detach()` it from the autograd graph and move it to
        CPU/numpy for the arithmetic against the numpy `returns`.

        Args:
            returns: Discounted returns G_t, shape [batch].
            observations: Observations, shape [batch, observation_dim].

        Returns:
            Advantages, shape [batch].
        """
        observations = np2torch(observations)
        baseline = self(observations).detach().cpu().numpy()
        advantages = returns - baseline
        return advantages

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray) -> None:
        """Train the value network one step to better predict the returns.

        This is a plain supervised regression: minimize the mean-squared error between
        the predicted values V(s_t) and the observed returns G_t. Improving V here makes
        it a tighter (lower-variance) baseline on the next iteration.

        Args:
            returns: Regression targets G_t, shape [batch].
            observations: Inputs, shape [batch, observation_dim].
        """
        returns = np2torch(returns)
        observations = np2torch(observations)

        self.optimizer.zero_grad()
        loss = nn.functional.mse_loss(self(observations), returns)
        loss.backward()
        self.optimizer.step()
