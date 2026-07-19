"""Policy models: the part of the agent that turns an observation into an action.

A "policy" answers one question: *given what I currently see, how should I act?*
It does this probabilistically — it defines a probability distribution over actions
and samples from it. Sampling (rather than always taking the single best action) is
what lets the agent EXPLORE, which reinforcement learning depends on.

Two concrete policies live here, sharing one interface:

    CategoricalPolicy -- for DISCRETE actions (pick one of N choices, e.g. left/right)
    GaussianPolicy    -- for CONTINUOUS actions (real-valued vectors, e.g. joint torques)

Both are built on the same `BasePolicy` contract, so the training code can treat them
interchangeably: it only ever calls `action_distribution(...)` and `act(...)`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as ptd
import torch.nn as nn

from network_utils import np2torch


class BasePolicy(ABC):
    """The shared contract every policy must fulfil.

    This is an *abstract base class* (ABC): it defines the methods a policy must
    provide but doesn't implement the model-specific one itself. Subclasses fill in
    `action_distribution`; in return they inherit `act` for free. The training loop
    programs against *this* interface, so it never needs to know whether it's driving
    a discrete or a continuous policy.
    """

    @abstractmethod
    def action_distribution(self, observations: torch.Tensor) -> ptd.Distribution:
        """Return the action distribution p(action | observation) — MODEL-SPECIFIC.

        This is the one piece each policy defines differently: it maps a batch of
        observations to a torch `Distribution` object over actions (a `Categorical`
        for discrete actions, a Gaussian for continuous ones). The returned
        distribution is *batched* — one independent distribution per observation in
        the batch — so downstream `.sample()` / `.log_prob(...)` act per-row.

        Marked `@abstractmethod`: Python refuses to instantiate any subclass that
        forgets to implement it, catching the mistake early instead of at call time.

        Args:
            observations: Tensor of shape [batch_size, observation_dim].

        Returns:
            A torch distribution with batch shape [batch_size].
        """
        raise NotImplementedError

    def act(
        self,
        observations: np.ndarray,
        return_log_prob: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Choose actions for a batch of observations (the environment-facing entry point).

        This is what the rollout loop calls to actually *play*. The pipeline is:

            numpy observations  ->  tensor  ->  distribution  ->  sample  ->  numpy actions

        Concretely it:
          1. Converts the incoming numpy observations to a tensor on the right device.
          2. Builds the action distribution via the subclass's `action_distribution`.
          3. Samples one action per observation (this is the exploration step).
          4. Optionally records each action's log-probability — PPO needs this later
             as the "old" policy probability to compute its importance ratio.
          5. Converts everything back to numpy for the environment / replay buffer.

        Why `.detach().cpu().numpy()` on the way out:
          - `.detach()` cuts the tensor out of the autograd graph. We are *acting*, not
            training, so we don't want PyTorch tracking gradients through sampled actions
            (that would leak memory and is meaningless here).
          - `.cpu()` moves the tensor off the GPU, because numpy only lives on the CPU;
            calling `.numpy()` on a GPU tensor would otherwise error.

        Args:
            observations: numpy array of shape [batch_size, observation_dim].
            return_log_prob: If True, also return the log-probability of each action.

        Returns:
            sampled_actions of shape [batch_size, *action_shape]; and, if
            `return_log_prob` is True, a tuple (sampled_actions, log_probs) where
            log_probs has shape [batch_size].
        """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()
        # Compute log-prob BEFORE converting the action to numpy — log_prob needs the tensor.
        log_probs = distribution.log_prob(sampled_actions)

        sampled_actions = sampled_actions.detach().cpu().numpy()
        if return_log_prob:
            return sampled_actions, log_probs.detach().cpu().numpy()
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    """Policy for DISCRETE action spaces — choose one of N actions.

    The network outputs one raw score ("logit") per possible action; a Categorical
    distribution turns those scores into a probability for each action (via softmax)
    and lets us sample one. Example: CartPole, where the two actions are push-left /
    push-right.

    It inherits from both `BasePolicy` (the policy contract) and `nn.Module` (so its
    network's weights are tracked as trainable parameters by the optimizer).
    """

    def __init__(self, network: nn.Module):
        """Store the network that maps observations -> per-action logits.

        `nn.Module.__init__` must run first so PyTorch sets up its internal parameter
        bookkeeping; assigning `self.network` then auto-registers that network's
        weights as trainable parameters of this policy.
        """
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations: torch.Tensor) -> ptd.Categorical:
        """Build the categorical distribution over discrete actions.

        The network produces a logit per action; passing them as `logits=` lets
        PyTorch apply the softmax internally (numerically safer than doing it by hand).

        Args:
            observations: Tensor of shape [batch_size, observation_dim].

        Returns:
            A `Categorical` with batch shape [batch_size] and N categories.
        """
        logits = self.network(observations)
        return ptd.Categorical(logits=logits)


class GaussianPolicy(BasePolicy, nn.Module):
    """Policy for CONTINUOUS action spaces — output real-valued action vectors.

    Instead of picking from a menu, this policy proposes a real vector (e.g. the
    torque for each joint). It models actions as a *diagonal Gaussian*: the network
    predicts the MEAN action for the current observation, and a separate learned
    parameter controls the spread (how much random exploration to add around the mean).

    "Diagonal" means each action dimension is independent (no cross-correlations) — a
    simple, standard choice that makes the math cheap.
    """

    def __init__(self, network: nn.Module, action_dim: int):
        """Store the mean-network and create the learnable spread parameter.

        `self.log_std` is the *logarithm* of the standard deviation, one value per
        action dimension. We learn log(std) rather than std directly so that:
          - it can range over all real numbers (no need to constrain it positive), and
          - exponentiating it (see `std`) always yields a valid positive std.
        It is state-INDEPENDENT: the same exploration spread is used for every
        observation, and it's a `nn.Parameter` so the optimizer trains it alongside the
        network weights. Initializing to 0 means the initial std is exp(0) = 1.

        Args:
            network: Maps observations -> mean action, shape [batch, action_dim].
            action_dim: Number of continuous action dimensions.
        """
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def std(self) -> torch.Tensor:
        """Return the per-dimension standard deviations (always positive).

        Exponentiating `log_std` recovers the actual standard deviation and guarantees
        it's positive, no matter what value training pushes `log_std` to.

        Returns:
            Tensor of shape [action_dim] with the std of each action dimension.
        """
        return torch.exp(self.log_std)

    def action_distribution(self, observations: torch.Tensor) -> ptd.Distribution:
        """Build the diagonal Gaussian action distribution.

        The network gives the mean (`loc`); `std()` gives the spread. We assemble a
        multi-dimensional Gaussian with independent dimensions using
        `Independent(Normal(...), 1)`:
          - `Normal(loc, scale)` makes one independent 1-D Gaussian per action element.
          - `Independent(..., 1)` reinterprets the last dimension as a single joint
            action, so `.log_prob(action)` returns ONE number per observation (the sum
            of the per-dimension log-probs) instead of one per dimension.

        This is equivalent to a `MultivariateNormal` with a diagonal covariance, but
        much cheaper: it never builds or inverts a full covariance matrix. The std is
        broadcast across the batch, giving every observation the same spread.

        Args:
            observations: Tensor of shape [batch_size, observation_dim].

        Returns:
            A distribution with batch shape [batch_size] over action vectors of
            length `action_dim`.
        """
        loc = self.network(observations)
        return ptd.Independent(ptd.Normal(loc=loc, scale=self.std()), 1)
