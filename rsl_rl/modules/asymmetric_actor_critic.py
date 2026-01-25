# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal, Distribution, constraints
from typing import Any, NoReturn, Optional

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory
from rsl_rl.utils import unpad_trajectories


class GSDENoiseDistributionRecurrent(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE) with recurrent networks.
    
    This is a modified version of GSDENoiseDistribution that handles batched sequences
    from recurrent networks using torch.matmul instead of torch.mm.

    Paper: https://arxiv.org/abs/2005.05719
    """

    has_rsample = True
    arg_constraints = {
        "mean_actions": constraints.real,
        "log_std": constraints.real,
        "latent_features": constraints.real,
    }
    _validate_args = False

    def __init__(
        self,
        action_dim: int,
        epsilon: float = 1e-6,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self._base_distribution: Optional[Normal] = None
        self._latent_features: Optional[torch.Tensor] = None
        self._exploration_matrix: Optional[torch.Tensor] = None
        self._exploration_matrices: Optional[torch.Tensor] = None
        self._weights_distribution: Optional[Normal] = None
        super().__init__(batch_shape, event_shape, validate_args)

    def _std_from_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_std)

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        std = self._std_from_log_std(log_std)
        weights_distribution = Normal(torch.zeros_like(std), std)
        self._weights_distribution = weights_distribution
        self._exploration_matrix = weights_distribution.rsample()
        self._exploration_matrices = weights_distribution.rsample((batch_size,))

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> "GSDENoiseDistributionRecurrent":
        self._latent_features = latent_features
        # move exploration matrices to the correct device
        if self._exploration_matrix is not None:
            self._exploration_matrix = self._exploration_matrix.to(latent_features.device)
        if self._exploration_matrices is not None:
            self._exploration_matrices = self._exploration_matrices.to(latent_features.device)
        # variance per action: (phi(s)^2) @ (sigma^2)
        # Use matmul to handle batched sequences from recurrent networks
        variance = torch.matmul(latent_features**2, self._std_from_log_std(log_std) ** 2)
        self._base_distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(actions)
        return self._base_distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self._base_distribution.entropy()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if self._base_distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self._base_distribution.rsample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        if self._base_distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self._base_distribution.mean

    @property
    def mode(self) -> torch.Tensor:
        if self._base_distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self._base_distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        if self._base_distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self._base_distribution.variance

    @property
    def stddev(self) -> torch.Tensor:
        if self._base_distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self._base_distribution.stddev

    @property
    def support(self) -> constraints.Constraint:
        return constraints.real

    def expand(self, batch_shape: torch.Size, _instance=None) -> "GSDENoiseDistributionRecurrent":
        new = self._get_checked_instance(GSDENoiseDistributionRecurrent, _instance)
        new.action_dim = self.action_dim
        new.epsilon = self.epsilon
        new._base_distribution = self._base_distribution
        new._latent_features = self._latent_features
        new._exploration_matrix = self._exploration_matrix
        new._exploration_matrices = self._exploration_matrices
        new._weights_distribution = self._weights_distribution
        super(GSDENoiseDistributionRecurrent, new).__init__(batch_shape, self._event_shape, validate_args=False)
        return new

    def get_noise(self, latent_features: torch.Tensor) -> torch.Tensor:
        if (
            self._exploration_matrices is None
            or len(latent_features) == 1
            or len(latent_features) != len(self._exploration_matrices)
        ):
            # Use matmul to handle batched inputs
            return torch.matmul(latent_features, self._exploration_matrix)
        latent_features = latent_features.unsqueeze(dim=1)
        noise = torch.bmm(latent_features, self._exploration_matrices)
        return noise.squeeze(dim=1)


class AsymmetricActorCritic(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "AsymmetricActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The AsymmetricActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The AsymmetricActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        # Actor
        self.memory_a = Memory(num_actor_obs, rnn_hidden_dim, rnn_num_layers, rnn_type)
        if self.state_dependent_std:
            self.actor = MLP(rnn_hidden_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            elif self.noise_std_type == "gsde":
                self.log_std = nn.Parameter(
                    torch.ones(actor_hidden_dims[-1], num_actions) * torch.log(torch.tensor(init_noise_std))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar', 'log', or 'gsde'")

        # Action distribution (populated in update_distribution)
        if self.noise_std_type == "gsde":
            self.distribution = GSDENoiseDistributionRecurrent(action_dim=num_actions)
            self.distribution.sample_weights(self.log_std)
        else:
            self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _update_distribution(self, rnn_out: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(rnn_out)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(rnn_out)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            elif self.noise_std_type == "gsde":
                pass  # gSDE handles std internally via log_std and features
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar', 'log', or 'gsde'")
        # Create distribution
        if self.noise_std_type == "gsde":
            features = self.actor[:-1](rnn_out)
            self.distribution.proba_distribution(mean, self.log_std, features)
        else:
            self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs).squeeze(0)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        else:
            return self.actor(out_mem)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        masks = kwargs.get('masks', None)
        # Critic is a simple MLP (like ActorCritic)
        # In recurrent mode, process padded trajectories and unpad like the actor RNN does
        if masks is not None:  # Recurrent mini-batch mode
            # obs shape: [time, num_trajectories, features]
            batch_shape = obs.shape[:2]  # (time, num_trajectories)
            obs_flat = obs.flatten(0, 1)  # [time*num_trajectories, features]
            values = self.critic(obs_flat)  # [time*num_trajectories, 1]
            values = values.view(*batch_shape, -1)  # [time, num_trajectories, 1]
            # Unpad to match environment-based shape like actor RNN output
            return unpad_trajectories(values, masks)  # [time, num_envs, 1]
        else:  # Non-recurrent mode
            return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, None]:
        return self.memory_a.hidden_state, None

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
