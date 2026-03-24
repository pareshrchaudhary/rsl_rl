# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, HiddenState


class BCImageActorCritic(nn.Module):
    """Actor-Critic where the actor is initialized from a BC-trained MLPImagePolicy.

    - Actor receives obs["policy"] (images + low-dim with history), processes through
      BC encoder/trunk/heads to produce a Gaussian action distribution.
    - Critic receives obs["critic"] (privileged state, flat vector) through a fresh MLP.

    The BC policy operates in normalized observation/action space internally.
    The distribution is transformed to environment action space for PPO.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        bc_checkpoint_path: str = "",
        critic_hidden_dims: list[int] | tuple[int] = (512, 256, 128),
        critic_activation: str = "elu",
        freeze_encoder: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(f"BCImageActorCritic: ignoring unexpected kwargs: {list(kwargs.keys())}")
        super().__init__()

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.freeze_encoder = freeze_encoder

        # -- Load BC policy from checkpoint --
        assert bc_checkpoint_path, "bc_checkpoint_path is required."
        bc_policy = self._load_bc_policy(bc_checkpoint_path)

        assert bc_policy.action_dim == num_actions, (
            f"BC action_dim {bc_policy.action_dim} != env num_actions {num_actions}"
        )
        self.n_obs_steps = bc_policy.n_obs_steps
        self.obs_feature_dim = bc_policy.obs_feature_dim

        self.obs_encoder = bc_policy.obs_encoder
        self.trunk = bc_policy.trunk
        self.mean_head = bc_policy.mean_head
        self.log_std_head = bc_policy.log_std_head
        self.log_std_limits = bc_policy.log_std_limits

        self.bc_normalizer = bc_policy.normalizer
        for p in self.bc_normalizer.parameters():
            p.requires_grad_(False)

        action_params = self.bc_normalizer.params_dict["action"]
        self.register_buffer("action_scale", action_params["scale"].clone())
        self.register_buffer("action_offset", action_params["offset"].clone())

        # Strip BC augmentation transforms (ColorJitter, RandomCrop, etc.).
        # Isaac Lab handles visual domain randomization. Keep only resize + ImageNet norm.
        import torchvision
        for key in self.obs_encoder.rgb_keys:
            old_tf = self.obs_encoder.key_transform_map[key]
            keep = [m for m in old_tf if isinstance(m, (torchvision.transforms.Resize, torchvision.transforms.Normalize))]
            self.obs_encoder.key_transform_map[key] = nn.Sequential(*(keep or [nn.Identity()]))

        if freeze_encoder:
            self.obs_encoder.eval()
            for p in self.obs_encoder.parameters():
                p.requires_grad_(False)
            print("BCImageActorCritic: encoder frozen.")

        # -- Critic: fresh MLP on privileged state --
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, (
                f"Critic obs '{obs_group}' must be 2D (B, D), got shape {obs[obs_group].shape}"
            )
            num_critic_obs += obs[obs_group].shape[-1]
        assert num_critic_obs > 0, "Critic obs dimension is 0. Check obs_groups['critic']."

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, critic_activation)
        print(f"BCImageActorCritic: critic MLP {num_critic_obs} -> {critic_hidden_dims} -> 1")
        print(f"BCImageActorCritic: actor from BC checkpoint, feature_dim={self.obs_feature_dim}, "
              f"n_obs_steps={self.n_obs_steps}, action_dim={num_actions}, "
              f"freeze_encoder={freeze_encoder}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _load_bc_policy(checkpoint_path: str):
        """Load MLPImagePolicy from a diffusion_policy checkpoint."""
        import dill
        if "diffusion_policy" not in sys.path:
            sys.path.insert(0, "diffusion_policy")
        from hydra.utils import instantiate

        payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=dill)
        assert "cfg" in payload and "state_dicts" in payload, (
            f"Checkpoint must contain 'cfg' and 'state_dicts' keys, got {list(payload.keys())}"
        )
        cfg = payload["cfg"]
        policy = instantiate(cfg.policy)
        policy.load_state_dict(payload["state_dicts"]["model"])
        return policy

    def _normalize_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize observations using BC normalizer."""
        result = {}
        for key, value in obs_dict.items():
            if key in self.bc_normalizer.params_dict:
                from diffusion_policy.model.common.normalizer import _normalize
                result[key] = _normalize(value, self.bc_normalizer.params_dict[key], forward=True)
            else:
                result[key] = value
        return result

    def _get_policy_obs(self, obs: TensorDict) -> TensorDict:
        """Extract the policy observation group from the full obs TensorDict."""
        policy_keys = self.obs_groups.get("policy", ["policy"])
        assert len(policy_keys) == 1, f"Expected single policy group, got {policy_keys}"
        return obs[policy_keys[0]]

    def get_actor_obs(self, policy_obs: TensorDict) -> torch.Tensor:
        """Encode policy observations through the BC pipeline.

        Returns:
            Flat feature tensor (B, n_obs_steps * obs_feature_dim) ready for trunk.
        """
        obs_dict = {k: policy_obs[k] for k in policy_obs.keys()}

        first_val = next(iter(obs_dict.values()))
        B = first_val.shape[0]
        T = self.n_obs_steps

        nobs = self._normalize_obs(obs_dict)

        flat_nobs = {}
        for key, val in nobs.items():
            assert val.shape[0] == B, f"get_actor_obs: {key} batch dim {val.shape[0]} != {B}"
            assert val.shape[1] >= T, f"get_actor_obs: {key} time dim {val.shape[1]} < n_obs_steps={T}"
            flat_nobs[key] = val[:, :T].reshape(-1, *val.shape[2:])

        if self.freeze_encoder:
            with torch.no_grad():
                features = self.obs_encoder(flat_nobs)
            features = features.detach()
        else:
            features = self.obs_encoder(flat_nobs)

        assert features.shape == (B * T, self.obs_feature_dim), (
            f"Encoder output {features.shape} != expected ({B * T}, {self.obs_feature_dim})"
        )
        return features.reshape(B, T * self.obs_feature_dim)

    def _update_distribution(self, trunk_features: torch.Tensor) -> None:
        """Compute action distribution in env space from trunk features."""
        h = self.trunk(trunk_features)
        mean_norm = self.mean_head(h)
        log_std_norm = self.log_std_head(h).clamp(
            min=self.log_std_limits[0], max=self.log_std_limits[1]
        )
        std_norm = torch.exp(log_std_norm)

        mean_env = (mean_norm - self.action_offset) / self.action_scale
        std_env = std_norm / self.action_scale

        mean_env = torch.nan_to_num(mean_env, nan=0.0)
        std_env = torch.nan_to_num(std_env, nan=1.0, posinf=1e3)

        self.distribution = Normal(mean_env, std_env.clamp(min=1e-6))

    def act(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        policy_obs = self._get_policy_obs(obs)
        features = self.get_actor_obs(policy_obs)
        self._update_distribution(features)
        actions = self.distribution.sample()
        assert actions.shape[-1] == self.num_actions, (
            f"act() output dim {actions.shape[-1]} != {self.num_actions}"
        )
        return actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self._get_policy_obs(obs)
        features = self.get_actor_obs(policy_obs)
        h = self.trunk(features)
        mean_norm = self.mean_head(h)
        mean_env = (mean_norm - self.action_offset) / self.action_scale
        return mean_env

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[g] for g in self.obs_groups["critic"]]
        result = torch.cat(obs_list, dim=-1)
        assert result.ndim == 2, f"Critic obs must be 2D, got {result.ndim}D with shape {result.shape}"
        return result

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        value = self.critic(critic_obs)
        assert value.shape[-1] == 1, f"Critic output dim {value.shape[-1]} != 1"
        return value

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None, "Must call act() before get_actions_log_prob()."
        assert actions.shape[-1] == self.num_actions
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.obs_encoder.eval()
        return self

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return (None, None)

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
