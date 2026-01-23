# Copyright (c) 2026, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class SimplePPO:
    """Simplified PPO: actor-only bandit updates with PPO-style ratio clipping.

    This is intended for the adversary in a one-step (bandit) setting:
    - No critic/value learning (no value loss, no GAE). Baseline is an EMA of past mean returns.
    - Still uses PPO-style ratio clipping and optional KL-adaptive LR for stability.
    - Does NOT call policy.evaluate().
    """

    policy: ActorCritic

    def __init__(
        self,
        policy: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        schedule: str = "fixed",
        desired_kl: float | None = None,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # Baseline is an EMA of past returns; start at 0 so constant-regret rewards still learn.
        baseline_momentum: float = 0.9,
        multi_gpu_cfg: dict | None = None,
        **_: object,
    ) -> None:
        self.device = device
        self.policy = policy.to(self.device)

        self.clip_param = float(clip_param)
        self.entropy_coef = float(entropy_coef)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.max_grad_norm = float(max_grad_norm)

        self.schedule = str(schedule)
        self.desired_kl = desired_kl
        self.learning_rate = float(learning_rate)
        self.normalize_advantage_per_mini_batch = bool(normalize_advantage_per_mini_batch)
        self.baseline_momentum = float(baseline_momentum)
        self._running_return_mean = torch.tensor(0.0, device=self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Dummy value (RolloutStorage requires it for training_type='rl')
        self.transition.values = torch.zeros(
            (self.transition.actions.shape[0], 1), dtype=torch.float, device=self.device
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        assert self.storage is not None
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Bandit: no GAE/bootstrapping.
        return

    def update(self) -> dict[str, float]:
        assert self.storage is not None

        # Returns (bandit): rewards
        returns = self.storage.rewards.flatten(0, 1)  # [B, 1]
        batch_mean_return = returns.mean()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(batch_mean_return, op=torch.distributed.ReduceOp.SUM)
            batch_mean_return /= self.gpu_world_size

        baseline = self._running_return_mean
        # Update running baseline (EMA)
        self._running_return_mean = (
            self.baseline_momentum * self._running_return_mean + (1.0 - self.baseline_momentum) * batch_mean_return
        )

        generator = self.storage.bandit_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        num_updates = 0
        for (
            obs_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            del dones_batch
            del hidden_states_batch
            del masks_batch

            advantages_batch = rewards_batch - baseline
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute distribution under current params
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            entropy_batch = self.policy.entropy

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            entropy_term = entropy_batch.mean()
            loss = surrogate_loss - self.entropy_coef * entropy_term

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # KL divergence: compute *post-update* KL(old || new) for logging/adaptive LR.
            # This makes KL meaningful even when num_updates == 1 (common for bandit adversary).
            kl_mean = None
            if self.desired_kl is not None:
                with torch.inference_mode():
                    self.policy.act(obs_batch)
                    new_mu_batch = self.policy.action_mean
                    new_sigma_batch = self.policy.action_std
                    kl = torch.sum(
                        torch.log(new_sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - new_mu_batch))
                        / (2.0 * torch.square(new_sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                # Optional KL-adaptive LR (applies to future updates)
                if self.schedule == "adaptive":
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = float(lr_tensor.item())
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy_term.item())
            if kl_mean is not None:
                mean_kl += float(kl_mean.item())
            num_updates += 1

        if num_updates > 0:
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            if self.desired_kl is not None:
                mean_kl /= num_updates

        self.storage.clear()
        loss_dict = {
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        if self.desired_kl is not None:
            loss_dict["kl"] = mean_kl
        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (for distributed training)."""
        for param in self.policy.parameters():
            torch.distributed.broadcast(param.data, src=0)

