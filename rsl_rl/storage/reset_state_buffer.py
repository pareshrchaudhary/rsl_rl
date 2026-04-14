# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU ring buffer for validated scene states (same format as OmniReset .pt files).

Each slot bundles together everything needed to (a) reset an env to a validated
state in Phase B, and (b) train the adversary on the action that produced it:

  scene_state, adv_obs, action, log_prob, mu, sigma, gen_reward

The adversary tuple (obs, action, log_prob, mu, sigma) is captured at the
moment the adversary actually sampled the action in Phase A — this is what
SimplePPO needs for an honest on-policy ratio-clipped update.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class ResetStateBuffer:
    """Stores validated scene states bundled with the full adversary PPO transition."""

    def __init__(self, capacity: int, device: str, num_envs: int) -> None:
        self.capacity = capacity
        self.device = device
        self.num_envs = num_envs
        self._write_idx = 0
        self._size = 0
        self._storage: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
        # Adversary PPO transition fields, allocated lazily on first push.
        self._adv_obs: dict[str, torch.Tensor] | None = None  # flat TD: key -> (capacity, *inner)
        self._actions: torch.Tensor | None = None     # (capacity, action_dim)
        self._adv_log_prob: torch.Tensor | None = None  # (capacity, 1)
        self._adv_mu: torch.Tensor | None = None        # (capacity, action_dim)
        self._adv_sigma: torch.Tensor | None = None     # (capacity, action_dim)
        self._gen_rewards: torch.Tensor | None = None   # (capacity,)
        # (num_envs,) buffer slot pinned to each env; -1 = unassigned (drawn on next sample).
        self._per_env_slot: torch.Tensor = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    @property
    def occupancy(self) -> int:
        return self._size

    def push(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: torch.Tensor,
        *,
        adv_obs: TensorDict | dict[str, torch.Tensor] | None = None,
        actions: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        mus: torch.Tensor | None = None,
        sigmas: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ) -> int:
        """Push validated state + adversary PPO transition for given env_ids.

        All adversary fields (adv_obs, actions, log_probs, mus, sigmas, rewards)
        should correspond to the action that produced this state in Phase A.
        Each tensor argument is indexed by ``env_ids`` and written into the
        ring buffer in lockstep, so all fields stay aligned per slot.
        """
        n = env_ids.shape[0]
        if n == 0:
            return 0
        if self._storage is None:
            self._storage = self._allocate_storage(state)

        write_indices = (torch.arange(n, device=self.device) + self._write_idx) % self.capacity
        for category in state:
            for asset_name in state[category]:
                for field_name, tensor in state[category][asset_name].items():
                    self._storage[category][asset_name][field_name][write_indices] = tensor[env_ids]

        if adv_obs is not None:
            if self._adv_obs is None:
                self._adv_obs = {
                    k: torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device)
                    for k, v in adv_obs.items()
                }
            for k, v in adv_obs.items():
                self._adv_obs[k][write_indices] = v[env_ids]

        if actions is not None:
            if self._actions is None:
                self._actions = torch.zeros(self.capacity, actions.shape[-1], device=self.device)
            self._actions[write_indices] = actions[env_ids]

        if log_probs is not None:
            if self._adv_log_prob is None:
                self._adv_log_prob = torch.zeros(self.capacity, 1, device=self.device)
            self._adv_log_prob[write_indices] = log_probs[env_ids].view(-1, 1)

        if mus is not None:
            if self._adv_mu is None:
                self._adv_mu = torch.zeros(self.capacity, mus.shape[-1], device=self.device)
            self._adv_mu[write_indices] = mus[env_ids]

        if sigmas is not None:
            if self._adv_sigma is None:
                self._adv_sigma = torch.zeros(self.capacity, sigmas.shape[-1], device=self.device)
            self._adv_sigma[write_indices] = sigmas[env_ids]

        if rewards is not None:
            if self._gen_rewards is None:
                self._gen_rewards = torch.zeros(self.capacity, device=self.device)
            self._gen_rewards[write_indices] = rewards[env_ids]

        self._write_idx = (self._write_idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        return n

    def get_adversary_tuples_for_envs(self) -> dict | None:
        """Return per-env adversary PPO tuples based on ``_per_env_slot``.

        Used by Phase B's adversary update: each env i is mapped to the buffer
        slot it was reset from, so we get back the (obs, action, log_prob, mu,
        sigma, gen_reward) for the action that produced env i's reset state.
        Returns None if any required field is missing.
        """
        if self._adv_obs is None or self._actions is None:
            return None
        if self._adv_log_prob is None or self._adv_mu is None or self._adv_sigma is None:
            return None
        if self._gen_rewards is None:
            return None
        slots = self._per_env_slot
        return {
            "obs": {k: v[slots].clone() for k, v in self._adv_obs.items()},
            "action": self._actions[slots].clone(),
            "log_prob": self._adv_log_prob[slots].clone(),
            "mu": self._adv_mu[slots].clone(),
            "sigma": self._adv_sigma[slots].clone(),
            "gen_reward": self._gen_rewards[slots].clone(),
        }

    def sample_for_envs(self, env_ids: torch.Tensor) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Return states for specific env_ids, pinning each env to a fixed buffer slot.

        Each env gets assigned a slot the first time it's sampled after a
        ``reset_per_env_tracking`` call; every subsequent sample for the same
        env returns the *same* slot. This gives the adversary regret estimator
        N attempts on one validated state, so ``max - mean`` is meaningful and
        the adversary tuple looked up via ``_per_env_slot`` matches the
        returns that produced it.
        """
        n = env_ids.shape[0]
        if self._size < n:
            raise RuntimeError(f"Requested {n} samples but only {self._size} available.")
        if self._storage is None:
            raise RuntimeError("No data pushed yet.")

        current = self._per_env_slot[env_ids]
        unassigned = current < 0
        if unassigned.any():
            new_indices = torch.randint(0, self._size, (int(unassigned.sum().item()),), device=self.device)
            current[unassigned] = new_indices
            self._per_env_slot[env_ids] = current
        indices = current

        result: dict = {}
        for category, assets in self._storage.items():
            result[category] = {}
            for asset_name, fields in assets.items():
                result[category][asset_name] = {
                    k: v[indices].clone() for k, v in fields.items()
                }
        return result

    def reset_per_env_tracking(self) -> None:
        """Clear per-env slot assignments so the next sample draws fresh slots."""
        self._per_env_slot.fill_(-1)

    def get_per_env_gen_rewards(self) -> torch.Tensor | None:
        """Get generation rewards for all envs based on per-env buffer slot tracking.

        Returns shape ``(num_envs,)`` — one gen_reward per env.
        """
        if self._gen_rewards is None:
            return None
        return self._gen_rewards[self._per_env_slot].clone()

    def clear(self) -> None:
        self._write_idx = 0
        self._size = 0

    def _allocate_storage(self, template: dict) -> dict:
        storage: dict = {}
        for category, assets in template.items():
            storage[category] = {}
            for asset_name, fields in assets.items():
                storage[category][asset_name] = {}
                for field_name, tensor in fields.items():
                    storage[category][asset_name][field_name] = torch.zeros(
                        (self.capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=self.device,
                    )
        return storage

    def __repr__(self) -> str:
        return f"ResetStateBuffer(capacity={self.capacity}, occupancy={self._size})"
