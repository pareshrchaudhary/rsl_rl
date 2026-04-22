# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inline-settling multi-agent runner.

Every env maintains a mode in ``{LIVE, SETTLING}`` and transitions between
them continuously. LIVE envs are driven
by the protagonist and contribute gradient to PPO. SETTLING envs are driven
by the adversary for a short window (``settle_max_steps`` control steps);
their transitions are stored with ``valid_mask=0`` so PPO ignores them. On
settle-valid, the pre-success scene state is written back to sim and the env
flips to LIVE for ``regret_k`` episodes, then flips back to SETTLING with a
fresh adversary proposal. Teacher tuples (one per proposal) accumulate in a
commit ring that drains into an adversary PPO update once every rank has
``num_envs`` commits.

The ``episode_length_buf`` manipulation at LIVE→SETTLING reuses Isaac's
natural time_out + full reset-event chain to close each settle window, so
adversary validation runs through the same reset-event path the protagonist
uses — just per-env instead of global.
"""

from __future__ import annotations

import os
import time
import torch
import warnings
from collections import deque
from dataclasses import dataclass
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms.ppo_cage import PPO
from rsl_rl.algorithms.simple_ppo import SimplePPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, AsymmetricActorCritic, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.utils.logger import log_iter_metrics


# Per-env mode flag; uint8 on device. LIVE envs contribute PPO gradient;
# SETTLING envs are adversary-driven and masked out of the loss.
MODE_LIVE = 0
MODE_SETTLING = 1


@dataclass
class InlineSettlingConfig:
    """Production parameters for inline settling."""
    settle_max_steps: int = 20               # 2.0s window at 0.1s control step
    invalid_settle_penalty: float = -1.0     # reward for a forced-LIVE teacher tuple
    max_resample_retries: int = 5            # per-env settle attempts before giving up
    adversary_update_batch_size: int | None = None  # None ⇒ num_envs (per-rank)

    @classmethod
    def from_cfg(cls, cfg_dict: dict) -> "InlineSettlingConfig":
        raw = cfg_dict.get("inline_settling", {}) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


class MultiAgentRunner:
    """Continuous per-env LIVE/SETTLING loop with masked PPO + rolling adversary updates."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.alg_adversary_cfg_raw = train_cfg["adversary_algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.policy_adversary_cfg_raw = train_cfg["adversary_policy"]
        self.obs_groups_raw = train_cfg["obs_groups"]
        self.adversary_obs_groups_raw = train_cfg["adversary_obs_groups"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Action split: student drives first N-adversary_action_dim entries,
        # adversary drives the tail. Concatenated per-step with per-env mode
        # routing (zero student on SETTLING, zero adversary on LIVE).
        self.adversary_action_dim = self.cfg["adversary_robot_parameters"]
        self.policy_action_dim = int(self.env.num_actions - self.adversary_action_dim)

        obs = self.env.get_observations()

        self.cfg["obs_groups"] = resolve_obs_groups(obs, dict(self.obs_groups_raw), ["critic"])

        adversary_obs_groups_raw = dict(self.adversary_obs_groups_raw)
        if "critic" not in adversary_obs_groups_raw:
            adversary_obs_groups_raw["critic"] = list(adversary_obs_groups_raw.get("policy", []))
        self.adversary_obs_groups = resolve_obs_groups(obs, adversary_obs_groups_raw, ["critic"])

        # IPPO: independent PPO for protagonist; SimplePPO bandit for adversary.
        self.alg = self._construct_agent_algorithm(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            policy_cfg=self.policy_cfg,
            alg_cfg=self.alg_cfg,
            action_dim=self.policy_action_dim,
            storage_horizon=self.num_steps_per_env,
        )
        self.alg_adversary = self._construct_agent_algorithm(
            obs=obs,
            obs_groups=self.adversary_obs_groups,
            policy_cfg=self.policy_adversary_cfg_raw,
            alg_cfg=self.alg_adversary_cfg_raw,
            action_dim=self.adversary_action_dim,
            storage_horizon=1,
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos: list[str] = [str(rsl_rl.__file__)]

        self.beta_gen_reward = float(self.cfg.get("beta_gen_reward", 1.0))
        self.adversary_kl_penalty_coef = float(self.cfg.get("adversary_kl_penalty_coef", 0.0))
        self.alg_adversary.kl_penalty_coef = self.adversary_kl_penalty_coef
        self.regret_k = int(self.cfg.get("regret_k", 6))

        self.inline = InlineSettlingConfig.from_cfg(self.cfg)

        # Per-env mode + settle bookkeeping. All resident on device, mutated
        # in-place each step. At 65k envs these are ~2MB total; cheap.
        n = self.env.num_envs
        self.env_mode = torch.full((n,), MODE_LIVE, dtype=torch.uint8, device=self.device)
        self.settle_remaining = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.settle_retries = torch.zeros(n, dtype=torch.int32, device=self.device)
        # Pre-success scene snapshot + proposal gen_reward, filled on
        # success-in-settle and restored on settle-valid. Shape matches
        # scene.get_state() output; allocated lazily on first stash.
        self._pre_success_state: dict | None = None
        self._pending_teacher = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pending_gen_reward = torch.zeros(n, dtype=torch.float, device=self.device)
        # K-episode pinning: protagonist gets ``regret_k`` LIVE episodes on the
        # same validated start state before the env flips to a new proposal.
        self._live_episodes_since_settle = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._per_env_live_returns: list[list[float]] = [[] for _ in range(n)]
        # Teacher commit ring — drained into an adversary PPO update once all
        # ranks have ``_teacher_batch_size_target`` tuples.
        self._teacher_commit: dict[str, list] = {
            "obs": [], "action": [], "log_prob": [], "mu": [], "sigma": [],
            "gen_reward": [], "reward": [],
        }
        # Latest adversary-update snapshot, emitted per-iter under
        # ``Adversary/*`` so WandB panels configured for the old cycle-based
        # logger keep populating.
        self._inline_last_adv_loss: dict[str, float] = {}
        self._inline_last_adv_rewards: dict[str, float] = {}

        # Per-iter running buffers of K-pin commit stats. Reset each iter,
        # averaged into ``Metrics/adversary/*`` — one scalar per iter, mean
        # over every K-pin commit that fired during the iter.
        self._iter_commit_stats: dict[str, list[float]] = {
            "regret": [], "max_batch_returns": [], "mean_batch_returns": [],
        }

        self._success_term_idx: int | None = None

    # =====================================================================
    # Adversary sampling + manual transition population
    # =====================================================================

    def _sample_adversary_capture(
        self, obs, env_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample adversary actions and snapshot ``(obs, action, log_prob, mu, sigma)``
        into per-env scratch tensors. Uses ``policy.act`` (stochastic) so the PPO
        ratio at update time is honest. ``env_indices=None`` refreshes all envs;
        otherwise only those slots are overwritten.
        """
        with torch.inference_mode():
            action = self.alg_adversary.policy.act(obs).detach()
            log_prob = self.alg_adversary.policy.get_actions_log_prob(action).detach()
            mu = self.alg_adversary.policy.action_mean.detach().clone()
            sigma_raw = self.alg_adversary.policy.action_std.detach()
            if sigma_raw.dim() == 1:
                sigma = sigma_raw.unsqueeze(0).expand(self.env.num_envs, -1).clone()
            else:
                sigma = sigma_raw.clone()

        log_prob_2d = log_prob.view(-1, 1)

        if env_indices is None:
            self._scratch_adv_obs = obs.clone()
            self._scratch_adv_action = action.clone()
            self._scratch_adv_log_prob = log_prob_2d.clone()
            self._scratch_adv_mu = mu.clone()
            self._scratch_adv_sigma = sigma.clone()
        else:
            self._scratch_adv_obs[env_indices] = obs[env_indices]
            self._scratch_adv_action[env_indices] = action[env_indices]
            self._scratch_adv_log_prob[env_indices] = log_prob_2d[env_indices]
            self._scratch_adv_mu[env_indices] = mu[env_indices]
            self._scratch_adv_sigma[env_indices] = sigma[env_indices]

        return action

    def _adversary_update_from_tuples(
        self, tuples: dict, rewards: torch.Tensor, last_obs
    ) -> dict:
        """SimplePPO update that bypasses ``alg.act()`` and uses the
        log_prob/mu/sigma stashed at action-time — so the ratio is correct for
        action samples drawn under π_{n-1}.
        """
        alg = self.alg_adversary
        n = tuples["action"].shape[0]

        alg.transition.observations = tuples["obs"]
        alg.transition.actions = tuples["action"]
        alg.transition.actions_log_prob = tuples["log_prob"]
        alg.transition.action_mean = tuples["mu"]
        alg.transition.action_sigma = tuples["sigma"]
        alg.transition.values = torch.zeros((n, 1), dtype=torch.float, device=self.device)

        dummy_dones = torch.ones((n, 1), dtype=torch.float, device=self.device)
        alg.process_env_step(tuples["obs"], rewards, dummy_dones, {})
        alg.compute_returns(last_obs)  # no-op for SimplePPO
        return alg.update()

    def _resolve_success_term_idx(self) -> int:
        if self._success_term_idx is not None:
            return self._success_term_idx
        term_mgr = self.env.unwrapped.termination_manager
        idx = term_mgr._term_name_to_term_idx.get("success")
        if idx is None:
            raise ValueError(
                f"Env must expose a 'success' termination term. "
                f"Available: {list(term_mgr._term_name_to_term_idx.keys())}"
            )
        self._success_term_idx = idx
        return idx

    # =====================================================================
    # Teacher commit ring + rolling adversary update
    # =====================================================================

    def _snapshot_teacher_tuple(self, env_ids: torch.Tensor) -> dict:
        """Extract the adversary scratch slots for ``env_ids`` as detached clones."""
        return {
            "obs": {k: v[env_ids].detach().clone() for k, v in self._scratch_adv_obs.items()},
            "action": self._scratch_adv_action[env_ids].detach().clone(),
            "log_prob": self._scratch_adv_log_prob[env_ids].detach().clone(),
            "mu": self._scratch_adv_mu[env_ids].detach().clone(),
            "sigma": self._scratch_adv_sigma[env_ids].detach().clone(),
        }

    def _commit_teacher(self, env_id: int, gen_reward: float, combined_reward: float) -> None:
        """Append a single teacher tuple to the commit ring using the env's
        current scratch slot. Clears the pending flag."""
        idx_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
        tup = self._snapshot_teacher_tuple(idx_t)
        self._teacher_commit["obs"].append(tup["obs"])
        self._teacher_commit["action"].append(tup["action"])
        self._teacher_commit["log_prob"].append(tup["log_prob"])
        self._teacher_commit["mu"].append(tup["mu"])
        self._teacher_commit["sigma"].append(tup["sigma"])
        self._teacher_commit["gen_reward"].append(float(gen_reward))
        self._teacher_commit["reward"].append(float(combined_reward))
        self._pending_teacher[env_id] = False
        self._pending_gen_reward[env_id] = 0.0

    def _teacher_batch_size_target(self) -> int:
        bs = self.inline.adversary_update_batch_size
        return int(bs) if bs is not None else self.env.num_envs

    def _maybe_fire_adversary_update(self, last_obs) -> dict | None:
        """Drain the commit ring into an adversary PPO update when every rank
        has at least ``_teacher_batch_size_target`` tuples.

        MIN-syncs the ready decision across ranks so every rank fires together
        or skips together — otherwise the NCCL all_reduce inside ``alg.update()``
        would deadlock against a rank still doing protagonist rollout.
        """
        target = self._teacher_batch_size_target()
        local_ready = int(len(self._teacher_commit["reward"]) >= target)
        if self.is_distributed:
            ready_t = torch.tensor(local_ready, dtype=torch.int, device=self.device)
            torch.distributed.all_reduce(ready_t, op=torch.distributed.ReduceOp.MIN)
            global_ready = bool(ready_t.item())
        else:
            global_ready = bool(local_ready)
        if not global_ready:
            return None

        def _pop_n(lst: list, n: int) -> list:
            head = lst[:n]
            del lst[:n]
            return head

        obs_list = _pop_n(self._teacher_commit["obs"], target)
        action_list = _pop_n(self._teacher_commit["action"], target)
        log_prob_list = _pop_n(self._teacher_commit["log_prob"], target)
        mu_list = _pop_n(self._teacher_commit["mu"], target)
        sigma_list = _pop_n(self._teacher_commit["sigma"], target)
        gen_reward_list = _pop_n(self._teacher_commit["gen_reward"], target)
        reward_list = _pop_n(self._teacher_commit["reward"], target)

        stacked_obs = {k: torch.cat([o[k] for o in obs_list], dim=0) for k in obs_list[0].keys()}
        adv_tuples = {
            "obs": stacked_obs,
            "action": torch.cat(action_list, dim=0),
            "log_prob": torch.cat(log_prob_list, dim=0),
            "mu": torch.cat(mu_list, dim=0),
            "sigma": torch.cat(sigma_list, dim=0),
            "gen_reward": torch.tensor(gen_reward_list, dtype=torch.float, device=self.device),
        }
        combined_rewards = torch.tensor(reward_list, dtype=torch.float, device=self.device).unsqueeze(-1)

        self._prepare_adversary_storage(n=target, obs_spec=adv_tuples["obs"])
        loss_dict = self._adversary_update_from_tuples(adv_tuples, combined_rewards, last_obs=last_obs)

        self._inline_last_adv_loss = {k: float(v) for k, v in loss_dict.items()}
        self._inline_last_adv_rewards = {
            "mean_total_reward": float(combined_rewards.mean().item()),
            "mean_gen_reward": float(adv_tuples["gen_reward"].mean().item()),
            "mean_regret": float(
                (combined_rewards.squeeze(-1) - self.beta_gen_reward * adv_tuples["gen_reward"]).mean().item()
            ),
        }
        return loss_dict

    def _emit_inline_iter_metrics(self, writer, it: int) -> None:
        """Push the most-recent adversary-update stats under ``Adversary/*`` and
        the per-iter K-pin commit stats under ``Metrics/adversary/*`` — the
        latter alongside ``Metrics/task_command/*`` so they aggregate on the
        same dashboard. Values are means over every commit fired during the
        iter. Buffer is reset after emission."""
        if writer is not None:
            for k, v in self._inline_last_adv_loss.items():
                writer.add_scalar(f"Adversary/{k}", v, it)
            for k, v in self._inline_last_adv_rewards.items():
                writer.add_scalar(f"Adversary/{k}", v, it)

        commit_means: dict[str, float] = {}
        for k, vals in self._iter_commit_stats.items():
            if vals:
                commit_means[k] = sum(vals) / len(vals)
        if commit_means:
            if writer is not None:
                for k, v in commit_means.items():
                    writer.add_scalar(f"Metrics/adversary/{k}", v, it)
            line = " | ".join(f"{k}: {v:.4f}" for k, v in commit_means.items())
            print(f"{'Metrics/adversary:':>35} {line}")
        for vals in self._iter_commit_stats.values():
            vals.clear()

    # =====================================================================
    # Pre-success state capture + restore
    # =====================================================================

    def _write_state_to_sim(self, state: dict, env_ids: torch.Tensor) -> None:
        """Restore the saved scene state for ``env_ids``: articulation root +
        velocity + joint state, rigid object root + velocity, env-relative
        xyz with ``env_origins`` added back in. ``scene.write_data_to_sim()``
        flushes the writes."""
        if env_ids.numel() == 0:
            return
        scene = self.env.unwrapped.scene
        env_origins = scene.env_origins[env_ids]
        with torch.inference_mode():
            for category, assets in state.items():
                if category == "articulation":
                    for asset_name, fields in assets.items():
                        if asset_name not in scene._articulations:
                            continue
                        articulation = scene._articulations[asset_name]
                        root_pose = fields["root_pose"][env_ids].clone()
                        root_pose[:, :3] += env_origins
                        articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                        articulation.write_root_velocity_to_sim(
                            fields["root_velocity"][env_ids].clone(), env_ids=env_ids
                        )
                        joint_position = fields["joint_position"][env_ids].clone()
                        joint_velocity = fields["joint_velocity"][env_ids].clone()
                        articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
                        articulation.set_joint_position_target(joint_position, env_ids=env_ids)
                        articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
                elif category == "rigid_object":
                    for asset_name, fields in assets.items():
                        if asset_name not in scene._rigid_objects:
                            continue
                        rigid_object = scene._rigid_objects[asset_name]
                        root_pose = fields["root_pose"][env_ids].clone()
                        root_pose[:, :3] += env_origins
                        rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                        rigid_object.write_root_velocity_to_sim(
                            fields["root_velocity"][env_ids].clone(), env_ids=env_ids
                        )
            scene.write_data_to_sim()

    def _stash_pre_success_for(self, state: dict, env_ids: torch.Tensor) -> None:
        """Write a per-env slice of ``state`` into ``self._pre_success_state``.
        Lazily allocates the full-env-size scratch on first call."""
        if env_ids.numel() == 0:
            return
        if self._pre_success_state is None:
            self._pre_success_state = {}
            for category, assets in state.items():
                self._pre_success_state[category] = {}
                for asset_name, fields in assets.items():
                    self._pre_success_state[category][asset_name] = {
                        k: torch.zeros_like(v) for k, v in fields.items()
                    }
        for category, assets in state.items():
            for asset_name, fields in assets.items():
                for k, v in fields.items():
                    self._pre_success_state[category][asset_name][k][env_ids] = v[env_ids].clone()

    # =====================================================================
    # Per-step mode transitions
    # =====================================================================

    def _step_modes(
        self,
        dones: torch.Tensor,
        term_mgr,
        success_idx: int,
        scene_state_pre: dict,
        rewards: torch.Tensor,
        obs: TensorDict,
    ) -> bool:
        """Resolve per-env mode transitions after a sim-step.

        Logic:
          1. If a SETTLING env saw ``success`` this step, snapshot the pre-step
             scene state + the step's reward as the pending proposal's ``gen_reward``.
          2. Any done during SETTLING is a resolution event. Success → valid;
             anything else (abnormal / natural time_out) → invalid.
          3. Valid: write the stashed pre-success state back to sim, flip to
             LIVE, clear K-episode counter.
          4. Invalid with retries left: push ``episode_length_buf`` near
             ``max_episode_length`` so Isaac's natural time_out fires
             ``settle_max_steps`` later; resample adversary.
          5. Invalid with retries exhausted: commit pending tuple with the
             fixed penalty, force the env LIVE.
          6. A LIVE done with a pending teacher: if fewer than ``regret_k``
             returns banked, re-write the same pre-success state and stay LIVE
             (K-episode pinning); on the Kth, compute regret, commit, flip to
             SETTLING with a fresh proposal.
          7. A LIVE done without a pending teacher (bootstrap): flip to
             SETTLING with a fresh proposal.

        Returns True iff any sim state was written — the caller must refresh
        ``obs`` so the next action sample reflects the restored configuration.
        """
        state_was_written = False
        max_ep_len = int(self.env.max_episode_length)
        settle_start = max(0, max_ep_len - self.inline.settle_max_steps)

        settling_mask = (self.env_mode == MODE_SETTLING)
        live_mask = ~settling_mask
        dones_bool = dones.to(torch.bool).view(-1)

        # ---- (1) Stash pre-success state for SETTLING envs that saw success.
        success_this_step = term_mgr._last_episode_dones[:, success_idx].to(torch.bool) & dones_bool
        success_in_settle = success_this_step & settling_mask
        if success_in_settle.any():
            success_ids = success_in_settle.nonzero(as_tuple=False).squeeze(-1)
            self._stash_pre_success_for(scene_state_pre, success_ids)
            self._pending_gen_reward[success_ids] = rewards[success_ids].view(-1).float()

        # ---- (2) Resolution: any SETTLING done this step, or timer hit 0.
        done_settling = settling_mask & dones_bool
        timed_out_no_done = settling_mask & (self.settle_remaining == 0) & ~dones_bool
        resolved = done_settling | timed_out_no_done

        if resolved.any():
            valid = resolved & success_in_settle
            invalid = resolved & ~valid

            # ---- (3) Valid settles flip to LIVE on pre-success state.
            if valid.any():
                valid_ids = valid.nonzero(as_tuple=False).squeeze(-1)
                if self._pre_success_state is not None:
                    self._write_state_to_sim(self._pre_success_state, valid_ids)
                    state_was_written = True
                self.env_mode[valid_ids] = MODE_LIVE
                self.settle_remaining[valid_ids] = 0
                self.settle_retries[valid_ids] = 0
                self._live_episodes_since_settle[valid_ids] = 0
                for eid in valid_ids.tolist():
                    self._per_env_live_returns[eid] = []

            # ---- (4,5) Invalid settles: retry if budget left, else forced-LIVE.
            if invalid.any():
                invalid_ids = invalid.nonzero(as_tuple=False).squeeze(-1)
                can_retry = self.settle_retries[invalid_ids] < self.inline.max_resample_retries
                retry_ids = invalid_ids[can_retry]
                giveup_ids = invalid_ids[~can_retry]

                if retry_ids.numel() > 0:
                    with torch.inference_mode():
                        self.env.episode_length_buf[retry_ids] = settle_start
                    self.settle_remaining[retry_ids] = self.inline.settle_max_steps
                    self.settle_retries[retry_ids] += 1
                    self._sample_adversary_capture(obs, env_indices=retry_ids)

                if giveup_ids.numel() > 0:
                    for eid in giveup_ids.tolist():
                        if bool(self._pending_teacher[eid].item()):
                            self._commit_teacher(
                                env_id=eid,
                                gen_reward=float(self._pending_gen_reward[eid].item()),
                                combined_reward=self.inline.invalid_settle_penalty,
                            )
                    self.env_mode[giveup_ids] = MODE_LIVE
                    self.settle_remaining[giveup_ids] = 0
                    self.settle_retries[giveup_ids] = 0
                    self._live_episodes_since_settle[giveup_ids] = 0

        # ---- (6,7) LIVE dones: K-episode pinning or fresh-proposal flip.
        live_done = live_mask & dones_bool
        if live_done.any():
            ids = live_done.nonzero(as_tuple=False).squeeze(-1)
            for eid in ids.tolist():
                if bool(self._pending_teacher[eid].item()):
                    self._live_episodes_since_settle[eid] += 1
                    if int(self._live_episodes_since_settle[eid].item()) < self.regret_k:
                        # Keep the env on the validated start state for
                        # another LIVE episode.
                        if self._pre_success_state is not None:
                            self._write_state_to_sim(
                                self._pre_success_state,
                                torch.tensor([eid], device=self.device, dtype=torch.long),
                            )
                            state_was_written = True
                    else:
                        returns = self._per_env_live_returns[eid][:self.regret_k]
                        max_batch_returns = max(returns)
                        mean_batch_returns = sum(returns) / self.regret_k
                        regret = max_batch_returns - mean_batch_returns
                        self._iter_commit_stats["regret"].append(regret)
                        self._iter_commit_stats["max_batch_returns"].append(max_batch_returns)
                        self._iter_commit_stats["mean_batch_returns"].append(mean_batch_returns)
                        gen_r = float(self._pending_gen_reward[eid].item())
                        combined = self.beta_gen_reward * gen_r + regret
                        self._commit_teacher(env_id=eid, gen_reward=gen_r, combined_reward=combined)
                        self.env_mode[eid] = MODE_SETTLING
                        self.settle_remaining[eid] = self.inline.settle_max_steps
                        self.settle_retries[eid] = 0
                        self._live_episodes_since_settle[eid] = 0
                        self._per_env_live_returns[eid] = []
                        with torch.inference_mode():
                            self.env.episode_length_buf[eid] = settle_start
                        single_id = torch.tensor([eid], device=self.device, dtype=torch.long)
                        self._sample_adversary_capture(obs, env_indices=single_id)
                        self._pending_teacher[eid] = True
                else:
                    # Bootstrap LIVE-done (no prior proposal, or just forced-LIVE):
                    # flip to SETTLING and sample a fresh adversary proposal.
                    self.env_mode[eid] = MODE_SETTLING
                    self.settle_remaining[eid] = self.inline.settle_max_steps
                    self.settle_retries[eid] = 0
                    self._live_episodes_since_settle[eid] = 0
                    with torch.inference_mode():
                        self.env.episode_length_buf[eid] = settle_start
                    single_id = torch.tensor([eid], device=self.device, dtype=torch.long)
                    self._sample_adversary_capture(obs, env_indices=single_id)
                    self._pending_teacher[eid] = True

        return state_was_written

    def _prepare_adversary_storage(self, n: int, obs_spec) -> None:
        self.alg_adversary.init_storage(
            "rl", num_envs=n, num_transitions_per_env=1,
            obs=obs_spec, actions_shape=[self.adversary_action_dim],
        )

    # =====================================================================
    # Main training loop
    # =====================================================================

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            self.alg.broadcast_parameters()
            self.alg_adversary.broadcast_parameters()

        # Term indices + manager handle are resolved once; cached on the instance.
        success_idx = self._resolve_success_term_idx()
        unwrapped_env = self.env.unwrapped
        term_mgr_handle = unwrapped_env.termination_manager

        # Optional handle for the task_command term — used to override the
        # alignment metrics with LIVE-only versions before they're logged.
        try:
            command_term_handle = unwrapped_env.command_manager.get_term("task_command")
        except (AttributeError, KeyError):
            command_term_handle = None

        # Allocate adversary scratch up-front so per-env resample calls
        # inside _step_modes can index into it without reallocating.
        self._sample_adversary_capture(obs, env_indices=None)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Zero template for the adversary slice of the action vector: LIVE
        # envs keep the zeros; SETTLING envs overwrite their row with the
        # scratch adversary action inside the rollout loop.
        adv_action_zero_template = torch.zeros(
            (self.env.num_envs, self.adversary_action_dim), device=self.device
        )

        for it in range(start_iter, tot_iter):
            collection_start = time.time()

            for _ in range(self.num_steps_per_env):
                with torch.inference_mode():
                    policy_actions = self.alg.act(obs)

                    settling_mask = (self.env_mode == MODE_SETTLING)
                    live_mask = ~settling_mask

                    # Student slice: zero on SETTLING, policy action on LIVE.
                    student_actions = policy_actions.clone()
                    if settling_mask.any():
                        student_actions[settling_mask] = 0.0

                    # Adversary slice: zero on LIVE, scratch action on SETTLING.
                    adv_slice = adv_action_zero_template
                    if settling_mask.any():
                        adv_slice = adv_action_zero_template.clone()
                        adv_slice[settling_mask] = self._scratch_adv_action[settling_mask]

                    actions = torch.cat([student_actions, adv_slice], dim=-1)

                    valid_mask = live_mask.float().unsqueeze(-1)
                    scene_state_pre = unwrapped_env.scene.get_state(is_relative=True)

                    # Tell the task_command term which rows are LIVE this
                    # step — its `_update_metrics` and `reset` will skip
                    # SETTLING rows so Isaac's `Metrics/task_command/*`
                    # emits clean protagonist-only means.
                    if command_term_handle is not None:
                        command_term_handle._live_mask.copy_(
                            live_mask.to(command_term_handle._live_mask.device)
                        )

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device),
                    )

                    self.alg.process_env_step(obs, rewards, dones, extras, valid_mask=valid_mask)

                    # Tick settle timers on SETTLING envs.
                    settling_mask_now = (self.env_mode == MODE_SETTLING)
                    if settling_mask_now.any():
                        self.settle_remaining[settling_mask_now] = torch.clamp(
                            self.settle_remaining[settling_mask_now] - 1, min=0
                        )

                    if self.log_dir is not None and dones.any():
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    cur_reward_sum += rewards
                    cur_episode_length += 1

                if done_ids.numel() > 0:
                    done_env_indices = done_ids[:, 0]
                    ep_returns = cur_reward_sum[done_ids][:, 0]
                    cur_lens = cur_episode_length[done_ids][:, 0]

                    done_env_list = done_env_indices.cpu().numpy().tolist()
                    ep_returns_list = ep_returns.cpu().numpy().tolist()
                    cur_lens_list = cur_lens.cpu().numpy().tolist()

                    # Bank LIVE-mode episode returns into the per-env regret
                    # ring before mode resolution rotates this env. Only LIVE
                    # returns feed both rewbuffer and _per_env_live_returns —
                    # SETTLING returns (adversary-driven) are not the
                    # protagonist's responsibility.
                    counted_rets: list[float] = []
                    counted_lens: list[float] = []
                    for env_idx, ret, ep_len in zip(done_env_list, ep_returns_list, cur_lens_list):
                        if int(self.env_mode[env_idx].item()) != MODE_LIVE:
                            continue
                        self._per_env_live_returns[env_idx].append(float(ret))
                        counted_rets.append(float(ret))
                        counted_lens.append(float(ep_len))

                    if self.log_dir is not None and counted_rets:
                        rewbuffer.extend(counted_rets)
                        lenbuffer.extend(counted_lens)

                    cur_reward_sum[done_ids] = 0
                    cur_episode_length[done_ids] = 0

                # Mode transitions AFTER all per-step bookkeeping.
                state_was_written = self._step_modes(
                    dones=dones,
                    term_mgr=term_mgr_handle,
                    success_idx=success_idx,
                    scene_state_pre=scene_state_pre,
                    rewards=rewards,
                    obs=obs,
                )
                if state_was_written:
                    obs = self.env.get_observations().to(self.device)

            collection_time = time.time() - collection_start

            learn_start = time.time()
            with torch.inference_mode():
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()

            # Drain the teacher ring into an adversary update when every rank
            # has enough commits (MIN-sync inside _maybe_fire_adversary_update).
            self._maybe_fire_adversary_update(last_obs=obs)

            learn_time = time.time() - learn_start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
                self.tot_timesteps += collection_size
                self.tot_time += collection_time + learn_time

                log_iter_metrics(
                    writer=self.writer,
                    device=self.device,
                    num_steps_per_env=self.num_steps_per_env,
                    num_envs=self.env.num_envs,
                    gpu_world_size=self.gpu_world_size,
                    alg=self.alg,
                    logger_type=self.logger_type,
                    tot_timesteps=self.tot_timesteps,
                    tot_time=self.tot_time,
                    locs=locals(),
                )
                self._emit_inline_iter_metrics(self.writer, it)

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and self.log_dir is not None and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        save_file_fn = getattr(self.writer, "save_file", None)
                        if callable(save_file_fn):
                            save_file_fn(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # =====================================================================
    # Save / load
    # =====================================================================

    def save(self, path: str, infos: dict | None = None) -> None:
        torch.save({
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

        adversary_path = path.replace(".pt", "_adversary.pt")
        torch.save({
            "model_state_dict": self.alg_adversary.policy.state_dict(),
            "optimizer_state_dict": self.alg_adversary.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, adversary_path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)  # type: ignore
            self.writer.save_model(adversary_path, self.current_learning_iteration)  # type: ignore

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        if resumed_training and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        adversary_path = path.replace(".pt", "_adversary.pt")
        if os.path.exists(adversary_path):
            adversary_loaded_dict = torch.load(adversary_path, weights_only=False, map_location=map_location)
            adversary_resumed_training = self.alg_adversary.policy.load_state_dict(
                adversary_loaded_dict["model_state_dict"]
            )
            if load_optimizer and adversary_resumed_training:
                self.alg_adversary.optimizer.load_state_dict(adversary_loaded_dict["optimizer_state_dict"])

        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        adv_dim = self.adversary_action_dim

        def padded_policy(obs):
            actions = policy(obs)
            pad = torch.zeros(actions.shape[0], adv_dim, device=actions.device)
            return torch.cat([actions, pad], dim=-1)

        return padded_policy

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.alg_adversary.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg_adversary.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    # =====================================================================
    # Infrastructure
    # =====================================================================

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' >= world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' >= world size '{self.gpu_world_size}'."
            )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
            )
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_agent_algorithm(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        policy_cfg: dict,
        alg_cfg: dict,
        action_dim: int,
        storage_horizon: int,
    ) -> PPO:
        alg_cfg = dict(alg_cfg)
        policy_cfg = dict(policy_cfg)

        if "rnd_cfg" in alg_cfg:
            alg_cfg["rnd_cfg"] = None
        alg_cfg = resolve_symmetry_config(alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Use `actor_obs_normalization` "
                "and `critic_obs_normalization` under `policy` instead.",
                DeprecationWarning,
            )
            if policy_cfg.get("actor_obs_normalization") is None:
                policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if policy_cfg.get("critic_obs_normalization") is None:
                policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        actor_critic_class = eval(policy_cfg["class_name"])
        policy_kwargs = {k: v for k, v in policy_cfg.items() if k != "class_name"}
        actor_critic: ActorCritic | ActorCriticRecurrent | AsymmetricActorCritic = actor_critic_class(
            obs, obs_groups, action_dim, **policy_kwargs
        ).to(self.device)

        alg_class = eval(alg_cfg["class_name"])
        alg_kwargs = {k: v for k, v in alg_cfg.items() if k != "class_name"}
        alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs, multi_gpu_cfg=self.multi_gpu_cfg)

        alg.init_storage("rl", self.env.num_envs, storage_horizon, obs, [action_dim])
        return alg

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is None or self.writer is not None or self.disable_logs:
            return
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(
                self.env.cfg, self.cfg,
                {"policy": self.alg_cfg, "adversary": self.alg_adversary_cfg_raw},
                {"policy": self.policy_cfg, "adversary": self.policy_adversary_cfg_raw},
            )
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(
                self.env.cfg, self.cfg,
                {"policy": self.alg_cfg, "adversary": self.alg_adversary_cfg_raw},
                {"policy": self.policy_cfg, "adversary": self.policy_adversary_cfg_raw},
            )
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Use 'neptune', 'wandb', or 'tensorboard'.")
