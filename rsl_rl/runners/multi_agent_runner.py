# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
import h5py
from collections import deque
from dataclasses import dataclass
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms.ppo_cage import PPO
from rsl_rl.algorithms.reinforce import Reinforce
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, AsymmetricActorCritic, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.utils.logger import log_iter_metrics

# LIVE contributes PPO gradient; SETTLING is adversary-driven, masked out.
MODE_LIVE = 0
MODE_SETTLING = 1

@dataclass
class InlineSettlingConfig:
    settle_max_steps: int = 20               # 2.0s window at 0.1s control step
    invalid_settle_penalty: float = -1.0     # teacher reward for exhausted settle proposals
    max_resample_retries: int = 5            # per-env settle attempts before giving up
    force_live_after_max_retries: bool = True
    # Legacy path: failed settle becomes LIVE after retries.
    live_handoff_hold_steps: int = 0         # mask PPO while holding the reset pose after SETTLING
    adversary_update_batch_size: int | None = None  # None ⇒ num_envs (per-rank)
    settling_gripper_default_action: float = -1.0
    # Parameter-only adversary: no SETTLING/LIVE gate or K-pin.
    skip_settling: bool = False

    @classmethod
    def from_cfg(cls, cfg_dict: dict) -> "InlineSettlingConfig":
        raw = cfg_dict.get("inline_settling", {}) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


class MultiAgentRunner:
    """PPO student plus reset-state adversary."""

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
        self.record_parameters = self.cfg.get("record_parameters", False)

        self.adversary_action_dim = self.cfg["adversary_robot_parameters"]
        self.policy_action_dim = int(self.env.num_actions - self.adversary_action_dim)

        obs = self.env.get_observations()

        self.cfg["obs_groups"] = resolve_obs_groups(obs, dict(self.obs_groups_raw), ["critic"])

        adversary_obs_groups_raw = dict(self.adversary_obs_groups_raw)
        if "critic" not in adversary_obs_groups_raw:
            adversary_obs_groups_raw["critic"] = list(adversary_obs_groups_raw.get("policy", []))
        self.adversary_obs_groups = resolve_obs_groups(obs, adversary_obs_groups_raw, ["critic"])

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
        self.regret_k = int(self.cfg.get("regret_k", 6))

        self.inline = InlineSettlingConfig.from_cfg(self.cfg)

        n = self.env.num_envs
        self.env_mode = torch.full((n,), MODE_LIVE, dtype=torch.uint8, device=self.device)
        self.settle_remaining = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.settle_retries = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.live_handoff_remaining = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._pre_success_state: dict | None = None
        self._pending_teacher = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._pending_gen_reward = torch.zeros(n, dtype=torch.float, device=self.device)
        self._live_episodes_since_settle = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._per_env_live_returns: list[list[float]] = [[] for _ in range(n)]
        self._teacher_commit: dict[str, list] = {
            "obs": [], "action": [], "log_prob": [], "mu": [], "sigma": [],
            "gen_reward": [], "reward": [], "regret": [],
        }
        self._inline_last_adv_loss: dict[str, float] = {}
        self._inline_last_adv_rewards: dict[str, float] = {}

        self._success_term_idx: int | None = None

        self._install_live_filtered_reward_logging()

    # Logging

    def _record_packet_to_cpu(self, packet) -> dict | None:
        if not self.record_parameters:
            return None
        if not isinstance(packet, dict):
            return None

        raw_datasets = packet.get("datasets")
        if not isinstance(raw_datasets, dict):
            return None
        datasets = {
            key: value.detach().cpu()
            for key, value in raw_datasets.items()
            if isinstance(key, str) and isinstance(value, torch.Tensor) and value.numel() > 0
        }
        if not datasets:
            return None
        attrs = packet.get("attrs", {})
        file_name = packet.get("file_name", "adversary_records.h5")
        return {
            "datasets": datasets,
            "attrs": attrs if isinstance(attrs, dict) else {},
            "file_name": str(file_name),
        }

    def _merge_record_packets(self, packets: list[dict | None]) -> dict | None:
        non_empty = [packet for packet in packets if isinstance(packet, dict) and packet.get("datasets")]
        if not non_empty:
            return None

        keys = sorted({key for packet in non_empty for key in packet["datasets"]})
        datasets: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [
                packet["datasets"][key]
                for packet in non_empty
                if key in packet["datasets"] and packet["datasets"][key].numel() > 0
            ]
            if values:
                datasets[key] = torch.cat(values, dim=0)
        if not datasets:
            return None

        attrs: dict = {}
        for packet in non_empty:
            attrs.update(packet.get("attrs", {}))
        file_name = str(non_empty[0].get("file_name", "adversary_records.h5"))
        return {"datasets": datasets, "attrs": attrs, "file_name": file_name}

    def _collect_adversary_records(self) -> dict | None:
        if not self.record_parameters or not self.cfg.get("save_accepted_omnireset_datasets", True):
            return None
        local = self._record_packet_to_cpu(
            self._call_multi_agent_env_hook("consume_adversary_hdf5_records", default=None)
        )

        if not self.is_distributed:
            return local

        gathered: list[dict | None] = [None for _ in range(self.gpu_world_size)]
        torch.distributed.all_gather_object(gathered, local)
        if self.gpu_global_rank != 0:
            return None
        return self._merge_record_packets(gathered)

    def _adversary_record_path(self, records: dict) -> str | None:
        if self.log_dir is None or self.disable_logs:
            return None
        file_name = os.path.basename(str(records.get("file_name", "adversary_records.h5")))
        if not file_name:
            file_name = "adversary_records.h5"
        return os.path.join(self.log_dir, file_name)

    @staticmethod
    def _write_h5_attr(h5_file, key: str, value) -> None:
        if isinstance(value, (list, tuple)):
            h5_file.attrs[key] = [str(v).encode("utf-8") for v in value]
        elif isinstance(value, str):
            h5_file.attrs[key] = value
        elif isinstance(value, (int, float, bool)):
            h5_file.attrs[key] = value

    def _write_adversary_records(self, it: int, records: dict | None) -> None:
        if records is None or self.disable_logs:
            return
        datasets = records.get("datasets", {})
        if not isinstance(datasets, dict) or not datasets:
            return
        path = self._adversary_record_path(records)
        if path is None:
            return
        with h5py.File(path, "a") as f:
            for key, value in records.get("attrs", {}).items():
                if isinstance(key, str):
                    self._write_h5_attr(f, key, value)
            group = f.create_group(f"iteration_{it}")
            num_samples = 0
            for key, value in datasets.items():
                if not isinstance(key, str) or not isinstance(value, torch.Tensor) or value.numel() == 0:
                    continue
                group.create_dataset(key, data=value.numpy())
                if value.ndim > 0:
                    num_samples = max(num_samples, int(value.shape[0]))
            group.attrs["num_samples"] = num_samples

    def _install_live_filtered_reward_logging(self) -> None:
        """Filter reward logs to LIVE episodes."""
        reward_mgr = self.env.unwrapped.reward_manager
        env_mode = self.env_mode

        def reset(env_ids=None):
            if env_ids is None:
                env_ids_t = torch.arange(reward_mgr.num_envs, device=reward_mgr.device)
            elif isinstance(env_ids, torch.Tensor):
                env_ids_t = env_ids
            else:
                env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=reward_mgr.device)
            live_ids = env_ids_t[env_mode[env_ids_t] == MODE_LIVE]
            max_ep = reward_mgr._env.max_episode_length_s
            extras: dict[str, torch.Tensor] = {}
            for key, sums in reward_mgr._episode_sums.items():
                if live_ids.numel() > 0:
                    extras["Episode_Reward/" + key] = torch.mean(sums[live_ids]) / max_ep
                else:
                    extras["Episode_Reward/" + key] = torch.empty(0, device=reward_mgr.device)
                sums[env_ids_t] = 0.0
            for term_cfg in reward_mgr._class_term_cfgs:
                term_cfg.func.reset(env_ids=env_ids_t)
            return extras

        reward_mgr.reset = reset

    # Adversary sampling

    def _update_adversary_previous_action_obs(self, action: torch.Tensor, env_indices: torch.Tensor | None) -> None:
        action_term = getattr(getattr(self.env, "unwrapped", self.env).action_manager, "_terms", {}).get("adversaryaction")
        if action_term is None or not hasattr(action_term, "set_previous_actions"):
            return
        action_term.set_previous_actions(action.detach(), env_indices)

    def _set_adversary_raw_actions_for_reset(self, env_indices: torch.Tensor) -> None:
        action_term = getattr(getattr(self.env, "unwrapped", self.env).action_manager, "_terms", {}).get("adversaryaction")
        raw_actions = getattr(action_term, "raw_actions", None)
        if not isinstance(raw_actions, torch.Tensor):
            return
        source_ids = env_indices.to(self._scratch_adv_action.device)
        target_ids = env_indices.to(raw_actions.device)
        raw_actions[target_ids] = self._scratch_adv_action[source_ids].to(raw_actions.device)

    def _sample_adversary_capture(
        self, obs, env_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample adversary actions into per-env scratch."""
        hook_env_ids = None if env_indices is None else env_indices.to(self.env.device)
        hooks = self._multi_agent_env_hooks()
        prepare_context = getattr(hooks, "prepare_adversary_proposal_context", None)
        if callable(prepare_context):
            restore_env_ids = (
                torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
                if env_indices is None
                else env_indices
            )
            restore_state = self.env.unwrapped.scene.get_state(is_relative=True)
            with torch.inference_mode():
                context_changed = bool(prepare_context(hook_env_ids))
            if context_changed:
                obs = self.env.get_observations().to(self.device)
                self._write_state_to_sim(restore_state, restore_env_ids, notify_hooks=False)

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

        self._update_adversary_previous_action_obs(action, env_indices)
        return action

    def _adversary_update_from_tuples(
        self, tuples: dict, rewards: torch.Tensor, last_obs
    ) -> dict:
        """Run one adversary update from stashed action-time tuples."""
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
        alg.compute_returns(last_obs)  # no-op for Reinforce
        return alg.update()

    def _resolve_success_term_idx(self) -> int:
        if self.inline.skip_settling:
            return -1
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

    def _settling_episode_start_len(self) -> int:
        """Episode counter offset so ``check_reset_state_success`` times out with settle_max_steps."""
        max_ep_len = int(self.env.max_episode_length)
        return max(0, max_ep_len - self.inline.settle_max_steps)

    # Adversary updates

    def _snapshot_teacher_tuple(self, env_ids: torch.Tensor) -> dict:
        return {
            "obs": {k: v[env_ids].detach().clone() for k, v in self._scratch_adv_obs.items()},
            "action": self._scratch_adv_action[env_ids].detach().clone(),
            "log_prob": self._scratch_adv_log_prob[env_ids].detach().clone(),
            "mu": self._scratch_adv_mu[env_ids].detach().clone(),
            "sigma": self._scratch_adv_sigma[env_ids].detach().clone(),
        }

    def _commit_teacher(
        self,
        env_id: int,
        gen_reward: float,
        combined_reward: float,
        regret: float | None = None,
    ) -> None:
        idx_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
        tup = self._snapshot_teacher_tuple(idx_t)
        self._teacher_commit["obs"].append(tup["obs"])
        self._teacher_commit["action"].append(tup["action"])
        self._teacher_commit["log_prob"].append(tup["log_prob"])
        self._teacher_commit["mu"].append(tup["mu"])
        self._teacher_commit["sigma"].append(tup["sigma"])
        self._teacher_commit["gen_reward"].append(float(gen_reward))
        self._teacher_commit["reward"].append(float(combined_reward))
        self._teacher_commit["regret"].append(regret)
        self._pending_teacher[env_id] = False
        self._pending_gen_reward[env_id] = 0.0

    def _teacher_batch_size_target(self) -> int:
        bs = self.inline.adversary_update_batch_size
        return int(bs) if bs is not None else self.env.num_envs

    def _maybe_fire_adversary_update(self, last_obs) -> dict | None:
        """Update adversary once every rank has enough tuples."""
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
        regret_list = _pop_n(self._teacher_commit["regret"], target)

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

        kpin_regrets = [r for r in regret_list if r is not None]
        self._inline_last_adv_loss = {k: float(v) for k, v in loss_dict.items()}
        self._inline_last_adv_rewards = {
            "mean_total_reward": float(combined_rewards.mean().item()),
            "mean_gen_reward": float(adv_tuples["gen_reward"].mean().item()),
            "mean_regret": float(sum(kpin_regrets) / len(kpin_regrets)) if kpin_regrets else float("nan"),
        }
        return loss_dict

    def _emit_inline_iter_metrics(self, writer, it: int) -> None:
        if writer is not None:
            for k, v in self._inline_last_adv_loss.items():
                writer.add_scalar(f"Adversary/{k}", v, it)
            for k, v in self._inline_last_adv_rewards.items():
                writer.add_scalar(f"Adversary/{k}", v, it)

    def _multi_agent_env_hooks(self):
        return getattr(self.env.unwrapped, "_multi_agent_runner_hooks", None)

    def _call_multi_agent_env_hook(self, hook_name: str, *args, default=None, **kwargs):
        hooks = self._multi_agent_env_hooks()
        hook = getattr(hooks, hook_name, None)
        if not callable(hook):
            return default
        return hook(*args, **kwargs)

    def _settling_gripper_targets(self) -> torch.Tensor:
        target = self._call_multi_agent_env_hook(
            "settling_gripper_targets",
            float(self.inline.settling_gripper_default_action),
            default=None,
        )
        if isinstance(target, torch.Tensor) and target.numel() == self.env.num_envs:
            return target.to(self.device, dtype=torch.float).view(-1)
        return torch.full(
            (self.env.num_envs,),
            float(self.inline.settling_gripper_default_action),
            dtype=torch.float,
            device=self.device,
        )

    # State restore

    def _write_state_to_sim(self, state: dict, env_ids: torch.Tensor, notify_hooks: bool = True) -> None:
        """Restore saved scene state."""
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
            if notify_hooks:
                self._call_multi_agent_env_hook("on_runner_state_written", env_ids.to(self.env.device))

    def _stash_pre_success_for(self, state: dict, env_ids: torch.Tensor) -> None:
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

    def _consume_reset_success_state(self, env_ids: torch.Tensor, fallback_state: dict) -> dict:
        """Allow task hooks to replace the fallback state with a cached success state."""
        success_state = self._call_multi_agent_env_hook(
            "consume_reset_success_state",
            env_ids.to(self.env.device),
            fallback_state,
            default=fallback_state,
        )
        return success_state if isinstance(success_state, dict) else fallback_state

    def _clear_live_episode_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._live_episodes_since_settle[env_ids] = 0
        for eid in env_ids.tolist():
            self._per_env_live_returns[eid] = []

    def _activate_live_envs(self, env_ids: torch.Tensor, handoff_steps: int = 0) -> None:
        if env_ids.numel() == 0:
            return
        self.env_mode[env_ids] = MODE_LIVE
        self.settle_remaining[env_ids] = 0
        self.settle_retries[env_ids] = 0
        self.live_handoff_remaining[env_ids] = max(0, int(handoff_steps))
        # Full OmniReset-length LIVE segment: settling must not consume the 16s budget.
        self.env.episode_length_buf[env_ids] = 0
        self._clear_live_episode_state(env_ids)
        self._call_multi_agent_env_hook("on_live_anchor_start", env_ids.to(self.env.device))

    def _restart_settling_envs(
        self,
        env_ids: torch.Tensor,
        obs: TensorDict | None,
        *,
        pending_teacher: bool = True,
    ) -> bool:
        if env_ids.numel() == 0:
            return False
        self.env_mode[env_ids] = MODE_SETTLING
        self.settle_remaining[env_ids] = self.inline.settle_max_steps
        self.settle_retries[env_ids] = 0
        self.live_handoff_remaining[env_ids] = 0
        self._clear_live_episode_state(env_ids)
        self._call_multi_agent_env_hook("on_live_anchor_clear", env_ids.to(self.env.device))
        if obs is None:
            obs = self.env.get_observations().to(self.device)
        self._sample_adversary_capture(obs, env_indices=env_ids)
        self._set_adversary_raw_actions_for_reset(env_ids)
        reset_ids = env_ids.to(self.env.device)
        with torch.inference_mode():
            self.env.unwrapped._reset_idx(reset_ids)
            self.env.episode_length_buf[env_ids] = self._settling_episode_start_len()
        if pending_teacher:
            self._pending_teacher[env_ids] = True
        return True

    # Mode transitions

    def _step_modes_no_settling(
        self,
        dones: torch.Tensor,
        obs: TensorDict,
    ) -> bool:
        """K-pin regret without SETTLING validation."""
        state_was_written = False
        dones_bool = dones.to(torch.bool).view(-1)
        if not dones_bool.any():
            return state_was_written
        done_ids = dones_bool.nonzero(as_tuple=False).squeeze(-1)

        post_reset_state = self.env.unwrapped.scene.get_state(is_relative=True)

        for eid in done_ids.tolist():
            single_id = torch.tensor([eid], device=self.device, dtype=torch.long)
            if not bool(self._pending_teacher[eid].item()):
                self._stash_pre_success_for(post_reset_state, single_id)
                self._pending_teacher[eid] = True
                self._clear_live_episode_state(single_id)
                continue

            self._live_episodes_since_settle[eid] += 1
            count = int(self._live_episodes_since_settle[eid].item())
            if count < self.regret_k:
                if self._pre_success_state is not None:
                    self._write_state_to_sim(self._pre_success_state, single_id)
                    state_was_written = True
            else:
                returns = self._per_env_live_returns[eid][:self.regret_k]
                regret = max(returns) - sum(returns) / self.regret_k
                combined = self.beta_gen_reward * 0.0 + regret
                self._commit_teacher(
                    env_id=eid, gen_reward=0.0, combined_reward=combined, regret=regret,
                )
                self._clear_live_episode_state(single_id)
                self._sample_adversary_capture(obs, env_indices=single_id)

        return state_was_written

    def _step_modes(
        self,
        dones: torch.Tensor,
        term_mgr,
        success_idx: int,
        scene_state_pre: dict,
        rewards: torch.Tensor,
        obs: TensorDict,
        cur_reward_sum: torch.Tensor,
        cur_episode_length: torch.Tensor,
    ) -> bool:
        """Resolve LIVE/SETTLING transitions."""
        state_was_written = False
        settling_mask = (self.env_mode == MODE_SETTLING)
        handoff_mask = (~settling_mask) & (self.live_handoff_remaining > 0)
        active_live_mask = (~settling_mask) & ~handoff_mask
        dones_bool = dones.to(torch.bool).view(-1)

        success_this_step = term_mgr._last_episode_dones[:, success_idx].to(torch.bool) & dones_bool
        success_in_settle = success_this_step & settling_mask
        if success_in_settle.any():
            success_ids = success_in_settle.nonzero(as_tuple=False).squeeze(-1)
            success_state = self._consume_reset_success_state(success_ids, scene_state_pre)
            self._stash_pre_success_for(success_state, success_ids)
            self._pending_gen_reward[success_ids] = rewards[success_ids].view(-1).float()

        done_settling = settling_mask & dones_bool
        timed_out_no_done = settling_mask & (self.settle_remaining == 0) & ~dones_bool
        resolved = done_settling | timed_out_no_done

        if resolved.any():
            valid = resolved & success_in_settle
            invalid = resolved & ~valid

            if valid.any():
                valid_ids = valid.nonzero(as_tuple=False).squeeze(-1)
                if self._pre_success_state is not None:
                    self._write_state_to_sim(self._pre_success_state, valid_ids)
                    state_was_written = True
                self._activate_live_envs(valid_ids, handoff_steps=self.inline.live_handoff_hold_steps)
                self._reset_student_rnn(valid_ids)
                cur_reward_sum[valid_ids] = 0.0
                cur_episode_length[valid_ids] = 0.0

            if invalid.any():
                invalid_ids = invalid.nonzero(as_tuple=False).squeeze(-1)
                can_retry = self.settle_retries[invalid_ids] < self.inline.max_resample_retries
                retry_ids = invalid_ids[can_retry]
                giveup_ids = invalid_ids[~can_retry]

                if retry_ids.numel() > 0:
                    self.settle_remaining[retry_ids] = self.inline.settle_max_steps
                    self.settle_retries[retry_ids] += 1
                    self._sample_adversary_capture(obs, env_indices=retry_ids)
                    self._set_adversary_raw_actions_for_reset(retry_ids)
                    with torch.inference_mode():
                        self.env.unwrapped._reset_idx(retry_ids.to(self.env.device))
                        self.env.episode_length_buf[retry_ids] = self._settling_episode_start_len()
                    state_was_written = True
                    self._reset_student_rnn(retry_ids)

                if giveup_ids.numel() > 0:
                    for eid in giveup_ids.tolist():
                        if bool(self._pending_teacher[eid].item()):
                            self._commit_teacher(
                                env_id=eid,
                                gen_reward=float(self._pending_gen_reward[eid].item()),
                                combined_reward=self.inline.invalid_settle_penalty,
                            )
                    if self.inline.force_live_after_max_retries:
                        with torch.inference_mode():
                            self.env.unwrapped._reset_idx(giveup_ids.to(self.env.device))
                        state_was_written = True
                        self._activate_live_envs(giveup_ids)
                    else:
                        state_was_written |= self._restart_settling_envs(giveup_ids, obs=None)
                        self._pending_gen_reward[giveup_ids] = 0.0

                    self._reset_student_rnn(giveup_ids)
                    cur_reward_sum[giveup_ids] = 0.0
                    cur_episode_length[giveup_ids] = 0.0

        handoff_done = handoff_mask & dones_bool
        if handoff_done.any():
            handoff_done_ids = handoff_done.nonzero(as_tuple=False).squeeze(-1)
            state_was_written |= self._restart_settling_envs(handoff_done_ids, obs)

        live_done = active_live_mask & dones_bool
        if live_done.any():
            ids = live_done.nonzero(as_tuple=False).squeeze(-1)
            for eid in ids.tolist():
                if bool(self._pending_teacher[eid].item()):
                    self._live_episodes_since_settle[eid] += 1
                    if int(self._live_episodes_since_settle[eid].item()) < self.regret_k:
                        if self._pre_success_state is not None:
                            self._write_state_to_sim(
                                self._pre_success_state,
                                torch.tensor([eid], device=self.device, dtype=torch.long),
                            )
                            state_was_written = True
                    else:
                        returns = self._per_env_live_returns[eid][:self.regret_k]
                        regret = max(returns) - sum(returns) / self.regret_k
                        gen_r = float(self._pending_gen_reward[eid].item())
                        combined = self.beta_gen_reward * gen_r + regret
                        self._commit_teacher(
                            env_id=eid, gen_reward=gen_r, combined_reward=combined, regret=regret,
                        )
                        single_id = torch.tensor([eid], device=self.device, dtype=torch.long)
                        state_was_written |= self._restart_settling_envs(single_id, obs)
                else:
                    single_id = torch.tensor([eid], device=self.device, dtype=torch.long)
                    state_was_written |= self._restart_settling_envs(single_id, obs)

        return state_was_written

    def _prepare_adversary_storage(self, n: int, obs_spec) -> None:
        self.alg_adversary.init_storage(
            "rl", num_envs=n, num_transitions_per_env=1,
            obs=obs_spec, actions_shape=[self.adversary_action_dim],
        )

    def _reset_student_rnn(self, env_ids: torch.Tensor) -> None:
        """Clear recurrent student state for selected envs."""
        if env_ids.numel() == 0:
            return
        if not getattr(self.alg.policy, "is_recurrent", False):
            return
        synthetic = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        synthetic[env_ids] = 1.0
        self.alg.policy.reset(synthetic)

    # Training loop

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

        success_idx = self._resolve_success_term_idx()
        unwrapped_env = self.env.unwrapped
        term_mgr_handle = unwrapped_env.termination_manager

        try:
            command_term_handle = unwrapped_env.command_manager.get_term("task_command")
        except (AttributeError, KeyError):
            command_term_handle = None

        self._sample_adversary_capture(obs, env_indices=None)

        max_ep_len = int(self.env.max_episode_length)
        student_zero = torch.zeros(
            self.env.num_envs, self.policy_action_dim, device=self.env.device
        )
        if not self.inline.skip_settling and self.policy_action_dim > 0:
            gripper_targets = self._settling_gripper_targets()
            student_zero[:, -1] = gripper_targets.to(self.env.device)
            self._call_multi_agent_env_hook(
                "set_settling_control_mask",
                torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device),
            )
        prime_actions = torch.cat(
            [student_zero, self._scratch_adv_action.to(self.env.device)], dim=-1
        )
        with torch.inference_mode():
            self.env.episode_length_buf.fill_(max_ep_len)
        obs, _, _, _ = self.env.step(prime_actions)
        obs = obs.to(self.device)

        if not self.inline.skip_settling:
            self.env_mode.fill_(MODE_SETTLING)
            self.settle_remaining.fill_(self.inline.settle_max_steps)
            self.settle_retries.zero_()
            self.live_handoff_remaining.zero_()
            self._pending_teacher.fill_(True)
            with torch.inference_mode():
                self.env.episode_length_buf.fill_(self._settling_episode_start_len())

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        adv_action_zero_template = torch.zeros(
            (self.env.num_envs, self.adversary_action_dim), device=self.device
        )

        for it in range(start_iter, tot_iter):
            collection_start = time.time()

            for _ in range(self.num_steps_per_env):
                with torch.inference_mode():
                    policy_actions = self.alg.act(obs)

                    settling_mask = (self.env_mode == MODE_SETTLING)
                    handoff_mask = (~settling_mask) & (self.live_handoff_remaining > 0)
                    live_mask = ~settling_mask
                    train_live_mask = live_mask & ~handoff_mask

                    if self.inline.skip_settling:
                        student_actions = policy_actions
                        adv_slice = self._scratch_adv_action
                    else:
                        student_actions = policy_actions.clone()
                        controlled_mask = settling_mask | handoff_mask
                        if controlled_mask.any():
                            student_actions[controlled_mask] = 0.0
                            if self.policy_action_dim > 0:
                                gripper_targets = self._settling_gripper_targets()
                                student_actions[controlled_mask, -1] = gripper_targets[controlled_mask]
                        self._call_multi_agent_env_hook(
                            "set_settling_control_mask",
                            controlled_mask.to(self.env.device),
                        )

                        adv_slice = adv_action_zero_template
                        if settling_mask.any():
                            adv_slice = adv_action_zero_template.clone()
                            adv_slice[settling_mask] = self._scratch_adv_action[settling_mask]

                    actions = torch.cat([student_actions, adv_slice], dim=-1)

                    if not self.inline.skip_settling and (settling_mask | handoff_mask).any():
                        pin_ids = (settling_mask | handoff_mask).nonzero(as_tuple=False).squeeze(-1)
                        self._call_multi_agent_env_hook(
                            "apply_settling_state_targets",
                            pin_ids.to(self.env.device),
                            write_state=True,
                        )

                    valid_mask = train_live_mask.float().unsqueeze(-1)
                    scene_state_pre = unwrapped_env.scene.get_state(is_relative=True)

                    if command_term_handle is not None:
                        command_term_handle._live_mask.copy_(
                            train_live_mask.to(command_term_handle._live_mask.device)
                        )

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device),
                    )

                    self.alg.process_env_step(obs, rewards, dones, extras, valid_mask=valid_mask)

                    settling_mask_now = (self.env_mode == MODE_SETTLING)
                    if settling_mask_now.any():
                        self.settle_remaining[settling_mask_now] = torch.clamp(
                            self.settle_remaining[settling_mask_now] - 1, min=0
                        )
                    dones_bool = dones.to(torch.bool).view(-1)
                    live_dones = dones_bool & train_live_mask
                    if self.log_dir is not None and live_dones.any():
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    live_mask_float = train_live_mask.to(cur_reward_sum.dtype)
                    cur_reward_sum += rewards * live_mask_float
                    cur_episode_length += live_mask_float

                if done_ids.numel() > 0:
                    done_env_indices = done_ids[:, 0]
                    ep_returns = cur_reward_sum[done_ids][:, 0]
                    cur_lens = cur_episode_length[done_ids][:, 0]

                    done_env_list = done_env_indices.cpu().numpy().tolist()
                    ep_returns_list = ep_returns.cpu().numpy().tolist()
                    cur_lens_list = cur_lens.cpu().numpy().tolist()

                    counted_rets: list[float] = []
                    counted_lens: list[float] = []
                    for env_idx, ret, ep_len in zip(done_env_list, ep_returns_list, cur_lens_list):
                        if not bool(train_live_mask[env_idx].item()):
                            continue
                        self._per_env_live_returns[env_idx].append(float(ret))
                        counted_rets.append(float(ret))
                        counted_lens.append(float(ep_len))

                    if self.log_dir is not None and counted_rets:
                        rewbuffer.extend(counted_rets)
                        lenbuffer.extend(counted_lens)

                    live_done_env_indices = done_env_indices[train_live_mask[done_env_indices]]
                    self._call_multi_agent_env_hook(
                        "on_live_episode_done",
                        live_done_env_indices.to(self.env.device),
                    )

                    cur_reward_sum[done_ids] = 0
                    cur_episode_length[done_ids] = 0

                if self.inline.skip_settling:
                    state_was_written = self._step_modes_no_settling(dones=dones, obs=obs)
                else:
                    state_was_written = self._step_modes(
                        dones=dones,
                        term_mgr=term_mgr_handle,
                        success_idx=success_idx,
                        scene_state_pre=scene_state_pre,
                        rewards=rewards,
                        obs=obs,
                        cur_reward_sum=cur_reward_sum,
                        cur_episode_length=cur_episode_length,
                    )
                if not self.inline.skip_settling:
                    handoff_mask_now = (self.env_mode == MODE_LIVE) & (self.live_handoff_remaining > 0)
                    if handoff_mask_now.any():
                        handoff_ids_before = handoff_mask_now.nonzero(as_tuple=False).squeeze(-1)
                        self.live_handoff_remaining[handoff_ids_before] = torch.clamp(
                            self.live_handoff_remaining[handoff_ids_before] - 1, min=0
                        )
                if state_was_written:
                    obs = self.env.get_observations().to(self.device)

            collection_time = time.time() - collection_start
            adversary_records = self._collect_adversary_records()

            learn_start = time.time()
            with torch.inference_mode():
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()

            self._maybe_fire_adversary_update(last_obs=obs)
            env_iter_metrics = self._call_multi_agent_env_hook(
                "consume_iter_metrics",
                is_distributed=self.is_distributed,
                default={},
            )

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
                for key, value in env_iter_metrics.items():
                    self.writer.add_scalar(key, value, it)

                self._write_adversary_records(it, adversary_records)

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

    # Save / load

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

    # Infrastructure

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
