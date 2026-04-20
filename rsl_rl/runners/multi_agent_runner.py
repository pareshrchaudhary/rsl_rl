# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.simple_ppo import SimplePPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, AsymmetricActorCritic, resolve_symmetry_config
from rsl_rl.runners.eval_runner import EvalRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.utils.logger import log_phase_a, log_phase_b
from rsl_rl.storage.reset_state_buffer import ResetStateBuffer

class MultiAgentRunner:
    """Alternates adversary generation (2s settling) and policy training (16s) phases."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        print("Initializing MultiAgentRunner...")
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

        # Action split: policy controls robot, adversary controls last `adversary_action_dim` entries.
        self.adversary_action_dim = self.cfg["adversary_robot_parameters"]
        self.policy_action_dim = int(self.env.num_actions - self.adversary_action_dim)

        obs = self.env.get_observations()

        self.cfg["obs_groups"] = resolve_obs_groups(obs, dict(self.obs_groups_raw), ["critic"])

        adversary_obs_groups_raw = dict(self.adversary_obs_groups_raw)
        if "critic" not in adversary_obs_groups_raw:
            adversary_obs_groups_raw["critic"] = list(adversary_obs_groups_raw.get("policy", []))
        self.adversary_obs_groups = resolve_obs_groups(obs, adversary_obs_groups_raw, ["critic"])

        # IPPO: independent PPO per agent.
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

        # Env is created with the training episode length; snapshot it and swap to
        # generation_episode_length_s only during Phase A.
        self.generation_max_steps = self.cfg.get("generation_max_steps", 1000)
        self.training_episode_length_s = self.env.unwrapped.cfg.episode_length_s
        self.generation_episode_length_s = self.cfg.get("generation_episode_length_s", 2.0)
        # 0 disables validity/geometry shaping, leaving regret as the sole signal.
        self.beta_gen_reward = float(self.cfg.get("beta_gen_reward", 1.0))

        self.state_buffer = ResetStateBuffer(
            capacity=self.env.num_envs, device=self.device, num_envs=self.env.num_envs
        )

        self._success_term_idx: int | None = None

        print(f"[MultiAgentRunner] num_envs={self.env.num_envs}, "
              f"gen={self.generation_episode_length_s}s/{self.generation_max_steps}steps, "
              f"train={self.training_episode_length_s}s")

        self.eval_runner = EvalRunner(self)

    # =====================================================================
    # Adversary sampling + manual transition population
    # =====================================================================

    def _sample_adversary_capture(
        self, obs, env_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample stochastic adversary actions and snapshot (obs, action, log_prob, mu, sigma)
        into scratch tensors. Uses ``policy.act`` (not ``act_inference``) so PPO's ratio is
        honest at update time. ``env_indices=None`` snapshots all envs; otherwise only those.
        """
        with torch.inference_mode():
            action = self.alg_adversary.policy.act(obs).detach()
            log_prob = self.alg_adversary.policy.get_actions_log_prob(action).detach()
            mu = self.alg_adversary.policy.action_mean.detach().clone()
            sigma_raw = self.alg_adversary.policy.action_std.detach()
            # Broadcast shared-std case (action_dim,) to (num_envs, action_dim).
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
        """SimplePPO update using pre-captured tuples from Phase A, bypassing ``alg_adversary.act()``
        so the log_prob/mu/sigma stored at action-time are used directly.
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
                f"Generation env must have a 'success' termination term. "
                f"Available: {list(term_mgr._term_name_to_term_idx.keys())}"
            )
        self._success_term_idx = idx
        return idx

    # =====================================================================
    # Cycle helpers
    # =====================================================================

    def _run_phase_a_refill(self, obs):
        """Run Phase A, refill buffer, re-attach for async Phase B pulls. Returns (obs, gen_stats).

        No synchronized reset at the boundary — envs transition to buffer states
        organically via the natural auto-reset path (``reset_from_state_buffer``
        event term fires inside ``env.step`` when each env's leftover Phase A
        episode ends). This avoids the sync burst that synchronized resets
        produce and matches OmniReset's async reset pattern.
        """
        self.env.unwrapped.cfg.episode_length_s = self.generation_episode_length_s
        self.env.unwrapped.reset_state_buffer = None

        gen_stats = self._run_generation_loop(obs)
        obs = gen_stats["final_obs"]

        self.env.unwrapped.cfg.episode_length_s = self.training_episode_length_s
        self.env.unwrapped.reset_state_buffer = self.state_buffer
        self.state_buffer.reset_per_env_tracking()

        return obs, gen_stats

    def _prepare_adversary_storage(self, n: int, obs_spec) -> None:
        self.alg_adversary.init_storage(
            "rl",
            num_envs=n,
            num_transitions_per_env=1,
            obs=obs_spec,
            actions_shape=[self.adversary_action_dim],
        )

    def _select_adv_tuples(self, adv_tuples_full: dict, idx_t: torch.Tensor) -> dict:
        return {
            "obs":        {k: v[idx_t].clone() for k, v in adv_tuples_full["obs"].items()},
            "action":     adv_tuples_full["action"][idx_t].clone(),
            "log_prob":   adv_tuples_full["log_prob"][idx_t].clone(),
            "mu":         adv_tuples_full["mu"][idx_t].clone(),
            "sigma":      adv_tuples_full["sigma"][idx_t].clone(),
            "gen_reward": adv_tuples_full["gen_reward"][idx_t].clone(),
        }

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

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        cycle_adv_loss_sums: dict[str, float] = {}
        cycle_adv_loss_count: int = 0
        cycle_adv_total_rewards: list[float] = []
        cycle_adv_gen_rewards: list[float] = []
        cycle_adv_regrets: list[float] = []
        cycle_id: int = 0

        regret_k = int(self.cfg.get("regret_k", 3))
        max_iters_per_cycle = int(self.cfg.get("max_iters_per_cycle", 50))

        # Phase B adversary slice is inert (no obs/reward term reads it, buffer
        # reset ignores raw_actions) — zeros just satisfy action-width.
        phase_b_adversary_actions = torch.zeros(
            (self.env.num_envs, self.adversary_action_dim), device=self.device
        )

        obs, _gen_stats_initial = self._run_phase_a_refill(obs)
        cur_reward_sum.zero_()
        cur_episode_length.zero_()

        if self.log_dir is not None and not self.disable_logs:
            log_phase_a(
                writer=self.writer,
                alg_adversary=self.alg_adversary,
                cycle_id=cycle_id,
                step=start_iter,
                gen_stats=_gen_stats_initial,
                buffer_diag=None,
                cycle_adv_loss_sums={},
                cycle_adv_loss_count=0,
                cycle_adv_total_rewards=[],
                cycle_adv_gen_rewards=[],
                cycle_adv_regrets=[],
            )
        cycle_id += 1

        per_slot_returns: list[list[float]] = [[] for _ in range(self.env.num_envs)]
        # First done per env post-refill is the leftover Phase A episode
        # continuing into Phase B (the env hasn't yet pinned to a buffer slot).
        # Drop it from metrics and adversary credit — only buffer-pinned
        # episodes should count toward per-slot regret.
        first_post_refill: list[bool] = [True] * self.env.num_envs
        iters_in_cycle: int = 0
        num_cycles_completed: int = 0

        for it in range(start_iter, tot_iter):
            gen_stats = None
            iters_in_cycle += 1

            # Phase B: protagonist rollout.
            collection_start = time.time()

            for _ in range(self.num_steps_per_env):
                with torch.inference_mode():
                    policy_actions = self.alg.act(obs)
                    actions = torch.cat([policy_actions, phase_b_adversary_actions], dim=-1)

                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    self.alg.process_env_step(obs, rewards, dones, extras)

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

                    counted_rets: list[float] = []
                    counted_lens: list[float] = []
                    for env_idx, ret, ep_len in zip(done_env_list, ep_returns_list, cur_lens_list):
                        if first_post_refill[env_idx]:
                            first_post_refill[env_idx] = False
                            continue
                        per_slot_returns[env_idx].append(ret)
                        counted_rets.append(ret)
                        counted_lens.append(ep_len)

                    if self.log_dir is not None and counted_rets:
                        rewbuffer.extend(counted_rets)
                        lenbuffer.extend(counted_lens)

                    cur_reward_sum[done_ids] = 0
                    cur_episode_length[done_ids] = 0

            collection_time = time.time() - collection_start

            learn_start = time.time()
            with torch.inference_mode():
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()

            returns_flat = [r for slot in per_slot_returns for r in slot]
            if returns_flat:
                batch_episode_count = len(returns_flat)
                max_batch_total_reward = float(max(returns_flat))
                mean_batch_total_reward = float(sum(returns_flat) / batch_episode_count)
                regret = max_batch_total_reward - mean_batch_total_reward
            else:
                batch_episode_count = 0
                max_batch_total_reward = 0.0
                mean_batch_total_reward = 0.0
                regret = 0.0

            # Cycle ends when every slot has banked K episodes. MIN-sync across
            # ranks so all refill together.
            local_complete = int(all(len(s) >= regret_k for s in per_slot_returns))
            local_cap = int(iters_in_cycle >= max_iters_per_cycle)
            if self.is_distributed:
                cc_t = torch.tensor(local_complete, dtype=torch.int, device=self.device)
                fc_t = torch.tensor(local_cap, dtype=torch.int, device=self.device)
                torch.distributed.all_reduce(cc_t, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(fc_t, op=torch.distributed.ReduceOp.MAX)
                cycle_complete = bool(cc_t.item())
                force_refill = bool(fc_t.item())
            else:
                cycle_complete = bool(local_complete)
                force_refill = bool(local_cap)

            if cycle_complete or force_refill:
                # Max-mean over first K returns when slot has K, else 0. Starved
                # slots still train on gen_reward so they're marked as bad targets.
                regrets: list[float] = []
                zero_regret_count = 0
                for i in range(self.env.num_envs):
                    slot = per_slot_returns[i]
                    if len(slot) >= regret_k:
                        first_k = slot[:regret_k]
                        regrets.append(max(first_k) - sum(first_k) / regret_k)
                    else:
                        regrets.append(0.0)
                        zero_regret_count += 1

                if force_refill and not cycle_complete:
                    print(
                        f"[cycle-cap] force-refilling after {iters_in_cycle} iters; "
                        f"{zero_regret_count}/{self.env.num_envs} slots had zero regret"
                    )
                else:
                    total_eps = sum(len(s) for s in per_slot_returns)
                    print(
                        f"[cycle-complete] iter={it} cycle={num_cycles_completed} "
                        f"iters={iters_in_cycle} K={regret_k} "
                        f"total_episodes={total_eps} → refilling Phase A"
                    )

                # MIN-sync so all ranks fire the grad all_reduce together (or skip it).
                adv_tuples_full = self.state_buffer.get_adversary_tuples_for_envs()
                local_ok = int(adv_tuples_full is not None)
                ok_t = torch.tensor(local_ok, dtype=torch.int, device=self.device)
                if self.is_distributed:
                    torch.distributed.all_reduce(ok_t, op=torch.distributed.ReduceOp.MIN)

                if ok_t.item():
                    all_idx_t = torch.arange(
                        self.env.num_envs, dtype=torch.long, device=self.device
                    )
                    regret_t = torch.tensor(regrets, dtype=torch.float, device=self.device)
                    adv_tuples = self._select_adv_tuples(adv_tuples_full, all_idx_t)
                    combined_rewards = (
                        self.beta_gen_reward * adv_tuples["gen_reward"] + regret_t
                    ).unsqueeze(-1)
                    self._prepare_adversary_storage(
                        n=self.env.num_envs, obs_spec=adv_tuples["obs"]
                    )
                    adv_loss_dict = self._adversary_update_from_tuples(
                        adv_tuples, combined_rewards, last_obs=obs,
                    )
                    for k, v in adv_loss_dict.items():
                        cycle_adv_loss_sums[k] = cycle_adv_loss_sums.get(k, 0.0) + float(v)
                    cycle_adv_loss_count += 1
                    cycle_adv_total_rewards.extend(combined_rewards.squeeze(-1).tolist())
                    cycle_adv_gen_rewards.extend(adv_tuples["gen_reward"].tolist())
                    cycle_adv_regrets.extend(regret_t.tolist())

                # Snapshot regret distribution before the buffer is discarded.
                buffer_diag = self._compute_buffer_diagnostics(per_slot_returns)

                # Refill with the just-updated adversary.
                obs, gen_stats = self._run_phase_a_refill(obs)

                if self.log_dir is not None and not self.disable_logs:
                    log_phase_a(
                        writer=self.writer,
                        alg_adversary=self.alg_adversary,
                        cycle_id=cycle_id,
                        step=it,
                        gen_stats=gen_stats,
                        buffer_diag=buffer_diag,
                        cycle_adv_loss_sums=cycle_adv_loss_sums,
                        cycle_adv_loss_count=cycle_adv_loss_count,
                        cycle_adv_total_rewards=cycle_adv_total_rewards,
                        cycle_adv_gen_rewards=cycle_adv_gen_rewards,
                        cycle_adv_regrets=cycle_adv_regrets,
                    )

                cycle_adv_loss_sums = {}
                cycle_adv_loss_count = 0
                cycle_adv_total_rewards = []
                cycle_adv_gen_rewards = []
                cycle_adv_regrets = []
                cycle_id += 1

                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                per_slot_returns = [[] for _ in range(self.env.num_envs)]
                first_post_refill = [True] * self.env.num_envs
                num_cycles_completed += 1
                iters_in_cycle = 0

            learn_time = time.time() - learn_start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
                self.tot_timesteps += collection_size
                self.tot_time += collection_time + learn_time

                log_phase_b(
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

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            self.eval_runner.run(it)
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
    # Phase A: Adversary Generation Loop
    # =====================================================================

    def _run_generation_loop(self, obs) -> dict:
        """Adversary action → physics settles → validity check. Successful
        (state, adv tuple, gen_reward) bundles are pushed to the buffer.
        """
        success_idx = self._resolve_success_term_idx()
        self.state_buffer.clear()

        unwrapped_env = self.env.unwrapped
        term_mgr = unwrapped_env.termination_manager
        reward_mgr = unwrapped_env.reward_manager

        adversary_actions = self._sample_adversary_capture(obs, env_indices=None)

        zero_policy_actions = torch.zeros(
            (self.env.num_envs, self.policy_action_dim), device=self.device
        )

        total_steps = 0
        num_episodes_done = 0
        num_successes = 0

        collected_rewards = []

        num_reward_terms = len(reward_mgr.active_terms)
        per_term_success_total = torch.zeros(num_reward_terms, device=self.device)
        success_count = 0

        log_interval = 100

        while total_steps < self.generation_max_steps and self.state_buffer.occupancy < self.state_buffer.capacity:
            with torch.inference_mode():
                # Snapshot BEFORE the step — this is the validated state pushed
                # to the buffer if the episode terminates as success this step.
                last_scene_state = unwrapped_env.scene.get_state(is_relative=True)
                actions = torch.cat([zero_policy_actions, adversary_actions], dim=-1)
                obs, rewards, dones, _ = self.env.step(actions.to(self.env.device))
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                total_steps += 1
                done_ids = (dones > 0).nonzero(as_tuple=False)
                if done_ids.numel() > 0:
                    done_env_indices = done_ids[:, 0]
                    num_episodes_done += done_env_indices.numel()

                    collected_rewards.append(rewards[done_env_indices].clone())

                    success_mask = term_mgr._last_episode_dones[done_env_indices, success_idx]
                    if success_mask.any():
                        success_env_ids = done_env_indices[success_mask]
                        num_successes += success_env_ids.numel()
                        success_count += success_env_ids.numel()
                        # Scratch holds the tuple the adversary used to *generate*
                        # this state — that's what PPO trains on.
                        self.state_buffer.push(
                            last_scene_state, success_env_ids,
                            adv_obs=self._scratch_adv_obs,
                            actions=self._scratch_adv_action,
                            log_probs=self._scratch_adv_log_prob,
                            mus=self._scratch_adv_mu,
                            sigmas=self._scratch_adv_sigma,
                            rewards=rewards,
                        )
                        per_term_success_total += reward_mgr._step_reward[success_env_ids].sum(dim=0)

                    # Re-sample for reset envs and refresh scratch for next step.
                    new_adv_actions = self._sample_adversary_capture(obs, env_indices=done_env_indices)
                    adversary_actions[done_env_indices] = new_adv_actions[done_env_indices]

                if total_steps % log_interval == 0:
                    validity = num_successes / max(num_episodes_done, 1)
                    print(f"  [Gen] step={total_steps}/{self.generation_max_steps}, "
                          f"episodes={num_episodes_done}, successes={num_successes}, "
                          f"validity={validity:.3f}, buffer={self.state_buffer.occupancy}/{self.state_buffer.capacity}")

        if collected_rewards:
            all_rewards = torch.cat(collected_rewards, dim=0)
        else:
            all_rewards = torch.zeros(0, device=self.device)

        mean_validity = num_successes / max(num_episodes_done, 1)
        mean_reward = all_rewards.mean().item() if all_rewards.numel() > 0 else 0.0
        mean_success_quality = all_rewards[all_rewards > 0].mean().item() if (all_rewards > 0).any() else 0.0

        reward_term_names = reward_mgr.active_terms
        per_term_success_means = (per_term_success_total / max(success_count, 1)).cpu().tolist()

        buffer_fill_pct = 100 * self.state_buffer.occupancy / max(self.state_buffer.capacity, 1)

        return {
            "final_obs": obs,
            "total_steps": total_steps,
            "num_episodes_done": num_episodes_done,
            "num_successes": num_successes,
            "mean_validity_rate": mean_validity,
            "mean_reward": mean_reward,
            "mean_state_quality": mean_success_quality,
            "collected_rewards": all_rewards,
            "reward_term_names": reward_term_names,
            "reward_term_success_means": per_term_success_means,
            "buffer_fill_pct": buffer_fill_pct,
        }

    # =====================================================================
    # Buffer diagnostics
    # =====================================================================

    def _compute_buffer_diagnostics(
        self, per_env_episode_returns: list[list[float]]
    ) -> dict | None:
        """Per-slot regret distribution. kept_top50_over_mean >= 1.5 means
        high-regret slots are being wasted. Returns None if no slot has returns.
        """
        returns_with_data = [r for r in per_env_episode_returns if len(r) > 0]
        n = len(returns_with_data)
        if n == 0:
            return None

        max_t = torch.tensor([max(r) for r in returns_with_data], dtype=torch.float, device=self.device)
        mean_t = torch.tensor(
            [sum(r) / len(r) for r in returns_with_data], dtype=torch.float, device=self.device
        )
        regret_t = max_t - mean_t

        regret_mean = regret_t.mean().item()
        regret_max = regret_t.max().item()
        regret_p90 = torch.sort(regret_t).values[min(int(0.9 * n), n - 1)].item()

        k = max(1, n // 2)
        kept_top50 = torch.topk(regret_t, k, largest=True).values.mean().item()
        kept_top50_over_mean = kept_top50 / max(regret_mean, 1e-8)

        print(
            f"[BufferDiag] regret: mean={regret_mean:.3f} p90={regret_p90:.3f} "
            f"max={regret_max:.3f} | kept_top50/mean={kept_top50_over_mean:.3f}"
        )

        return {
            "regret_mean": regret_mean,
            "regret_p90": regret_p90,
            "regret_max": regret_max,
            "kept_top50_over_mean": kept_top50_over_mean,
        }

    # =====================================================================
    # Save / load
    # =====================================================================

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        adversary_path = path.replace(".pt", "_adversary.pt")
        adversary_saved_dict = {
            "model_state_dict": self.alg_adversary.policy.state_dict(),
            "optimizer_state_dict": self.alg_adversary.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(adversary_saved_dict, adversary_path)

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
            adversary_resumed_training = self.alg_adversary.policy.load_state_dict(adversary_loaded_dict["model_state_dict"])
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
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
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
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
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

        alg.init_storage(
            "rl",
            self.env.num_envs,
            storage_horizon,
            obs,
            [action_dim],
        )
        return alg

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            os.makedirs(self.log_dir, exist_ok=True)

            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(
                    self.env.cfg,
                    self.cfg,
                    {"policy": self.alg_cfg, "adversary": self.alg_adversary_cfg_raw},
                    {"policy": self.policy_cfg, "adversary": self.policy_adversary_cfg_raw},
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(
                    self.env.cfg,
                    self.cfg,
                    {"policy": self.alg_cfg, "adversary": self.alg_adversary_cfg_raw},
                    {"policy": self.policy_cfg, "adversary": self.policy_adversary_cfg_raw},
                )
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
