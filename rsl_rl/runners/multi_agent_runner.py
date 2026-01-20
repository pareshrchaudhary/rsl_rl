# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
import h5py
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.simple_ppo import SimplePPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state


class MultiAgentRunner:
    """Multi-agent runner for training and evaluation of multi-agent actor-critic methods."""

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

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.adversary_update_every_k_steps = self.cfg["adversary_update_every_k_steps"]
        self.save_interval = self.cfg["save_interval"]

        # Action split: policy controls robot, adversary controls last `adversary_action_dim` entries.
        # NOTE: For UWLab adversarial tasks this is 9 (see AdversaryActionCfg.action_dim).
        self.adversary_action_dim = int(self.cfg.get("adversary_action_dim", 9))
        if self.env.num_actions <= self.adversary_action_dim:
            raise ValueError(
                f"env.num_actions ({self.env.num_actions}) must be > adversary_action_dim ({self.adversary_action_dim})."
            )
        self.policy_action_dim = int(self.env.num_actions - self.adversary_action_dim)
        # Resolve parameter names for logging
        self.randomized_param_names = self._resolve_randomized_param_names()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()

        # Resolve observation group mappings per agent.
        self.cfg["obs_groups"] = resolve_obs_groups(obs, dict(self.obs_groups_raw), ["critic"])

        adversary_obs_groups_raw = dict(self.adversary_obs_groups_raw)
        if "critic" not in adversary_obs_groups_raw:
            adversary_obs_groups_raw["critic"] = list(adversary_obs_groups_raw.get("policy", []))
        self.adversary_obs_groups = resolve_obs_groups(obs, adversary_obs_groups_raw, ["critic"])

        # Create the algorithms (IPPO).
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

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos: list[str] = [str(rsl_rl.__file__)]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        adv_rewbuffer = deque(maxlen=100)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            self.alg_adversary.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        param_h5_path = None
        if self.log_dir is not None:
            param_h5_path = os.path.join(self.log_dir, "adversary_params.h5")
            # Initialize HDF5 file with metadata if it doesn't exist (only from rank 0)
            if not self.disable_logs and not os.path.exists(param_h5_path):
                with h5py.File(param_h5_path, "w") as f:
                    f.attrs["param_names"] = [n.encode("utf-8") for n in self.randomized_param_names]
        
        raw_params_list = []  # Store raw parameters before aggregation
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Reset adversary update window per rollout so we update exactly every k steps within num_steps_per_env.
            adv_step = 0
            adv_reward_window: list[float] = []
            # Rollout-batch *episode* reward stats (per-iteration, episode-based):
            # We compute these over episodes that TERMINATE during this rollout collection window.
            # In distributed mode, these are aggregated across ranks to match the full batch.
            batch_episode_reward_sum = torch.tensor(0.0, dtype=torch.float, device=self.device)
            batch_episode_count = torch.tensor(0.0, dtype=torch.float, device=self.device)
            batch_episode_reward_max = torch.tensor(float("-inf"), dtype=torch.float, device=self.device)

            # Track adversary updates within this learning iteration.
            adv_updates = 0
            adv_loss_sum: dict[str, float] = {}

            # Rollout
            for _ in range(self.num_steps_per_env):
                # Agent stepping is inference-only; updates happen outside.
                with torch.inference_mode():
                    # Sample actions from both agents (each uses its own obs_groups mapping).
                    _, adversary_actions, actions = self._act_both(obs)

                    # Step the environment with combined actions.
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # Extract and track randomized parameters
                    randomized_params = self._extract_randomized_params()
                    # Store raw parameters before aggregation (keep on GPU for distributed gather)
                    raw_params_list.append(randomized_params.detach())

                    # Adversary reward: regret over the last k policy step-mean rewards.
                    # regret = max(window) - mean(window), computed when window reaches k steps.
                    adv_rewards, adv_step, adv_reward_window, do_adv_update, adv_regret = self._compute_adversary_rewards(
                        rewards, adv_step, adv_reward_window
                    )

                    # Process env step for both agents.
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Bandit-style adversary: only store the transition on the update step.
                    if do_adv_update:
                        self.alg_adversary.process_env_step(obs, adv_rewards, dones, extras)

                    # Book keeping (like OnPolicyRunner)
                    if self.log_dir is not None:
                        done_ids = (dones > 0).nonzero(as_tuple=False)
                        batch_episode_reward_sum, batch_episode_count, batch_episode_reward_max = (
                            self._update_rollout_bookkeeping(
                                extras,
                                ep_infos,
                                rewards,
                                done_ids,
                                cur_reward_sum,
                                cur_episode_length,
                                rewbuffer,
                                lenbuffer,
                                batch_episode_reward_sum,
                                batch_episode_count,
                                batch_episode_reward_max,
                            )
                        )

                    if do_adv_update:
                        # Store the regret value for logging (one value per adversary update).
                        adv_rewbuffer.append(adv_regret)
                        self.alg_adversary.compute_returns(obs)

                if do_adv_update:
                    adv_loss_dict = self.alg_adversary.update()
                    for k, v in adv_loss_dict.items():
                        adv_loss_sum[k] = adv_loss_sum.get(k, 0.0) + float(v)
                    adv_updates += 1

            # Compute rollout-batch EPISODE regret metrics for this iteration.
            # Note: In distributed mode, we aggregate across ranks.
            if self.is_distributed:
                torch.distributed.all_reduce(batch_episode_reward_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(batch_episode_count, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(batch_episode_reward_max, op=torch.distributed.ReduceOp.MAX)

            # Convert to Python floats for logging
            batch_episode_count_f = float(batch_episode_count.item())
            if batch_episode_count_f > 0:
                max_batch_total_reward = float(batch_episode_reward_max.item())
                mean_batch_total_reward = float((batch_episode_reward_sum / batch_episode_count_f).item())
                regret = float(max_batch_total_reward - mean_batch_total_reward)
            else:
                # No episode ended during this rollout window.
                max_batch_total_reward = 0.0
                mean_batch_total_reward = 0.0
                regret = 0.0
            # Replace tensor with float for logging compatibility (same as OnPolicyRunner).
            batch_episode_count = batch_episode_count_f

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Compute randomized parameter stats directly from collected batch
            all_params_cpu = None  # Store for HDF5 saving
            if raw_params_list:
                raw_params_tensor = torch.cat(raw_params_list, dim=0)  # (local_samples, 18) on GPU
                
                if self.is_distributed:
                    # Gather from all ranks
                    gathered = [torch.zeros_like(raw_params_tensor) for _ in range(self.gpu_world_size)]
                    torch.distributed.all_gather(gathered, raw_params_tensor)
                    # Concatenate all ranks' data
                    if self.gpu_global_rank == 0:
                        all_params = torch.cat(gathered, dim=0)
                        all_params_cpu = all_params.cpu()  # Store for HDF5 saving
                    else:
                        all_params = None
                else:
                    all_params = raw_params_tensor
                    all_params_cpu = all_params.cpu()  # Store for HDF5 saving
                
                # Compute statistics from the batch (on rank 0 or non-distributed)
                if all_params is not None and all_params.shape[0] > 0:
                    param_mean = all_params.mean(dim=0).cpu().tolist()
                    param_std = all_params.std(dim=0).cpu().tolist()
                    param_min_v = all_params.min(dim=0).values.cpu().tolist()
                    param_max_v = all_params.max(dim=0).values.cpu().tolist()
                else:
                    zeros = [0.0] * 18
                    param_mean = param_std = param_min_v = param_max_v = zeros
            else:
                zeros = [0.0] * 18
                param_mean = param_std = param_min_v = param_max_v = zeros

            # Compute returns
            with torch.inference_mode():
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()
            adv_loss_dict_mean = None
            if adv_updates > 0:
                adv_loss_dict_mean = {k: v / adv_updates for k, v in adv_loss_sum.items()}

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                # Log information (only from rank 0)
                if not self.disable_logs:
                    self.log(locals())
                    # Save model
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                # Save raw parameters to HDF5 (reuse gathered data from statistics computation)
                if all_params_cpu is not None and param_h5_path:
                    with h5py.File(param_h5_path, "a") as f:
                        group = f.create_group(f"iteration_{it}")
                        group.create_dataset("raw_params", data=all_params_cpu.numpy())
                        group.attrs["num_samples"] = all_params_cpu.shape[0]
                
                raw_params_list.clear()  # Clear after processing

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and self.log_dir is not None and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        save_file_fn = getattr(self.writer, "save_file", None)
                        if callable(save_file_fn):
                            save_file_fn(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        
        # Gather and save any remaining raw parameters (all ranks participate in gather)
        if raw_params_list and param_h5_path:
            raw_params_tensor = torch.cat(raw_params_list, dim=0)  # On GPU
            raw_params_list.clear()
            
            if self.is_distributed:
                # Gather from all ranks
                gathered = [torch.zeros_like(raw_params_tensor) for _ in range(self.gpu_world_size)]
                torch.distributed.all_gather(gathered, raw_params_tensor)
                
                # Concatenate all ranks' data on rank 0
                all_params = None
                if self.gpu_global_rank == 0:
                    all_params = torch.cat(gathered, dim=0).cpu()
            else:
                all_params = raw_params_tensor.cpu()
            
            if param_h5_path and (not self.is_distributed or self.gpu_global_rank == 0):
                with h5py.File(param_h5_path, "a") as f:
                    group = f.create_group(f"iteration_{self.current_learning_iteration}")
                    group.create_dataset("raw_params", data=all_params.numpy())
                    group.attrs["num_samples"] = all_params.shape[0]

    def _act_both(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample policy + adversary actions and concatenate for env.step()."""
        policy_actions = self.alg.act(obs)
        adversary_actions = self.alg_adversary.act(obs)
        actions = torch.cat([policy_actions, adversary_actions], dim=-1)
        return policy_actions, adversary_actions, actions

    def _compute_adversary_rewards(
        self,
        rewards: torch.Tensor,
        adv_step: int,
        adv_reward_window: list[float],
    ) -> tuple[torch.Tensor, int, list[float], bool, float]:
        """Adversary reward (bandit) emitted every k steps.

        For each env.step(), we compute:
            batch_max_t  = max(rewards over envs)
            batch_mean_t = mean(rewards over envs)

        When the window length reaches k (= adversary_update_every_k_steps), we compute:
            regret = max(batch_max_{t-k+1:t}) - batch_mean_t

        On the update step, we provide the logged scalar regret as a dense per-env bandit return:
            R_i = regret = window_max - mean(rewards)
        """
        batch_max = float(rewards.max().item())
        batch_mean = float(rewards.mean().item())
        adv_reward_window.append(batch_max)
        adv_step += 1

        if adv_step < self.adversary_update_every_k_steps:
            return torch.zeros_like(rewards), adv_step, adv_reward_window, False, 0.0

        # Compute regret over the window and emit as reward.
        window_max = max(adv_reward_window)
        regret = float(window_max - batch_mean)
        adv_rewards = torch.full_like(rewards, regret)
        # Reset window/counter for next adversary interval.
        return adv_rewards, 0, [], True, regret

    def _update_rollout_bookkeeping(
        self,
        extras: dict,
        ep_infos: list,
        rewards: torch.Tensor,
        done_ids: torch.Tensor,
        cur_reward_sum: torch.Tensor,
        cur_episode_length: torch.Tensor,
        rewbuffer: deque,
        lenbuffer: deque,
        batch_episode_reward_sum: torch.Tensor,
        batch_episode_count: torch.Tensor,
        batch_episode_reward_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update episode reward/length buffers and per-iteration episode metrics."""
        if "episode" in extras:
            ep_infos.append(extras["episode"])
        elif "log" in extras:
            ep_infos.append(extras["log"])

        cur_reward_sum += rewards
        cur_episode_length += 1

        if done_ids.numel() == 0:
            return batch_episode_reward_sum, batch_episode_count, batch_episode_reward_max

        ep_returns = cur_reward_sum[done_ids][:, 0]
        batch_episode_reward_sum += ep_returns.sum()
        batch_episode_count += float(ep_returns.numel())
        batch_episode_reward_max = torch.maximum(batch_episode_reward_max, ep_returns.max())

        rewbuffer.extend(cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[done_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[done_ids] = 0
        cur_episode_length[done_ids] = 0
        return batch_episode_reward_sum, batch_episode_count, batch_episode_reward_max

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        assert self.writer is not None
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    v = float(value.item())
                    self.writer.add_scalar(key, v, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {v:.4f}\n"""
                else:
                    v = float(value.item())
                    self.writer.add_scalar("Episode/" + key, v, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {v:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        adv_mean_std = self.alg_adversary.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if locs.get("adv_loss_dict_mean") is not None:
            for key, value in locs["adv_loss_dict_mean"].items():
                self.writer.add_scalar(f"Adversary/Loss/{key}", value, locs["it"])
            self.writer.add_scalar("Adversary/Loss/learning_rate", self.alg_adversary.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Adversary/mean_noise_std", adv_mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log adversary parameters (aggregated over rollout window)
        if (
            "param_mean" in locs
            and "param_std" in locs
            and "param_min_v" in locs
            and "param_max_v" in locs
        ):
            for i, (m, s, mn, mx) in enumerate(
                zip(locs["param_mean"], locs["param_std"], locs["param_min_v"], locs["param_max_v"])
            ):
                name = self.randomized_param_names[i] if i < len(self.randomized_param_names) else f"dim_{i}"
                sanitized = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in name)
                tag = f"action_{i:02d}_{sanitized}"
                self.writer.add_scalar(f"adversary_actions/{tag}/mean", float(m), locs["it"])
                self.writer.add_scalar(f"adversary_actions/{tag}/std", float(s), locs["it"])
                self.writer.add_scalar(f"adversary_actions/{tag}/min", float(mn), locs["it"])
                self.writer.add_scalar(f"adversary_actions/{tag}/max", float(mx), locs["it"])

        # Log rollout-batch metrics (independent of completed episodes)
        if (
            "batch_episode_count" in locs
            and "max_batch_total_reward" in locs
            and "mean_batch_total_reward" in locs
            and "regret" in locs
            and locs["batch_episode_count"] > 0
        ):
            self.writer.add_scalar("Metrics/max_batch_total_reward", locs["max_batch_total_reward"], locs["it"])
            self.writer.add_scalar("Metrics/mean_batch_total_reward", locs["mean_batch_total_reward"], locs["it"])
            self.writer.add_scalar("Metrics/regret", locs["regret"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if "adv_rewbuffer" in locs and len(locs["adv_rewbuffer"]) > 0:
                self.writer.add_scalar("Adversary/mean_reward", statistics.mean(locs["adv_rewbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            if locs.get("adv_loss_dict_mean") is not None:
                for key, value in locs["adv_loss_dict_mean"].items():
                    log_string += f"""{f"Mean adversary {key} loss:":>{pad}} {value:.4f}\n"""
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            if "adv_rewbuffer" in locs and len(locs["adv_rewbuffer"]) > 0:
                log_string += f"""{f"Mean adversary reward:":>{pad}} {statistics.mean(locs["adv_rewbuffer"]):.2f}\n"""
            # Print regret metrics
            log_string += f"""{"Max batch reward:":>{pad}} {locs["max_batch_total_reward"]:.2f}\n"""
            log_string += f"""{"Regret:":>{pad}} {locs["regret"]:.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
            if "max_batch_total_reward" in locs and "regret" in locs:
                log_string += f"""{"Max batch reward:":>{pad}} {locs["max_batch_total_reward"]:.2f}\n"""
                log_string += f"""{"Regret:":>{pad}} {locs["regret"]:.2f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save in the exact same format as OnPolicyRunner for downstream compatibility.
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration) # type: ignore

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Backward compat: previously saved as nested dicts under "protagonist"/"adversary".
        if "protagonist" in loaded_dict and "adversary" in loaded_dict:
            p = loaded_dict["protagonist"]
            resumed_training = self.alg.policy.load_state_dict(p["model_state_dict"])
            if load_optimizer and resumed_training:
                self.alg.optimizer.load_state_dict(p["optimizer_state_dict"])
            if resumed_training and "iter" in loaded_dict:
                self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict.get("infos")

        # Default: OnPolicyRunner-compatible keys.
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        if resumed_training and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        self.alg_adversary.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg_adversary.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
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
        """Construct one PPO pipeline for a single agent."""
        alg_cfg = dict(alg_cfg)
        policy_cfg = dict(policy_cfg)

        # Explicitly disable RND for this runner.
        if "rnd_cfg" in alg_cfg:
            alg_cfg["rnd_cfg"] = None
        # Resolve symmetry config
        alg_cfg = resolve_symmetry_config(alg_cfg, self.env)

        # Resolve deprecated normalization config
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

        # Initialize the policy
        actor_critic_class = eval(policy_cfg["class_name"])
        policy_kwargs = {k: v for k, v in policy_cfg.items() if k != "class_name"}
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, obs_groups, action_dim, **policy_kwargs
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(alg_cfg["class_name"])
        alg_kwargs = {k: v for k, v in alg_cfg.items() if k != "class_name"}
        alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            storage_horizon,
            obs,
            [action_dim],
        )
        return alg

    def _resolve_randomized_param_names(self) -> list[str]:
        cfg_names = self.cfg.get("randomized_param_names", None)
        if isinstance(cfg_names, (list, tuple)) and len(cfg_names) == 18:
            return [str(x) for x in cfg_names]
        return [
            "robot_static_friction", "robot_dynamic_friction",
            "insertive_object_static_friction", "insertive_object_dynamic_friction",
            "receptive_object_static_friction", "receptive_object_dynamic_friction",
            "table_static_friction", "table_dynamic_friction",
            "robot_mass_scale", "insertive_object_mass_scale",
            "receptive_object_mass_scale", "table_mass_scale",
            "robot_joint_friction_scale", "robot_joint_armature_scale",
            "gripper_stiffness_scale", "gripper_damping_scale",
            "osc_stiffness_scale", "osc_damping_scale",
        ]

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
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

    def _extract_randomized_params(self) -> torch.Tensor:
        # In distributed training, each rank should only extract from its assigned environments
        # If env.num_envs returns total, we need to determine per-rank subset
        total_envs = self.env.num_envs
        if self.is_distributed:
            # Calculate which environments belong to this rank
            # Typically: rank R handles environments [R * (N/W) : (R+1) * (N/W)]
            envs_per_rank = total_envs // self.gpu_world_size
            start_env = self.gpu_global_rank * envs_per_rank
            end_env = start_env + envs_per_rank if self.gpu_global_rank < self.gpu_world_size - 1 else total_envs
            num_envs = end_env - start_env
            env_indices = slice(start_env, end_env)
        else:
            num_envs = total_envs
            env_indices = slice(None)
        
        params = torch.zeros((num_envs, 18), dtype=torch.float, device=self.device)
        scene = self.env.unwrapped.scene

        robot = scene["robot"]
        materials = robot.root_physx_view.get_material_properties()[env_indices]
        params[:, 0] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 1] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = robot.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = robot.data.default_mass[env_indices].to(self.device)
        params[:, 8] = (current_masses / (default_masses + 1e-8)).mean(dim=1)
        current_friction = robot.data.joint_friction_coeff[env_indices].to(self.device)
        default_friction = robot.data.default_joint_friction_coeff[env_indices].to(self.device)
        params[:, 12] = (current_friction / (default_friction + 1e-8)).mean(dim=1)
        current_armature = robot.data.joint_armature[env_indices].to(self.device)
        default_armature = robot.data.default_joint_armature[env_indices].to(self.device)
        params[:, 13] = (current_armature / (default_armature + 1e-8)).mean(dim=1)
        current_stiffness = robot.data.joint_stiffness[env_indices].to(self.device)
        default_stiffness = robot.data.default_joint_stiffness[env_indices].to(self.device)
        params[:, 14] = (current_stiffness / (default_stiffness + 1e-8)).mean(dim=1)
        current_damping = robot.data.joint_damping[env_indices].to(self.device)
        default_damping = robot.data.default_joint_damping[env_indices].to(self.device)
        params[:, 15] = (current_damping / (default_damping + 1e-8)).mean(dim=1)

        insertive_obj = scene["insertive_object"]
        materials = insertive_obj.root_physx_view.get_material_properties()[env_indices]
        params[:, 2] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 3] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = insertive_obj.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = insertive_obj.data.default_mass[env_indices].to(self.device)
        params[:, 9] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        receptive_obj = scene["receptive_object"]
        materials = receptive_obj.root_physx_view.get_material_properties()[env_indices]
        params[:, 4] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 5] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = receptive_obj.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = receptive_obj.data.default_mass[env_indices].to(self.device)
        params[:, 10] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        table = scene["table"]
        materials = table.root_physx_view.get_material_properties()[env_indices]
        params[:, 6] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 7] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = table.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = table.data.default_mass[env_indices].to(self.device)
        params[:, 11] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        osc_action_term = self.env.unwrapped.action_manager._terms.get("arm")
        controller = osc_action_term._osc
        # Get current stiffness gains (diagonal of motion_p_gains_task)
        current_stiffness_diag = torch.diagonal(controller._motion_p_gains_task[env_indices], dim1=-2, dim2=-1)
        # Default stiffness from config (6 values: xyz, rpy)
        default_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=self.device)
        # Extract scale from xyz block (use first element or mean)
        current_stiffness_xyz = current_stiffness_diag[:, 0]  # First xyz element
        default_stiffness_xyz = default_stiffness[0]
        params[:, 16] = current_stiffness_xyz / (default_stiffness_xyz + 1e-8)

        # Get current damping gains and compute damping ratio
        current_damping_diag = torch.diagonal(controller._motion_d_gains_task[env_indices], dim1=-2, dim2=-1)
        # Damping = 2 * sqrt(stiffness) * damping_ratio, so damping_ratio = damping / (2 * sqrt(stiffness))
        current_damping_ratio_xyz = current_damping_diag[:, 0] / (2 * current_stiffness_xyz.sqrt() + 1e-8)
        default_damping_ratio = torch.tensor(controller.cfg.motion_damping_ratio_task, device=self.device)
        default_damping_ratio_xyz = default_damping_ratio[0]
        params[:, 17] = current_damping_ratio_xyz / (default_damping_ratio_xyz + 1e-8)

        return params