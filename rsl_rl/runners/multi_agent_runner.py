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
from rsl_rl.utils.logger import resolve_randomized_param_names, extract_randomized_params, log_multi_agent


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
        self.record_parameters = self.cfg.get("record_parameters", False)

        # Action split: policy controls robot, adversary controls last `adversary_action_dim` entries.
        # NOTE: For UWLab adversarial tasks this is 9 (see AdversaryActionCfg.action_dim).
        self.adversary_action_dim = int(self.cfg.get("adversary_action_dim", 9))
        if self.env.num_actions <= self.adversary_action_dim:
            raise ValueError(
                f"env.num_actions ({self.env.num_actions}) must be > adversary_action_dim ({self.adversary_action_dim})."
            )
        self.policy_action_dim = int(self.env.num_actions - self.adversary_action_dim)
        # Resolve parameter names for logging
        self.randomized_param_names = resolve_randomized_param_names(self.cfg)

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
        if self.log_dir is not None and self.record_parameters:
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

            # Extract parameters once per iteration
            randomized_params = extract_randomized_params(
                self.env, self.device, self.is_distributed, self.gpu_world_size, self.gpu_global_rank
            )
            # Store raw parameters (keep on GPU for distributed gather)
            raw_params_list.append(randomized_params.detach())

            # Rollout
            for _ in range(self.num_steps_per_env):
                # Agent stepping is inference-only; updates happen outside.
                with torch.inference_mode():
                    # Sample actions from both agents (each uses its own obs_groups mapping).
                    _, adversary_actions, actions = self._act_both(obs)

                    # Step the environment with combined actions.
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

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
                # Extract the single parameter tensor (one per environment per iteration)
                raw_params_tensor = raw_params_list[0]  # (num_envs, 18) on GPU
                
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
                    # Update total timesteps and time
                    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
                    self.tot_timesteps += collection_size
                    self.tot_time += collection_time + learn_time
                    # Log metrics
                    log_multi_agent(
                        writer=self.writer,
                        device=self.device,
                        num_steps_per_env=self.num_steps_per_env,
                        num_envs=self.env.num_envs,
                        gpu_world_size=self.gpu_world_size,
                        alg=self.alg,
                        alg_adversary=self.alg_adversary,
                        randomized_param_names=self.randomized_param_names,
                        logger_type=self.logger_type,
                        tot_timesteps=self.tot_timesteps,
                        tot_time=self.tot_time,
                        locs=locals(),
                    )
                    # Save model
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                # Save raw parameters to HDF5 (reuse gathered data from statistics computation)
                if self.record_parameters and all_params_cpu is not None and param_h5_path:
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
        if self.record_parameters and raw_params_list and param_h5_path:
            raw_params_tensor = raw_params_list[0]  # Single tensor (num_envs, 18) on GPU
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
            
            if param_h5_path and all_params is not None and (not self.is_distributed or self.gpu_global_rank == 0):
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

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration) # type: ignore
            self.writer.save_model(adversary_path, self.current_learning_iteration) # type: ignore

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        if resumed_training and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]

        # Load adversary from separate file if it exists
        adversary_path = path.replace(".pt", "_adversary.pt")
        if os.path.exists(adversary_path):
            adversary_loaded_dict = torch.load(adversary_path, weights_only=False, map_location=map_location)
            adversary_resumed_training = self.alg_adversary.policy.load_state_dict(adversary_loaded_dict["model_state_dict"])
            if load_optimizer and adversary_resumed_training:
                self.alg_adversary.optimizer.load_state_dict(adversary_loaded_dict["optimizer_state_dict"])

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
