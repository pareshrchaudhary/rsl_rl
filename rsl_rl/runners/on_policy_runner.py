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
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Resolve parameter names for logging
        self.randomized_param_names = self._resolve_randomized_param_names()

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

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        param_h5_path = None
        if self.log_dir is not None:
            param_h5_path = os.path.join(self.log_dir, "raw_params.h5")
            # Initialize HDF5 file with metadata if it doesn't exist (only from rank 0)
            if not self.disable_logs and not os.path.exists(param_h5_path):
                with h5py.File(param_h5_path, "w") as f:
                    f.attrs["param_names"] = [n.encode("utf-8") for n in self.randomized_param_names]
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Track randomized parameters over this rollout window (for logging)
            param_sum = torch.zeros(16, dtype=torch.float, device=self.device)
            param_sumsq = torch.zeros(16, dtype=torch.float, device=self.device)
            param_min = torch.full((16,), float("inf"), dtype=torch.float, device=self.device)
            param_max = torch.full((16,), float("-inf"), dtype=torch.float, device=self.device)
            param_count = 0
            raw_params_list = []  # Store raw parameters before aggregation

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Extract and track randomized parameters
                    randomized_params = self._extract_randomized_params()
                    p = randomized_params.reshape(-1, randomized_params.shape[-1])
                    # Store raw parameters before aggregation (keep on GPU for distributed gather)
                    raw_params_list.append(randomized_params.detach())
                    param_sum = param_sum + p.sum(dim=0)
                    param_sumsq = param_sumsq + (p * p).sum(dim=0)
                    param_min = torch.minimum(param_min, p.min(dim=0).values)
                    param_max = torch.maximum(param_max, p.max(dim=0).values)
                    param_count += int(p.shape[0])

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                if self.is_distributed:
                    # Aggregate randomized parameter stats across ranks
                    torch.distributed.all_reduce(param_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(param_sumsq, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(param_min, op=torch.distributed.ReduceOp.MIN)
                    torch.distributed.all_reduce(param_max, op=torch.distributed.ReduceOp.MAX)
                    param_count_t = torch.tensor(float(param_count), dtype=torch.float, device=self.device)
                    torch.distributed.all_reduce(param_count_t, op=torch.distributed.ReduceOp.SUM)
                    param_count = int(param_count_t.item())


                stop = time.time()
                collection_time = stop - start
                start = stop

                # Finalize randomized parameter stats for logging
                if param_count > 0:
                    param_mean = (param_sum / float(param_count)).detach().cpu().tolist()
                    var_t = (param_sumsq / float(param_count)) - (param_sum / float(param_count)) * (param_sum / float(param_count))
                    param_std = torch.sqrt(torch.clamp(var_t, min=0.0)).detach().cpu().tolist()
                    param_min_v = param_min.detach().cpu().tolist()
                    param_max_v = param_max.detach().cpu().tolist()
                else:
                    zeros = [0.0] * 16
                    param_mean = param_std = param_min_v = param_max_v = zeros

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

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
                # Gather and concatenate raw parameters from all ranks, then save
                if raw_params_list:
                    raw_params_tensor = torch.cat(raw_params_list, dim=0)  # (local_samples, 16) on GPU
                    raw_params_list.clear()  # Clear after concatenation
                    
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
                    
                    # Save from rank 0 only (or non-distributed)
                    if param_h5_path and (not self.is_distributed or self.gpu_global_rank == 0):
                        with h5py.File(param_h5_path, "a") as f:
                            group = f.create_group(f"iteration_{it}")
                            group.create_dataset("raw_params", data=all_params.numpy())
                            group.attrs["num_samples"] = all_params.shape[0]

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

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

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
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
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log randomized parameters (aggregated over rollout window)
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

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
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
            # Print rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
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
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

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

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
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
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _resolve_randomized_param_names(self) -> list[str]:
        cfg_names = self.cfg.get("randomized_param_names", None)
        if isinstance(cfg_names, (list, tuple)) and len(cfg_names) == 16:
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
        ]

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
        
        params = torch.zeros((num_envs, 16), dtype=torch.float, device=self.device)
        scene = self.env.unwrapped.scene

        robot = scene["robot"]
        materials = robot.root_physx_view.get_material_properties()[env_indices]
        params[:, 0] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 1] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = robot.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = robot.data.default_mass.to(self.device)
        params[:, 8] = (current_masses / (default_masses + 1e-8)).mean(dim=1)
        current_friction = robot.data.joint_friction_coeff[env_indices].to(self.device)
        default_friction = robot.data.default_joint_friction_coeff.to(self.device)
        params[:, 12] = (current_friction / (default_friction + 1e-8)).mean(dim=1)
        current_armature = robot.data.joint_armature[env_indices].to(self.device)
        default_armature = robot.data.default_joint_armature.to(self.device)
        params[:, 13] = (current_armature / (default_armature + 1e-8)).mean(dim=1)
        current_stiffness = robot.data.joint_stiffness[env_indices].to(self.device)
        default_stiffness = robot.data.default_joint_stiffness.to(self.device)
        params[:, 14] = (current_stiffness / (default_stiffness + 1e-8)).mean(dim=1)
        current_damping = robot.data.joint_damping[env_indices].to(self.device)
        default_damping = robot.data.default_joint_damping.to(self.device)
        params[:, 15] = (current_damping / (default_damping + 1e-8)).mean(dim=1)

        insertive_obj = scene["insertive_object"]
        materials = insertive_obj.root_physx_view.get_material_properties()[env_indices]
        params[:, 2] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 3] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = insertive_obj.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = insertive_obj.data.default_mass.to(self.device)
        params[:, 9] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        receptive_obj = scene["receptive_object"]
        materials = receptive_obj.root_physx_view.get_material_properties()[env_indices]
        params[:, 4] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 5] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = receptive_obj.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = receptive_obj.data.default_mass.to(self.device)
        params[:, 10] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        table = scene["table"]
        materials = table.root_physx_view.get_material_properties()[env_indices]
        params[:, 6] = materials[:, :, 0].to(self.device).mean(dim=1)
        params[:, 7] = materials[:, :, 1].to(self.device).mean(dim=1)
        current_masses = table.root_physx_view.get_masses()[env_indices].to(self.device)
        default_masses = table.data.default_mass.to(self.device)
        params[:, 11] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

        return params