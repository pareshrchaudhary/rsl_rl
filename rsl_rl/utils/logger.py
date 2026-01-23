# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import statistics
import time
import torch


def resolve_randomized_param_names(cfg: dict) -> list[str]:
    """Resolve randomized parameter names from config or return defaults.
    
    Args:
        cfg: Configuration dictionary that may contain 'randomized_param_names'
        
    Returns:
        List of 18 parameter names
    """
    cfg_names = cfg.get("randomized_param_names", None)
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


def extract_randomized_params(
    env,
    device: str | torch.device,
    is_distributed: bool = False,
    gpu_world_size: int = 1,
    gpu_global_rank: int = 0,
) -> torch.Tensor:
    """Extract randomized parameters from environment.
    
    Args:
        env: Environment instance with unwrapped.scene attribute
        device: Device to place tensors on
        is_distributed: Whether running in distributed mode
        gpu_world_size: Total number of GPU ranks
        gpu_global_rank: Current GPU rank
        
    Returns:
        Tensor of shape (num_envs, 18) containing randomized parameters
    """
    # In distributed training, each rank should only extract from its assigned environments
    # If env.num_envs returns total, we need to determine per-rank subset
    total_envs = env.num_envs
    if is_distributed:
        # Calculate which environments belong to this rank
        # Typically: rank R handles environments [R * (N/W) : (R+1) * (N/W)]
        envs_per_rank = total_envs // gpu_world_size
        start_env = gpu_global_rank * envs_per_rank
        end_env = start_env + envs_per_rank if gpu_global_rank < gpu_world_size - 1 else total_envs
        num_envs = end_env - start_env
        env_indices = slice(start_env, end_env)
    else:
        num_envs = total_envs
        env_indices = slice(None)
    
    params = torch.zeros((num_envs, 18), dtype=torch.float, device=device)
    scene = env.unwrapped.scene

    robot = scene["robot"]
    materials = robot.root_physx_view.get_material_properties()[env_indices]
    params[:, 0] = materials[:, :, 0].to(device).mean(dim=1)
    params[:, 1] = materials[:, :, 1].to(device).mean(dim=1)
    current_masses = robot.root_physx_view.get_masses()[env_indices].to(device)
    default_masses = robot.data.default_mass[env_indices].to(device)
    params[:, 8] = (current_masses / (default_masses + 1e-8)).mean(dim=1)
    current_friction = robot.data.joint_friction_coeff[env_indices].to(device)
    default_friction = robot.data.default_joint_friction_coeff[env_indices].to(device)
    params[:, 12] = (current_friction / (default_friction + 1e-8)).mean(dim=1)
    current_armature = robot.data.joint_armature[env_indices].to(device)
    default_armature = robot.data.default_joint_armature[env_indices].to(device)
    params[:, 13] = (current_armature / (default_armature + 1e-8)).mean(dim=1)
    current_stiffness = robot.data.joint_stiffness[env_indices].to(device)
    default_stiffness = robot.data.default_joint_stiffness[env_indices].to(device)
    params[:, 14] = (current_stiffness / (default_stiffness + 1e-8)).mean(dim=1)
    current_damping = robot.data.joint_damping[env_indices].to(device)
    default_damping = robot.data.default_joint_damping[env_indices].to(device)
    params[:, 15] = (current_damping / (default_damping + 1e-8)).mean(dim=1)

    insertive_obj = scene["insertive_object"]
    materials = insertive_obj.root_physx_view.get_material_properties()[env_indices]
    params[:, 2] = materials[:, :, 0].to(device).mean(dim=1)
    params[:, 3] = materials[:, :, 1].to(device).mean(dim=1)
    current_masses = insertive_obj.root_physx_view.get_masses()[env_indices].to(device)
    default_masses = insertive_obj.data.default_mass[env_indices].to(device)
    params[:, 9] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

    receptive_obj = scene["receptive_object"]
    materials = receptive_obj.root_physx_view.get_material_properties()[env_indices]
    params[:, 4] = materials[:, :, 0].to(device).mean(dim=1)
    params[:, 5] = materials[:, :, 1].to(device).mean(dim=1)
    current_masses = receptive_obj.root_physx_view.get_masses()[env_indices].to(device)
    default_masses = receptive_obj.data.default_mass[env_indices].to(device)
    params[:, 10] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

    table = scene["table"]
    materials = table.root_physx_view.get_material_properties()[env_indices]
    params[:, 6] = materials[:, :, 0].to(device).mean(dim=1)
    params[:, 7] = materials[:, :, 1].to(device).mean(dim=1)
    current_masses = table.root_physx_view.get_masses()[env_indices].to(device)
    default_masses = table.data.default_mass[env_indices].to(device)
    params[:, 11] = (current_masses / (default_masses + 1e-8)).mean(dim=1)

    osc_action_term = env.unwrapped.action_manager._terms.get("arm")
    controller = osc_action_term._osc
    # Get current stiffness gains (diagonal of motion_p_gains_task)
    current_stiffness_diag = torch.diagonal(controller._motion_p_gains_task[env_indices], dim1=-2, dim2=-1)
    # Default stiffness from config (6 values: xyz, rpy)
    default_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=device)
    # Extract scale from xyz block (use first element or mean)
    current_stiffness_xyz = current_stiffness_diag[:, 0]  # First xyz element
    default_stiffness_xyz = default_stiffness[0]
    params[:, 16] = current_stiffness_xyz / (default_stiffness_xyz + 1e-8)

    # Get current damping gains and compute damping ratio
    current_damping_diag = torch.diagonal(controller._motion_d_gains_task[env_indices], dim1=-2, dim2=-1)
    # Damping = 2 * sqrt(stiffness) * damping_ratio, so damping_ratio = damping / (2 * sqrt(stiffness))
    current_damping_ratio_xyz = current_damping_diag[:, 0] / (2 * current_stiffness_xyz.sqrt() + 1e-8)
    default_damping_ratio = torch.tensor(controller.cfg.motion_damping_ratio_task, device=device)
    default_damping_ratio_xyz = default_damping_ratio[0]
    params[:, 17] = current_damping_ratio_xyz / (default_damping_ratio_xyz + 1e-8)

    return params


def log_multi_agent(
    writer,
    device: str | torch.device,
    num_steps_per_env: int,
    num_envs: int,
    gpu_world_size: int,
    alg,
    alg_adversary,
    randomized_param_names: list[str],
    logger_type: str,
    tot_timesteps: int,
    tot_time: float,
    locs: dict,
    width: int = 80,
    pad: int = 35,
) -> None:
    """Log training metrics for multi-agent runner.
    
    Args:
        writer: Summary writer for logging
        device: Device to place tensors on
        num_steps_per_env: Number of steps per environment per iteration
        num_envs: Number of environments
        gpu_world_size: Total number of GPU ranks
        alg: Main algorithm instance
        alg_adversary: Adversary algorithm instance
        randomized_param_names: List of randomized parameter names
        logger_type: Type of logger (tensorboard, wandb, neptune)
        tot_timesteps: Total timesteps so far
        tot_time: Total time elapsed so far
        locs: Dictionary containing logging data
        width: Width of log output
        pad: Padding for log output
        
    Returns:
        Tuple of (updated tot_timesteps, updated tot_time)
    """
    # Compute the collection size
    collection_size = num_steps_per_env * num_envs * gpu_world_size
    iteration_time = locs["collection_time"] + locs["learn_time"]

    # Log episode information
    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            infotensor = torch.tensor([], device=device)
            for ep_info in locs["ep_infos"]:
                # Handle scalar and zero dimensional tensor infos
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(device)))
            value = torch.mean(infotensor)
            # Log to logger and terminal
            if "/" in key:
                v = float(value.item())
                writer.add_scalar(key, v, locs["it"])
                ep_string += f"""{f"{key}:":>{pad}} {v:.4f}\n"""
            else:
                v = float(value.item())
                writer.add_scalar("Episode/" + key, v, locs["it"])
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {v:.4f}\n"""

    mean_std = alg.policy.action_std.mean()
    adv_mean_std = alg_adversary.policy.action_std.mean()
    fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

    # Log losses
    for key, value in locs["loss_dict"].items():
        writer.add_scalar(f"Loss/{key}", value, locs["it"])
    writer.add_scalar("Loss/learning_rate", alg.learning_rate, locs["it"])
    if locs.get("adv_loss_dict_mean") is not None:
        for key, value in locs["adv_loss_dict_mean"].items():
            writer.add_scalar(f"Adversary/Loss/{key}", value, locs["it"])
        writer.add_scalar("Adversary/Loss/learning_rate", alg_adversary.learning_rate, locs["it"])

    # Log noise std
    writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
    writer.add_scalar("Adversary/mean_noise_std", adv_mean_std.item(), locs["it"])

    # Log performance
    writer.add_scalar("Perf/total_fps", fps, locs["it"])
    writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

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
            name = randomized_param_names[i] if i < len(randomized_param_names) else f"dim_{i}"
            sanitized = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in name)
            tag = f"action_{i:02d}_{sanitized}"
            writer.add_scalar(f"adversary_actions/{tag}/mean", float(m), locs["it"])
            writer.add_scalar(f"adversary_actions/{tag}/std", float(s), locs["it"])
            writer.add_scalar(f"adversary_actions/{tag}/min", float(mn), locs["it"])
            writer.add_scalar(f"adversary_actions/{tag}/max", float(mx), locs["it"])

    # Log rollout-batch metrics (independent of completed episodes)
    if (
        "batch_episode_count" in locs
        and "max_batch_total_reward" in locs
        and "mean_batch_total_reward" in locs
        and "regret" in locs
        and locs["batch_episode_count"] > 0
    ):
        writer.add_scalar("Metrics/max_batch_total_reward", locs["max_batch_total_reward"], locs["it"])
        writer.add_scalar("Metrics/mean_batch_total_reward", locs["mean_batch_total_reward"], locs["it"])
        writer.add_scalar("Metrics/regret", locs["regret"], locs["it"])

    # Log training
    if len(locs["rewbuffer"]) > 0:
        # Everything else
        writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        if "adv_rewbuffer" in locs and len(locs["adv_rewbuffer"]) > 0:
            writer.add_scalar("Adversary/mean_regret", statistics.mean(locs["adv_rewbuffer"]), locs["it"])
        if logger_type != "wandb":  # wandb does not support non-integer x-axis logging
            writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), tot_time)
            writer.add_scalar(
                "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), tot_time
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
            log_string += f"""{f"Mean adversary regret:":>{pad}} {statistics.mean(locs["adv_rewbuffer"]):.2f}\n"""
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
        f"""{"Total timesteps:":>{pad}} {tot_timesteps}\n"""
        f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
        f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(tot_time))}\n"""
        f"""{"ETA:":>{pad}} {
            time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    tot_time
                    / (locs["it"] - locs["start_iter"] + 1)
                    * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                ),
            )
        }\n"""
    )
    print(log_string)
