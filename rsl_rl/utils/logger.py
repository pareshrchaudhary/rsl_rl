# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import statistics
import time
import torch


def extract_adversary_params(
    env,
    device: str | torch.device,
    is_distributed: bool = False,
    gpu_world_size: int = 1,
    gpu_global_rank: int = 0,
    adversary_param_extractor_fn: callable | None = None,
) -> torch.Tensor | None:
    """Extract adversary-chosen parameters from environment.

    Args:
        env: Environment instance.
        device: Device to place tensors on.
        is_distributed: Whether running in distributed mode.
        gpu_world_size: Total number of GPU ranks.
        gpu_global_rank: Current GPU rank.
        adversary_param_extractor_fn: Callable that extracts parameters from the
            environment for HDF5 storage. Signature:
            (env, device, is_distributed, world_size, rank) -> Tensor(num_envs, N).
            If None, extraction is skipped.

    Returns:
        Tensor of shape (num_envs, N) containing adversary parameters, or None
        if no extractor is provided.
    """
    if adversary_param_extractor_fn is None:
        return None
    return adversary_param_extractor_fn(env, device, is_distributed, gpu_world_size, gpu_global_rank)


def extract_cage_physics_params(
    env,
    device: str | torch.device,
    is_distributed: bool = False,
    gpu_world_size: int = 1,
    gpu_global_rank: int = 0,
) -> torch.Tensor:
    """Extract CAGE physics parameters from environment scene.

    Returns a tensor of shape (num_envs, 18) with columns:
        0-1: robot static/dynamic friction
        2-3: insertive object static/dynamic friction
        4-5: receptive object static/dynamic friction
        6-7: table static/dynamic friction
        8: robot mass scale
        9: insertive object mass scale
        10: receptive object mass scale
        11: table mass scale
        12: robot joint friction scale
        13: robot joint armature scale
        14: gripper stiffness scale
        15: gripper damping scale
        16: OSC stiffness scale
        17: OSC damping scale
    """
    total_envs = env.num_envs
    if is_distributed:
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
    if osc_action_term is not None:
        controller = getattr(osc_action_term, "_osc", None)
        if controller is not None and hasattr(controller, "_motion_p_gains_task"):
            current_stiffness_diag = torch.diagonal(
                controller._motion_p_gains_task[env_indices], dim1=-2, dim2=-1
            )
            default_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=device)
            current_stiffness_xyz = current_stiffness_diag[:, 0]
            default_stiffness_xyz = default_stiffness[0]
            params[:, 16] = current_stiffness_xyz / (default_stiffness_xyz + 1e-8)

            current_damping_diag = torch.diagonal(
                controller._motion_d_gains_task[env_indices], dim1=-2, dim2=-1
            )
            current_damping_ratio_xyz = current_damping_diag[:, 0] / (
                2 * current_stiffness_xyz.sqrt() + 1e-8
            )
            default_damping_ratio = torch.tensor(
                controller.cfg.motion_damping_ratio_task, device=device
            )
            default_damping_ratio_xyz = default_damping_ratio[0]
            params[:, 17] = current_damping_ratio_xyz / (default_damping_ratio_xyz + 1e-8)
        elif hasattr(osc_action_term, "_kp") and hasattr(osc_action_term, "_kd"):
            kp_s = osc_action_term._kp[env_indices]
            kd_s = osc_action_term._kd[env_indices]
            current_stiffness_xyz = kp_s[:, 0]
            default_stiffness_xyz = osc_action_term._kp_default[0]
            params[:, 16] = current_stiffness_xyz / (default_stiffness_xyz + 1e-8)
            current_damping_ratio_xyz = kd_s[:, 0] / (2 * current_stiffness_xyz.sqrt() + 1e-8)
            default_damping_ratio_xyz = osc_action_term._damping_ratio_default[0]
            params[:, 17] = current_damping_ratio_xyz / (default_damping_ratio_xyz + 1e-8)

    return params


def log_phase_a(
    writer,
    alg_adversary,
    cycle_id: int,
    step: int,
    gen_stats: dict | None,
    buffer_diag: dict | None,
    cycle_adv_loss_sums: dict[str, float],
    cycle_adv_loss_count: int,
    cycle_adv_total_rewards: list[float],
    cycle_adv_gen_rewards: list[float],
    cycle_adv_regrets: list[float],
    width: int = 80,
    pad: int = 35,
) -> None:
    """Log Phase A (generator + adversary) metrics at the current iteration step.

    Called once per refill event: the bootstrap refill and each in-loop refill
    thereafter. Does not log any protagonist metrics; tag namespaces
    (``Adversary/*``, ``Generator/*``, ``Generator_Reward_Success/*``,
    ``BufferDiag/*``) keep Phase A curves separate from Phase B on the shared
    iteration x-axis. Because Phase A fires only at cycle boundaries, its
    points are naturally sparse on that axis — no carry-forward between cycles.

    Adversary reward is decomposed into three scalars so plots show which
    component drives PPO:
        total  = beta * gen_reward + regret   (what PPO actually trains on)
        gen    = gen_reward                   (raw validity/shaping signal)
        regret = max(returns) - mean(returns) (0 for slots with <K episodes)

    Args:
        writer: Summary writer for logging.
        alg_adversary: Adversary algorithm instance (for noise std + lr).
        cycle_id: Refill index, used only in the console banner.
        step: Monotonic step (iteration index) used as the writer x-axis. This
            must be monotonically non-decreasing across all writer calls so
            wandb's global-step constraint is satisfied.
        gen_stats: Dict returned by ``_run_generation_loop`` (or None).
        buffer_diag: Dict of per-slot regret distribution stats (or None).
        cycle_adv_loss_sums: Running per-key loss sums for the cycle.
        cycle_adv_loss_count: Number of adversary updates that fired this cycle.
        cycle_adv_total_rewards: Per-update mean of beta*gen_reward + regret.
        cycle_adv_gen_rewards: Per-update mean of raw gen_reward.
        cycle_adv_regrets: Per-update mean regret (0 for starved-slot entries).
        width: Console line width.
        pad: Right-alignment pad for the console block.
    """
    if gen_stats is not None:
        writer.add_scalar("Generator/validity_rate", gen_stats["mean_validity_rate"], step)
        writer.add_scalar("Generator/mean_reward", gen_stats["mean_reward"], step)
        writer.add_scalar("Generator/state_quality", gen_stats["mean_state_quality"], step)
        writer.add_scalar("Generator/num_successes", gen_stats["num_successes"], step)
        writer.add_scalar("Generator/num_episodes_done", gen_stats["num_episodes_done"], step)
        writer.add_scalar("Generator/buffer_fill_pct", gen_stats["buffer_fill_pct"], step)

        for tname, tval_suc in zip(
            gen_stats["reward_term_names"],
            gen_stats["reward_term_success_means"],
        ):
            writer.add_scalar(f"Generator_Reward_Success/{tname}", tval_suc, step)

    if buffer_diag is not None:
        writer.add_scalar("BufferDiag/regret_mean", buffer_diag["regret_mean"], step)
        writer.add_scalar("BufferDiag/regret_p90", buffer_diag["regret_p90"], step)
        writer.add_scalar("BufferDiag/regret_max", buffer_diag["regret_max"], step)
        writer.add_scalar("BufferDiag/kept_top50_over_mean", buffer_diag["kept_top50_over_mean"], step)

    adv_loss_means: dict[str, float] = {}
    if cycle_adv_loss_count > 0:
        adv_loss_means = {k: v / cycle_adv_loss_count for k, v in cycle_adv_loss_sums.items()}
        for key, value in adv_loss_means.items():
            writer.add_scalar(f"Adversary/Loss/{key}", value, step)

    writer.add_scalar("Adversary/Loss/learning_rate", alg_adversary.learning_rate, step)
    adv_mean_std = alg_adversary.policy.action_std.mean()
    writer.add_scalar("Adversary/mean_noise_std", adv_mean_std.item(), step)

    mean_adv_total: float | None = None
    mean_adv_gen: float | None = None
    mean_adv_regret: float | None = None
    if cycle_adv_total_rewards:
        mean_adv_total = statistics.mean(cycle_adv_total_rewards)
        writer.add_scalar("Adversary/mean_total_reward", mean_adv_total, step)
    if cycle_adv_gen_rewards:
        mean_adv_gen = statistics.mean(cycle_adv_gen_rewards)
        writer.add_scalar("Adversary/mean_gen_reward", mean_adv_gen, step)
    if cycle_adv_regrets:
        mean_adv_regret = statistics.mean(cycle_adv_regrets)
        writer.add_scalar("Adversary/mean_regret", mean_adv_regret, step)

    header = f" \033[1m Phase A cycle {cycle_id} \033[0m "
    log_string = f"""{"#" * width}\n{header.center(width, " ")}\n\n"""
    if gen_stats is not None:
        log_string += f"""{"Validity rate:":>{pad}} {gen_stats["mean_validity_rate"]:.3f}\n"""
        log_string += f"""{"Buffer fill pct:":>{pad}} {gen_stats["buffer_fill_pct"]:.2f}\n"""
        log_string += f"""{"Num successes:":>{pad}} {int(gen_stats["num_successes"])}\n"""
        log_string += f"""{"Num episodes done:":>{pad}} {int(gen_stats["num_episodes_done"])}\n"""
    log_string += f"""{"Adversary noise std:":>{pad}} {adv_mean_std.item():.2f}\n"""
    if adv_loss_means:
        for key, value in adv_loss_means.items():
            log_string += f"""{f"Mean adversary {key} loss:":>{pad}} {value:.4f}\n"""
        log_string += f"""{"Adversary updates in cycle:":>{pad}} {cycle_adv_loss_count}\n"""
    if mean_adv_total is not None:
        log_string += f"""{"Mean adversary total reward:":>{pad}} {mean_adv_total:.4f}\n"""
    if mean_adv_gen is not None:
        log_string += f"""{"Mean adversary gen reward:":>{pad}} {mean_adv_gen:.4f}\n"""
    if mean_adv_regret is not None:
        log_string += f"""{"Mean adversary regret:":>{pad}} {mean_adv_regret:.4f}\n"""
    log_string += f"""{"-" * width}\n"""
    print(log_string)


def log_phase_b(
    writer,
    device: str | torch.device,
    num_steps_per_env: int,
    num_envs: int,
    gpu_world_size: int,
    alg,
    logger_type: str,
    tot_timesteps: int,
    tot_time: float,
    locs: dict,
    width: int = 80,
    pad: int = 35,
) -> None:
    """Log Phase B (protagonist) metrics on the iteration axis.

    Called every iteration. Does not log any adversary/generator metrics; those
    are owned by ``log_phase_a`` on the cycle_id axis.

    Args:
        writer: Summary writer for logging.
        device: Device for ``ep_infos`` tensor concatenation.
        num_steps_per_env: Rollout length per env per iter.
        num_envs: Number of parallel envs.
        gpu_world_size: Total GPU ranks (for collection-size / fps).
        alg: Protagonist algorithm instance (for noise std + lr).
        logger_type: Logger backend (tensorboard / wandb / neptune).
        tot_timesteps: Running total timesteps so far.
        tot_time: Running total wall time so far.
        locs: ``locals()`` dict from the runner's training loop.
        width: Console line width.
        pad: Right-alignment pad for the console block.
    """
    collection_size = num_steps_per_env * num_envs * gpu_world_size
    iteration_time = locs["collection_time"] + locs["learn_time"]

    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            infotensor = torch.tensor([], device=device)
            for ep_info in locs["ep_infos"]:
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(device)))
            value = torch.mean(infotensor)
            if "/" in key:
                v = float(value.item())
                writer.add_scalar(key, v, locs["it"])
                ep_string += f"""{f"{key}:":>{pad}} {v:.4f}\n"""
            else:
                v = float(value.item())
                writer.add_scalar("Episode/" + key, v, locs["it"])
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {v:.4f}\n"""

    mean_std = alg.policy.action_std.mean()
    fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

    for key, value in locs["loss_dict"].items():
        writer.add_scalar(f"Loss/{key}", value, locs["it"])
    writer.add_scalar("Loss/learning_rate", alg.learning_rate, locs["it"])

    writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

    writer.add_scalar("Perf/total_fps", fps, locs["it"])
    writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

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

    if len(locs["rewbuffer"]) > 0:
        writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        if logger_type != "wandb":  # wandb does not support non-integer x-axis logging
            writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), tot_time)
            writer.add_scalar(
                "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), tot_time
            )

    header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

    if len(locs["rewbuffer"]) > 0:
        log_string = (
            f"""{"#" * width}\n"""
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                locs["learn_time"]:.3f}s)\n"""
            f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
        log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
        log_string += f"""{"Batch episode count:":>{pad}} {int(locs.get("batch_episode_count", 0))}\n"""
        log_string += f"""{"Max batch reward:":>{pad}} {locs["max_batch_total_reward"]:.4f}\n"""
        log_string += f"""{"Mean batch reward:":>{pad}} {locs["mean_batch_total_reward"]:.4f}\n"""
        log_string += f"""{"Regret (batch max-mean):":>{pad}} {locs["regret"]:.4f}\n"""
        log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
    else:
        log_string = (
            f"""{"#" * width}\n"""
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                locs["learn_time"]:.3f}s)\n"""
            f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
        if "max_batch_total_reward" in locs and "regret" in locs:
            log_string += f"""{"Batch episode count:":>{pad}} {int(locs.get("batch_episode_count", 0))}\n"""
            log_string += f"""{"Max batch reward:":>{pad}} {locs["max_batch_total_reward"]:.4f}\n"""
            log_string += f"""{"Mean batch reward:":>{pad}} {locs["mean_batch_total_reward"]:.4f}\n"""
            log_string += f"""{"Regret (batch max-mean):":>{pad}} {locs["regret"]:.4f}\n"""

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
