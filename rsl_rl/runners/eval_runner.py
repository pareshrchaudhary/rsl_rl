# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import statistics

import torch


class EvalRunner:
    """Adds periodic deterministic evaluation to a training runner.

    Expects the host runner to have: ``env``, ``device``, ``alg``, ``writer``,
    ``disable_logs``, ``eval_mode()``, ``train_mode()``, ``get_inference_policy()``.
    """

    def __init__(self, runner) -> None:
        self.runner = runner
        self.eval_interval = runner.cfg.get("eval_interval", None)
        self.eval_num_episodes = runner.cfg.get("eval_num_episodes", 100)
        self._eval_configs: list[tuple[str, callable, callable]] = []

    def add_eval_config(self, name: str, enter_eval_fn: callable, exit_eval_fn: callable) -> None:
        """Add a named eval config with custom enter/exit callbacks."""
        self._eval_configs.append((name, enter_eval_fn, exit_eval_fn))

    def add_eval_env_cfg(self, name: str, eval_env_cfg) -> None:
        """Add a named eval config from an env cfg object.

        Automatically builds enter/exit callbacks that swap reset-mode event term
        params on the training env.
        """
        eval_term_params = {}
        for attr_name in vars(type(eval_env_cfg.events)):
            if attr_name.startswith("_"):
                continue
            term = getattr(eval_env_cfg.events, attr_name, None)
            if term is not None and hasattr(term, "mode") and term.mode == "reset":
                eval_term_params[attr_name] = term.params

        saved = {}

        def enter_eval(vec_env):
            em = vec_env.unwrapped.event_manager
            for term_name, p in eval_term_params.items():
                try:
                    cfg = em.get_term_cfg(term_name)
                    saved[term_name] = cfg.params
                    cfg.params = p
                except ValueError:
                    pass

        def exit_eval(vec_env):
            em = vec_env.unwrapped.event_manager
            for term_name, p in saved.items():
                cfg = em.get_term_cfg(term_name)
                cfg.params = p
            saved.clear()

        self._eval_configs.append((name, enter_eval, exit_eval))

    def run(self, iteration: int) -> None:
        """Call every training iteration. Runs registered evals when ``iteration`` hits ``eval_interval``."""
        if self.eval_interval is None or not self._eval_configs:
            return
        if iteration % self.eval_interval != 0 or self.runner.disable_logs:
            return
        for eval_name, enter_fn, exit_fn in self._eval_configs:
            self._run_eval(iteration, eval_name, enter_fn, exit_fn)

    def _run_eval(self, iteration: int, name: str, enter_fn: callable, exit_fn: callable) -> None:
        """Run evaluation rollouts on the training env (with eval config) and log metrics."""
        runner = self.runner
        env = runner.env
        device = runner.device
        alg = runner.alg
        writer = runner.writer

        # Save recurrent hidden states from training
        saved_hidden = None
        if alg.policy.is_recurrent:
            mem_a = alg.policy.memory_a.hidden_state
            mem_c = alg.policy.memory_c.hidden_state

            def _clone(hs):
                if hs is None:
                    return None
                if isinstance(hs, tuple):
                    return tuple(h.clone() for h in hs)
                return hs.clone()

            saved_hidden = (_clone(mem_a), _clone(mem_c))

        # Swap env to eval config
        enter_fn(env)

        # Switch policy to eval mode and get the inference policy
        policy = runner.get_inference_policy()
        obs = env.get_observations().to(device)
        alg.policy.reset()

        episode_rewards = torch.zeros(env.num_envs, dtype=torch.float, device=device)
        episode_lengths = torch.zeros(env.num_envs, dtype=torch.float, device=device)
        completed_rewards = []
        completed_lengths = []

        max_ep_len = env.max_episode_length
        if isinstance(max_ep_len, torch.Tensor):
            max_ep_len = max_ep_len.max().item()
        max_steps = int(self.eval_num_episodes / max(env.num_envs, 1) * max_ep_len) + int(max_ep_len)

        with torch.inference_mode():
            for _ in range(max_steps):
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)

                episode_rewards += rewards
                episode_lengths += 1

                done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    completed_rewards.extend(episode_rewards[done_ids].cpu().tolist())
                    completed_lengths.extend(episode_lengths[done_ids].cpu().tolist())
                    episode_rewards[done_ids] = 0
                    episode_lengths[done_ids] = 0

                alg.policy.reset(dones)

                if len(completed_rewards) >= self.eval_num_episodes:
                    break

        # Restore env to train config
        exit_fn(env)

        # Log eval metrics
        if completed_rewards and writer is not None:
            mean_reward = statistics.mean(completed_rewards[:self.eval_num_episodes])
            mean_length = statistics.mean(completed_lengths[:self.eval_num_episodes])

            writer.add_scalar(f"Eval/{name}/mean_reward", mean_reward, iteration)
            writer.add_scalar(f"Eval/{name}/mean_episode_length", mean_length, iteration)

            print(f"  [Eval:{name}] iter {iteration}: mean_reward={mean_reward:.2f}, "
                  f"mean_ep_len={mean_length:.1f}, episodes={len(completed_rewards)}")

        # Restore policy to train mode
        runner.train_mode()

        # Restore recurrent hidden states
        if saved_hidden is not None:
            alg.policy.memory_a.reset(hidden_state=saved_hidden[0])
            alg.policy.memory_c.reset(hidden_state=saved_hidden[1])
