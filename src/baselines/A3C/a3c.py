# src/baselines/A3C/a3c.py
"""
Independent Asynchronous Advantage Actor-Critic (A3C) baseline
(Mnih et al., 2016, "Asynchronous Methods for Deep Reinforcement Learning").

Unlike AC (single-process, per-step online) and A2C (single-process,
batched rollout), A3C runs multiple worker PROCESSES in parallel, each
stepping its own independent copy of the environment. Each worker
periodically syncs a local copy of the network from a shared global
network, computes gradients locally against an n-step rollout, then
applies those gradients directly to the shared global network and steps
a shared optimizer -- lock-free ("Hogwild"-style asynchronous SGD), exactly
as the original paper describes. Workers run on CPU: A3C's entire premise
is parallelism from many CPU cores instead of a GPU (safely sharing one
CUDA context across OS processes needs CUDA IPC and buys little for what
A3C is designed around -- GPU work wants batching, which is what A2C
already gives you).

Reuses AC's ActorCriticNetwork (shared trunk, policy head + value head) --
generic, no AC-specific behavior baked in, and it's the same architecture
the original A3C paper uses.

Because BaseAlgorithm.__init__(self, env, config) only receives one
already-built env, and A3C fundamentally needs one independent env PER
WORKER, A3C requires one additional config key beyond every other
baseline: `env_fn`, a zero-argument callable that builds a fresh
environment. The passed-in `env` is still used the normal way (to infer
state_dim/action_dim/agent_ids, and for BaseAlgorithm.evaluate()).

`env_fn` must be picklable -- on Windows (and anywhere using the 'spawn'
start method), a lambda closing over a config dict will NOT pickle. Use a
module-level function or a callable class instance instead (see
scripts/run_a3c.py's EnvFactory).
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributions import Categorical
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register
from baselines.AC.network import ActorCriticNetwork
from baselines.AC.return_normalizer import RunningReturnNormalizer
from baselines.A3C.shared_adam import SharedAdam

LOGGER = logging.getLogger("a3c")


def _encode(obs, env) -> np.ndarray:
    encoded = env.observation_encoder(obs, env)
    return np.asarray(encoded, dtype=np.float32).reshape(-1)


def _worker_loop(
    worker_id: int,
    agent_ids: list,
    state_dim: int,
    action_dim: int,
    hidden_layers: list,
    global_networks: dict,
    optimizers: dict,
    env_fn,
    gamma: float,
    n_steps: int,
    entropy_coef: float,
    value_coef: float,
    grad_clip: float,
    episode_counter,
    max_episodes: int,
    counter_lock,
    csv_lock,
    csv_path: Optional[str],
    log_interval: int,
    verbose: bool,
    seed: Optional[int],
    normalize_returns: bool = False,
    return_norm_decay: float = 0.999,
) -> None:
    """One A3C worker: runs entirely in its own process. Builds its own
    env via env_fn(), builds local (non-shared) networks, and repeatedly
    syncs from the shared global networks, collects an n-step rollout,
    and pushes gradients into the shared networks -- until the shared
    episode_counter reaches max_episodes."""
    if seed is not None:
        torch.manual_seed(seed)
    default_rng(seed)

    env = env_fn()
    local_networks = {
        aid: ActorCriticNetwork(state_dim, hidden_layers, action_dim)
        for aid in agent_ids
    }
    # One normalizer per agent, LOCAL to this worker process (not shared or
    # synchronized across workers -- each estimates the same underlying
    # return distribution independently, which converges close enough in
    # practice; a fully shared, lock-protected estimate would need per-step
    # lock contention this doesn't attempt). See return_normalizer.py.
    return_normalizers = {
        aid: RunningReturnNormalizer(decay=return_norm_decay) for aid in agent_ids
    }

    csv_file = open(csv_path, "a", newline="") if csv_path else None
    csv_writer = None
    if csv_file:
        reward_cols = [f"{aid}_reward" for aid in agent_ids]
        loss_cols = [f"{aid}_loss" for aid in agent_ids]
        entropy_cols = [f"{aid}_entropy" for aid in agent_ids]
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["episode", "worker"] + reward_cols + loss_cols + entropy_cols,
        )

    try:
        while True:
            with counter_lock:
                if episode_counter.value >= max_episodes:
                    break
                episode_counter.value += 1
                episode_num = episode_counter.value

            observations, _ = env.reset()
            episode_rewards = {aid: 0.0 for aid in agent_ids}
            episode_losses = {aid: [] for aid in agent_ids}
            episode_entropies = {aid: [] for aid in agent_ids}
            done = False
            terminated = False
            step_count = 0

            while not done:
                for aid in agent_ids:
                    local_networks[aid].load_state_dict(
                        global_networks[aid].state_dict()
                    )

                rollout = {
                    aid: {"log_probs": [], "values": [], "rewards": [], "entropies": []}
                    for aid in agent_ids
                }

                for _ in range(n_steps):
                    actions = {}
                    for aid in agent_ids:
                        state = _encode(observations[aid], env)
                        state_t = torch.from_numpy(state).float().unsqueeze(0)
                        logits, value = local_networks[aid](state_t)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        actions[aid] = int(action.item())
                        rollout[aid]["log_probs"].append(dist.log_prob(action))
                        rollout[aid]["values"].append(value)
                        rollout[aid]["entropies"].append(dist.entropy())

                    step_out = env.step(actions)
                    next_observations = step_out["obs"]
                    rewards = step_out["reward"]
                    terminated = bool(step_out["terminated"])
                    truncated = bool(step_out["truncated"])
                    done = terminated or truncated

                    for aid in agent_ids:
                        r = float(rewards[aid])
                        rollout[aid]["rewards"].append(r)
                        episode_rewards[aid] += r

                    observations = next_observations
                    step_count += 1

                    if done:
                        break

                for aid in agent_ids:
                    if not rollout[aid]["rewards"]:
                        continue

                    normalizer = return_normalizers[aid]
                    if terminated:
                        bootstrap_value = 0.0
                    else:
                        with torch.no_grad():
                            next_state = _encode(observations[aid], env)
                            next_state_t = (
                                torch.from_numpy(next_state).float().unsqueeze(0)
                            )
                            _, next_value = local_networks[aid](next_state_t)
                            bootstrap_value = float(next_value.item())
                            if normalize_returns:
                                # values_t (collected below) are on a
                                # NORMALIZED scale once this is enabled --
                                # un-normalize against the PRIOR estimate to
                                # bootstrap in raw units, same two-phase
                                # pattern as AC's _update().
                                bootstrap_value = normalizer.denormalize(
                                    bootstrap_value
                                )

                    returns = []
                    R = bootstrap_value
                    for r in reversed(rollout[aid]["rewards"]):
                        R = r + gamma * R
                        returns.insert(0, R)

                    if normalize_returns:
                        for r in returns:
                            normalizer.update(r)
                        returns_t = torch.tensor(
                            [normalizer.normalize(r) for r in returns],
                            dtype=torch.float32,
                        )
                    else:
                        returns_t = torch.tensor(returns, dtype=torch.float32)

                    values_t = torch.cat(rollout[aid]["values"])
                    log_probs_t = torch.cat(rollout[aid]["log_probs"])
                    entropies_t = torch.cat(rollout[aid]["entropies"])

                    advantages = returns_t - values_t.detach()

                    actor_loss = (
                        -(log_probs_t * advantages).mean()
                        - entropy_coef * entropies_t.mean()
                    )
                    critic_loss = nn.SmoothL1Loss()(values_t, returns_t)
                    loss = actor_loss + value_coef * critic_loss

                    local_networks[aid].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        local_networks[aid].parameters(), grad_clip
                    )

                    # Hogwild: push local gradients straight onto the shared
                    # global parameters (no lock) and step the shared
                    # optimizer -- other workers may be doing the same
                    # concurrently; that staleness is the point of A3C.
                    for local_p, global_p in zip(
                        local_networks[aid].parameters(),
                        global_networks[aid].parameters(),
                    ):
                        global_p._grad = local_p.grad
                    optimizers[aid].step()

                    episode_losses[aid].append(float(loss.item()))
                    episode_entropies[aid].append(float(entropies_t.mean().item()))

            if verbose and episode_num % log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={v:.2f}" for aid, v in episode_rewards.items()
                )
                entropy_str = ", ".join(
                    f"{aid}={np.mean(ent):.5f}" for aid, ent in episode_entropies.items()
                )
                print(
                    f"[A3C][worker {worker_id}] Episode {episode_num}/{max_episodes} | "
                    f"steps={step_count} | rewards: {reward_str} | "
                    f"avg_entropy: {entropy_str}"
                )

            if csv_writer:
                row: dict = {"episode": episode_num, "worker": worker_id}
                for aid in agent_ids:
                    row[f"{aid}_reward"] = round(episode_rewards[aid], 4)
                    losses = episode_losses[aid]
                    row[f"{aid}_loss"] = (
                        round(sum(losses) / len(losses), 6) if losses else ""
                    )
                    entropies = episode_entropies[aid]
                    row[f"{aid}_entropy"] = (
                        round(sum(entropies) / len(entropies), 6) if entropies else ""
                    )
                with csv_lock:
                    csv_writer.writerow(row)
                    csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()
        env.close()


class A3C(BaseAlgorithm):
    """Independent A3C: one shared network + shared optimizer per agent,
    updated asynchronously by multiple worker processes."""

    def __init__(self, env, config: dict):
        super().__init__(env, config)

        self.gamma = float(config.get("gamma", 0.99))
        self.episodes = int(config.get("episodes", 1000))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.hidden_layers = [int(v) for v in config.get("hidden_layers", [128, 128])]
        self.n_steps = int(config.get("n_steps", 5))
        self.entropy_coef = float(config.get("entropy_coef", 0.05))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.grad_clip = float(config.get("grad_clip", 5.0))
        # L2 penalty on the policy head only (see A2C's actor_weight_decay,
        # docs/algorithms/a2c.md). A3C reuses AC's shared-trunk network, so
        # decay is scoped to policy_head's own parameters via a dedicated
        # optimizer param group -- decaying the shared trunk would also
        # regularize the critic's features across every worker, not just the
        # actor's. 0.0 preserves the original single-group behavior.
        self.actor_weight_decay = float(config.get("actor_weight_decay", 0.0))
        # See baselines/AC/return_normalizer.py and AC's own normalize_returns
        # (docs/algorithms/actor-critic.md) -- same fix, applied per-worker
        # (each worker keeps its own local running estimate; see _worker_loop).
        self.normalize_returns = bool(config.get("normalize_returns", False))
        self.return_norm_decay = float(config.get("return_norm_decay", 0.999))
        self.num_workers = int(config.get("num_workers", 4))
        self.verbose = bool(config.get("verbose", True))
        self.log_interval = int(config.get("log_interval", 10))
        self.save_path = config.get("save_path", None)
        self.curves_path: Optional[str] = config.get("curves_path", None)

        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")

        self.env_fn = config.get("env_fn", None)
        if self.env_fn is None or not callable(self.env_fn):
            raise ValueError(
                "A3C requires config['env_fn']: a zero-argument, picklable "
                "callable that builds a fresh environment for each worker "
                "process (a lambda will NOT pickle under the 'spawn' start "
                "method -- use a module-level function or a callable class "
                "instance; see scripts/run_a3c.py's EnvFactory)."
            )

        self.seed = config.get("seed", None)
        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        self.observation_encoder = getattr(self.env, "observation_encoder", None)
        if self.observation_encoder is None:
            raise ValueError(
                "Environment is missing observation_encoder. Attach an observation "
                "builder's encode() method to env.observation_encoder before "
                "constructing A3C (see run_from_config.build_environment)."
            )

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.state_dim = self._encode_observation(initial_obs[self.agent_ids[0]]).shape[
            0
        ]
        self.action_dim = self._resolve_action_dim(config)

        self._build_learners()

        self._debug(
            "Initialized A3C | "
            f"agents={self.agent_ids} | state_dim={self.state_dim} | "
            f"action_dim={self.action_dim} | num_workers={self.num_workers} | "
            f"n_steps={self.n_steps} | entropy_coef={self.entropy_coef} | "
            f"actor_weight_decay={self.actor_weight_decay} | "
            f"normalize_returns={self.normalize_returns} | device=cpu"
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_action_dim(self, config: dict) -> int:
        plugin = getattr(self.env, "action_space_plugin", None)
        plugin_n_actions = (
            int(plugin.n_actions)
            if plugin is not None
            else int(self.env.action_space.n)
        )

        configured = config.get("action_dim", None)
        if configured is None:
            return plugin_n_actions

        configured = int(configured)
        if configured != plugin_n_actions:
            raise ValueError(
                f"A3C config error: 'action_dim'={configured} does not match the "
                f"environment's action space size ({plugin_n_actions}). Fix "
                "'action_dim' in your experiment YAML, or remove it entirely so "
                "it gets inferred automatically."
            )
        return configured

    def _build_learners(self) -> None:
        self.networks = {}
        self.optimizers = {}

        for agent_id in self.agent_ids:
            network = ActorCriticNetwork(
                self.state_dim, self.hidden_layers, self.action_dim
            )
            network.share_memory()
            self.networks[agent_id] = network
            self.optimizers[agent_id] = SharedAdam(
                [
                    {
                        "params": network.policy_head.parameters(),
                        "weight_decay": self.actor_weight_decay,
                    },
                    {
                        "params": list(network.trunk.parameters())
                        + list(network.value_head.parameters()),
                    },
                ],
                lr=self.learning_rate,
            )

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[A3C] {message}")

    def _encode_observation(self, obs) -> np.ndarray:
        encoded = self.observation_encoder(obs, self.env)
        return np.asarray(encoded, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # Action selection (BaseAlgorithm interface) -- main process only,
    # against the shared global networks (used by evaluate()).
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            state = self._encode_observation(obs)
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.networks[agent_id](state_t)
            dist = Categorical(logits=logits)
            actions[agent_id] = int(dist.sample().item())
        return actions

    # ------------------------------------------------------------------
    # Training (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def train(self):
        if self.curves_path:
            os.makedirs(os.path.dirname(self.curves_path) or ".", exist_ok=True)
            reward_cols = [f"{aid}_reward" for aid in self.agent_ids]
            loss_cols = [f"{aid}_loss" for aid in self.agent_ids]
            entropy_cols = [f"{aid}_entropy" for aid in self.agent_ids]
            with open(self.curves_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["episode", "worker"]
                    + reward_cols
                    + loss_cols
                    + entropy_cols,
                )
                writer.writeheader()

        self._debug(
            f"Starting training for {self.episodes} episodes across "
            f"{self.num_workers} workers"
        )

        episode_counter = mp.Value("i", 0)
        counter_lock = mp.Lock()
        csv_lock = mp.Lock() if self.curves_path else None

        processes = []
        for worker_id in range(self.num_workers):
            worker_seed = None if self.seed is None else int(self.seed) + worker_id
            p = mp.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    self.agent_ids,
                    self.state_dim,
                    self.action_dim,
                    self.hidden_layers,
                    self.networks,
                    self.optimizers,
                    self.env_fn,
                    self.gamma,
                    self.n_steps,
                    self.entropy_coef,
                    self.value_coef,
                    self.grad_clip,
                    episode_counter,
                    self.episodes,
                    counter_lock,
                    csv_lock,
                    self.curves_path,
                    self.log_interval,
                    self.verbose,
                    worker_seed,
                    self.normalize_returns,
                    self.return_norm_decay,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if self.save_path:
            self.save(self.save_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # env_fn is a live callable tied to this session, not meaningful
        # checkpoint data -- and may not even be picklable (e.g. a lambda),
        # so it's dropped before persisting rather than saved wholesale.
        config_to_save = {k: v for k, v in self.config.items() if k != "env_fn"}
        payload = {
            "config": config_to_save,
            "agent_ids": self.agent_ids,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_dicts": {
                aid: net.state_dict() for aid, net in self.networks.items()
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved A3C checkpoint -> %s", path)
        self._debug(f"Saved A3C checkpoint -> {path}")

    @classmethod
    def load(cls, env, config: dict, path: str) -> "A3C":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        for agent_id in instance.agent_ids:
            if agent_id in payload["state_dicts"]:
                instance.networks[agent_id].load_state_dict(
                    payload["state_dicts"][agent_id]
                )

        LOGGER.info("Loaded A3C checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("a3c", A3C)


# ------------------------------------------------------------------
# Standalone CLI -- python -m baselines.A3C.a3c [--mode train|eval]
# (mirrors the other baselines' own CLI for quick manual testing)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from multi_agent_package.core.agent import Agent
    from multi_agent_package.core.gridworld import GridWorldEnv
    from multi_agent_package.registry import get_action_space, get_observation_builder

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    class _SimpleEnvFactory:
        """Picklable env builder for the standalone CLI (spawn-safe)."""

        def __init__(self, size, predators, preys, observation, action_space, seed):
            self.size = size
            self.predators = predators
            self.preys = preys
            self.observation = observation
            self.action_space = action_space
            self.seed = seed

        def __call__(self):
            agents = []
            for i in range(1, self.preys + 1):
                agents.append(
                    Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey")
                )
            for i in range(1, self.predators + 1):
                agents.append(
                    Agent(
                        agent_name=f"predator_{i}",
                        agent_team=i,
                        agent_type="predator",
                    )
                )
            env = GridWorldEnv(
                agents=agents,
                render_mode=None,
                size=self.size,
                perc_num_obstacle=10,
                seed=self.seed,
            )
            observation_builder = get_observation_builder(self.observation)
            env.observation_builder = observation_builder.build
            env.observation_encoder = observation_builder.encode
            env.action_space_plugin = get_action_space(self.action_space)
            return env

    p = argparse.ArgumentParser("Independent A3C — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--observation", type=str, default="local_only")
    p.add_argument("--action-space", type=str, default="discrete_5")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_a3c.pkl")
    p.add_argument("--load-path", type=str, default=None)
    args = p.parse_args()

    env_fn = _SimpleEnvFactory(
        args.size,
        args.predators,
        args.preys,
        args.observation,
        args.action_space,
        args.seed,
    )
    env = env_fn()

    config = {
        "hidden_layers": args.hidden_layers,
        "num_workers": args.num_workers,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
        "episodes": args.episodes,
        "seed": args.seed,
        "env_fn": env_fn,
    }

    if args.mode == "train":
        algo = A3C(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = A3C.load(env, config, args.load_path)
        print(algo.evaluate(episodes=args.episodes))

    env.close()
