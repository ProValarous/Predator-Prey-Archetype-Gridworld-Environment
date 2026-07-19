# src/baselines/DQN/dqn.py
"""
Independent Deep Q-Network baseline.

Mirrors IQL/CQL's independent-learning pattern -- one Q-network, target
network, and replay buffer per agent -- with a neural net standing in
for the Q-table. Observations are converted to fixed-length numeric
vectors via the environment's observation_encoder: attach an
observation builder's encode() method to env.observation_encoder before
constructing DQN (run_from_config.build_environment does this).
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register
from baselines.DQN.q_network import QNetwork
from baselines.DQN.replay_buffer import ReplayBuffer

LOGGER = logging.getLogger("dqn")


class DQN(BaseAlgorithm):
    """Independent DQN: one Q-network, target network, and replay buffer per agent."""

    def __init__(self, env, config: dict):
        super().__init__(env, config)

        self.gamma = float(config.get("gamma", 0.99))
        self.epsilon = float(config.get("epsilon", 1.0))
        self.epsilon_decay = float(config.get("epsilon_decay", 0.995))
        self.min_epsilon = float(config.get("min_epsilon", 0.05))
        self.episodes = int(config.get("episodes", 1000))
        self.batch_size = int(config.get("batch_size", 32))
        self.buffer_size = int(config.get("buffer_size", 10000))
        self.min_replay_size = int(config.get("min_replay_size", self.batch_size))
        self.target_update_interval = int(config.get("target_update_interval", 100))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.hidden_layers = [int(v) for v in config.get("hidden_layers", [128, 128])]
        self.grad_clip = float(config.get("grad_clip", 5.0))
        self.device = torch.device(config.get("device", "cpu"))
        self.verbose = bool(config.get("verbose", True))
        self.log_interval = int(config.get("log_interval", 10))
        self.debug_first_episode = bool(config.get("debug_first_episode", True))
        self.save_path = config.get("save_path", None)
        self.curves_path: Optional[str] = config.get("curves_path", None)

        seed = config.get("seed", None)
        self.rng = default_rng(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

        # -- observation encoder contract (attached by build_environment) --
        self.observation_encoder = getattr(self.env, "observation_encoder", None)
        if self.observation_encoder is None:
            raise ValueError(
                "Environment is missing observation_encoder. Attach an observation "
                "builder's encode() method to env.observation_encoder before "
                "constructing DQN (see run_from_config.build_environment)."
            )

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.state_dim = self._encode_observation(initial_obs[self.agent_ids[0]]).shape[
            0
        ]
        self.action_dim = self._resolve_action_dim(config)

        if self.buffer_size < self.batch_size:
            warnings.warn(
                f"buffer_size ({self.buffer_size}) is smaller than batch_size "
                f"({self.batch_size}) -- training will never fill a full batch.",
                stacklevel=2,
            )

        self._build_learners(seed)
        self._train_steps = 0

        self._debug(
            "Initialized DQN | "
            f"agents={self.agent_ids} | state_dim={self.state_dim} | "
            f"action_dim={self.action_dim} | device={self.device} | "
            f"batch_size={self.batch_size} | min_replay_size={self.min_replay_size}"
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_action_dim(self, config: dict) -> int:
        """Infer action_dim from the env's action plugin, or validate if configured."""
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
                f"DQN config error: 'action_dim'={configured} does not match the "
                f"environment's action space size ({plugin_n_actions}). Fix "
                "'action_dim' in your experiment YAML, or remove it entirely so "
                "it gets inferred automatically."
            )
        return configured

    def _build_learners(self, seed) -> None:
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}

        for i, agent_id in enumerate(self.agent_ids):
            buffer_seed = None if seed is None else int(seed) + i

            self.q_networks[agent_id] = QNetwork(
                self.state_dim, self.hidden_layers, self.action_dim
            ).to(self.device)
            self.target_networks[agent_id] = QNetwork(
                self.state_dim, self.hidden_layers, self.action_dim
            ).to(self.device)
            self.target_networks[agent_id].load_state_dict(
                self.q_networks[agent_id].state_dict()
            )
            self.target_networks[agent_id].eval()

            self.optimizers[agent_id] = optim.Adam(
                self.q_networks[agent_id].parameters(), lr=self.learning_rate
            )
            self.replay_buffers[agent_id] = ReplayBuffer(
                self.buffer_size, self.state_dim, seed=buffer_seed
            )

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[DQN] {message}")

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_observation(self, obs) -> np.ndarray:
        encoded = self.observation_encoder(obs, self.env)
        return np.asarray(encoded, dtype=np.float32).reshape(-1)

    def _validate_state_shape(self, agent_id: str, state: np.ndarray) -> None:
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"DQN expected state_dim={self.state_dim}, but agent {agent_id!r} "
                f"produced state_dim={state.shape[0]}. This usually means the "
                "observation plugin is returning variable-length observations."
            )

    # ------------------------------------------------------------------
    # Action selection (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            if self.rng.random() < self.epsilon:
                actions[agent_id] = int(self.rng.integers(self.action_dim))
                continue

            state = self._encode_observation(obs)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_networks[agent_id](state_t)
            actions[agent_id] = int(torch.argmax(q_values, dim=1).item())
        return actions

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _optimize_agent(self, agent_id: str) -> float | None:
        buffer = self.replay_buffers[agent_id]
        if len(buffer) < max(self.min_replay_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        q_values = self.q_networks[agent_id](states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_networks[agent_id](next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_q_values

        loss = nn.SmoothL1Loss()(q_values, targets)

        optimizer = self.optimizers[agent_id]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_networks[agent_id].parameters(), self.grad_clip
        )
        optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self._sync_targets()

        return float(loss.item())

    def _sync_targets(self) -> None:
        for agent_id in self.agent_ids:
            self.target_networks[agent_id].load_state_dict(
                self.q_networks[agent_id].state_dict()
            )

    # ------------------------------------------------------------------
    # Training loop (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def train(self):
        csv_file = None
        csv_writer = None
        if self.curves_path:
            os.makedirs(os.path.dirname(self.curves_path) or ".", exist_ok=True)
            csv_file = open(self.curves_path, "w", newline="")
            reward_cols = [f"{aid}_reward" for aid in self.agent_ids]
            loss_cols = [f"{aid}_loss" for aid in self.agent_ids]
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=["episode", "epsilon"] + reward_cols + loss_cols
            )
            csv_writer.writeheader()

        self._debug(f"Starting training for {self.episodes} episodes")
        try:
            self._train_loop(csv_writer)
        finally:
            if csv_file:
                csv_file.close()

        if self.save_path:
            self.save(self.save_path)

    def _train_loop(self, csv_writer) -> None:
        for episode in range(self.episodes):
            observations, _ = self.env.reset()
            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            episode_losses = {agent_id: [] for agent_id in self.agent_ids}
            done = False
            step_count = 0

            while not done:
                actions = self.select_actions(observations)
                step_out = self.env.step(actions)
                next_observations = step_out["obs"]
                rewards = step_out["reward"]
                done = bool(step_out["terminated"] or step_out["truncated"])

                for agent_id in self.agent_ids:
                    state = self._encode_observation(observations[agent_id])
                    next_state = self._encode_observation(next_observations[agent_id])
                    self._validate_state_shape(agent_id, state)
                    self._validate_state_shape(agent_id, next_state)

                    self.replay_buffers[agent_id].push(
                        state,
                        int(actions[agent_id]),
                        float(rewards[agent_id]),
                        next_state,
                        done,
                    )
                    episode_rewards[agent_id] += float(rewards[agent_id])
                    loss = self._optimize_agent(agent_id)
                    if loss is not None:
                        episode_losses[agent_id].append(loss)

                    if episode == 0 and step_count == 0 and self.debug_first_episode:
                        self._debug(
                            f"First transition | agent={agent_id} | "
                            f"state_shape={state.shape} | "
                            f"action={actions[agent_id]} | "
                            f"reward={float(rewards[agent_id]):.3f} | "
                            f"done={done}"
                        )

                observations = next_observations
                step_count += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (episode + 1) % self.log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={value:.2f}" for aid, value in episode_rewards.items()
                )
                loss_str = ", ".join(
                    f"{aid}={np.mean(losses):.5f}" if losses else f"{aid}=warmup"
                    for aid, losses in episode_losses.items()
                )
                self._debug(
                    f"Episode {episode + 1}/{self.episodes} | eps={self.epsilon:.3f} | "
                    f"steps={step_count} | rewards: {reward_str} | avg_loss: {loss_str}"
                )

            if csv_writer:
                row: dict = {"episode": episode + 1, "epsilon": round(self.epsilon, 4)}
                for aid in self.agent_ids:
                    row[f"{aid}_reward"] = round(episode_rewards[aid], 4)
                    losses = episode_losses[aid]
                    row[f"{aid}_loss"] = (
                        round(sum(losses) / len(losses), 6) if losses else ""
                    )
                csv_writer.writerow(row)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": self.config,
            "agent_ids": self.agent_ids,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "q_state_dicts": {
                aid: net.state_dict() for aid, net in self.q_networks.items()
            },
            "target_state_dicts": {
                aid: net.state_dict() for aid, net in self.target_networks.items()
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved DQN checkpoint -> %s", path)
        self._debug(f"Saved DQN checkpoint -> {path}")

    @classmethod
    def load(cls, env, config: dict, path: str) -> "DQN":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        for agent_id in instance.agent_ids:
            if agent_id in payload["q_state_dicts"]:
                instance.q_networks[agent_id].load_state_dict(
                    payload["q_state_dicts"][agent_id]
                )
            if agent_id in payload["target_state_dicts"]:
                instance.target_networks[agent_id].load_state_dict(
                    payload["target_state_dicts"][agent_id]
                )

        LOGGER.info("Loaded DQN checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("dqn", DQN)


# ------------------------------------------------------------------
# Standalone CLI -- python -m baselines.DQN.dqn [--mode train|eval]
# (mirrors IQL/CQL's own CLI for quick manual testing without YAML configs)
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

    p = argparse.ArgumentParser("Independent DQN — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--observation", type=str, default="local_only")
    p.add_argument("--action-space", type=str, default="discrete_5")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_dqn.pkl")
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    agents = []
    for i in range(1, args.preys + 1):
        agents.append(Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey"))
    for i in range(1, args.predators + 1):
        agents.append(
            Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator")
        )

    render = "human" if (args.mode == "eval" and args.render) else None
    env = GridWorldEnv(
        agents=agents,
        render_mode=render,
        size=args.size,
        perc_num_obstacle=10,
        seed=args.seed,
    )

    observation_builder = get_observation_builder(args.observation)
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode
    env.action_space_plugin = get_action_space(args.action_space)

    config = {
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "epsilon": 1.0 if args.mode == "train" else 0.0,
        "seed": args.seed,
    }

    if args.mode == "train":
        algo = DQN(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = DQN.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
