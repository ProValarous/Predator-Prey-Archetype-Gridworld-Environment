"""
Vanilla Deep Q-Network baseline.

This implementation is intentionally minimal and uses the repo's existing
plugin-based environment contract:
- observations come from GridWorldEnv.reset()/step()
- actions are discrete integers from the action-space plugin
- rewards come from GridWorldEnv.step()

The first version is independent per agent: each agent gets its own replay
buffer, policy network, and target network.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("dqn")


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN(BaseAlgorithm):
    """Independent vanilla DQN with one learner per agent."""

    def __init__(self, env, config):
        super().__init__(env, config)

        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.min_epsilon = config.get("min_epsilon", 0.05)
        self.episodes = config.get("episodes", 500)
        self.batch_size = config.get("batch_size", 32)
        self.buffer_size = config.get("buffer_size", 10000)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.target_update_interval = config.get("target_update_interval", 20)
        self.min_replay_size = config.get("min_replay_size", self.batch_size)
        self.device = torch.device(config.get("device", "cpu"))
        self.verbose = bool(config.get("verbose", True))
        self.log_interval = int(config.get("log_interval", 1))
        self.debug_first_episode = bool(config.get("debug_first_episode", True))
        self.save_path = config.get("save_path", None)
        self.rng = np.random.default_rng(config.get("seed", None))

        self.action_dim = self._infer_action_dim()
        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.state_dim = self._flatten_observation(initial_obs[self.agent_ids[0]]).shape[0]

        self._build_learners()

        self._debug(
            "Initialized DQN | "
            f"agents={self.agent_ids} | "
            f"state_dim={self.state_dim} | "
            f"action_dim={self.action_dim} | "
            f"device={self.device} | "
            f"batch_size={self.batch_size} | "
            f"min_replay_size={self.min_replay_size}"
        )

    def _build_learners(self) -> None:
        self.policy_networks = {
            agent_id: QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            for agent_id in self.agent_ids
        }
        self.target_networks = {
            agent_id: QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            for agent_id in self.agent_ids
        }
        self.optimizers = {
            agent_id: optim.Adam(self.policy_networks[agent_id].parameters(), lr=self.learning_rate)
            for agent_id in self.agent_ids
        }
        self.replay_buffers = {
            agent_id: ReplayBuffer(self.buffer_size)
            for agent_id in self.agent_ids
        }

        for agent_id in self.agent_ids:
            self.target_networks[agent_id].load_state_dict(self.policy_networks[agent_id].state_dict())
            self.target_networks[agent_id].eval()

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[DQN] {message}")

    def _infer_action_dim(self) -> int:
        action_space = getattr(self.env, "action_space_plugin", None)
        if action_space is not None and hasattr(action_space, "n_actions"):
            return int(action_space.n_actions)
        return int(self.env.action_space.n)

    def _flatten_observation(self, obs) -> np.ndarray:
        values: List[float] = []

        def _collect(item):
            if item is None:
                return
            if isinstance(item, dict):
                for key in sorted(item.keys()):
                    _collect(item[key])
                return
            if isinstance(item, (list, tuple)):
                for value in item:
                    _collect(value)
                return
            if hasattr(item, "tolist"):
                converted = item.tolist()
                if isinstance(converted, (list, tuple)):
                    _collect(converted)
                else:
                    values.append(float(converted))
                return
            values.append(float(item))

        _collect(obs)
        return np.asarray(values, dtype=np.float32)

    def _validate_state_shape(self, agent_id: str, state: np.ndarray) -> None:
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"DQN expected state_dim={self.state_dim}, but agent {agent_id!r} "
                f"produced state_dim={state.shape[0]}. This usually means the "
                "observation plugin is returning variable-length observations."
            )

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        return torch.from_numpy(self._flatten_observation(obs)).float().unsqueeze(0).to(self.device)

    def _sync_target(self, agent_id: str) -> None:
        self.target_networks[agent_id].load_state_dict(self.policy_networks[agent_id].state_dict())

    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            if self.rng.random() < self.epsilon:
                actions[agent_id] = int(self.rng.integers(self.action_dim))
                continue

            with torch.no_grad():
                q_values = self.policy_networks[agent_id](self._obs_to_tensor(obs))
                actions[agent_id] = int(torch.argmax(q_values, dim=1).item())

        return actions

    def _optimize_agent(self, agent_id: str) -> float | None:
        buffer = self.replay_buffers[agent_id]
        if len(buffer) < max(self.min_replay_size, self.batch_size):
            return None

        batch = buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_networks[agent_id](states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_networks[agent_id](next_states).max(dim=1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = nn.SmoothL1Loss()(q_values, targets)

        optimizer = self.optimizers[agent_id]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())

    def train(self):
        self._debug(f"Starting training for {self.episodes} episodes")
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
                    state = self._flatten_observation(observations[agent_id])
                    next_state = self._flatten_observation(next_observations[agent_id])
                    self._validate_state_shape(agent_id, state)
                    self._validate_state_shape(agent_id, next_state)
                    transition = Transition(
                        state=state,
                        action=int(actions[agent_id]),
                        reward=float(rewards[agent_id]),
                        next_state=next_state,
                        done=done,
                    )
                    self.replay_buffers[agent_id].push(transition)
                    episode_rewards[agent_id] += float(rewards[agent_id])
                    loss = self._optimize_agent(agent_id)
                    if loss is not None:
                        episode_losses[agent_id].append(loss)

                    if episode == 0 and step_count == 0 and self.debug_first_episode:
                        self._debug(
                            f"First transition | agent={agent_id} | "
                            f"state_shape={state.shape} | action={actions[agent_id]} | "
                            f"reward={float(rewards[agent_id]):.3f} | "
                            f"next_state_shape={next_state.shape} | done={done}"
                        )

                observations = next_observations
                step_count += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (episode + 1) % self.target_update_interval == 0:
                for agent_id in self.agent_ids:
                    self._sync_target(agent_id)
                self._debug(f"Synced target networks at episode {episode + 1}")

            if (episode + 1) % self.log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={value:.2f}" for aid, value in episode_rewards.items()
                )
                loss_str = ", ".join(
                    f"{aid}={np.mean(losses):.5f}" if losses else f"{aid}=warmup"
                    for aid, losses in episode_losses.items()
                )
                buffer_str = ", ".join(
                    f"{aid}={len(self.replay_buffers[aid])}"
                    for aid in self.agent_ids
                )
                self._debug(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"eps={self.epsilon:.3f} | steps={step_count} | "
                    f"rewards: {reward_str} | avg_loss: {loss_str} | "
                    f"replay: {buffer_str}"
                )

        if self.save_path:
            self.save(self.save_path)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": self.config,
            "agent_ids": self.agent_ids,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "policy_state_dicts": {
                agent_id: network.state_dict() for agent_id, network in self.policy_networks.items()
            },
            "target_state_dicts": {
                agent_id: network.state_dict() for agent_id, network in self.target_networks.items()
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
            if agent_id in payload["policy_state_dicts"]:
                instance.policy_networks[agent_id].load_state_dict(payload["policy_state_dicts"][agent_id])
            if agent_id in payload["target_state_dicts"]:
                instance.target_networks[agent_id].load_state_dict(payload["target_state_dicts"][agent_id])

        LOGGER.info("Loaded DQN checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("dqn", DQN)
