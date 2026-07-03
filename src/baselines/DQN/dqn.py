"""
Independent Deep Q-Networks for the predator-prey gridworld.

The implementation keeps the repo's separation of concerns intact:
- the environment remains a black box
- observations are converted into structured numeric vectors for replay storage
- each agent gets its own DQN, target network, and replay buffer
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("dqn")


# transition experience tuple (s, a, r, s', done)
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# used deque for efficient FIFO behavior; sample() picks random indices from the buffer without replacement, returning a list of transitions; as_list() and load() allow saving and restoring the buffer contents for checkpointing
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[Transition]:
        batch_size = min(int(batch_size), len(self._buffer))
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        items = list(self._buffer)
        return [items[int(idx)] for idx in indices]

    def as_list(self) -> List[Transition]:
        return list(self._buffer)

    def load(self, transitions: Iterable[Transition]) -> None:
        self._buffer.clear()
        for transition in transitions:
            self._buffer.append(transition)


# vanilla MLP for Q-value approximation; input_dim = structured observation vector size (auto-computed from max_other_agents and max_visible_obstacles), hidden_layers = list of hidden layer sizes, output_dim = number of actions; ReLU activations between layers, no activation on the output (raw Q-values); can be made deeper/wider in the YAML as needed
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Deep Q-Network implementation
class DQN(BaseAlgorithm):
    """Independent DQN: one network and replay buffer per agent."""

    # set up every hyperparameter from YAML config; self.env.reset() runs the environment once to discover agent IDs; each agent gets its own Q-network, target network, optimizer, and replay buffer;
    # obs_dim is auto-computed as 2 + max_other_agents*4 + max_visible_obstacles*3 unless input_dim is explicitly overridden in config;
    # curves_path: if set, writes one CSV row per episode with rewards and losses (None = no file written)
    def __init__(self, env, config: dict):
        super().__init__(env, config)

        self.gamma = float(config.get("gamma", 0.99))
        self.epsilon = float(config.get("epsilon", 1.0))
        self.epsilon_decay = float(config.get("epsilon_decay", 0.995))
        self.min_epsilon = float(config.get("min_epsilon", 0.05))
        self.episodes = int(config.get("episodes", 1000))
        self.batch_size = int(config.get("batch_size", 64))
        self.buffer_size = int(config.get("buffer_size", 10000))
        self.target_update_interval = int(config.get("target_update_interval", 100))
        self.learning_rate = float(config.get("learning_rate", config.get("alpha", 1e-3)))
        self.action_dim = int(config.get("action_dim", getattr(self.env.action_space, "n", 5)))
        self.hidden_layers = [int(v) for v in config.get("hidden_layers", [128, 128])]
        self.device = torch.device(config.get("device", "cpu"))
        self.curves_path: Optional[str] = config.get("curves_path", None)

        seed = config.get("seed", None)
        self.rng = default_rng(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())

        # obs_dim layout: 2 (self x,y) + max_other_agents * 5 (rel_x, rel_y, dist, is_prey, speed) + max_visible_obstacles * 3 (rel_x, rel_y, dist)
        # this matches the local_radius observation structure exactly; set input_dim in YAML to override
        self.max_other_agents = int(config.get("max_other_agents", 5))
        self.max_visible_obstacles = int(config.get("max_visible_obstacles", 5))
        computed_dim = 2 + self.max_other_agents * 5 + self.max_visible_obstacles * 3
        self.obs_dim = int(config.get("input_dim", computed_dim))
        if self.obs_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        self.q_networks: Dict[str, QNetwork] = {}
        self.target_networks: Dict[str, QNetwork] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.replay_buffers: Dict[str, ReplayBuffer] = {
            agent_id: ReplayBuffer(self.buffer_size)
            for agent_id in self.agent_ids
        }

        for agent_id in self.agent_ids:
            self.q_networks[agent_id] = self._build_network()
            self.target_networks[agent_id] = self._build_network()
            self.target_networks[agent_id].load_state_dict(
                self.q_networks[agent_id].state_dict()
            )
            self.target_networks[agent_id].eval()
            self.optimizers[agent_id] = optim.Adam(
                self.q_networks[agent_id].parameters(), lr=self.learning_rate
            )

        self._loss_fn = nn.MSELoss()
        self._train_steps = 0

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    # structured flat encoder for local_radius observations
    #
    # vector layout (total length = obs_dim):
    #   [0]   self_x
    #   [1]   self_y
    #   [2]   agent_slot_0_rel_x   } first visible agent (sorted alphabetically by name)
    #   [3]   agent_slot_0_rel_y   }
    #   [4]   agent_slot_0_dist    }
    #   [5]   agent_slot_0_is_prey } 1.0 if prey, 0.0 if predator
    #   [6]   agent_slot_0_speed   } cells moved per logical step (from agent_speed in agents.yaml)
    #   ...repeated for max_other_agents slots (zero-padded if agent not visible)...
    #   [2 + max_other_agents*5 + 0]  obstacle_slot_0_rel_x  } closest visible obstacle
    #   [2 + max_other_agents*5 + 1]  obstacle_slot_0_rel_y  }
    #   [2 + max_other_agents*5 + 2]  obstacle_slot_0_dist   }
    #   ...repeated for max_visible_obstacles slots (zero-padded if obstacle not visible)...
    #
    # absent slots are filled with zeros; the network therefore sees a consistent-size input
    # regardless of how many agents/obstacles are currently in radius

    def _encode_observation(self, obs: dict) -> np.ndarray:
        vector = np.zeros(self.obs_dim, dtype=np.float32)

        # --- own position: indices 0-1 ---
        local = obs.get("local", None)
        if local is not None:
            # relative/absolute observations wrap position in a sub-dict; local_radius gives a raw array
            pos = local.get("pos", np.zeros(2)) if isinstance(local, dict) else local
            if hasattr(pos, "__len__") and len(pos) >= 2:
                vector[0] = float(pos[0])
                vector[1] = float(pos[1])
        idx = 2

        # --- other agents: max_other_agents slots of 5 values each ---
        # local_radius obs uses "visible_agents"; sorted by name so the ordering is deterministic
        visible = obs.get("visible_agents", {})
        slots_used = 0
        for name in sorted(visible.keys()):
            if slots_used >= self.max_other_agents:
                break
            entry = visible[name]
            rel = entry.get("rel_pos", (0, 0))
            vector[idx]     = float(rel[0])
            vector[idx + 1] = float(rel[1])
            vector[idx + 2] = float(entry.get("dist", 0.0))
            vector[idx + 3] = 1.0 if entry.get("type", "") == "prey" else 0.0
            vector[idx + 4] = float(entry.get("speed", 1.0))
            idx += 5
            slots_used += 1
        idx += (self.max_other_agents - slots_used) * 5  # advance past unused agent slots

        # --- visible obstacles: max_visible_obstacles slots of 3 values each ---
        # sorted closest-first so slot 0 always holds the nearest obstacle
        visible_obstacles = obs.get("visible_obstacles", {})
        sorted_obstacles = sorted(visible_obstacles.values(), key=lambda o: o.get("dist", 0.0))
        for i, entry in enumerate(sorted_obstacles):
            if i >= self.max_visible_obstacles:
                break
            rel = entry.get("rel_pos", (0, 0))
            vector[idx]     = float(rel[0])
            vector[idx + 1] = float(rel[1])
            vector[idx + 2] = float(entry.get("dist", 0.0))
            idx += 3

        return vector

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _build_network(self) -> QNetwork:
        return QNetwork(self.obs_dim, self.hidden_layers, self.action_dim).to(self.device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    # standard epsilon-greedy action selection; roll random number, if less than epsilon, pick random action (explore); otherwise encode observation into a tensor, run through Q-network, and pick the action with the highest Q-value (exploit); returns a dict of agent_id -> action integer
    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            if self.rng.random() < self.epsilon:
                actions[agent_id] = int(self.rng.integers(self.action_dim))
                continue

            encoded = torch.tensor(
                self._encode_observation(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():  # no gradient tracking during action selection, only during training
                q_values = self.q_networks[agent_id](encoded)
            actions[agent_id] = int(torch.argmax(q_values, dim=1).item())
        return actions

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _store_transition(
        self,
        agent_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffers[agent_id].add(
            Transition(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
            )
        )

    # core learning step:
    # 1. skip if buffer has fewer than batch_size samples (not enough data yet)
    # 2. sample a random mini-batch from the replay buffer (breaks temporal correlation)
    # 3. stack transitions into tensors (states, actions, rewards, next_states, dones)
    # 4. compute q_pred: what the online Q-network predicts for Q(s, a_taken) using .gather()
    # 5. compute target: r + gamma * (1-done) * max_a' Q_target(s', a') using the frozen target network
    # 6. MSE loss between q_pred and target, backpropagate, clip gradients, step optimizer
    def _update_agent(self, agent_id: str) -> Optional[float]:
        buffer = self.replay_buffers[agent_id]
        if len(buffer) < self.batch_size:
            return None

        batch = buffer.sample(self.batch_size, self.rng)
        states = torch.tensor(
            np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [1.0 if item.done else 0.0 for item in batch], dtype=torch.float32, device=self.device
        )

        # predicted Q-value for the action actually taken
        q_pred = self.q_networks[agent_id](states).gather(1, actions).squeeze(1)

        # Bellman target using the frozen target network — torch.no_grad() is critical here
        # so the target doesn't accumulate gradients (we're not updating the target network here)
        with torch.no_grad():
            next_q = self.target_networks[agent_id](next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self._loss_fn(q_pred, target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_networks[agent_id].parameters(), 5.0)  # prevents exploding gradients
        self.optimizers[agent_id].step()
        return float(loss.item())

    # hard copy of Q-network weights into target network (called every target_update_interval steps);
    # "hard" means copy all weights at once, as opposed to "soft" Polyak averaging (tau * new + (1-tau) * old)
    def _sync_targets(self) -> None:
        for agent_id in self.agent_ids:
            self.target_networks[agent_id].load_state_dict(
                self.q_networks[agent_id].state_dict()
            )

    # for each episode: reset env, loop until done: select actions, step env, store transitions per agent, update each agent's Q-network, sync target networks every target_update_interval global steps;
    # decay epsilon after each episode (multiplicative: eps = max(min_eps, eps * decay));
    # log a summary every 50 episodes; write one CSV row per episode if curves_path is set
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

        try:
            for episode in range(self.episodes):
                observations, _ = self.env.reset()
                done = False
                episode_reward = {agent_id: 0.0 for agent_id in self.agent_ids}
                # collect losses per agent per episode so we can report mean loss per episode
                episode_losses: Dict[str, List[float]] = {agent_id: [] for agent_id in self.agent_ids}

                while not done:
                    actions = self.select_actions(observations)
                    step_out = self.env.step(actions)
                    next_observations = step_out["obs"]
                    rewards = step_out["reward"]
                    done = bool(step_out["terminated"] or step_out["truncated"])

                    for agent_id in self.agent_ids:
                        state = self._encode_observation(observations[agent_id])
                        next_state = self._encode_observation(next_observations[agent_id])
                        reward = float(rewards[agent_id])
                        episode_reward[agent_id] += reward

                        self._store_transition(
                            agent_id=agent_id,
                            state=state,
                            action=actions[agent_id],
                            reward=reward,
                            next_state=next_state,
                            done=done,
                        )
                        loss = self._update_agent(agent_id)
                        if loss is not None:
                            episode_losses[agent_id].append(loss)

                    observations = next_observations
                    self._train_steps += 1
                    if self._train_steps % self.target_update_interval == 0:
                        self._sync_targets()

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

                if (episode + 1) % 50 == 0:
                    reward_str = ", ".join(f"{aid}={val:.2f}" for aid, val in episode_reward.items())
                    LOGGER.info(
                        "Episode %d/%d | eps=%.3f | %s",
                        episode + 1,
                        self.episodes,
                        self.epsilon,
                        reward_str,
                    )

                if csv_writer:
                    row: dict = {"episode": episode + 1, "epsilon": round(self.epsilon, 4)}
                    for aid in self.agent_ids:
                        row[f"{aid}_reward"] = round(episode_reward[aid], 4)
                        losses = episode_losses[aid]
                        row[f"{aid}_loss"] = round(sum(losses) / len(losses), 6) if losses else ""
                    csv_writer.writerow(row)
        finally:
            if csv_file:
                csv_file.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # save the entire DQN state to a file using pickle; includes config, agent IDs, obs_dim, epsilon, Q-network states, target network states, optimizer states (Adam momentum — important for resuming training correctly), replay buffers, and training step counter; creates directories if needed
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": self.config,
            "agent_ids": self.agent_ids,
            "obs_dim": self.obs_dim,
            "epsilon": self.epsilon,
            "q_state": {aid: net.state_dict() for aid, net in self.q_networks.items()},
            "target_state": {aid: net.state_dict() for aid, net in self.target_networks.items()},
            "optimizer_state": {aid: opt.state_dict() for aid, opt in self.optimizers.items()},
            "replay_buffers": {aid: buf.as_list() for aid, buf in self.replay_buffers.items()},
            "train_steps": self._train_steps,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved DQN checkpoint -> %s", path)

    @classmethod
    def load(cls, env, config: dict, path: str) -> "DQN":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)

        instance.epsilon = float(payload.get("epsilon", instance.epsilon))
        instance._train_steps = int(payload.get("train_steps", 0))

        for agent_id in instance.agent_ids:
            if agent_id in payload.get("q_state", {}):
                instance.q_networks[agent_id].load_state_dict(payload["q_state"][agent_id])
            if agent_id in payload.get("target_state", {}):
                instance.target_networks[agent_id].load_state_dict(
                    payload["target_state"][agent_id]
                )
            if agent_id in payload.get("optimizer_state", {}):
                instance.optimizers[agent_id].load_state_dict(
                    payload["optimizer_state"][agent_id]
                )
            if agent_id in payload.get("replay_buffers", {}):
                instance.replay_buffers[agent_id].load(payload["replay_buffers"][agent_id])

        LOGGER.info("Loaded DQN checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("dqn", DQN)


if __name__ == "__main__":
    import argparse
    import logging

    from multi_agent_package.scripts.run_from_config import build_environment, load_all_configs

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    parser = argparse.ArgumentParser("Independent DQN — train or evaluate")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-file", default="experiment_dqn.yaml")
    parser.add_argument("--save-path", default="trained_dqn.pkl")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir, args.experiment_file)
    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = "human" if args.render else None

    env = build_environment(configs)
    algo_params = configs["experiment"]["experiment"].get("algorithm", {}).get("params", {})

    if args.mode == "train":
        algo = DQN(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = DQN.load(env, algo_params, args.load_path)
        algo.evaluate()

    env.close()
