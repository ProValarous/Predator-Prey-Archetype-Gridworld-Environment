# src/baselines/AC/actor_critic.py
"""
Independent one-step online Actor-Critic baseline (Sutton & Barto,
Reinforcement Learning: An Introduction, 2nd ed., Algorithm 13.5,
"Actor-Critic (episodic), for estimating pi_theta").

Mirrors DQN's independent-per-agent structure -- one network and
optimizer per agent -- but learns fully on-policy: every single
env.step() immediately triggers one gradient update from the TD(0)
error, with no replay buffer and no target network. Observations are
converted to fixed-length numeric vectors via the environment's
observation_encoder, exactly as DQN requires: attach an observation
builder's encode() method to env.observation_encoder before
constructing ActorCritic (run_from_config.build_environment does this).
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register
from baselines.AC.network import ActorCriticNetwork

LOGGER = logging.getLogger("actor_critic")


class ActorCritic(BaseAlgorithm):
    """Independent one-step online actor-critic: one network per agent,
    no replay buffer."""

    def __init__(self, env, config: dict):
        super().__init__(env, config)

        self.gamma = float(config.get("gamma", 0.99))
        self.episodes = int(config.get("episodes", 1000))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.hidden_layers = [int(v) for v in config.get("hidden_layers", [128, 128])]
        self.value_coef = float(config.get("value_coef", 0.5))
        self.entropy_coef = float(config.get("entropy_coef", 0.0))
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
                "constructing ActorCritic (see run_from_config.build_environment)."
            )

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.state_dim = self._encode_observation(initial_obs[self.agent_ids[0]]).shape[
            0
        ]
        self.action_dim = self._resolve_action_dim(config)

        self._build_learners()

        self._debug(
            "Initialized ActorCritic | "
            f"agents={self.agent_ids} | state_dim={self.state_dim} | "
            f"action_dim={self.action_dim} | device={self.device}"
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
                f"ActorCritic config error: 'action_dim'={configured} does not match "
                f"the environment's action space size ({plugin_n_actions}). Fix "
                "'action_dim' in your experiment YAML, or remove it entirely so "
                "it gets inferred automatically."
            )
        return configured

    def _build_learners(self) -> None:
        self.networks = {}
        self.optimizers = {}

        for agent_id in self.agent_ids:
            self.networks[agent_id] = ActorCriticNetwork(
                self.state_dim, self.hidden_layers, self.action_dim
            ).to(self.device)
            self.optimizers[agent_id] = optim.Adam(
                self.networks[agent_id].parameters(), lr=self.learning_rate
            )

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[ActorCritic] {message}")

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_observation(self, obs) -> np.ndarray:
        encoded = self.observation_encoder(obs, self.env)
        return np.asarray(encoded, dtype=np.float32).reshape(-1)

    def _validate_state_shape(self, agent_id: str, state: np.ndarray) -> None:
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"ActorCritic expected state_dim={self.state_dim}, but agent "
                f"{agent_id!r} produced state_dim={state.shape[0]}. This usually "
                "means the observation plugin is returning variable-length "
                "observations."
            )

    # ------------------------------------------------------------------
    # Action selection (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            state = self._encode_observation(obs)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.networks[agent_id](state_t)
            dist = Categorical(logits=logits)
            actions[agent_id] = int(dist.sample().item())
        return actions

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _update(
        self,
        agent_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        discount: float,
    ) -> float:
        """One TD(0) actor-critic step (Sutton & Barto, Algorithm 13.5).

        `terminal` reflects true termination only (not truncation), so the
        target keeps bootstrapping through timeout cut-offs -- same
        distinction DQN's training loop makes. `discount` is the running
        I <- gamma * I factor the caller decays once per step; it weights
        the actor's gradient by how many steps into the episode we are.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        action_t = torch.tensor([action], device=self.device)

        logits, value = self.networks[agent_id](state_t)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()

        with torch.no_grad():
            _, next_value = self.networks[agent_id](next_state_t)
            next_value = next_value * (0.0 if terminal else 1.0)
            td_target = reward + self.gamma * next_value

        delta = td_target - value

        actor_loss = -(discount * delta.detach() * log_prob).mean()
        critic_loss = delta.pow(2).mean()
        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy.mean()
        )

        optimizer = self.optimizers[agent_id]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.networks[agent_id].parameters(), self.grad_clip
        )
        optimizer.step()

        return float(loss.item())

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
                csv_file, fieldnames=["episode"] + reward_cols + loss_cols
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
            discount = {agent_id: 1.0 for agent_id in self.agent_ids}
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
                # True termination only -- keep bootstrapping through
                # timeout truncation, same distinction DQN's loop makes.
                terminal = bool(step_out["terminated"])

                for agent_id in self.agent_ids:
                    state = self._encode_observation(observations[agent_id])
                    next_state = self._encode_observation(next_observations[agent_id])
                    self._validate_state_shape(agent_id, state)
                    self._validate_state_shape(agent_id, next_state)

                    loss = self._update(
                        agent_id,
                        state,
                        int(actions[agent_id]),
                        float(rewards[agent_id]),
                        next_state,
                        terminal,
                        discount[agent_id],
                    )
                    discount[agent_id] *= self.gamma
                    episode_rewards[agent_id] += float(rewards[agent_id])
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

            if (episode + 1) % self.log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={value:.2f}" for aid, value in episode_rewards.items()
                )
                loss_str = ", ".join(
                    f"{aid}={np.mean(losses):.5f}"
                    for aid, losses in episode_losses.items()
                )
                self._debug(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"steps={step_count} | rewards: {reward_str} | avg_loss: {loss_str}"
                )

            if csv_writer:
                row: dict = {"episode": episode + 1}
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
            "state_dicts": {
                aid: net.state_dict() for aid, net in self.networks.items()
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved ActorCritic checkpoint -> %s", path)
        self._debug(f"Saved ActorCritic checkpoint -> {path}")

    @classmethod
    def load(cls, env, config: dict, path: str) -> "ActorCritic":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        for agent_id in instance.agent_ids:
            if agent_id in payload["state_dicts"]:
                instance.networks[agent_id].load_state_dict(
                    payload["state_dicts"][agent_id]
                )

        LOGGER.info("Loaded ActorCritic checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("actor_critic", ActorCritic)


# ------------------------------------------------------------------
# Standalone CLI -- python -m baselines.AC.actor_critic [--mode train|eval]
# (mirrors DQN's own CLI for quick manual testing without YAML configs)
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

    p = argparse.ArgumentParser("Independent Actor-Critic — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--observation", type=str, default="local_only")
    p.add_argument("--action-space", type=str, default="discrete_5")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_actor_critic.pkl")
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
        "episodes": args.episodes,
        "seed": args.seed,
    }

    if args.mode == "train":
        algo = ActorCritic(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = ActorCritic.load(env, config, args.load_path)
        print(algo.evaluate(episodes=args.episodes))

    env.close()
