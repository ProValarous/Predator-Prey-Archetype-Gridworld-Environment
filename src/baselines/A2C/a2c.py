# src/baselines/A2C/a2c.py
"""
Independent Advantage Actor-Critic (A2C) baseline.

Independent-learning pattern -- one learner PER AGENT --
each learner is now a (actor, critic) pair:

    A2C   : per agent -> actor network + critic network               (on-policy)

A2C collects a short ROLLOUT of `n_steps` transitions, computes an 
n-step bootstrapped return for each one, and updates immediately. 

Observations are converted to fixed-length numeric vectors via the
environment's observation_encoder: attach an
observation builder's encode() method to env.observation_encoder
before constructing A2C (run_from_config.build_environment does this).
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
from baselines.A2C.actor_network import ActorNetwork
from baselines.A2C.critic_network import CriticNetwork

LOGGER = logging.getLogger("a2c")


class A2C(BaseAlgorithm):
    """Independent A2C: one actor + one critic per agent, updated on an n-step rollout."""

    def __init__(self, env, config: dict):
        super().__init__(env, config)

        # ---- hyperparameters from config --------------------------------

        self.gamma = float(config.get("gamma", 0.99))
        self.episodes = int(config.get("episodes", 1000))
        self.hidden_layers = [int(v) for v in config.get("hidden_layers", [128, 128])]
        self.n_steps = int(config.get("n_steps", 5))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_loss_coef = float(config.get("value_loss_coef", 0.5))
        self.grad_clip = float(config.get("grad_clip", 5.0))

        # separate learning rates: the critic usually wants to learn faster/
        # more stably than the actor, since the actor's gradient direction
        # depends on the critic already being roughly correct.
        actor_lr = config.get("actor_learning_rate", config.get("learning_rate", 3e-4))
        critic_lr = config.get("critic_learning_rate", config.get("learning_rate", 1e-3))
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)

        self.device = torch.device(config.get("device", "cpu"))
        self.verbose = bool(config.get("verbose", True))
        self.log_interval = int(config.get("log_interval", 10))
        self.debug_first_episode = bool(config.get("debug_first_episode", True))
        self.save_path = config.get("save_path", None)
        self.curves_path: Optional[str] = config.get("curves_path", None)

        # If True, select_actions() acts greedily (argmax over the policy's
        # probabilities) instead of sampling. Exploration comes from sampling 
        # the stochastic policy. Evaluation scripts set this to True.
        self.greedy_eval = bool(config.get("greedy_eval", False))

        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")

        seed = config.get("seed", None)
        self.rng = default_rng(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

        # ---- observation encoder contract --------

        self.observation_encoder = getattr(self.env, "observation_encoder", None)
        if self.observation_encoder is None:
            raise ValueError(
                "Environment is missing observation_encoder. Attach an observation "
                "builder's encode() method to env.observation_encoder before "
                "constructing A2C (see run_from_config.build_environment)."
            )

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.state_dim = self._encode_observation(initial_obs[self.agent_ids[0]]).shape[0]
        self.action_dim = self._resolve_action_dim(config)

        self._build_learners()
        self._train_steps = 0

        self._debug(
            "Initialized A2C | "
            f"agents={self.agent_ids} | state_dim={self.state_dim} | "
            f"action_dim={self.action_dim} | device={self.device} | "
            f"n_steps={self.n_steps} | entropy_coef={self.entropy_coef}"
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_action_dim(self, config: dict) -> int:
        """Infer action_dim from the env's action plugin, or validate it if configured.
        Identical logic to DQN's, kept local so A2C stays self-contained."""
        plugin = getattr(self.env, "action_space_plugin", None)
        plugin_n_actions = (
            int(plugin.n_actions) if plugin is not None else int(self.env.action_space.n)
        )

        configured = config.get("action_dim", None)
        if configured is None:
            return plugin_n_actions

        configured = int(configured)
        if configured != plugin_n_actions:
            raise ValueError(
                f"A2C config error: 'action_dim'={configured} does not match the "
                f"environment's action space size ({plugin_n_actions}). Fix "
                "'action_dim' in your experiment YAML, or remove it entirely so "
                "it gets inferred automatically."
            )
        return configured

    def _build_learners(self) -> None:
        self.actors = {}
        self.critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        for agent_id in self.agent_ids:
            self.actors[agent_id] = ActorNetwork(
                self.state_dim, self.hidden_layers, self.action_dim
            ).to(self.device)
            self.critics[agent_id] = CriticNetwork(
                self.state_dim, self.hidden_layers
            ).to(self.device)

            self.actor_optimizers[agent_id] = optim.Adam(
                self.actors[agent_id].parameters(), lr=self.actor_lr
            )
            self.critic_optimizers[agent_id] = optim.Adam(
                self.critics[agent_id].parameters(), lr=self.critic_lr
            )

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[A2C] {message}")

    def _empty_rollout(self) -> dict:
        return {
            aid: {"log_probs": [], "values": [], "rewards": [], "entropies": []}
            for aid in self.agent_ids
        }

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_observation(self, obs) -> np.ndarray:
        encoded = self.observation_encoder(obs, self.env)
        return np.asarray(encoded, dtype=np.float32).reshape(-1)

    def _validate_state_shape(self, agent_id: str, state: np.ndarray) -> None:
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"A2C expected state_dim={self.state_dim}, but agent {agent_id!r} "
                f"produced state_dim={state.shape[0]}. This usually means the "
                "observation plugin is returning variable-length observations."
            )

    # ------------------------------------------------------------------
    # Action selection (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        """
        Public, gradient-free action selection -- used by BaseAlgorithm.evaluate()
        and by anything outside the training loop. The training loop itself does
        NOT call this method: it needs log_prob/value/entropy WITH gradients
        attached, so it runs the same forward pass directly (see _train_loop).
        """
        actions = {}
        with torch.no_grad():
            for agent_id, obs in observations.items():
                state = self._encode_observation(obs)
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                dist = self.actors[agent_id].distribution(state_t)
                if self.greedy_eval:
                    action = torch.argmax(dist.probs, dim=1)
                else:
                    action = dist.sample()
                actions[agent_id] = int(action.item())
        return actions

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _optimize_agent(self, agent_id: str, traj: dict, bootstrap_value: float):
        """
        Turns one agent's n-step rollout into an actor update and a critic
        update. Returns (actor_loss, critic_loss, mean_entropy) as floats for
        logging.

        n-step return:  R_t = r_t + gamma*r_{t+1} + ... + gamma^k * V(s_{t+k})
        Bootstrap value is 0 if the episode ended inside this rollout
        (nothing beyond a terminal state contributes future reward),
        otherwise it's the critic's own estimate of the state we stopped at.
        """
        rewards = traj["rewards"]

        returns = []
        R = bootstrap_value
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        values_t = torch.cat(traj["values"])      # keeps grad -> critic params
        log_probs_t = torch.cat(traj["log_probs"])  # keeps grad -> actor params
        entropies_t = torch.cat(traj["entropies"])

        # detach() is essential:
        # the actor should be pushed by the SIGN and SIZE of the advantage,
        # not by trying to change the critic's value through this path.
        advantages = returns_t - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean() - self.entropy_coef * entropies_t.mean()
        critic_loss = nn.MSELoss()(values_t, returns_t)

        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.grad_clip)
        self.actor_optimizers[agent_id].step()

        self.critic_optimizers[agent_id].zero_grad()
        (self.value_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.grad_clip)
        self.critic_optimizers[agent_id].step()

        self._train_steps += 1

        return float(actor_loss.item()), float(critic_loss.item()), float(entropies_t.mean().item())

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
            actor_loss_cols = [f"{aid}_actor_loss" for aid in self.agent_ids]
            critic_loss_cols = [f"{aid}_critic_loss" for aid in self.agent_ids]
            entropy_cols = [f"{aid}_entropy" for aid in self.agent_ids]
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=["episode"] + reward_cols + actor_loss_cols
                + critic_loss_cols + entropy_cols,
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

    def _act_with_grad(self, agent_id: str, obs) -> tuple:
        """Forward pass WITH gradient tracking -- used only inside the
        training loop, where log_prob/value/entropy must stay part of the
        autograd graph so _optimize_agent can backpropagate through them."""
        state = self._encode_observation(obs)
        self._validate_state_shape(agent_id, state)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        dist = self.actors[agent_id].distribution(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critics[agent_id](state_t).squeeze(1)

        return int(action.item()), log_prob, entropy, value

    def _train_loop(self, csv_writer) -> None:
        for episode in range(self.episodes):
            observations, _ = self.env.reset()
            episode_rewards = {aid: 0.0 for aid in self.agent_ids}
            episode_actor_losses = {aid: [] for aid in self.agent_ids}
            episode_critic_losses = {aid: [] for aid in self.agent_ids}
            episode_entropies = {aid: [] for aid in self.agent_ids}

            rollout = self._empty_rollout()
            done = False
            step_count = 0

            while not done:
                actions = {}
                for agent_id in self.agent_ids:
                    action, log_prob, entropy, value = self._act_with_grad(
                        agent_id, observations[agent_id]
                    )
                    actions[agent_id] = action
                    rollout[agent_id]["log_probs"].append(log_prob)
                    rollout[agent_id]["values"].append(value)
                    rollout[agent_id]["entropies"].append(entropy)

                step_out = self.env.step(actions)
                next_observations = step_out["obs"]
                rewards = step_out["reward"]
                done = bool(step_out["terminated"] or step_out["truncated"])

                for agent_id in self.agent_ids:
                    r = float(rewards[agent_id])
                    rollout[agent_id]["rewards"].append(r)
                    episode_rewards[agent_id] += r

                    if episode == 0 and step_count == 0 and self.debug_first_episode:
                        self._debug(
                            f"First transition | agent={agent_id} | "
                            f"action={actions[agent_id]} | reward={r:.3f} | done={done}"
                        )

                observations = next_observations
                step_count += 1

                # Update every n_steps, or immediately at episode end (bootstraps 
                # from 0 instead of from V(s_next)).
                ready_to_update = (step_count % self.n_steps == 0) or done
                if ready_to_update:
                    for agent_id in self.agent_ids:
                        if not rollout[agent_id]["rewards"]:
                            continue  # nothing collected for this agent yet

                        if done:
                            bootstrap_value = 0.0
                        else:
                            with torch.no_grad():
                                next_state = self._encode_observation(observations[agent_id])
                                next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                                bootstrap_value = float(
                                    self.critics[agent_id](next_state_t).item()
                                )

                        actor_loss, critic_loss, mean_entropy = self._optimize_agent(
                            agent_id, rollout[agent_id], bootstrap_value
                        )
                        episode_actor_losses[agent_id].append(actor_loss)
                        episode_critic_losses[agent_id].append(critic_loss)
                        episode_entropies[agent_id].append(mean_entropy)

                    rollout = self._empty_rollout()

            if (episode + 1) % self.log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={value:.2f}" for aid, value in episode_rewards.items()
                )
                self._debug(
                    f"Episode {episode + 1}/{self.episodes} | steps={step_count} | "
                    f"rewards: {reward_str}"
                )

            if csv_writer:
                row: dict = {"episode": episode + 1}
                for aid in self.agent_ids:
                    row[f"{aid}_reward"] = round(episode_rewards[aid], 4)
                    a_losses = episode_actor_losses[aid]
                    c_losses = episode_critic_losses[aid]
                    ent = episode_entropies[aid]
                    row[f"{aid}_actor_loss"] = round(float(np.mean(a_losses)), 6) if a_losses else ""
                    row[f"{aid}_critic_loss"] = round(float(np.mean(c_losses)), 6) if c_losses else ""
                    row[f"{aid}_entropy"] = round(float(np.mean(ent)), 6) if ent else ""
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
            "actor_state_dicts": {aid: net.state_dict() for aid, net in self.actors.items()},
            "critic_state_dicts": {aid: net.state_dict() for aid, net in self.critics.items()},
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved A2C checkpoint -> %s", path)
        self._debug(f"Saved A2C checkpoint -> {path}")

    @classmethod
    def load(cls, env, config: dict, path: str) -> "A2C":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        for agent_id in instance.agent_ids:
            if agent_id in payload["actor_state_dicts"]:
                instance.actors[agent_id].load_state_dict(payload["actor_state_dicts"][agent_id])
            if agent_id in payload["critic_state_dicts"]:
                instance.critics[agent_id].load_state_dict(payload["critic_state_dicts"][agent_id])

        LOGGER.info("Loaded A2C checkpoint from %s", path)
        return instance


if __name__ != "__main__":
    register("a2c", A2C)


# ------------------------------------------------------------------
# Standalone CLI -- python -m baselines.A2C.a2c [--mode train|eval]
# (for quick manual testing without YAML configs)
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

    p = argparse.ArgumentParser("Independent A2C — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--observation", type=str, default="local_only")
    p.add_argument("--action-space", type=str, default="discrete_5")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128])
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_a2c.pkl")
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    agents = []
    for i in range(1, args.preys + 1):
        agents.append(Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey"))
    for i in range(1, args.predators + 1):
        agents.append(Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator"))

    render = "human" if (args.mode == "eval" and args.render) else None
    env = GridWorldEnv(
        agents=agents, render_mode=render, size=args.size, perc_num_obstacle=10, seed=args.seed
    )

    observation_builder = get_observation_builder(args.observation)
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode
    env.action_space_plugin = get_action_space(args.action_space)

    config = {
        "hidden_layers": args.hidden_layers,
        "n_steps": args.n_steps,
        "actor_learning_rate": args.actor_lr,
        "critic_learning_rate": args.critic_lr,
        "episodes": args.episodes,
        "seed": args.seed,
        "greedy_eval": args.mode == "eval",
    }

    if args.mode == "train":
        algo = A2C(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = A2C.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
