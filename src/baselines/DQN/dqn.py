"""
DQN Algorithm — Independent Deep Q-Learning for MARL.

This module implements the top-level algorithm class.
It is the only public interface the rest of the codebase
interacts with — run_from_config.py calls:

    algo = DQN(env, config)
    algo.train()

Architecture
------------
One DQNAgent per environment agent (Independent Q-Learning style).
All agents share one ReplayBuffer.

    DQN
    ├── agents: dict[str, DQNAgent]   one per env agent
    ├── buffer: ReplayBuffer          shared across all agents
    └── _obs_to_vector()              dict obs → float32 vector

Observation encoding
--------------------
The environment returns structured dict observations:
    obs[agent_name] = {
        "local":  np.ndarray   (x, y) position
        "global": dict         dist_agents, etc.
    }

_obs_to_vector() flattens this into a 1D float32 numpy array
that the network can consume. All environment-specific
structure is isolated here — changing observation format
only requires changing this one method.

Logging and checkpointing
--------------------------
train() handles all logging and checkpointing internally.
No external training script is needed beyond run_from_config.py.
TensorBoard and wandb logging mirrors the IQL implementation.

Config keys (all optional — safe defaults provided)
---------------------------------------------------
    hidden_units        list[int]   [128, 128]
    lr                  float       1e-3
    gamma               float       0.99
    epsilon             float       1.0
    epsilon_decay       float       0.995
    min_epsilon         float       0.01
    batch_size          int         64
    buffer_capacity     int         50_000
    target_update_freq  int         100
    use_double          bool        True
    use_dueling         bool        True
    episodes            int         5_000
    max_steps           int         200
    seed                int         0
    device              str         "cpu"
    save_path           str         "baselines/DQN/"
    log_interval        int         100

This module contains:
    - NO YAML parsing
    - NO argparse
    - NO environment construction
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import os 
import netrc

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register
from baselines.DQN.dqn_agent import DQNAgent
from baselines.DQN.replay_buffer import ReplayBuffer
from .utils import create_experiment_dir, write_experiment_md

LOGGER = logging.getLogger("dqn")


class DQN(BaseAlgorithm):
    """
    Independent Deep Q-Learning for multi-agent environments.

    Each environment agent gets its own DQNAgent (online + target
    network, optimiser). All agents push transitions into a single
    shared ReplayBuffer and sample from it independently.

    Inherits BaseAlgorithm — implements:
        select_actions(observations) -> dict
        train()                      -> None

    evaluate() is inherited from BaseAlgorithm and uses
    select_actions() automatically.

    Parameters
    ----------
    env : GridWorldEnv
        The environment instance. Treated as a black box —
        only env.reset() and env.step() are called.
    config : dict
        Hyperparameter dict, typically loaded from experiment.yaml
        by run_from_config.py. All keys are optional.

    Examples
    --------
    >>> algo = DQN(env, config={"lr": 1e-3, "episodes": 1000})
    >>> algo.train()
    """

    def __init__(self, env, config: dict) -> None:
        super().__init__(env, config)

        # ── Hyperparameters ───────────────────────────────────────────────
        self.episodes          = config.get("episodes",           5_000)
        self.max_steps         = config.get("max_steps",            200)
        self.batch_size        = config.get("batch_size",             64)
        self.epsilon           = config.get("epsilon",               1.0)
        self.epsilon_decay     = config.get("epsilon_decay",       0.995)
        self.min_epsilon       = config.get("min_epsilon",          0.01)
        self.seed              = config.get("seed",                    0)
        self.save_path         = config.get("save_path",   "baselines/DQN/")
        self.log_interval      = config.get("log_interval",          100)

        hidden_units           = config.get("hidden_units",    [128, 128])
        lr                     = config.get("lr",                   1e-3)
        gamma                  = config.get("gamma",                0.99)
        use_double             = config.get("use_double",           True)
        use_dueling            = config.get("use_dueling",          True)
        target_update_freq     = config.get("target_update_freq",    100)
        buffer_capacity        = config.get("buffer_capacity",    50_000)
        device_str             = config.get("device",              "cpu")
        

        self.device = torch.device(device_str)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ── Discover agents and observation dimensions ─────────────────────
        # Reset once to find agent IDs and infer state_dim from a
        # real observation. Mirrors how IQL discovers its agents.
        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        

        if config.get("state_dim") is not None:
            self.state_dim = config["state_dim"]
        else:
            sample_vec = self._obs_to_vector(initial_obs[self.agent_ids[0]])
            self.state_dim = sample_vec.shape[0]

        self.action_dim = config.get("action_dim", 5)

        # ── Shared replay buffer ──────────────────────────────────────────
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=self.seed)

        # ── One DQNAgent per environment agent ────────────────────────────
        self.agents: dict = {
            agent_id: DQNAgent(
                agent_id=agent_id,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_units=hidden_units,
                lr=lr,
                gamma=gamma,
                use_double=use_double,
                use_dueling=use_dueling,
                target_update_freq=target_update_freq,
                device=self.device,
            )
            for agent_id in self.agent_ids
        }

    # ── Observation encoding ──────────────────────────────────────────────────

    def _obs_to_vector(self, obs_dict: dict) -> np.ndarray:
        """
        Flatten a single agent's local-radius observation into a
        fixed-size float32 vector.

        Observation structure:
            local            : [x, y]          own position
            radius           : int             perception radius
            visible_agents   : dict            agents within radius
            visible_obstacles: dict            obstacles within radius

        Encoding (fixed size = 24):
            [0:2]   own position (x, y)
            [2]     radius
            [3:6]   nearest predator (dist, dx, dy) — zeros if none visible
            [6:9]   2nd nearest predator
            [9:12]  3rd nearest predator
            [12:15] nearest prey (dist, dx, dy) — zeros if none visible
            [15:18] 2nd nearest prey
            [18:21] 3rd nearest prey
            [21:24] nearest obstacle (dist, dx, dy) — zeros if none visible

        If fewer agents/obstacles than slots are visible, remaining
        slots are filled with zeros. This guarantees a fixed output
        size of 23 regardless of what is actually visible.

        Parameters
        ----------
        obs_dict : dict
            Single agent observation from env.reset() or env.step().

        Returns
        -------
        np.ndarray
            Flat float32 vector, shape (24,).
        """
        vec = np.zeros(24, dtype=np.float32)

        # ── Own position ──────────────────────────────────────────────────
        local = obs_dict.get("local", [0, 0])
        vec[0] = float(local[0])
        vec[1] = float(local[1])

        # ── Radius ────────────────────────────────────────────────────────
        vec[2] = float(obs_dict.get("radius", 0))

        # ── Visible agents — split by type, sort by distance ─────────────
        visible_agents = obs_dict.get("visible_agents", {})

        predators_visible = sorted(
            [(v["dist"], v["rel_pos"][0], v["rel_pos"][1])
            for v in visible_agents.values()
            if v.get("type") == "predator"],
            key=lambda x: x[0]
        )
        preys_visible = sorted(
            [(v["dist"], v["rel_pos"][0], v["rel_pos"][1])
            for v in visible_agents.values()
            if v.get("type") == "prey"],
            key=lambda x: x[0]
        )

        # Fill predator slots (up to 3)
        for i, (dist, dx, dy) in enumerate(predators_visible[:3]):
            base = 3 + i * 3
            vec[base]     = float(dist)
            vec[base + 1] = float(dx)
            vec[base + 2] = float(dy)

        # Fill prey slots (up to 3)
        for i, (dist, dx, dy) in enumerate(preys_visible[:3]):
            base = 12 + i * 3
            vec[base]     = float(dist)
            vec[base + 1] = float(dx)
            vec[base + 2] = float(dy)

        # ── Visible obstacles — nearest only ──────────────────────────────
        visible_obstacles = obs_dict.get("visible_obstacles", {})
        if visible_obstacles:
            nearest_obs = min(
                visible_obstacles.values(),
                key=lambda o: o["dist"]
            )
            vec[21] = float(nearest_obs["dist"])
            vec[22] = float(nearest_obs["rel_pos"][0])
            vec[23] = float(nearest_obs["rel_pos"][1])

        return vec

    # ── BaseAlgorithm interface ───────────────────────────────────────────────

    def select_actions(self, observations: dict) -> dict:
        """
        Select actions for all agents given current observations.

        Called every step during both train() and evaluate().
        Uses epsilon-greedy — self.epsilon is managed by train().

        Parameters
        ----------
        observations : dict
            {agent_name: obs_dict} as returned by env.reset() / env.step().

        Returns
        -------
        dict
            {agent_name: action_int}
        """
        actions = {}
        for agent_id, obs in observations.items():
            state_vec = self._obs_to_vector(obs)
            actions[agent_id] = self.agents[agent_id].act(
                state_vec, self.epsilon
            )
        return actions

    def train(self) -> None:
        """
        Run the full training loop with logging and checkpointing.

        Episode structure (mirrors IQL train loop):
            1. env.reset()
            2. select_actions() → env.step()
            3. push transitions into shared buffer (one per agent)
            4. if buffer ready: sample batch, update each agent
            5. decay epsilon each episode
            6. log to TensorBoard + wandb every episode
            7. checkpoint every 1000 episodes

        Logging mirrors iql_train2v2.py:
            - per-agent mean reward (window=100)
            - per-agent mean loss  (window=100)
            - episode length
            - epsilon
            - captures
        """

        # ── wandb ─────────────────────────────────────────────────────────
        try:
            netrc_path = os.path.join(os.path.expanduser("~"), "_netrc")
            nrc = netrc.netrc(netrc_path)
            key = nrc.authenticators("api.wandb.ai")[2]
            os.environ["WANDB_API_KEY"] = "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"
        except Exception:
            pass
        
        wandb.init(project="MARL-Predator-Prey-Project", sync_tensorboard=True)

        # ── Experiment folder (mirrors iql_train2v2.py) ───────────────────
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
        )
        exp_dir, checkpoints_dir, logs_dir = create_experiment_dir(
            base=base_dir,
            name="dqn_run",
        )

        write_experiment_md(exp_dir, {
            "algorithm":         "DQN",
            "episodes":          self.episodes,
            "epsilon":           self.epsilon,
            "epsilon_decay":     self.epsilon_decay,
            "min_epsilon":       self.min_epsilon,
            "batch_size":        self.batch_size,
            "seed":              self.seed,
        })

        # ── TensorBoard ───────────────────────────────────────────────────
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        log_dir = os.path.join(logs_dir, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        checkpoint_path = os.path.join(
            checkpoints_dir, "dqn_checkpoint.pt"
        )

        # ── Bookkeeping ───────────────────────────────────────────────────
        per_agent_rewards: Dict[str, List[float]] = {
            aid: [] for aid in self.agent_ids
        }
        per_agent_losses: Dict[str, List[float]] = {
            aid: [] for aid in self.agent_ids
        }
        captures_per_ep: List[int] = []
        capture_count = 0
        window = 100

        # ── Training loop ─────────────────────────────────────────────────
        for ep in range(1, self.episodes + 1):

            obs, _ = self.env.reset()
            # sanity check — remove after first run
            # test_vec = self._obs_to_vector(obs[self.agent_ids[0]])
            # assert test_vec.shape[0] == 24, f"Expected 24, got {test_vec.shape[0]}"
            # print(f"state_dim confirmed: {test_vec.shape}")
            # print(f"agent_ids: {self.agent_ids}")
            # print(f"action_dim: {self.action_dim}")
            ep_rewards = {aid: 0.0 for aid in self.agent_ids}
            ep_losses  = {aid: [] for aid in self.agent_ids}
            ep_len = self.max_steps

            for t in range(self.max_steps):

                # Action selection
                actions = self.select_actions(obs)

                # Environment step
                step_out   = self.env.step(actions)
                next_obs   = step_out["obs"]
                rewards    = step_out["reward"]
                terminated = step_out["terminated"]
                done       = terminated or step_out.get("trunc", False)

                # Accumulate episode rewards
                for aid in self.agent_ids:
                    ep_rewards[aid] += float(rewards[aid])

                # Push transitions into shared buffer
                for aid in self.agent_ids:
                    s  = self._obs_to_vector(obs[aid])
                    s_ = self._obs_to_vector(next_obs[aid])
                    self.buffer.push(
                        state=s,
                        action=actions[aid],
                        reward=float(rewards[aid]),
                        next_state=s_,
                        done=done,
                    )

                # Update each agent once buffer is ready
                if self.buffer.is_ready(self.batch_size):
                    for aid in self.agent_ids:
                        batch = self.buffer.sample(
                            self.batch_size, self.device
                        )
                        loss = self.agents[aid].update(*batch)
                        ep_losses[aid].append(loss)

                if terminated:
                    capture_count += 1
                    ep_len = t + 1
                    break

                obs = next_obs

            # ── Epsilon decay ─────────────────────────────────────────────
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay,
            )

            # ── Episode bookkeeping ───────────────────────────────────────
            for aid in self.agent_ids:
                per_agent_rewards[aid].append(ep_rewards[aid])
                if ep_losses[aid]:
                    per_agent_losses[aid].append(
                        float(np.mean(ep_losses[aid]))
                    )

            captures_this_ep = int(getattr(self.env, "_captures_total", 0))
            captures_per_ep.append(captures_this_ep)

            # ── TensorBoard logging ───────────────────────────────────────
            writer.add_scalar("episode/length",   ep_len,         ep)
            writer.add_scalar("episode/epsilon",  self.epsilon,   ep)
            writer.add_scalar("episode/captures", captures_this_ep, ep)

            mean_captures = float(np.mean(captures_per_ep[-window:])) \
                if captures_per_ep else 0.0
            writer.add_scalar("mean/captures", mean_captures, ep)

            for aid in self.agent_ids:
                mean_reward = float(np.mean(
                    per_agent_rewards[aid][-window:]
                )) if per_agent_rewards[aid] else 0.0
                writer.add_scalar(f"mean/{aid}/reward", mean_reward, ep)

                if per_agent_losses[aid]:
                    mean_loss = float(np.mean(
                        per_agent_losses[aid][-window:]
                    ))
                    writer.add_scalar(f"mean/{aid}/loss", mean_loss, ep)

            # ── Console logging ───────────────────────────────────────────
            if ep % self.log_interval == 0:
                reward_str = ", ".join(
                    f"{aid}={np.mean(per_agent_rewards[aid][-100:]):.2f}"
                    for aid in self.agent_ids
                )
                LOGGER.info(
                    "Ep %d | eps=%.3f | rewards(last100): %s | "
                    "captures(last100)=%.2f",
                    ep, self.epsilon, reward_str, mean_captures,
                )

            # ── Periodic flush and checkpoint ─────────────────────────────
            if ep % 10 == 0:
                writer.flush()

            if ep % 100 == 0:
                self._save_checkpoint(
                    checkpoint_path, ep,
                    capture_count, per_agent_rewards,
                )

        # ── Final checkpoint ──────────────────────────────────────────────
        self._save_checkpoint(
            checkpoint_path, self.episodes,
            capture_count, per_agent_rewards,
        )

        writer.close()
        LOGGER.info(
            "Training complete. Total captures: %d. "
            "Final epsilon: %.4f. Checkpoint: %s",
            capture_count, self.epsilon, checkpoint_path,
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        path: str,
        ep: int,
        capture_count: int,
        per_agent_rewards: Dict[str, List[float]],
    ) -> None:
        """Save full training state to a .pt file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "algorithm_state":   self.state_dict(),
            "episode":           ep,
            "capture_count":     capture_count,
            "per_agent_rewards": per_agent_rewards,
            "state_dim":         self.state_dim,        # ← add
            "action_dim":        self.action_dim,       # ← add
            "hidden_units":      self.config.get("hidden_units", [128, 128]),  # ← add
            "use_double":        self.config.get("use_double", True),          # ← add
            "use_dueling":       self.config.get("use_dueling", True),         # ← add
        }, path)
        LOGGER.info("Checkpoint saved -> %s (episode %d)", path, ep)

    def state_dict(self) -> dict:
        """Return full algorithm state for checkpointing."""
        return {
            "agents": {
                aid: agent.state_dict()
                for aid, agent in self.agents.items()
            },
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore algorithm state from a checkpoint dict."""
        for aid, agent_state in state["agents"].items():
            self.agents[aid].load_state_dict(agent_state)
        self.epsilon = state["epsilon"]


register("dqn", DQN)