"""
Clean, modular and documented version of the tabular IQL trainer for a 1-predator / 1-prey
GridWorld experiment.

Features
- Uses compact joint state: (own_cell_index, discretized_distance_to_other)
- Modular functions for environment & agent creation, Q-table init, action selection, updates
- Checkpoint save/load (npz) with metadata and resume support
- Periodic checkpointing via --save-interval and final saves
- TensorBoard logging

Assumptions / external utilities required
- `multi_agent_package.gridworld.GridWorldEnv` is available and matches the API used below
- `multi_agent_package.agents.Agent` exists for agent objects
- `baselines.IQL.utils` provides: make_agents, global_joint_state_index, save_checkpoint, load_checkpoint
  (we import these explicitly; if your project defines different names, adapt the imports)

Usage (example):
    python iql_train_cleaned.py --episodes 10000 --size 8 --alpha 0.1 --gamma 0.99

"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

# Import utilities from your baselines package. These are expected to exist.
from baselines.IQL.utils import (
    make_agents,
    global_joint_state_index,
    save_checkpoint,
    load_checkpoint,
)


LOGGER = logging.getLogger("iql_trainer")


# ----------------------
# Helpers / utilities
# ----------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure module-level logging."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_env_and_meta(agents, grid_size: int, seed: int, perc_num_obstacle: int = 10) -> Tuple[GridWorldEnv, int, int, int, int]:
    """Create GridWorld environment and compute state/action space sizes.

    Returns
    -------
    env, n_cells, n_distance_bins, n_states, n_actions
    """
    # agent bookkeeping for rendering/visuals (not required for training but kept consistent)
    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=perc_num_obstacle, seed=seed)

    n_cells = grid_size * grid_size
    max_dist = math.ceil(math.sqrt(2) * (grid_size - 1))
    n_distance_bins = max_dist + 1
    n_states = n_cells * n_distance_bins
    n_actions = env.action_space.n

    return env, n_cells, n_distance_bins, n_states, n_actions


def init_q_tables(agents, n_states: int, n_actions: int) -> Dict[str, np.ndarray]:
    """Initialize zero Q-tables for each agent.

    Agent names are used as keys.
    """
    return {ag.agent_name: np.zeros((n_states, n_actions), dtype=np.float32) for ag in agents}


def save_qs_npz(path: str, Qs: Dict[str, np.ndarray]) -> None:
    """Save final Q-tables to an .npz file (not a checkpoint metadata wrapper).

    This mirrors the older script behavior which saved the plain Q matrices.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **Qs)
    LOGGER.info("Saved Q-tables to %s", path)


def epsilon_greedy_action(q_row: np.ndarray, n_actions: int, rng: np.random.Generator, eps: float) -> int:
    """Standard epsilon-greedy action selection."""
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    return int(int(np.argmax(q_row)))


# ----------------------
# Training loop
# ----------------------

def train(
    episodes: int = 5000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.95,
    grid_size: int = 8,
    save_path: str = "baselines/IQL/iql_qs.npz",
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    save_interval: int = 1000,
) -> None:
    """Main training loop for independent Q-learning (tabular) with two agents.

    High-level behaviour notes
    - Predator uses epsilon-greedy learning policy
    - Prey acts randomly (stochastic policy). This mirrors the earlier script where prey
      did not learn via Q-values. If you'd like prey to learn too, remove the special-casing.
    """
    rng = np.random.default_rng(seed)

    # create agents and environment metadata
    prey, predator = make_agents()
    agents = [prey, predator]
    env, n_cells, n_distance_bins, n_states, n_actions = make_env_and_meta(agents, grid_size, seed)

    # initialize Qs
    Qs = init_q_tables(agents, n_states, n_actions)

    # bookkeeping for logging, checkpointing and resume
    prey_episode_totals: list[float] = []
    predator_episode_totals: list[float] = []
    eps = float(eps_start)
    capture_count = 0
    start_ep = 1

    # Attempt to resume from checkpoint
    if resume and checkpoint_path:
        try:
            loaded_Qs, meta = load_checkpoint(checkpoint_path)
            # Validate shapes
            compatible = True
            for name, arr in loaded_Qs.items():
                if name in Qs and Qs[name].shape != arr.shape:
                    LOGGER.warning("Checkpoint shape mismatch for %s: expected %s, got %s", name, Qs[name].shape, arr.shape)
                    compatible = False
                    break
            if compatible:
                for name, arr in loaded_Qs.items():
                    if name in Qs:
                        Qs[name] = arr
                eps = float(meta.get("eps", eps))
                start_ep = int(meta.get("ep", 0)) + 1
                capture_count = int(meta.get("capture_count", capture_count))
                prey_episode_totals = list(meta.get("prey_totals", []))
                predator_episode_totals = list(meta.get("pred_totals", []))
                LOGGER.info("Resuming from checkpoint %s: start_ep=%d, eps=%.4f, capture_count=%d", checkpoint_path, start_ep, eps, capture_count)
            else:
                LOGGER.info("Incompatible checkpoint — starting fresh")
        except FileNotFoundError:
            LOGGER.info("No checkpoint found at %s — starting fresh", checkpoint_path)
        except Exception as e:
            LOGGER.exception("Failed to load checkpoint: %s", e)

    # Prepare saving/logging dirs
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(save_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    LOGGER.info("TensorBoard logs -> %s", log_dir)

    # ----- episode loop -----
    for ep in range(start_ep, episodes + 1):
        obs, info = env.reset()
        ep_agent_totals: Dict[str, float] = {ag.agent_name: 0.0 for ag in agents}

        for t in range(max_steps):
            # Build current state indices for each agent
            s_idx: Dict[str, int] = {}
            pos_map: Dict[str, np.ndarray] = {}

            # collect positions (local obs fallback)
            for ag in agents:
                local = obs.get(ag.agent_name, {}).get("local")
                if local is None:
                    local = getattr(ag, "_agent_location", np.array([0, 0]))
                pos_map[ag.agent_name] = np.asarray(local, dtype=int)

            # compute joint compact state index using provided util
            for ag in agents:
                other = next(o for o in agents if o.agent_name != ag.agent_name)
                global_info = obs.get(ag.agent_name, {}).get("global", {})
                dist_agents = (global_info or {}).get("dist_agents", {})
                s_idx[ag.agent_name] = global_joint_state_index(pos_map[ag.agent_name], dist_agents, other.agent_name, grid_size)

            # select actions
            actions: Dict[str, int] = {}
            for ag in agents:
                q_row = Qs[ag.agent_name][s_idx[ag.agent_name]]
                # Prey uses a random (stochastic) policy; predator learns with epsilon-greedy.
                if getattr(ag, "agent_type", "") == "prey":
                    actions[ag.agent_name] = int(rng.integers(0, n_actions))
                else:
                    actions[ag.agent_name] = epsilon_greedy_action(q_row, n_actions, rng, eps)

            # step environment
            mgp = env.step(actions)
            next_obs = mgp["obs"]
            rewards_from_env = mgp["reward"]

            # accumulate raw rewards
            for ag in agents:
                ep_agent_totals[ag.agent_name] += float(rewards_from_env.get(ag.agent_name, 0.0))

            # compute next-state indices
            s_next_idx: Dict[str, int] = {}
            next_pos_map: Dict[str, np.ndarray] = {}
            for ag in agents:
                local = next_obs.get(ag.agent_name, {}).get("local")
                if local is None:
                    local = getattr(ag, "_agent_location", np.array([0, 0]))
                next_pos_map[ag.agent_name] = np.asarray(local, dtype=int)

            for ag in agents:
                other = next(o for o in agents if o.agent_name != ag.agent_name)
                global_info_next = next_obs.get(ag.agent_name, {}).get("global", {})
                dist_agents_next = (global_info_next or {}).get("dist_agents", {})
                s_next_idx[ag.agent_name] = global_joint_state_index(next_pos_map[ag.agent_name], dist_agents_next, other.agent_name, grid_size)

            # IQL Q-table updates (per-agent)
            for ag in agents:
                s = s_idx[ag.agent_name]
                a = int(actions[ag.agent_name])
                r = float(rewards_from_env.get(ag.agent_name, 0.0))
                s2 = s_next_idx[ag.agent_name]
                qvals = Qs[ag.agent_name]
                td_target = r + gamma * float(np.max(qvals[s2]))
                td_error = td_target - qvals[s, a]
                qvals[s, a] += alpha * td_error

            # termination
            if mgp.get("terminated", False):
                break

            obs = next_obs

        # Episode finished: aggregate totals using heuristics on agent names
        ep_prey_total = sum(v for n, v in ep_agent_totals.items() if "prey" in n.lower() or n.lower().startswith("py"))
        ep_pred_total = sum(v for n, v in ep_agent_totals.items() if "predator" in n.lower() or n.lower().startswith("pd"))

        prey_episode_totals.append(ep_prey_total)
        predator_episode_totals.append(ep_pred_total)

        # decay epsilon
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)

        # running stats
        mean_prey = float(np.mean(prey_episode_totals)) if prey_episode_totals else 0.0
        mean_pred = float(np.mean(predator_episode_totals)) if predator_episode_totals else 0.0
        mean_diff = abs(mean_pred - mean_prey)

        # tensorboard
        writer.add_scalar("captures", getattr(env, "_captures_total", 0), ep)
        writer.add_scalar("mean/prey", mean_prey, ep)
        writer.add_scalar("mean/predator", mean_pred, ep)
        writer.add_scalar("mean_diff/pred_minus_prey", mean_diff, ep)

        if ep % 100 == 0:
            LOGGER.info(
                "Episode %d/%d | eps=%.4f | captures=%s | total_captures=%s | last_prey_total=%.3f | last_pred_total=%.3f | mean_prey=%.3f | mean_pred=%.3f",
                ep,
                episodes,
                getattr(env, "_captures_this_step", "N/A"),
                getattr(env, "_captures_total", "N/A"),
                ep_prey_total,
                ep_pred_total,
                mean_prey,
                mean_pred,
            )

        # periodic checkpointing
        if checkpoint_path and (ep % save_interval == 0):
            try:
                save_checkpoint(checkpoint_path, Qs, eps=eps, ep=ep, capture_count=capture_count, prey_totals=prey_episode_totals, pred_totals=predator_episode_totals)
                LOGGER.info("Saved checkpoint to %s at episode %d", checkpoint_path, ep)
            except Exception:
                LOGGER.exception("Checkpoint save failed at episode %d", ep)

        if ep % 10 == 0:
            writer.flush()

    # Training finished: final saves and summary
    save_qs_npz(save_path, Qs)
    LOGGER.info("Training finished. Total captures: %s", capture_count)

    if checkpoint_path:
        try:
            save_checkpoint(checkpoint_path, Qs, eps=eps, ep=episodes, capture_count=capture_count, prey_totals=prey_episode_totals, pred_totals=predator_episode_totals)
            LOGGER.info("Final checkpoint saved to %s", checkpoint_path)
        except Exception:
            LOGGER.exception("Final checkpoint save failed")

    mean_prey = float(np.mean(prey_episode_totals)) if prey_episode_totals else 0.0
    mean_pred = float(np.mean(predator_episode_totals)) if predator_episode_totals else 0.0
    mean_diff = mean_pred - mean_prey

    LOGGER.info("FINAL SUMMARY: Episodes run=%d", episodes)
    LOGGER.info("  Total prey reward (sum over episodes): %.3f", float(np.sum(prey_episode_totals)))
    LOGGER.info("  Total predator reward (sum over episodes): %.3f", float(np.sum(predator_episode_totals)))
    LOGGER.info("  Mean prey reward per episode: %.3f", mean_prey)
    LOGGER.info("  Mean predator reward per episode: %.3f", mean_pred)
    LOGGER.info("  Mean difference (predator - prey): %.3f", mean_diff)

    writer.flush()
    writer.close()


# ----------------------
# CLI
# ----------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train IQL (tabular) predator-prey")
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="baselines/IQL/iql_checkpoint.npz", help="Path to checkpoint (.npz) to save/load")
    p.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available")
    p.add_argument("--save-interval", type=int, default=1000, help="Episodes between checkpoint saves")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        save_interval=args.save_interval,
    )
