"""
DQN Test / Evaluation Script.

Loads a saved .pt checkpoint and runs the trained agents
in greedy mode (epsilon=0) with environment rendering.

Mirrors test_iql2v2.py in structure and CLI style.

The script reconstructs the DuelingMLP networks from the
checkpoint, then runs episodes using purely greedy action
selection — no exploration.

Usage
-----
cd src
python -m baselines.DQN.test_dqn --checkpoint path/to/dqn_checkpoint.pt \\
                                  --size 8 --episodes 5

Notes
-----
- Checkpoint must have been saved by dqn_train.py
- Network architecture flags (use_dueling, hidden_units) must
  match what was used during training — these are saved in the
  checkpoint and restored automatically
- Prey and predator counts must match the trained checkpoint
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.core.agent import Agent

from baselines.DQN.dqn import DQN

LOGGER = logging.getLogger("test_dqn")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


# ── Agent construction ────────────────────────────────────────────────────────

def make_agents(num_predators: int = 2, num_preys: int = 2) -> List[Agent]:
    """Create agents with same naming convention as dqn_train.py."""
    agents = []
    for i in range(1, num_preys + 1):
        agents.append(Agent(
            agent_name=f"prey_{i}",
            agent_team=i,
            agent_type="prey",
        ))
    for i in range(1, num_predators + 1):
        agents.append(Agent(
            agent_name=f"predator_{i}",
            agent_team=i,
            agent_type="predator",
        ))
    return agents


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    """
    Load a .pt checkpoint saved by dqn_train.py.

    Parameters
    ----------
    path : str
        Path to the .pt file.

    Returns
    -------
    dict
        Full checkpoint dict with algorithm_state, episode, etc.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    LOGGER.info(
        "Loaded checkpoint from %s (trained for %d episodes)",
        path, checkpoint.get("episode", "?"),
    )
    return checkpoint


# ── Greedy action selection ───────────────────────────────────────────────────

def select_greedy_actions(algorithm: DQN, observations: dict) -> dict:
    """
    Select actions greedily — epsilon=0, pure exploitation.

    Uses the algorithm's select_actions() with epsilon
    forced to 0.0 for the duration of this call.

    Parameters
    ----------
    algorithm : DQN
        Loaded and restored DQN instance.
    observations : dict
        {agent_name: obs_dict} from env.

    Returns
    -------
    dict
        {agent_name: action_int}
    """
    original_epsilon = algorithm.epsilon
    algorithm.epsilon = 0.0
    actions = algorithm.select_actions(observations)
    algorithm.epsilon = original_epsilon
    return actions


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_test(
    checkpoint_path: str,
    grid_size: int       = 8,
    num_predators: int   = 2,
    num_preys: int       = 2,
    episodes: int        = 5,
    max_steps: int       = 250,
    pause: float         = 0.05,
    hidden_units: Optional[List[int]] = None,
    use_double: bool     = True,
    use_dueling: bool    = True,
) -> None:
    """
    Load a trained DQN checkpoint and run greedy evaluation episodes.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint from dqn_train.py.
    grid_size : int
        Must match the grid size used during training.
    num_predators : int
        Must match the number used during training.
    num_preys : int
        Must match the number used during training.
    episodes : int
        Number of evaluation episodes to run.
    max_steps : int
        Maximum steps per episode.
    pause : float
        Seconds to wait between steps for rendering visibility.
    hidden_units : list of int, optional
        Must match what was used during training. Defaults to [128, 128].
    use_double : bool
        Must match training flag.
    use_dueling : bool
        Must match training flag.
    """
    if hidden_units is None:
        hidden_units = [128, 128]

    # ── Load checkpoint ───────────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)

    # ── Build environment with rendering ──────────────────────────────────
    agents = make_agents(num_predators=num_predators, num_preys=num_preys)
    agent_names = [ag.agent_name for ag in agents]

    env = GridWorldEnv(
        agents=agents,
        render_mode="human",
        size=grid_size,
        perc_num_obstacle=10,
    )

    # ── Reconstruct algorithm with same config as training ────────────────
    # epsilon=0 from the start — pure greedy evaluation
    state_dim    = checkpoint.get("state_dim", None)
    action_dim   = checkpoint.get("action_dim", 5)
    hidden_units = checkpoint.get("hidden_units", hidden_units)
    use_double   = checkpoint.get("use_double",   use_double)
    use_dueling  = checkpoint.get("use_dueling",  use_dueling)

    config = {
        "hidden_units": hidden_units,
        "use_double":   use_double,
        "use_dueling":  use_dueling,
        "epsilon":      0.0,
        "min_epsilon":  0.0,
        "episodes":     episodes,
        "max_steps":    max_steps,
        "state_dim":    state_dim,
        "action_dim":   action_dim,
    }
    algorithm = DQN(env, config)

    # ── Restore trained weights ───────────────────────────────────────────
    algorithm.load_state_dict(checkpoint["algorithm_state"])
    LOGGER.info("Restored weights for agents: %s", agent_names)

    # Set all agents' networks to eval mode
    for agent in algorithm.agents.values():
        agent.online_net.eval()
        agent.target_net.eval()

    # ── Evaluation loop ───────────────────────────────────────────────────
    episode_rewards: Dict[str, List[float]] = {n: [] for n in agent_names}
    capture_count = 0

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            ep_reward = {n: 0.0 for n in agent_names}
            LOGGER.info("Evaluation episode %d / %d", ep, episodes)

            for t in range(1, max_steps + 1):

                # Pure greedy — epsilon forced to 0
                actions = select_greedy_actions(algorithm, obs)

                step_out   = env.step(actions)
                next_obs   = step_out["obs"]
                rewards    = step_out["reward"]
                terminated = step_out["terminated"]
                trunc      = step_out.get("trunc", False)

                # Accumulate rewards
                for name in agent_names:
                    ep_reward[name] += float(rewards[name])

                print(f"  Step {t:3d} | rewards: { {n: f'{rewards[n]:.2f}' for n in agent_names} }")

                time.sleep(pause)

                if terminated or trunc:
                    capture_count += 1
                    LOGGER.info(
                        "  Capture at step %d (episode %d)", t, ep
                    )
                    break

                obs = next_obs

            # Episode summary
            for name in agent_names:
                episode_rewards[name].append(ep_reward[name])

            reward_str = ", ".join(
                f"{n}: {ep_reward[n]:.2f}" for n in agent_names
            )
            LOGGER.info("Episode %d done | %s", ep, reward_str)
            time.sleep(0.3)

    finally:
        try:
            env.close()
        except Exception:
            LOGGER.debug("Failed to close env cleanly", exc_info=True)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes run   : {episodes}")
    print(f"Total captures : {capture_count}")
    for name in agent_names:
        mean_r = np.mean(episode_rewards[name]) if episode_rewards[name] else 0.0
        print(f"Mean reward [{name:>12s}] : {mean_r:.3f}")
    print("="*50)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate trained DQN agents")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to .pt checkpoint from dqn_train.py")
    p.add_argument("--size",        type=int, default=10,
                   help="Grid size (must match training)")
    p.add_argument("--predators",   type=int, default=2)
    p.add_argument("--preys",       type=int, default=2)
    p.add_argument("--episodes",    type=int, default=5)
    p.add_argument("--max-steps",   type=int, default=250)
    p.add_argument("--pause",       type=float, default=0.05,
                   help="Seconds between rendered steps")
    p.add_argument("--no-double",   action="store_true",
                   help="Must match training flag")
    p.add_argument("--no-dueling",  action="store_true",
                   help="Must match training flag")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_test(
        checkpoint_path=args.checkpoint,
        grid_size=args.size,
        num_predators=args.predators,
        num_preys=args.preys,
        episodes=args.episodes,
        max_steps=args.max_steps,
        pause=args.pause,
        use_double=not args.no_double,
        use_dueling=not args.no_dueling,
    )