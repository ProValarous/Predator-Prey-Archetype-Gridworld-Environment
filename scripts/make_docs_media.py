#!/usr/bin/env python
"""
Generate the images used by the documentation site.

Everything here runs **offscreen** (matplotlib's Agg backend) with fixed seeds,
so it is deterministic and works headless (CI, no display). It trains a small
tabular Q-learning **predator** (the prey moves randomly) on a tiny grid, then
writes two assets into ``docs/assets/images/``:

  * ``learning_curve.png``    - predator return and episode length vs. training
  * ``episode_trained.gif``   - one greedy episode of the trained predator

A predator-only learner chasing a random-moving prey is used deliberately: it is
a clean, reliably-learnable pursuit task that shows a clear learning signal
(episode length drops as the predator improves), unlike a symmetric setup where
an equally-fast evading prey can avoid capture indefinitely.

Run from the repository root (needs ``pip install -e ".[docs]"``)::

    python scripts/make_docs_media.py

The generated files are committed so the docs build without regenerating them;
re-run this script whenever the environment's dynamics change.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # offscreen; must be set before pyplot import

from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

from multi_agent_package.core.agent import Agent  # noqa: E402
from multi_agent_package.core.gridworld import GridWorldEnv  # noqa: E402
from multi_agent_package.rewards.base_reward import BaseReward  # noqa: E402
from multi_agent_package.rewards.predator_distance import (  # noqa: E402
    PredatorDistanceReward,
)

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "images"
GRID_SIZE = 6
MAX_STEPS = 60
SEED = 7
N_ACTIONS = 5
PRED = "predator_1"
PREY = "prey_1"

PRED_COLOR = "#d62728"
PREY_COLOR = "#2ca02c"
OBSTACLE_COLOR = "#444444"


def make_env(seed: int) -> GridWorldEnv:
    """Small 1v1 env with base reward + predator-distance shaping attached."""
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name=PRED),
        Agent(agent_type="prey", agent_team="prey_1", agent_name=PREY),
    ]
    env = GridWorldEnv(
        agents=agents,
        size=GRID_SIZE,
        perc_num_obstacle=5,
        render_mode=None,
        seed=seed,
        max_steps=MAX_STEPS,
    )
    base = BaseReward(weight=1.0)
    shaping = PredatorDistanceReward(weight=0.3)

    def reward_fn(e):
        total = base.compute(e)
        for k, v in shaping.compute(e).items():
            total[k] += v
        return total

    env.reward_fn = reward_fn
    return env


def _state(env: GridWorldEnv) -> tuple:
    """Fully-observable tabular state: (pred_x, pred_y, prey_x, prey_y)."""
    pred = next(a for a in env.agents if a.agent_type.startswith("predator"))
    prey = next(a for a in env.agents if a.agent_type.startswith("prey"))
    return (
        int(pred._agent_location[0]),
        int(pred._agent_location[1]),
        int(prey._agent_location[0]),
        int(prey._agent_location[1]),
    )


def _rollout(env, Q, rng, greedy_eps):
    """Run one episode; predator uses Q (ε=greedy_eps), prey moves randomly.
    Returns (predator_return, episode_length)."""
    env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        s = _state(env)
        if rng.random() < greedy_eps:
            a_pred = int(rng.integers(N_ACTIONS))
        else:
            a_pred = int(np.argmax(Q[s]))
        a_prey = int(rng.integers(N_ACTIONS))
        out = env.step({PRED: a_pred, PREY: a_prey})
        total_r += out["reward"][PRED]
        steps += 1
        done = out["terminated"] or out["truncated"]
    return total_r, steps


def train_and_measure(seed: int):
    """Predator-only tabular Q-learning against a random prey."""
    env = make_env(seed)
    rng = np.random.default_rng(seed)
    Q: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
    alpha, gamma = 0.3, 0.95
    eps, eps_decay, eps_min = 1.0, 0.9985, 0.05

    chunk, n_chunks = 100, 30  # 3000 training episodes total
    xs, returns, lengths = [], [], []
    trained = 0
    for _ in range(n_chunks):
        for _ in range(chunk):
            env.reset()
            done = False
            while not done:
                s = _state(env)
                a_pred = (
                    int(rng.integers(N_ACTIONS))
                    if rng.random() < eps
                    else int(np.argmax(Q[s]))
                )
                a_prey = int(rng.integers(N_ACTIONS))
                out = env.step({PRED: a_pred, PREY: a_prey})
                r = out["reward"][PRED]
                terminal = out["terminated"]
                done = terminal or out["truncated"]
                s2 = _state(env)
                q_next = 0.0 if terminal else float(np.max(Q[s2]))
                Q[s][a_pred] += alpha * (r + gamma * q_next - Q[s][a_pred])
            eps = max(eps_min, eps * eps_decay)
        trained += chunk
        # greedy measurement over a few fresh episodes
        eval_rng = np.random.default_rng(1000 + trained)
        rs, ls = zip(*[_rollout(env, Q, eval_rng, 0.0) for _ in range(10)])
        xs.append(trained)
        returns.append(float(np.mean(rs)))
        lengths.append(float(np.mean(ls)))
    return Q, env, xs, returns, lengths


def _draw(env: GridWorldEnv, ax) -> None:
    n = env.size
    ax.clear()
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # y increases downward, matching the env convention
    for obs in env._obstacle_location:
        ax.add_patch(
            plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color=OBSTACLE_COLOR)
        )
    for ag in env.agents:
        x, y = ag._agent_location
        color = PRED_COLOR if ag.agent_type.startswith("predator") else PREY_COLOR
        ax.add_patch(plt.Circle((x, y), 0.32, color=color))
        ax.text(
            x, y, ag.agent_name[0].upper(), ha="center", va="center",
            color="white", fontsize=9, fontweight="bold",
        )


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_learning_curve(xs, returns, lengths, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(xs, returns, color=PRED_COLOR)
    ax1.set_title("Predator return (greedy)")
    ax1.set_xlabel("training episodes")
    ax1.set_ylabel("mean return")
    ax1.grid(True, alpha=0.3)
    ax2.plot(xs, lengths, color="#1f77b4")
    ax2.set_title("Episode length (steps to capture)")
    ax2.set_xlabel("training episodes")
    ax2.set_ylabel("mean steps")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_episode_gif(Q, env: GridWorldEnv, path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    env.reset()
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    _draw(env, ax)
    frames = [_grab(fig)]
    done = False
    while not done:
        s = _state(env)
        a_pred = int(np.argmax(Q[s]))
        a_prey = int(rng.integers(N_ACTIONS))
        out = env.step({PRED: a_pred, PREY: a_prey})
        done = out["terminated"] or out["truncated"]
        _draw(env, ax)
        frames.append(_grab(fig))
    plt.close(fig)
    frames.extend([frames[-1]] * 6)  # hold the final (capture) frame
    imageio.mimsave(path, frames, fps=4, loop=0)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"Training tabular Q-learning predator (seed={SEED}) ...")
    Q, env, xs, returns, lengths = train_and_measure(SEED)

    curve_path = ASSETS / "learning_curve.png"
    write_learning_curve(xs, returns, lengths, curve_path)
    print(f"wrote {curve_path}")

    # pick a demo seed that produces a capture with the trained policy
    gif_path = ASSETS / "episode_trained.gif"
    write_episode_gif(Q, env, gif_path, seed=3)
    print(f"wrote {gif_path}")

    print(f"done. final greedy return={returns[-1]:.1f}, length={lengths[-1]:.1f}")


if __name__ == "__main__":
    main()
