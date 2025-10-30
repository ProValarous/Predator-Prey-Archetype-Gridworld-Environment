"""Updated multi-agent tester for tabular IQL Q-tables.

Matches the coding style used by the trainer script in this project:
- logging setup function
- argparse CLI with mutually exclusive Q-table inputs
- simple, imperative control flow

Features added compared to earlier tester:
- Accepts a single combined .npz, multiple per-agent .npz files,
    or a directory (scanned recursively).
- If a per-agent file is stored inside a per-agent folder
    (e.g. ``predator_1/iql_q_table.npz``), the loader will infer the
    agent name from the parent folder when necessary.
- Falls back to random actions for agents missing a Q-table and
    logs a warning.

Usage examples:
    cd src
    python -m baselines.IQL.test_iql2v2.py --file baselines/IQL/all_qs.npz \\
            --size 6 --preys 2 --predators 2
    python -m baselines.IQL.test_iql2v2.py --q-files \\
            baselines/IQL/predator_1/iql_q_table.npz \\
            baselines/IQL/predator_2/iql_q_table.npz --size 6
    python -m baselines.IQL.test_iql2v2.py --q-dir baselines/IQL --size 6
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from multi_agent_package.agents import Agent
from multi_agent_package.gridworld import GridWorldEnv

LOGGER = logging.getLogger("test_iql")


def setup_logging(level: int = logging.INFO) -> None:
    """Random docstring"""
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


def make_agents(num_preys: int = 2, num_predators: int = 2) -> Dict[str, Agent]:
    """Random docstring"""
    agents: Dict[str, Agent] = {}
    for i in range(1, num_preys + 1):
        name = f"prey_{i}"
        agents[name] = Agent(agent_name=name, agent_team=i, agent_type="prey")
    for i in range(1, num_predators + 1):
        name = f"predator_{i}"
        agents[name] = Agent(agent_name=name, agent_team=i, agent_type="predator")
    return agents


def joint_state_index(positions: List[Tuple[int, int]], grid_size: int) -> int:
    """Encode a list of (x,y) positions into a single integer index.

    Each agent's cell index = x * grid_size + y. The combined index treats
    those cell indices as digits in base `n_cells`.
    """
    n_cells = grid_size * grid_size
    idx = 0
    for cell_pos in positions:
        cell_index = int(cell_pos[0]) * grid_size + int(cell_pos[1])
        idx = idx * n_cells + cell_index
    return int(idx)


# ----------------- Q loading helpers -----------------


def _extract_agent_name_from_filename(path: Path) -> str:
    """
    Heuristic: extract agent name (e.g. 'prey_1')
    from filename stem or parent folder.
    """
    stem = path.stem.lower()
    for base in ("prey", "predator"):
        if base in stem:
            m = re.search(rf"({base}[_\-]?\d+)", stem)
            if m:
                return m.group(1)
            return base
    # fallback to parent folder name if it looks like an agent id
    parent = path.parent.name.lower()
    if any(x in parent for x in ("prey", "predator")):
        return parent
    return stem


def _load_qs_from_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load any 2D arrays from a .npz file and return mapping name->array.

    If arrays inside the file are keyed with agent-names, those keys are used.
    Otherwise the caller can infer a name from filename / parent folder.
    """
    qs: Dict[str, np.ndarray] = {}
    with np.load(str(path), allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            if arr.ndim == 3:
                name = key[2:] if key.startswith("Q_") else key
                print(name)
                qs[name] = arr.astype(np.float32)
    return qs


def load_q_tables(
    single_file: Optional[str] = None,
    files: Optional[List[str]] = None,
    directory: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load Q-tables from a single .npz, a list of per-agent files, or a dir.

    Directory mode searches recursively and will infer agent names from file
    content or parent folder.
    """
    qs: Dict[str, np.ndarray] = {}

    # 1) single combined .npz
    if single_file:
        p = Path(single_file)
        if not p.exists():
            raise FileNotFoundError(single_file)
        loaded = _load_qs_from_npz(p)
        if loaded:
            qs.update(loaded)

    # 2) explicit per-agent files
    if files:
        for f in files:
            p = Path(f)
            if not p.exists():
                LOGGER.warning("Q-file not found: %s (skipping)", f)
                continue

            try:
                loaded = _load_qs_from_npz(p)
            except Exception as exc:
                LOGGER.warning("Failed loading %s: %s (skipping)", p, exc)
                continue

            if loaded:
                qs.update(loaded)
                continue

            # map first 2D array to name derived from filename or parent folder
            with np.load(str(p), allow_pickle=False) as data:
                found = False
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 2:
                        agent_name = _extract_agent_name_from_filename(p)
                        qs[agent_name] = arr.astype(np.float32)
                        found = True
                        break
                if not found:
                    LOGGER.warning("No 2D array found in %s", f)

    # 3) directory: recursive scan
    if directory:
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(directory)

        for p in sorted(d.rglob("*.npz")):
            try:
                loaded = _load_qs_from_npz(p)
            except Exception as exc:
                LOGGER.warning("Failed loading %s: %s (skipping)", p, exc)
                continue

            if loaded:
                qs.update(loaded)
                continue

            # fallback: map first 2D array to name derived from filename/parent
            with np.load(str(p), allow_pickle=False) as data:
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 2:
                        agent_name = _extract_agent_name_from_filename(p)
                        qs[agent_name] = arr.astype(np.float32)
                        break

    if not qs:
        raise RuntimeError("No Q-tables loaded from provided inputs.")

    LOGGER.info("Loaded Q-tables for agents: %s", list(qs.keys()))
    return qs


# ---------------- action selection -----------------


def choose_greedy_action_from_q(
    q_table: np.ndarray, s_idx: int, rng: np.random.Generator
) -> int:
    """Random docstring"""

    if s_idx < 0 or s_idx >= q_table.shape[0]:
        LOGGER.warning(
            "State index %d out of range for q_table with shape %s; "
            "choosing random action",
            s_idx,
            q_table.shape,
        )
        return int(rng.integers(0, q_table.shape[1]))
    row = q_table[s_idx]
    best = float(np.max(row))
    best_actions = [int(i) for i, v in enumerate(row) if np.isclose(v, best)]
    return int(rng.choice(best_actions))


# ---------------- runner -----------------


def run_test(
    q_file: Optional[str] = None,
    q_files: Optional[List[str]] = None,
    q_dir: Optional[str] = None,
    size: int = 6,
    preys: int = 2,
    predators: int = 2,
    episodes: int = 3,
    max_steps: int = 250,
    pause: float = 0.05,
) -> None:
    """Random docstring"""

    # load Qs (prefer explicit files > dir > single file)
    qs = load_q_tables(single_file=q_file, files=q_files, directory=q_dir)

    agent_map = make_agents(num_preys=preys, num_predators=predators)
    agent_names = list(agent_map.keys())
    agents = list(agent_map.values())  # env still expects a list

    env = GridWorldEnv(
        agents=agents, render_mode="human", size=size, perc_num_obstacle=10
    )
    rng = np.random.default_rng(0)

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            LOGGER.info("Test episode %d/%d", ep, episodes)

            for t in range(1, max_steps + 1):
                positions = [tuple(obs[name]["local"]) for name in agent_names]
                s_idx = joint_state_index(positions, size)

                actions: Dict[str, int] = {}

                for ag in agents:
                    q_table = qs.get(ag.agent_name)
                    if q_table is None:
                        # try alternate keys (case-insensitive)
                        for k in list(qs.keys()):
                            k_lower = k.lower()
                            name_lower = ag.agent_name.lower()
                            ends_with = k_lower.endswith(name_lower)
                            starts_with = k_lower.startswith(name_lower)
                            if ends_with or starts_with:
                                q_table = qs[k]
                                break

                    if q_table is None:
                        LOGGER.debug(
                            "No Q-table for %s: acting randomly", ag.agent_name
                        )
                        actions[ag.agent_name] = int(
                            rng.integers(0, env.action_space.n)
                        )
                    else:
                        actions[ag.agent_name] = choose_greedy_action_from_q(
                            q_table, s_idx, rng
                        )

                mgp = env.step(actions)
                obs = mgp["obs"]

                print(mgp.get("reward", {}))

                time.sleep(pause)

                if mgp.get("terminated", False):
                    print(f"Capture at step {t} (episode {ep})")
                    break

            time.sleep(0.25)

    finally:
        try:
            env.close()
        except Exception:
            LOGGER.debug("Failed to close env cleanly", exc_info=True)


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    """Random docstring"""

    p = argparse.ArgumentParser("Test multi-agent IQL-trained Q-tables")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Single .npz containing Q-tables for all agents",
    )
    group.add_argument(
        "--q-files",
        type=str,
        nargs="+",
        help="List of per-agent .npz files (order not important)",
    )
    group.add_argument(
        "--q-dir",
        type=str,
        help="Directory containing per-agent .npz files (searched recursively)",
    )
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--preys", type=int, default=2)
    p.add_argument("--predators", type=int, default=2)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_test(
        q_file=args.file,
        q_files=args.q_files,
        q_dir=args.q_dir,
        size=args.size,
        preys=args.preys,
        predators=args.predators,
        episodes=args.episodes,
        max_steps=args.max_steps,
        pause=args.pause,
    )
