"""
DQN Utility Functions.

This module contains utilities specific to the DQN baseline.
It is intentionally minimal — only functions that are
directly used by dqn.py live here.

Contents:
    - Experiment folder creation
    - Experiment README writing
    - Checkpoint save / load (PyTorch .pt format)

Note: checkpoint utilities here are PyTorch-based (torch.save /
torch.load) unlike the IQL utils which use numpy .npz format.
This is because DQN saves neural network weights, not Q-tables.
"""

import logging
import os
import time
from typing import Dict, List, Tuple

import torch

LOGGER = logging.getLogger("dqn.utils")


# ── Experiment folder ─────────────────────────────────────────────────────────

def create_experiment_dir(
    base: str,
    name: str = "dqn_run",
) -> Tuple[str, str, str]:
    """
    Create a timestamped experiment folder with checkpoints and logs
    subdirectories. Mirrors the IQL utils function exactly.

    Parameters
    ----------
    base : str
        Root directory under which the experiment folder is created.
        Typically the DQN save_path from config.
    name : str
        Human-readable name appended to the timestamp.

    Returns
    -------
    exp_dir : str
        Path to the experiment root folder.
    checkpoints_dir : str
        Path to the checkpoints subfolder.
    logs_dir : str
        Path to the logs subfolder (used by TensorBoard).

    Examples
    --------
    >>> exp_dir, ckpt_dir, log_dir = create_experiment_dir(
    ...     base="baselines/DQN", name="dqn_run"
    ... )
    """
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = name.strip().replace(" ", "_")
    exp_dir = os.path.join(base, f"{now}_{safe_name}")
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    logs_dir = os.path.join(exp_dir, "logs")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return exp_dir, checkpoints_dir, logs_dir


def write_experiment_md(exp_dir: str, params: dict) -> None:
    """
    Write a human-readable README.md into the experiment folder
    describing the run configuration. Mirrors the IQL utils function.

    Parameters
    ----------
    exp_dir : str
        Path to the experiment root folder.
    params : dict
        Key-value pairs to document. A "command" key is rendered
        inside a code block; all other keys are rendered as a list.
    """
    md_path = os.path.join(exp_dir, "README.md")
    lines = [
        f"# Experiment: {os.path.basename(exp_dir)}",
        "",
        f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Command",
        "```",
        params.get("command", ""),
        "```",
        "",
        "## Parameters",
    ]
    for k, v in params.items():
        if k == "command":
            continue
        lines.append(f"- **{k}**: `{v}`")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    LOGGER.info("Experiment README written -> %s", md_path)


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    algorithm_state: dict,
    ep: int,
    capture_count: int,
    per_agent_rewards: Dict[str, List[float]],
) -> None:
    """
    Save DQN training state to a .pt file using torch.save.

    Saves:
        - All agent network weights (online + target)
        - All optimiser states
        - Current epsilon
        - Episode number
        - Capture count
        - Per-agent reward history

    Parameters
    ----------
    path : str
        Full file path to save to. Parent directories are
        created automatically.
    algorithm_state : dict
        Output of DQN.state_dict() — contains agent weights
        and current epsilon.
    ep : int
        Current episode number.
    capture_count : int
        Total captures accumulated so far.
    per_agent_rewards : dict
        {agent_name: [reward_per_episode]} full history.

    Examples
    --------
    >>> save_checkpoint(
    ...     path="experiments/dqn_run/checkpoints/dqn_checkpoint.pt",
    ...     algorithm_state=algorithm.state_dict(),
    ...     ep=1000,
    ...     capture_count=42,
    ...     per_agent_rewards={"predator_1": [0.1, 0.3, ...]},
    ... )
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save({
        "algorithm_state":   algorithm_state,
        "episode":           ep,
        "capture_count":     capture_count,
        "per_agent_rewards": per_agent_rewards,
    }, path)

    LOGGER.info("Checkpoint saved -> %s (episode %d)", path, ep)


def load_checkpoint(path: str) -> dict:
    """
    Load a DQN checkpoint from a .pt file.

    Returns the raw checkpoint dict. The caller is responsible
    for restoring state into the algorithm:

        ckpt = load_checkpoint(path)
        algorithm.load_state_dict(ckpt["algorithm_state"])
        start_ep = ckpt["episode"] + 1

    Parameters
    ----------
    path : str
        Path to the .pt checkpoint file.

    Returns
    -------
    dict
        Keys:
            algorithm_state   : dict  — pass to algorithm.load_state_dict()
            episode           : int   — last completed episode
            capture_count     : int   — total captures at checkpoint
            per_agent_rewards : dict  — full reward history

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.

    Examples
    --------
    >>> ckpt = load_checkpoint("experiments/dqn_run/checkpoints/dqn_checkpoint.pt")
    >>> algorithm.load_state_dict(ckpt["algorithm_state"])
    >>> start_ep = ckpt["episode"] + 1
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # map_location="cpu" ensures checkpoints saved on GPU
    # can be loaded on a CPU machine without errors
    checkpoint = torch.load(path, map_location="cpu")

    LOGGER.info(
        "Checkpoint loaded <- %s (episode %d)",
        path, checkpoint.get("episode", "?"),
    )
    return checkpoint