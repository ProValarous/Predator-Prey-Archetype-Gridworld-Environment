from typing import Any
import sys
import subprocess
import json
import time
import os
from typing import Dict, Tuple, Optional

import math
import numpy as np

from multi_agent_package.agents import Agent

# TODO : create training configuration function here and import in iql_train2v2.py

# ----------------------
# Checkpoint utilities
# ----------------------


def save_checkpoint(
    path: str,
    Qs: Dict[str, np.ndarray],
    eps: float,
    ep: int,
    capture_count: int,
    prey_totals: list,
    pred_totals: list,
) -> None:
    """Save Q-tables and lightweight metadata to an .npz checkpoint file.

    Qs are saved under keys prefixed with 'Q_'. Metadata are saved as small
    numpy arrays so np.load can read them back.
    """
    save_dict = {}
    for name, arr in Qs.items():
        save_dict[f"Q_{name}"] = arr

    # metadata
    save_dict["eps"] = np.array(eps)
    save_dict["ep"] = np.array(ep)
    save_dict["capture_count"] = np.array(capture_count)
    save_dict["prey_totals"] = np.array(prey_totals, dtype=float)
    save_dict["pred_totals"] = np.array(pred_totals, dtype=float)

    # ensure parent dir exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savez(path, **save_dict)
    print(f"[checkpoint] saved -> {path}")


def load_checkpoint(path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load checkpoint .npz and return (Qs, metadata_dict).

    Returns empty dicts if path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    Qs: Dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("Q_"):
            name = k[2:]
            Qs[name] = data[k]

    metadata = {}
    # safe reads with defaults
    metadata["eps"] = float(data["eps"].tolist()) if "eps" in data.files else None
    metadata["ep"] = int(data["ep"].tolist()) if "ep" in data.files else None
    metadata["capture_count"] = (
        int(data["capture_count"].tolist()) if "capture_count" in data.files else 0
    )
    metadata["prey_totals"] = (
        data["prey_totals"].tolist() if "prey_totals" in data.files else []
    )
    metadata["pred_totals"] = (
        data["pred_totals"].tolist() if "pred_totals" in data.files else []
    )

    print(f"[checkpoint] loaded <- {path} (episodes so far: {metadata.get('ep')})")
    return Qs, metadata


# ----------------------
# Environment/Agent utilities
# ----------------------


def make_agents() -> Tuple[Agent, Agent]:
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    """Encode (x,y) into a single integer state index (row-major)."""
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def global_joint_state_index(
    own_pos: np.ndarray,
    dist_agents: Dict[str, float],
    other_name: str,
    size: int,
) -> int:
    """
    Create a compact joint-state index using:
      - own position (ax,ay) encoded as a_idx = ax*size + ay
      - distance to other agent (from dist_agents[other_name]) discretized to integer bins

    Returns index in range [0, n_cells * (max_dist+1) - 1].
    """
    # own cell index
    a_idx = state_index(own_pos, size)

    # estimate maximum possible Euclidean distance on grid and number of bins
    max_dist = math.ceil(math.sqrt(2) * (size - 1))
    # read distance (fallback to max_dist if missing)
    d = None
    if dist_agents is not None:
        try:
            d = float(dist_agents.get(other_name, max_dist))
        except Exception:
            d = max_dist
    else:
        d = max_dist

    # discretize/clamp distance to integer bin [0..max_dist]
    dist_bin = int(min(max_dist, max(0, int(round(d)))))

    return a_idx * (max_dist + 1) + dist_bin


# ----------------------
# Training experiment folder functions
# ----------------------


def create_experiment_dir(
    base: str = "experiments", name: str | None = None, params: dict | None = None
) -> Tuple[str, str, str]:
    """Create timestamped experiment folder and return (exp_dir, checkpoints_dir, logs_dir)."""
    params = params or {}
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = (name or "run").strip().replace(" ", "_")
    exp_dir = os.path.join(base, f"{now}_{safe_name}")
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return exp_dir, checkpoints_dir, logs_dir


def write_experiment_md(exp_dir: str, params: dict) -> None:
    """Write a human-readable README.md describing the run"""
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
