# src/baselines/DQN/curve_recorder.py
"""
Per-episode training-curve CSV writer.

Owns the file handle and the column layout so the training loop can stay
pure RL logic: one row per episode with each agent's total reward and mean
loss, plus the episode number and (post-decay) epsilon. Column order is
[episode, epsilon, <agent>_reward..., <agent>_loss...]; a warmup episode
with no optimizer steps writes an empty loss cell.
"""

import csv
import os
from typing import List


class CurveRecorder:
    def __init__(self, path: str, agent_ids: List[str]):
        self.agent_ids = list(agent_ids)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        reward_cols = [f"{aid}_reward" for aid in self.agent_ids]
        loss_cols = [f"{aid}_loss" for aid in self.agent_ids]
        self._writer = csv.DictWriter(
            self._file, fieldnames=["episode", "epsilon"] + reward_cols + loss_cols
        )
        self._writer.writeheader()

    def record(
        self, episode: int, epsilon: float, episode_rewards: dict, episode_losses: dict
    ) -> None:
        """Write one episode's row. `episode` is 1-based; `epsilon` is the
        value in effect after this episode's decay."""
        row: dict = {"episode": episode, "epsilon": round(epsilon, 4)}
        for aid in self.agent_ids:
            row[f"{aid}_reward"] = round(episode_rewards[aid], 4)
            losses = episode_losses[aid]
            row[f"{aid}_loss"] = round(sum(losses) / len(losses), 6) if losses else ""
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()
