# Guide: Writing a Custom Reward Function

Full working example from file to config-driven.

---

## Step 1 — Create the file

```
src/multi_agent_package/rewards/my_reward.py
```

```python
from multi_agent_package.rewards.base import RewardFunction


class TeamProximityReward(RewardFunction):
    """
    Rewards predators that are close to each other (teamwork shaping).
    
    Params (from YAML):
        weight: float — scales the entire reward (inherited from RewardFunction)
    
    Logic: predators closer than `radius` cells to a teammate get +weight;
           all other agents get 0.
    """

    def compute(self, env) -> dict:
        rewards = {ag.agent_name: 0.0 for ag in env.agents}

        predators = [a for a in env.agents if a.agent_type.startswith("predator")]

        for pred in predators:
            px, py = pred._agent_location
            for other in predators:
                if other.agent_name == pred.agent_name:
                    continue
                ox, oy = other._agent_location
                dist = abs(px - ox) + abs(py - oy)
                if dist <= 3:
                    rewards[pred.agent_name] += self.weight

        return rewards
```

**Rules:**
- Always initialize all agent names to `0.0` before computing — never leave an agent out
- `self.weight` is set by the registry from your YAML `weight:` entry
- Never call `env.step()` or modify agent positions
- Return `Dict[str, float]` — one entry per agent

---

## Step 2 — Register it

Open `src/multi_agent_package/registry/reward_registry.py`:

```python
from multi_agent_package.rewards.my_reward import TeamProximityReward

_REWARD_REGISTRY = {
    "base":               BaseReward,
    "predator_distance":  PredatorDistanceReward,
    "survival":           SurvivalReward,
    "team_proximity":     TeamProximityReward,   # ← add this line
}
```

---

## Step 3 — Configure it

```yaml
# configs/rewards.yaml
rewards:
  base:
    enabled: true

  shaping:
    - name: predator_distance
      weight: 0.5
    - name: team_proximity       # ← add your reward
      weight: 1.0
```

> **Note:** `applies_to:` is accepted in YAML but currently ignored — all reward functions apply to all agents. If you only want your reward to affect certain agents, filter by agent type inside `compute()`.

---

## Step 4 — Test it

```python
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.rewards.my_reward import TeamProximityReward
import numpy as np

agents = [
    Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
    Agent(agent_type="predator", agent_team="predator_2", agent_name="pred_2"),
    Agent(agent_type="prey",     agent_team="prey_1",     agent_name="prey_1"),
]
env = GridWorldEnv(agents=agents, size=8, perc_num_obstacle=0, render_mode=None, seed=0)
env.reset()

reward_fn = TeamProximityReward(weight=1.0)

# Place predators adjacent — should get reward
env.agents[0]._agent_location = np.array([3, 3], dtype=np.int32)
env.agents[1]._agent_location = np.array([3, 4], dtype=np.int32)  # 1 cell away

rewards = reward_fn.compute(env)
assert set(rewards.keys()) == {"pred_1", "pred_2", "prey_1"}
assert rewards["pred_1"] > 0
assert rewards["prey_1"] == 0.0

# Place predators far apart — no reward
env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
env.agents[1]._agent_location = np.array([7, 7], dtype=np.int32)
rewards = reward_fn.compute(env)
assert rewards["pred_1"] == 0.0

print("All assertions passed.")
```

---

## How combined rewards work

`run_from_config.py` stacks all enabled reward functions into a closure:

```python
def combined_reward(env):
    total = {ag.agent_name: 0.0 for ag in env.agents}
    for rf in reward_fns:          # [BaseReward, TeamProximityReward, ...]
        for k, v in rf.compute(env).items():
            total[k] += v
    return total
```

Your reward's output is **added** to the base reward, not replacing it. If you want your reward to stand alone (e.g., for ablation), disable `base: enabled: false` in `rewards.yaml`.

---

## Checklist

- [ ] Inherits from `RewardFunction`
- [ ] `compute(env)` initializes all agents to `0.0` before adding
- [ ] Returns `Dict[str, float]` — one key per agent
- [ ] Uses `self.weight` to scale output
- [ ] No writes to env or agent state
- [ ] Registered in `reward_registry.py` with matching YAML key
- [ ] Agent-type filtering (if needed) done inside `compute()`, not via `applies_to` in YAML
