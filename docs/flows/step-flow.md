# Flow: Per-Step Pipeline

One call to `env.step(actions)` — what happens inside.

---

## Inputs / Outputs

```
Input:  actions: Dict[str, int]
        e.g. {"predator_1": 0, "predator_2": 3, "prey_1": 2}

Output: dict
        {
            "obs":        Dict[str, dict],   ← per-agent observation
            "reward":     Dict[str, float],  ← per-agent reward
            "terminated": bool,              ← _captures_total >= capture_threshold
            "truncated":  bool,              ← _episode_steps >= max_steps
            "info":       dict,              ← per-agent metadata
        }
```

---

## Pipeline

```
env.step(actions)
│
├─ 1. RESET per-step trackers
│     _captured_this_step = []
│     _capturing_predators = set()
│     obstacle_set = {tuple(o) for o in _obstacle_location}   ← built once
│     already_captured = set(_captured_agents)                 ← snapshot
│
├─ 2. MOVE — for each agent in env.agents:
│     │
│     ├── if agent.agent_name in already_captured → skip (frozen)
│     │
│     ├── action = actions.get(agent.agent_name, 4)  ← default noop
│     │
│     ├── direction = action_space_plugin.to_direction(action)
│     │   #   if action_space_plugin is None → fallback to ag._actions_to_directions[action]
│     │   # DiscreteActionSpace: {0:[+1,0], 1:[0,+1], 2:[-1,0], 3:[0,-1], 4:[0,0]}
│     │   # NOTE: Y axis increases downward in this coordinate system.
│     │   #   action 1 ([0,+1]) moves toward higher Y → visually downward
│     │   #   action 3 ([0,-1]) moves toward lower Y  → visually upward
│     │
│     ├── candidate = clip(agent._agent_location + direction, 0, size-1)
│     │
│     ├── if block_agents_by_obstacles AND candidate in obstacle_set:
│     │     candidate = agent._agent_location  ← stay
│     │
│     └── agent._agent_location = candidate   ← applied immediately
│
├─ 3. CAPTURE DETECTION
│     position_map: Dict[tuple, List[Agent]] = group non-captured agents by position
│     │
│     for each cell with ≥ 2 agents:
│       predators_here = [a if a.type starts with "predator"]
│       prey_here      = [a if a.type starts with "prey"]
│       if predators_here and prey_here:
│         for prey in prey_here:
│           _captured_this_step.append(prey.name)
│         for predator in predators_here:
│           _capturing_predators.add(predator.name)
│
│     _captured_agents.extend(_captured_this_step)   ← persists across steps
│     _captures_total += len(_captured_this_step)
│     _episode_steps  += 1
│
├─ 4. REWARDS
│     rewards = base_reward()           ← uses _captured_this_step, _capturing_predators
│     if reward_fn: rewards += reward_fn(env)
│     → Dict[str, float]
│
├─ 5. TERMINATION CHECK
│     terminated = (_captures_total >= capture_threshold)
│     truncated  = (max_steps is not None and _episode_steps >= max_steps)
│
├─ 6. RENDER (if render_mode == "human")
│
├─ 7. OBSERVATIONS
│     obs = observation_builder(env)
│     → Dict[str, dict]  — includes ALL agents, even captured ones
│
└─ 8. RETURN dict
       {"obs": obs, "reward": rewards, "terminated": terminated,
        "truncated": truncated, "info": _get_info()}
```

---

## Simultaneous Movement — Key Properties

All agents move before capture is checked. This means:

**Scenario A: Predator walks onto prey's cell**
```
t=0:  P at [2,2],  R at [3,2]
action: P moves Right (0), R stays (4)
t=1:  P at [3,2],  R at [3,2]  → CAPTURE
```

**Scenario B: Prey walks onto predator's cell**
```
t=0:  P at [3,2],  R at [2,2]
action: P stays (4), R moves Right (0)
t=1:  P at [3,2],  R at [3,2]  → CAPTURE
```

**Scenario C: Simultaneous swap (position exchange)**
```
t=0:  P at [2,2],  R at [3,2]
action: P moves Right (0), R moves Left (2)
t=1:  P at [3,2],  R at [2,2]  → NO CAPTURE
      (they crossed each other; neither ends at the same cell)
```

Scenario C is the "crossing" edge case — agents swap positions without triggering a capture.

---

## What the Step Does NOT Do

- Does not validate that all agents are in `actions` (silently defaults to noop)
- Does not check for predator-predator or prey-prey collisions (same-type coexistence is always allowed)
- Does not remove captured agents from `env.agents` or from observations
- Does not call `env.render()` externally — if `render_mode == "human"`, `_render_frame()` is called internally

---

## MARL Note: Observation of Captured Agents

After capture, prey remain in the observation returned by `step()`. Downstream algorithms receive observations keyed by ALL agent names including captured ones. The algorithm is responsible for ignoring or filtering rewards/observations for captured agents if desired. IQL/CQL do not currently do this — they continue updating Q-tables for captured prey using zero-movement observations.
