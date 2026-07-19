# Concept: Agents

## What an Agent Is

An agent is an **entity that can observe, decide, and act** in the gridworld. Each agent has a fixed identity (type, team, name) and mutable state (position, stamina).

Implemented in `src/multi_agent_package/core/agent.py`.

---

## Agent Types

| Type | Role | Speed | Default Color | Objective |
|------|------|-------|--------------|-----------|
| `"predator"` | Hunter | 1 | Red | Capture prey |
| `"prey"` | Evader | 3 | Green | Survive |
| `"other"` | Custom | 1 | Blue | User-defined |

Speed is an **attribute** stored on the agent but is not currently used to allow multi-step movement in the core step loop. See [roadmap.md](../roadmap.md).

---

## Identity vs. State

| Property | Category | Mutable? |
|----------|----------|----------|
| `agent_name` | Identity | No |
| `agent_type` | Identity | No |
| `agent_team` | Identity | No |
| `agent_speed` | Identity | No |
| `_agent_location` | State | Yes (every step) |
| `stamina` | State | Not yet wired |

Agent identity is set at construction and never changes. The environment modifies `_agent_location` directly during `step()`.

---

## Action Space

`gymnasium.spaces.Discrete(5)` — defined by the `DiscreteActionSpace` plugin:

| Index | Label | Direction Vector | Visual movement |
|-------|-------|-----------------|-----------------|
| 0 | Right | `[+1,  0]` | Right |
| 1 | Up    | `[ 0, +1]` | **Down** (Y increases downward) |
| 2 | Left  | `[-1,  0]` | Left |
| 3 | Down  | `[ 0, -1]` | **Up** (Y increases downward) |
| 4 | Noop  | `[ 0,  0]` | Stay |

**Coordinate system caveat:** The origin `[0,0]` is at the top-left; Y increases downward. The code assigns the label "Up" to the vector `[0,+1]` (which moves the agent toward higher Y, i.e., visually *downward* on screen). The label and visual direction are inverted for actions 1 and 3. When reading agent behavior, reason from the **direction vectors**, not the action labels.

The action-to-direction mapping is owned by the active `ActionSpace` plugin (`env.action_space_plugin`). See [concepts/actions.md](actions.md).

---

## Team System

`agent_team` encodes both a **base type** and a **sub-team ID**, used to assign distinct visual appearances.

Accepted formats:

| Input | Parses to | Use case |
|-------|-----------|----------|
| `3` (int) | sub_id=3 | Numbered team |
| `"predator_2"` | base="predator", sub_id=2 | Named team |
| `"2"` (str) | sub_id=2 | String number |

Sub-team affects rendering only (color saturation, shape). It has no effect on gameplay.

---

## Visual Rendering

**Color** is derived from agent type via HSV:

| Type | Hue | Family |
|------|-----|--------|
| predator | 0° | Reds |
| prey | 120° | Greens |
| other | 240° | Blues |

Sub-teams within a type vary in saturation and brightness to stay visually distinct.

**Shape** cycles by sub-id:

| Sub-ID | Shape |
|--------|-------|
| 1 | Circle |
| 2 | Square |
| 3 | Triangle |
| 4 | Star |
| 5 | Diamond |
| 6+ | Cycles back to 1 |

---

## Lifecycle

```
Agent created by build_agents() from agents.yaml
  │
  ▼
env.reset()
  └► _agent_location assigned randomly (non-obstacle cell)

env.step() × N
  └► _agent_location updated by environment
     if agent captured → added to _captured_agents
                         position frozen, actions ignored

env.close()
  └► no agent cleanup needed
```

---

## What Agents Don't Do

Agents in this system are **passive data holders** during training. They do not:
- Select their own actions (the algorithm does this)
- Compute their own rewards (the reward functions do this)
- Build their own observations (the observation builders do this)

This separation is intentional. It keeps agents inspectable and the environment physics independent of learning behavior.
