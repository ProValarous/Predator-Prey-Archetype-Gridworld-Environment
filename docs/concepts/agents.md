# Concept: Agents

## What an Agent Is

An agent is an **entity that can observe, decide, and act** in the gridworld. Each agent has a fixed identity (type, team, name) and mutable state (position, stamina).

Implemented in `src/multi_agent_package/core/agent.py`.

---

## Agent Types

| Type | Role | Default Speed | Default Color | Objective |
|------|------|-------|--------------|-----------|
| `"predator"` | Hunter | 1 | Red | Capture prey |
| `"prey"` | Evader | 3 | Green | Survive |
| `"other"` | Custom | 1 | Blue | User-defined |

"Default Speed" is what `Agent.__init__` assigns if nothing overrides it (exact string match on `agent_type`, not `.startswith()` — an agent typed `"predator_fast"` would fall through to speed `1`). In practice, `build_agents()` in `run_from_config.py` always overwrites `agent.agent_speed` from `agents.yaml`'s `speed:` field right after construction, so the YAML value is what actually takes effect.

**`agent_speed` is not used by `GridWorldEnv.step()` itself** — the core environment always moves every agent exactly one cell per call, regardless of speed. It *is* used by `SpeedWrapper`, which `run_from_config.build_environment()` always applies as the outermost layer: when any agent's `agent_speed > 1`, the wrapper replays a logical step as multiple sub-steps so faster agents move further per turn. See [concepts/wrappers.md](wrappers.md).

---

## Identity vs. State

| Property | Category | Mutable? |
|----------|----------|----------|
| `agent_name` | Identity | No |
| `agent_type` | Identity | No |
| `agent_team` | Identity | No |
| `agent_speed` | Identity | Yes, once — overwritten by `build_agents()` from `agents.yaml` right after construction |
| `_agent_location` | State | Yes (every step) |
| `stamina` | State | Yes — depleted by `SpeedWrapper` (1 per sub-step taken), reset to max on `env.reset()` |

Agent identity is set at construction and never changes. The environment modifies `_agent_location` directly during `step()`.

---

## Action Space

`gymnasium.spaces.Discrete(5)` — defined by the `DiscreteActionSpace` plugin, the default of three shipped action spaces (see [concepts/actions.md](actions.md) for `cross` and `speed_discrete_5`):

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

Accepted formats (`Agent._parse_team()`):

| Input | Parses to | Use case |
|-------|-----------|----------|
| `None` | base=`agent_type`, sub_id=1 | No team specified |
| `3` (int) | base=`agent_type`, sub_id=3 | Numbered team |
| `"predator_2"` (str with `_`) | base="predator", sub_id=2 | Named team |
| `"2"` (digit str) | base=`agent_type`, sub_id=2 | String number |
| any other string, e.g. `"alpha"` | base="alpha", sub_id=1 | ⚠️ replaces the rendered base type entirely |

The last row is a real edge case worth knowing: if `agent_team` is a non-digit string with no underscore, `_parse_team()` uses that string as the base type for **color/shape purposes only** — an agent with `agent_type="prey"` and `agent_team="alpha"` renders with the "alpha" hue (falls back to red, since "alpha" isn't a recognized base type), not green. This never affects gameplay (capture detection, speed, rewards all read `agent.agent_type` directly, never `_parse_team()`'s output) — it's purely a rendering quirk. Sub-team otherwise only affects color saturation/brightness and shape; it has no effect on gameplay.

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
