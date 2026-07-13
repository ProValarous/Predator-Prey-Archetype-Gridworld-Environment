# Glossary

Definitions for terms used throughout this wiki and the codebase.

---

## Domain Terms

**Agent**
An entity in the grid that can observe, decide, and act. Every agent has a type (`predator`, `prey`, or `other`), a team identifier, and a unique name. Represented by `Agent` in `core/agent.py`.

**Predator**
An agent type whose objective is to capture prey by occupying the same cell. Moves at speed 1 (one cell per timestep). Rendered in red.

**Prey**
An agent type whose objective is to survive (avoid capture). Moves at speed 3 (up to three cells per timestep — though in the current implementation speed is an attribute, not a loop multiplier). Rendered in green.

**Capture**
The event that occurs when a predator and prey occupy the same cell at the end of a timestep. The prey is added to `_captured_agents` and can no longer move.

**Episode**
A single run of the environment from `reset()` until termination or truncation. Termination occurs when `_captures_total >= capture_threshold`. Truncation occurs when `_episode_steps >= max_steps`.

**Timestep (step)**
One call to `env.step(actions)`. All agents move simultaneously within a single timestep.

**Obstacle**
An impassable cell that blocks agent movement. Placed randomly at `reset()` on non-agent cells. Percentage controlled by `perc_num_obstacle`.

**Simultaneous Execution**
All agent moves are resolved at the same time within a step. No sequential ordering. This means two agents can swap positions in a single step (cross each other without a capture event if the swap is exact).

---

## Architecture Terms

**Core (immutable)**
The modules `core/gridworld.py` and `core/agent.py`. These define environment physics and are never modified by contributors. See [ADR-001](../decisions/ADR-001-immutable-core.md).

**Plugin**
Any `ObservationBuilder`, `RewardFunction`, or `ActionSpace` subclass. Plugins are loaded at runtime via the registry and injected into the environment. They may read but must not write env state.

**Registry**
A dict-based lookup table mapping string names (used in YAML) to Python classes. Three registries exist: `observation_registry`, `reward_registry`, and `action_registry`. See [ADR-002](../decisions/ADR-002-plugin-registry.md).

**Callable Injection**
The pattern where the environment holds a function pointer (`env.reward_fn`, `env.observation_builder`) rather than a class reference. The plugin's method is bound at wiring time in `run_from_config.py`. See [ADR-003](../decisions/ADR-003-callable-injection.md). The action space (`env.action_space_plugin`) follows the same injection pattern but is stored as an object rather than a bound method, because `to_direction()` requires an argument.

**Action Space Plugin**
An `ActionSpace` subclass that maps discrete integer actions to `[dx, dy]` direction vectors. Registered in `action_registry.py` and declared in `configs/actions.yaml`. Injected as `env.action_space_plugin` before training.

**Reward Shaping**
Additional reward signal layered on top of base rewards to guide learning. Implemented as separate `RewardFunction` subclasses combined via a closure. Does not replace base rewards.

**Base Rewards**
The hardcoded reward structure in `GridWorldEnv.base_reward()`: capture (+100), obstacle penalty (-200), step cost (-5). These are always active and cannot be disabled by config.

---

## Observation Terms

**Local Observation**
An agent's own position `[x, y]` in grid coordinates. Always available regardless of observation mode.

**Global Observation**
Information about other entities (agents, obstacles). What is included depends on the observation builder in use.

**Observability Mode**
The scope of what an agent can perceive:
- *Full observability*: sees all entities (Default, Absolute, Relative builders)
- *Partial observability*: sees only entities within a radius (LocalRadius)
- *Minimal observability*: sees only own position (LocalOnly)

**Egocentric Frame**
A coordinate system where the observing agent is always at position `[0, 0]`. All other positions are expressed as offsets. Used by `RelativeObservation`.

**World Frame**
A coordinate system using the grid's absolute coordinates (origin at top-left). Used by `AbsoluteObservation`.

**Manhattan Distance**
`|x1 - x2| + |y1 - y2|`. Used for radius filtering in `LocalRadiusObservation` and distance shaping in `PredatorDistanceReward`. Appropriate for grid movement.

---

## Algorithm Terms

**IQL (Independent Q-Learning)**
Each agent maintains its own Q-table and learns independently. No explicit modeling of other agents. Tabular: state must be hashable. Implemented in `baselines/IQL/iql.py`.

**CQL (Centralized Q-Learning)**
Q-learning with a shared (centralized) Q-table over the joint state-action space of all agents. A single table covers all agents collectively, enabling coordinated learning at the cost of state-space scaling. Implemented in `baselines/CQL/cql.py`.

**Q-Table**
A lookup table `Q[state][action] → expected_return`. In IQL, one table per agent. In CQL, one shared table over the joint state of all agents.

**MixedTrainer**
A multi-algorithm baseline where predators and prey are assigned different learning algorithms (IQL or CQL) via `predator_algo` / `prey_algo` config params. Registered as `"mixed"` in the algorithm registry.

**Epsilon-Greedy**
Exploration strategy: with probability `epsilon`, select a random action; otherwise select the greedy (highest Q-value) action. `epsilon` decays over episodes.

**State Encoding**
The process of converting an observation dict (which may contain numpy arrays and nested dicts) into a hashable tuple usable as a Q-table key. Implemented in `IQL._encode_state()`.

---

## Configuration Terms

**Config Directory**
The `configs/` directory containing five YAML files: `env.yaml`, `agents.yaml`, `observations.yaml`, `rewards.yaml`, `experiment.yaml`.

**Seed**
An integer passed to `np.random.default_rng(seed)` that initializes the environment's random number generator. The same seed produces the same obstacle layout, agent start positions, and (if the algorithm is deterministic) training trajectory.

**Capture Threshold**
The number of prey captures required to terminate an episode. Configured via `termination.capture_threshold` in `env.yaml` and passed to `GridWorldEnv(capture_threshold=N)`. Defaults to 1.
