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
The modules `core/gridworld.py` and `core/agent.py`. These define environment physics and are never modified by contributors — enforced by the `core-guard` CI check on every pull request. See [Contributing](../contributing.md)'s "Golden Rule."

**Plugin**
Any `ObservationBuilder`, `RewardFunction`, or `ActionSpace` subclass. Plugins are loaded at runtime via the registry and injected into the environment. They may read but must not write env state.

**Registry**
A dict-based lookup table mapping string names (used in YAML) to Python classes. Four registries exist: `observation_registry`, `reward_registry`, `action_registry` (all three under `multi_agent_package/registry/`), and `algorithm_registry` (under `baselines/registry/`).

**Callable Injection**
The pattern where the environment holds a function pointer (`env.reward_fn`, `env.observation_builder`) rather than a class reference. The plugin's method is bound at wiring time in `run_from_config.py`. The action space (`env.action_space_plugin`) follows the same injection pattern but is stored as an object rather than a bound method, because `to_direction()` requires an argument.

**Wrapper**
An object that wraps the environment (or another wrapper), overriding specific methods (`step()`/`reset()`) while proxying everything else through `__getattr__`. Not registry-driven — there's exactly one, `SpeedWrapper`, applied explicitly by `run_from_config.build_environment()` as the outermost layer. See [concepts/wrappers.md](../concepts/wrappers.md).

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

> ⚠️ **Naming collision:** this "CQL" is unrelated to the well-known offline-RL algorithm "Conservative Q-Learning" (Kumar et al., 2020) that shares the same acronym in the broader RL literature. This repo's CQL is plain online joint-action tabular Q-learning — no conservative/pessimistic regularization, no offline dataset, no distributional-shift penalty.

**Q-Table**
A lookup table `Q[state][action] → expected_return`. In IQL, one table per agent. In CQL, one shared table over the joint state of all agents.

**MixedTrainer**
A multi-algorithm baseline where predators and prey are assigned different learning algorithms (IQL or CQL) via `predator_algo` / `prey_algo` config params. Registered as `"mixed"` in the algorithm registry.

**DQN (Deep Q-Network)**
A PyTorch neural-network baseline: one independent `QNetwork` (or `DuelingQNetwork`) plus target network and replay buffer per agent — architecturally similar to IQL (independent per-agent learning) but with a function approximator instead of a table. Requires `env.observation_encoder` to be attached (an `encode(obs, env) -> np.ndarray` callable). Implemented in `baselines/DQN/dqn.py`.

**Double DQN**
A DQN variant (`double_dqn: true` config flag) that selects the bootstrap action using the *online* network but evaluates its Q-value using the *target* network, decoupling action selection from evaluation to reduce the max-operator's overestimation bias.

**Dueling DQN**
A DQN variant (`dueling: true` config flag) using `DuelingQNetwork`, which splits the network into a value stream `V(s)` and an advantage stream `A(s,a)`, recombined as `Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))`.

**Replay Buffer**
A fixed-capacity ring buffer of `(state, action, reward, next_state, done)` transitions, sampled without replacement in random batches for DQN's gradient updates. Implemented in `baselines/DQN/replay_buffer.py` as preallocated numpy arrays (not a `deque`), for O(1) vectorized batch sampling.

**Epsilon-Greedy**
Exploration strategy: with probability `epsilon`, select a random action; otherwise select the greedy (highest Q-value) action. `epsilon` decays over episodes. Used by all four baselines (IQL, CQL, MixedTrainer, DQN).

**State Encoding**
The process of converting an observation dict (which may contain numpy arrays and nested dicts) into a hashable tuple usable as a Q-table key. Implemented in `IQL._encode_state()` (and identically in `CQL`/`MixedTrainer`). DQN instead flattens observations to a numeric vector via `env.observation_encoder`, not a hashable tuple.

---

## Configuration Terms

**Config Directory**
The `configs/` directory containing six YAML files per experiment set: `env.yaml`, `agents.yaml`, `observations.yaml`, `rewards.yaml`, `actions.yaml`, and an experiment file (`experiment.yaml` or `experiment_{iql,cql,mixed,dqn}.yaml`). Ready-made DQN experiment sets also exist as subdirectories: `configs/dqn_1v1/`, `configs/dqn_speed1/`, `configs/dqn_speed2/`, `configs/dqn_speed3/`.

**Seed**
An integer passed to `np.random.default_rng(seed)` that initializes the environment's random number generator. The same seed produces the same obstacle layout, agent start positions, and (if the algorithm is deterministic) training trajectory.

**Capture Threshold**
The number of prey captures required to terminate an episode. Configured via `termination.capture_threshold` in `env.yaml` and passed to `GridWorldEnv(capture_threshold=N)`. Defaults to 1.
