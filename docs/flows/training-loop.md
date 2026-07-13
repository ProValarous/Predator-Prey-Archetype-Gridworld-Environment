# Flow: Training Loop

How IQL (and CQL by inheritance) progresses from an untrained state to a learned policy.

---

## Episode Structure

```
algorithm.train()
│
└── for episode in range(self.episodes):
    │
    ├── obs, info = env.reset()
    │   # New obstacle layout, new agent positions
    │   # Same layout as previous runs if seed is fixed
    │
    ├── done = False
    │
    └── while not done:
        │
        ├── actions = algorithm.select_actions(obs)
        │   │
        │   └── for each agent_name, agent_obs in obs.items():
        │         state = _encode_state(agent_obs)   ← hashable tuple
        │         if random() < epsilon:
        │           action = random int [0, 4]
        │         else:
        │           q_vals = q_tables[agent_name].get(state, [0]*5)
        │           action = argmax(q_vals)
        │
        ├── next_obs, rewards, terminated, truncated, info = env.step(actions)
        │
        ├── done = terminated or truncated
        │
        ├── Q-UPDATE — for each agent:
        │   │
        │   ├── state      = _encode_state(obs[agent])
        │   ├── next_state = _encode_state(next_obs[agent])
        │   ├── r          = rewards[agent]
        │   │
        │   ├── if done:
        │   │     td_target = r
        │   │   else:
        │   │     td_target = r + gamma * max(Q[agent][next_state])
        │   │
        │   └── Q[agent][state][action] +=
        │           alpha * (td_target - Q[agent][state][action])
        │
        └── obs = next_obs
    │
    └── epsilon = max(epsilon * epsilon_decay, epsilon_min)
```

---

## Key Loop Properties

**Independent learning:** Each agent's Q-table is updated using only its own reward and its own observations. Agents do not share gradients or Q-values.

**Epsilon decay happens per episode, not per step.** The first step of every episode uses the current epsilon, not a mid-episode-decayed value.

**New states are initialized lazily.** If `Q[agent][state]` has never been visited, `q_tables[agent].get(state, [0]*5)` returns zeros. This means Q-values start at zero and can go negative (step costs) or positive (capture bonus).

**Terminal Q-value:** On `terminated=True`, the TD target is just `r` (no bootstrap). On `truncated=True`, the target includes the bootstrap `+ gamma * max(Q[next_state])`, which may be zero for unseen states.

---

## State Space Growth

```
As training progresses:
  episode 1:   q_tables["predator_1"] = {state_A: [...], state_B: [...]}  ← few entries
  episode 100: q_tables["predator_1"] = {state_A, state_B, ... state_N}   ← many entries

State table grows until all reachable states are visited.
With LocalRadius obs (radius=3) on a 10×10 grid,
state space is large but bounded.
```

---

## CQL — Centralized Q-Learning

CQL replaces per-agent independent tables with **one shared Q-table over the joint state-action space**. The loop structure is the same as IQL but the update is centralized:

```
# Shared Q-table: joint_state → vector of length action_dim^n_agents
joint_state  = tuple(encode(obs[agent]) for agent in agent_ids)  ← all agents
joint_action = a0 * action_dim^(n-1) + a1 * action_dim^(n-2) + ...  ← single int

central_r = sum(rewards[agent] for agent in agent_ids)

q_current  = Q[joint_state][joint_action]
q_next_max = 0.0 if done else max(Q[joint_state_next])

Q[joint_state][joint_action] += alpha * (central_r + gamma * q_next_max - q_current)
```

Action selection marginalises the joint Q-tensor: for agent `i`, reshape the Q-vector to `(action_dim,)*n_agents`, then average over all axes except `i` to get per-agent Q-values.

**Trade-off:** CQL scales as `action_dim^n_agents` states per joint state. With 5 actions and 6 agents, each entry stores 5^6 = 15,625 floats. Keep grids and agent counts small.

---

## MixedTrainer — Per-Team Assignment

`MixedTrainer` assigns an algorithm (IQL or CQL) independently to the predator team and prey team. CQL teams share one joint Q-table over their team's joint state-action space. IQL agents keep individual tables. The training loop runs both update types within each step.

---

## Checkpoint Save/Load

`IQL`, `CQL`, and `MixedTrainer` all implement `save(path)` and `load(env, config, path)`:

```python
algo.save("checkpoints/run_1.pkl")
algo2 = IQL.load(env, config, "checkpoints/run_1.pkl")
algo2.evaluate(episodes=10)
```

---

## What Is Not Logged

The current training loop does not log:
- Per-episode reward totals
- Capture rates
- Epsilon value over time
- Q-table size / state coverage

If you need learning curves, wrap the training loop or add logging before calling `algorithm.train()`.
