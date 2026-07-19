# DQN Variants: Double DQN and Dueling DQN

The `dqn` baseline supports two optional improvements over vanilla DQN. They are
**independent** and each is toggled by a single config flag, so you can run any
of four combinations:

| `double_dqn` | `dueling` | Result |
| --- | --- | --- |
| `false` | `false` | Vanilla DQN |
| `true`  | `false` | Double DQN |
| `false` | `true`  | Dueling DQN |
| `true`  | `true`  | **D3QN** (Dueling Double DQN) |

Both default to `false`, so an experiment is vanilla DQN unless you opt in.

## Vanilla DQN target (baseline)

Each agent trains a Q-network against a slowly-updated target network using the
Bellman target:

```
y = r + γ · (1 − done) · max_a' Q_target(s', a')
```

The two variants below change **how the target is computed** (Double DQN) and
**how the network is structured** (Dueling DQN).

## Double DQN

**Problem it solves.** The `max` in the vanilla target both *selects* and
*evaluates* the next action with the same target network, which systematically
**overestimates** Q-values (the max of noisy estimates is biased upward).

**Fix.** Decouple selection from evaluation: pick the next action with the
**online** network, but read its value from the **target** network.

```
a*  = argmax_a' Q_online(s', a')
y   = r + γ · (1 − done) · Q_target(s', a*)
```

**Where it lives.** [`src/baselines/DQN/dqn.py`](../../src/baselines/DQN/dqn.py),
in `_optimize_agent`, guarded by `self.double_dqn`:

```python
if self.double_dqn:
    next_actions = self.q_networks[agent_id](next_states_t).argmax(dim=1, keepdim=True)
    next_q_values = (
        self.target_networks[agent_id](next_states_t).gather(1, next_actions).squeeze(1)
    )
else:
    next_q_values = self.target_networks[agent_id](next_states_t).max(dim=1)[0]
```

**Enable it:** `double_dqn: true` in the algorithm `params`.

## Dueling DQN

**Idea.** Split the network's final representation into two streams — a scalar
**state value** `V(s)` and a per-action **advantage** `A(s, a)` — then recombine
them into Q-values:

```
Q(s, a) = V(s) + ( A(s, a) − mean_a A(s, a) )
```

Subtracting the mean advantage keeps the decomposition identifiable. This lets
the network learn the value of a state independently of which action is taken,
which helps when many actions have similar value (common in a gridworld where
most cells are unremarkable).

**Where it lives.** [`src/baselines/DQN/q_network.py`](../../src/baselines/DQN/q_network.py),
class `DuelingQNetwork` (shared trunk → `value_head` + `advantage_head`, combined
in `forward`). DQN swaps it in for the plain `QNetwork` in `_build_learners`:

```python
network_cls = DuelingQNetwork if self.dueling else QNetwork
```

**Enable it:** `dueling: true` in the algorithm `params`.

## D3QN (both together)

Setting both flags gives Dueling Double DQN. A ready-made preset lives at
[`configs/d3qn/`](../../configs/d3qn) so the combination is discoverable and
reproducible. It is the same `dqn` algorithm — D3QN is not a separate registry
entry, just both flags on.

## Config example

Add the flags under `experiment.algorithm.params` (all other DQN hyperparameters
are unchanged):

```yaml
experiment:
  algorithm:
    name: dqn
    params:
      hidden_layers: [64, 64]
      learning_rate: 0.001
      gamma: 0.99
      target_update_interval: 100
      # --- variant flags ---
      double_dqn: true    # Double DQN target (reduces overestimation bias)
      dueling: true       # Dueling network (separate V(s) and A(s,a) streams)
```

Run it like any other config:

```bash
python -m multi_agent_package.scripts.run_dqn --config-dir configs/d3qn
```

## Where everything is

| Piece | File | Symbol |
| --- | --- | --- |
| Variant flags read from config | `src/baselines/DQN/dqn.py` | `self.double_dqn`, `self.dueling` |
| Double DQN target | `src/baselines/DQN/dqn.py` | `_optimize_agent` |
| Network selection | `src/baselines/DQN/dqn.py` | `_build_learners` (`network_cls`) |
| Dueling architecture | `src/baselines/DQN/q_network.py` | `DuelingQNetwork` |
| Plain architecture | `src/baselines/DQN/q_network.py` | `QNetwork` |
| D3QN preset | `configs/d3qn/` | `experiment_dqn.yaml` |
| Tests | `tests/test_baselines_dqn.py` | `TestDuelingQNetwork`, dueling/double training tests |

## Notes

- The two variants are orthogonal: Double DQN only changes the target
  computation; Dueling DQN only changes the network. They compose cleanly.
- Both are per-agent (independent learners), consistent with the rest of the
  DQN baseline.
- Runs are seeded and reproducible; the variant flags do not affect determinism.
