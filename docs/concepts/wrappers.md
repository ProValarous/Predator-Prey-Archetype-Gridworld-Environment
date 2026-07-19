# Concept: Wrappers

## What a Wrapper Is

A wrapper sits *around* the environment (or another wrapper) and can override its behavior while transparently forwarding everything it doesn't care about. Unlike observations/rewards/actions, wrappers are not a registry-driven plugin category — there's no `wrapper_registry.py` and no YAML `type:` key. There is currently exactly one wrapper: `SpeedWrapper`, applied explicitly (not config-driven) as the last step of `run_from_config.build_environment()`.

Implemented in `src/multi_agent_package/wrappers/speed.py`.

---

## Why It Exists

`GridWorldEnv.step()` always moves every agent exactly one cell, regardless of the `agent_speed` stored on each `Agent`. `SpeedWrapper` is what actually makes `agent_speed` (and `stamina`) matter: it replays one logical `step()` call as up to `max(agent_speed for all agents)` real sub-steps against the underlying environment, sending an agent's real action for as many sub-steps as its speed/stamina budget allows, and `NOOP` for the rest.

---

## Mechanics

```python
class SpeedWrapper:
    NOOP: int = 4

    def __init__(self, env):
        self.env = env
        self._speeds = {ag.agent_name: int(ag.agent_speed) for ag in env.agents}
        self._max_stamina = {ag.agent_name: int(ag.stamina) for ag in env.agents}
        self._max_speed = max(self._speeds.values(), default=1)
        self._plugin = env.action_space_plugin              # the configured action space
        # NOOP index read from that plugin (falls back to 4 if none is attached)
        self._noop_action = next(
            (a for a in range(self._plugin.n_actions) if self._plugin.is_noop(a)),
            self.NOOP,
        ) if self._plugin is not None else self.NOOP
        self._stamina = dict(self._max_stamina)
```

On `step(actions)`:

1. **Fast path:** if every agent's `agent_speed` is `1` (`self._max_speed == 1`), delegate straight to `self.env.step(actions)` — no sub-stepping overhead when speed isn't in play.
2. Otherwise, for each agent compute a sub-step budget: `0` if the configured plugin reports the action is a NOOP (`self._plugin.is_noop(action)`), else `min(max(speed, 1), max(stamina, 0))`.
3. Loop `sub` from `0` to `max_speed - 1`: each agent acts with its real action while `sub < n_steps[name]`, and `self._noop_action` (the plugin's NOOP index) for any remaining sub-steps in the shared budget. Each sub-step calls the real `self.env.step(sub_actions)`.
4. Rewards are **summed** across all sub-steps. Observations, `terminated`, and `truncated` come from the **final** sub-step that ran.
5. The loop breaks **immediately** if any sub-step signals `terminated` or `truncated` — a fast agent can't overshoot past a capture or a timeout mid-budget.
6. After the loop, stamina is deducted once per agent: `stamina -= n_steps[name]` (NOOP already costs `0` since it's excluded from `n_steps`).
7. `reset()` restores `self._stamina` to each agent's max and delegates to `self.env.reset()`.
8. `__getattr__` proxies any attribute not defined on the wrapper itself to the wrapped env — so `env.agents`, `env.action_space_plugin`, `env.observation_encoder`, etc. all pass through transparently. This is why `run_from_config.build_environment()` applies `SpeedWrapper` **last**: everything it proxies must already be attached to the inner env first.

---

## Worked Example

`configs/dqn_1v1`: predator `agent_speed=2`, prey `agent_speed=1`, both `stamina=9999`.

- `max_speed = 2`, so the fast path is skipped even though prey never gets extra sub-steps.
- Predator sends its real action for `min(2, 9999) = 2` sub-steps; prey sends its real action for `min(1, 9999) = 1` sub-step, then `NOOP` for the second sub-step.
- Net effect: the predator moves up to 2 cells per logical turn while the prey moves 1 — this 2x speed advantage is what the `configs/dqn_speed1/2/3` sweep configs vary and what `configs/dqn_1v1/agents.yaml`'s comment calls out as "enables consistent capture."

---

## Quirks Worth Knowing

> `SpeedWrapper` reads NOOP-ness from the **configured** action plugin
> (`env.action_space_plugin.is_noop(...)`) and derives the idle-slot action index
> from it, rather than hardcoding a `SpeedDiscreteActionSpace`. So the wrapper's
> sub-step counting uses the same action space that actually moves the agents, and
> a custom action space with a different NOOP index works correctly. (This closes
> the earlier decoupling gap; `SpeedDiscreteActionSpace.to_moves()` still exists as
> a standalone helper but is no longer used by the wrapper.)

---

## Writing a New Wrapper

There's no formal spec or registry for wrappers (unlike observations/rewards/actions), but `SpeedWrapper` establishes the pattern to follow:

1. Wrap an `env` in `__init__`, storing whatever per-agent state you need.
2. Override only `step()`/`reset()`/`close()` — whatever you actually need to change.
3. Add `__getattr__(self, name): return getattr(self.env, name)` so everything else passes through.
4. Apply it explicitly in `run_from_config.build_environment()` (or your own script) — **last**, after all plugins are attached, if your wrapper needs to see them via the proxy.
