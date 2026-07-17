"""
Speed wrapper for GridWorldEnv.

The core environment moves every agent exactly one cell per env.step() call,
regardless of the agent_speed field stored on each Agent. This wrapper honours
agent_speed by replaying each logical step as up to min(speed, stamina)
sub-steps:
  - if the action is NOOP (env.action_space_plugin.is_noop()), no sub-steps
    are taken and stamina is unchanged
  - otherwise the agent acts min(speed, stamina) times, each sub-step
    deducting 1 from stamina; remaining slots in the max-speed budget are
    filled with NOOP
  - stamina depletes across the episode and resets to its max on env.reset()

Rewards are summed across all sub-steps; observations and the done flag are
taken from the final sub-step. The episode ends as soon as any sub-step
signals terminated or truncated so faster agents cannot overshoot past a
capture or timeout.

Usage (automatic via run_from_config.build_environment):
    env.action_space_plugin = get_action_space(...)   # must be set first
    env = SpeedWrapper(env)
    # env.step / env.reset / all attributes work identically to GridWorldEnv
"""

from typing import Dict


class SpeedWrapper:
    # action integer sent for idle slots in the sub-step budget
    NOOP: int = 4

    def __init__(self, env):
        self.env = env
        self._speeds: Dict[str, int] = {
            ag.agent_name: int(ag.agent_speed) for ag in env.agents
        }
        self._max_stamina: Dict[str, int] = {
            ag.agent_name: int(ag.stamina) for ag in env.agents
        }
        self._max_speed: int = max(self._speeds.values(), default=1)
        # use the configured plugin for NOOP detection — avoids hardcoding a class
        self._plugin = env.action_space_plugin
        # stamina remaining this episode; reset on env.reset()
        self._stamina: Dict[str, int] = dict(self._max_stamina)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._stamina = dict(self._max_stamina)
        return self.env.reset(**kwargs)

    def step(self, actions: Dict[str, int]) -> dict:
        # fast path: no agent moves more than once — behave identically to base env
        if self._max_speed == 1:
            return self.env.step(actions)

        # compute sub-step budget: NOOP costs 0, movement costs min(speed, stamina)
        n_steps: Dict[str, int] = {
            name: (
                0 if self._plugin.is_noop(act)
                else min(max(self._speeds[name], 1), max(self._stamina[name], 0))
            )
            for name, act in actions.items()
        }

        accumulated: Dict[str, float] = {name: 0.0 for name in self._speeds}
        result: dict = {}

        for sub in range(self._max_speed):
            sub_actions = {
                name: (act if sub < n_steps[name] else self.NOOP)
                for name, act in actions.items()
            }
            result = self.env.step(sub_actions)
            for name, r in result["reward"].items():
                accumulated[name] += r
            if result["terminated"] or result["truncated"]:
                break

        # deduct 1 stamina per sub-step (NOOP costs 0, already excluded from n_steps)
        for name, n in n_steps.items():
            self._stamina[name] = max(0, self._stamina[name] - n)

        result["reward"] = accumulated
        return result

    def close(self):
        return self.env.close()

    # ------------------------------------------------------------------
    # Transparent attribute proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self.env, name)
