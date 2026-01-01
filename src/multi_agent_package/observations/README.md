# Observation Builders

This folder contains observation logic for GridWorld agents.

## How observations work

- Observations are built at the ENVIRONMENT level
- Agents do not compute their own observations
- Observation builders are pure functions

## What you are allowed to use

Inside `build(self, env)` you may read:
- agent.agent_name
- agent.agent_type
- agent._agent_location
- env.agents
- env._obstacle_location
- env.size

## What you are NOT allowed to do

- Modify agent positions
- Modify env state
- Add rewards
- Peek into learning algorithms

## How to add a new observation

1. Create a new file (e.g. `my_obs.py`)
2. Inherit from `ObservationBuilder`
3. Implement `build(self, env)`
4. Register it in `registry/observation_registry.py`
5. Select it via `observations.yaml`

## Design advice

If your observation:
- leaks global state → it’s invalid
- is hard to explain → simplify it
- cannot be ablated → rethink it
