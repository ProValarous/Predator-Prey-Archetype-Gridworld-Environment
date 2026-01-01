# Reward Functions

This folder contains reward logic for GridWorld experiments.

## Design philosophy

- Rewards are functions of state
- Rewards do NOT change environment dynamics
- Rewards do NOT terminate episodes
- Rewards do NOT access learning algorithms

## What you may use

Inside `compute(env)` you may read:
- agent.agent_name
- agent.agent_type
- agent._agent_location
- env.agents
- env._captures_this_step
- env._episode_steps

## What you may NOT do

- Modify agent positions
- Modify stamina or speed
- Call env.step()
- Change termination conditions
- Use future information

## How to add a reward

1. Create a new file (e.g. `my_reward.py`)
2. Inherit from `RewardFunction`
3. Implement `compute(env)`
4. Register it in `registry/reward_registry.py`
5. Enable it via `rewards.yaml`

## Red flags

If your reward:
- encodes movement rules → wrong
- encodes termination → wrong
- depends on actions → wrong
- is hard to explain → simplify it
