from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent
from multi_agent_package.helpers.helper import print_action, print_mgp_info

# Initialize agents with different types and unique names
agent101 = Agent("prey", 1,"PY101_Tom")
agent102 = Agent("prey", 2,"PY102_Garfield")


agent201 = Agent("prey", 3, "PD201_Jerry")
agent202 = Agent("predator", 1, "PD202_Stuart")

# agent301 = Agent("prey_3", "PD301_Spike")
# agent302 = Agent("prey_3", "PD302_Krypto")

# agent401 = Agent("prey_4", "PD402_Krypto")

# agent501 = Agent("prey_5", "PD502_Krypto")

# agent601 = Agent("predator_5", "PD502_Krypto")

agents = [agent101,agent102, agent201, agent202]#,agent301,agent302, agent401, agent501,agent601]

# Create the GridWorld environment with given agents and settings
env = GridWorldEnv(agents=agents, render_mode="human", size=10, perc_num_obstacle=10)

# Reset environment to start state
current_state, info = env.reset()

# Initialize dictionaries for storing next state, actions, and rewards
next_state = {}
action = {}
rewards = {}

# Main simulation loop for a fixed number of steps (episodes)
for i in range(100):
    # Sample a random action for each agent from the shared action space
    for ag in agents:
        action[ag.agent_name] = env.action_space.sample()

    # Step through the environment using the chosen actions
    mgp_tuple = env.step(action)

    # Print the state, action, reward, and other info for this step
    print_mgp_info(mgp_tuple, i, current_state, action)

    # Break the loop if a termination condition is met
    if mgp_tuple['terminated']:
        break

    # Update current state for the next iteration
    current_state = mgp_tuple['obs']

# Gracefully close the environment after simulation
env.close()
