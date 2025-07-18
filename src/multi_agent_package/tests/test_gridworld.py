from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.helpers.helper import print_mdp_tuple
from multi_agent_package.agents import Agent


agent1 = Agent("prey","Tom")
agent2 = Agent("predator","Jerry")
agent3 = Agent("judge","Spike")
agents = [agent1,agent2,agent3]
# Create an environment (e.g., CartPole)
env = GridWorldEnv(agents = agents, render_mode="human", size=10, perc_num_obstacle=10)

# Reset the environment
state, info = env.reset()
print(state,info)

# Example of an interaction loop
for i in range(100):
    # Render the environment
    env.render()

    # Sample random action from action space
    action = env.action_space.sample()
    
    # Step through the environment using the action
    mdp_tuple = env.step(action)
    next_state, reward, done,_, info = mdp_tuple

    print_mdp_tuple(i, state,action, mdp_tuple)

    # Break the loop if the episode is done
    if done:
        break

# Close the environment
env.close()