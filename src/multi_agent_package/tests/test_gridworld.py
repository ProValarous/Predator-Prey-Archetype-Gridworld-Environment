from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent
from multi_agent_package.helpers.helper import print_action, print_mgp_info


agent1 = Agent("prey","Tom")
agent2 = Agent("predator","Jerry")
agent3 = Agent("judge","Spike")
agents = [agent1,agent2,agent3]
# Create an environment (e.g., CartPole)
env = GridWorldEnv(agents = agents, render_mode="human", size=10, perc_num_obstacle=10)

# Reset the environment
current_state, info = env.reset()


next_state = {}
action = {}
rewards = {}


# Example of an interaction loop
for i in range(100):
    # print(f">>>>>>>>> Episode : {i+1} <<<<<<<<<<<")
    # Render the environment
    # env.render()

    # Sample random action from action space
    for ag in agents:
        action[ag.agent_name] = env.action_space.sample()

    # print_action(action)
    
    # action = env.action_space.sample()
    
    # Step through the environment using the action
    mgp_tuple = env.step(action)
    # print(i,action,mdp_tuple,sep='\n \n')
    # next_state, reward, done,_, info = mdp_tuple

    print_mgp_info(mgp_tuple,i,current_state,action)

    # Break the loop if the episode is done
    if mgp_tuple['terminated']:
        break
    
    current_state = mgp_tuple['obs']

# Close the environment
env.close()