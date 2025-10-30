from multi_agent_package.agents import Agent


agent1 = Agent("prey", "Tom")
agent2 = Agent("predator", "Jerry")
agent3 = Agent("judge", "Spike")
agents = [agent1, agent2, agent3]

for i in range(2):
    print(agents[i].agent_name)
    print(agents[i].agent_type)


num_agents = {"total": len(agents)}

for ag in agents:
    if str(ag.agent_type) in num_agents:
        num_agents[str(ag.agent_type)] += 1
    else:
        num_agents[str(ag.agent_type)] = 1

print(num_agents)
