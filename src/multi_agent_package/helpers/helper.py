def print_action(action):
    """
    Prints the action taken by each agent in a readable format.

    Args:
        action (dict): Dictionary mapping agent names to action indices.
    """
    num_to_dir = {
        0: 'UP',
        1: 'DOWN',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'NOOP'
    }
    for ag in action:
        act = action[ag]
        direction = num_to_dir.get(act, 'UNKNOWN')
        print(f"{ag}'s action : {direction} ({act})")

def print_mgp_info(mgp_info, episode, current_state, action):
    """
    Nicely formats and prints key information from the environment step for debugging.

    Args:
        mgp_info (dict): Output dictionary from environment's step() containing obs, reward, etc.
        episode (int): Current episode number.
        current_state (Any): Environment state representation.
        action (dict): Dictionary mapping agent names to their actions.
    """
    print(f" << episode --> {episode} >>\n")

    print(f">>> Current State:")
    print(current_state)
    print()

    print_action(action)
    print()

    for key in mgp_info:
        print(f">>> {key}:")
        if key == 'obs':
            for agent, obs in mgp_info['obs'].items():
                print(f"{agent} : {obs}")
        else:
            print(mgp_info[key])
        print()

    print("=====================================")
