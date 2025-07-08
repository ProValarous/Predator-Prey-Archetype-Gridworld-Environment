def print_action(action):
    num_to_dir = {0: 'UP', 1: 'DOWN', 2:'RIGHT', 3:'LEFT', 4:'NOOP'}
    for ag in action:
        print(f"{ag}'s action : {num_to_dir[action[ag]]} ({action[ag]})")


def print_mgp_info(mgp_info,episode,current_state,action):

    print(f" << episode --> {episode} >>")
    print()

    print(f">>> Current State:")
    print(f"{current_state}")
    print()

    print_action(action)
    print()

    for key in mgp_info:
        print(f">>> {key}:")
        print(mgp_info[key])
        print()
    
    
    
    # print(f"current state --> {state}")
    
    # for key in mdp_dict.keys():
    #     print(f"{key} --> {mdp_dict[key]}")

    print("=====================================")


