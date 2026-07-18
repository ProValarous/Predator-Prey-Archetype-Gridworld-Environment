"""
Sanity test for tabular baselines.

This verifies:
- Algorithm registry works
- Environment interaction works
- Q-tables update
- No import issues
"""

import sys
import traceback

# Ensure baselines register themselves
import baselines  # noqa: F401


from baselines.registry import get, list_algorithms
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv


def build_test_env():
    """
    Small deterministic environment.
    Keep it tiny for tabular sanity.
    """
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
        Agent(agent_type="predator", agent_team="predator_2", agent_name="pred_2"),
    ]

    env = GridWorldEnv(
        agents=agents,
        size=8,  # small grid
        perc_num_obstacle=10,  # light obstacle density
        render_mode=None,
        seed=42,
    )

    return env


def test_algorithm(algo_name):
    print(f"\n--- Testing algorithm: {algo_name} ---")

    env = build_test_env()

    algo_cls = get(algo_name)

    config = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.2,
        "episodes": 100,  # very small test
    }

    algo = algo_cls(env, config)

    print("Training...")
    algo.train()

    # Check Q-table structure — each algorithm exposes tables differently
    print("Checking Q-tables...")

    if hasattr(algo, "q_tables"):
        # IQL: one table per agent
        for agent_name, table in algo.q_tables.items():
            print(f"Agent: {agent_name}")
            print(f"  States learned: {len(table)}")
            if len(table) == 0:
                raise RuntimeError(
                    f"Q-table empty for {agent_name} — learning did not run."
                )
    elif hasattr(algo, "q_table"):
        # CQL: single shared joint-state table
        print(f"Shared joint Q-table: {len(algo.q_table)} joint states learned")
        if len(algo.q_table) == 0:
            raise RuntimeError("Shared Q-table empty — learning did not run.")
    elif hasattr(algo, "_iql_tables") or hasattr(algo, "_cql_tables"):
        # MixedTrainer: IQL per-agent + CQL per-team
        for aid, table in getattr(algo, "_iql_tables", {}).items():
            print(f"IQL agent {aid}: {len(table)} states")
            if len(table) == 0:
                raise RuntimeError(f"IQL Q-table empty for {aid}.")
        for tk, table in getattr(algo, "_cql_tables", {}).items():
            print(f"CQL team '{tk}': {len(table)} joint states")
            if len(table) == 0:
                raise RuntimeError(f"CQL joint table empty for team '{tk}'.")
    else:
        raise RuntimeError(f"Cannot inspect Q-tables for {algo_name}.")

    print("Evaluation run...")
    algo.evaluate(episodes=2)

    print(f"OK: {algo_name} passed structural test.")


def main():
    try:
        print("Registered algorithms:", list_algorithms())

        for name in list_algorithms():
            test_algorithm(name)

        print("\nALL TESTS PASSED.")
        print("Architecture integrity confirmed.")

    except Exception:
        print("\nTEST FAILED.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
