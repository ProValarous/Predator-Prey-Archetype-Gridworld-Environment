# Papers & Further Reading

Every algorithm and technique in this project traces back to published work. Use
this page as a map from the code to the literature. Entries are grouped by topic;
the "Used in" column points to where the idea appears in the docs/code.

## Foundations

| Reference | What it gives you | Used in |
| --- | --- | --- |
| Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press. | The whole foundation: MDPs, returns, value/Q-functions, TD, Q-learning. | [RL Foundations](../concepts/rl-foundations.md) |
| Bellman, R. (1957). *Dynamic Programming*, Princeton University Press. | The Bellman equation. | [RL Foundations](../concepts/rl-foundations.md#5-the-bellman-equation) |
| Watkins, C. J. C. H. & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3), 279–292. | Q-learning and its convergence proof. | [IQL](../algorithms/iql.md) |

## Multi-agent RL

| Reference | What it gives you | Used in |
| --- | --- | --- |
| Tan, M. (1993). *Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents*. ICML. | The original independent-learners idea behind IQL. | [IQL](../algorithms/iql.md) |
| Claus, C. & Boutilier, C. (1998). *The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems*. AAAI. | Joint-action learning dynamics. | [CQL](../algorithms/cql-mixed.md) |
| Busoniu, L., Babuska, R. & De Schutter, B. (2008). *A Comprehensive Survey of Multiagent Reinforcement Learning*. IEEE TSMC-C. | The independent-vs-centralized design axis. | [MARL Theory](../concepts/marl.md) |
| Tampuu, A. et al. (2017). *Multiagent cooperation and competition with deep reinforcement learning*. PLOS ONE. | Independent deep learners in a two-player game. | [IQL](../algorithms/iql.md) |
| Hernandez-Leal, P., Kaisers, M. et al. (2017). *A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity*. | Why non-stationarity is the core MARL problem. | [MARL Theory](../concepts/marl.md#non-stationarity) |

## Centralized training / decentralized execution (context — not implemented)

These are the modern CTDE methods this tabular + DQN testbed does **not**
implement, listed for context in [MARL Theory](../concepts/marl.md#centralized-training-decentralized-execution-ctde).

| Reference | Method |
| --- | --- |
| Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. NeurIPS. | MADDPG |
| Sunehag, P. et al. (2018). *Value-Decomposition Networks for Cooperative Multi-Agent Learning*. AAMAS. | VDN |
| Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation*. ICML. | QMIX |

## Deep Q-Networks

| Reference | What it gives you | Used in |
| --- | --- | --- |
| Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518. | DQN: experience replay + target network. | [DQN](../algorithms/dqn.md) |
| van Hasselt, H., Guez, A. & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI. | Double DQN (reduces overestimation). | [DQN Variants](../concepts/dqn-variants.md#double-dqn) |
| Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML. | Dueling DQN (separate V and A streams). | [DQN Variants](../concepts/dqn-variants.md#dueling-dqn) |

## Reward shaping

| Reference | What it gives you | Used in |
| --- | --- | --- |
| Ng, A. Y., Harada, D. & Russell, S. (1999). *Policy Invariance under Reward Transformations*. ICML. | Potential-based reward shaping (keeps the optimal policy). | [Rewards](../concepts/rewards.md) |
| Devlin, S. & Kudenko, D. (2012). *Dynamic Potential-Based Reward Shaping*. AAMAS. | Time-varying shaping potentials. | [Rewards](../concepts/rewards.md) |

## This environment in research

| Reference | Notes |
| --- | --- |
| Atif, M. A., Haji, N. N., Shaikh, M. S. & Atif, M. E. (2026). *Embodiment-Induced Coordination Regimes in Tabular Multi-Agent Q-Learning*. arXiv:2601.17454. | A controlled study that **uses this environment**: it compares IQL and CQL under speed/stamina embodiment constraints and finds centralized learning is not universally better. Motivates the [MixedTrainer](../algorithms/cql-mixed.md) asymmetry experiments. |

## Roadmap methods (not implemented)

Documented in the [Algorithms overview](../algorithms/index.md#roadmap) as future
directions; this repository does not implement them.

| Reference | Method |
| --- | --- |
| Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML. | A3C / A2C |
| Haarnoja, T. et al. (2018). *Soft Actor-Critic*. ICML. | SAC (a discrete-action variant would be needed here) |
