# Import algorithms so they auto-register
from baselines.IQL.iql import IQL  # noqa: F401
from baselines.CQL.cql import CQL  # noqa: F401
from baselines.MIXED.mix_train import MixedTrainer  # noqa: F401
from baselines.DQN.dqn import DQN  # noqa: F401
from baselines.AC.actor_critic import ActorCritic  # noqa: F401
