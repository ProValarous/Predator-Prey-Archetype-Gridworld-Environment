# Import algorithms so they auto-register
from ppage.baselines.IQL.iql import IQL  # noqa: F401
from ppage.baselines.CQL.cql import CQL  # noqa: F401
from ppage.baselines.MIXED.mix_train import MixedTrainer  # noqa: F401
from ppage.baselines.DQN.dqn import DQN  # noqa: F401
from ppage.baselines.AC.actor_critic import ActorCritic  # noqa: F401
from ppage.baselines.A2C.a2c import A2C  # noqa: F401
from ppage.baselines.A3C.a3c import A3C  # noqa: F401
