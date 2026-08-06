# Import algorithms so they auto-register.
#
# The tabular algorithms (IQL, CQL, MixedTrainer) depend only on the core
# install. The neural ones need PyTorch, which is an optional extra
# (pip install "ppage[baselines]"), so they register only when torch is
# importable; without it, requesting them by name raises the registry's
# "not registered" ValueError rather than an ImportError here.
from ppage.baselines.IQL.iql import IQL  # noqa: F401
from ppage.baselines.CQL.cql import CQL  # noqa: F401
from ppage.baselines.MIXED.mix_train import MixedTrainer  # noqa: F401

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    from ppage.baselines.DQN.dqn import DQN  # noqa: F401
    from ppage.baselines.AC.actor_critic import ActorCritic  # noqa: F401
    from ppage.baselines.A2C.a2c import A2C  # noqa: F401
    from ppage.baselines.A3C.a3c import A3C  # noqa: F401
