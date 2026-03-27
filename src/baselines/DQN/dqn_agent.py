"""
DQN Agent.

This module wraps a single agent's learning machinery.
One DQNAgent is created per environment agent inside DQN.

Responsibilities:
    - Holds the online network  (updated every train step)
    - Holds the target network  (periodically synced from online)
    - Holds the Adam optimizer
    - Selects actions (epsilon-greedy)
    - Computes the TD loss and runs one gradient step

This module contains:
    - NO environment interaction
    - NO replay buffer logic
    - NO episode loop
    - NO observation encoding

Those belong to dqn.py (the algorithm class).

Double DQN vs Vanilla — controlled by use_double flag:

    use_double=True  (default)
        Online network selects the best next action.
        Target network evaluates that action's Q-value.
        Reduces overestimation bias.

        target = r + γ · Q_target(s', argmax_a Q_online(s', a))

    use_double=False  (vanilla)
        Target network both selects and evaluates the best next action.

        target = r + γ · max_a Q_target(s', a)

Target network sync:
    Hard update — every `target_update_freq` steps, copy online
    weights directly into target. Simple and stable.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from baselines.DQN.dqn_network import DuelingMLP


class DQNAgent:
    """
    Single-agent DQN learner.

    Holds one online network and one target network (both DuelingMLP).
    Exposes act() for action selection and update() for one gradient step.

    Parameters
    ----------
    agent_id : str
        Name of the environment agent this learner controls.
        Used only for logging/debugging.
    state_dim : int
        Dimensionality of the flattened observation vector.
    action_dim : int
        Number of discrete actions available to this agent.
    hidden_units : list of int
        Hidden layer sizes passed to DuelingMLP.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor for future rewards.
    use_double : bool
        If True, use Double DQN update rule.
    use_dueling : bool
        If True, use Dueling network architecture.
    target_update_freq : int
        Number of update() calls between hard target network syncs.
    device : torch.device
        Device for all tensors and network parameters.

    Examples
    --------
    >>> agent = DQNAgent(
    ...     agent_id="predator_1",
    ...     state_dim=10,
    ...     action_dim=5,
    ...     hidden_units=[128, 128],
    ...     lr=1e-3,
    ...     gamma=0.99,
    ...     use_double=True,
    ...     use_dueling=True,
    ...     target_update_freq=100,
    ...     device=torch.device("cpu"),
    ... )
    >>> action = agent.act(obs_vector, epsilon=0.1)
    >>> loss = agent.update(states, actions, rewards, next_states, dones)
    """

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        hidden_units: List[int],
        lr: float,
        gamma: float,
        use_double: bool,
        use_dueling: bool,
        target_update_freq: int,
        device: torch.device,
    ) -> None:
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_double = use_double
        self.target_update_freq = target_update_freq
        self.device = device

        # ── Networks ──────────────────────────────────────────────────────
        # Both online and target share the same architecture.
        # Target starts as an exact copy of online.
        self.online_net = DuelingMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            use_dueling=use_dueling,
        ).to(device)

        self.target_net = DuelingMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            use_dueling=use_dueling,
        ).to(device)

        # Initialise target as exact copy of online — important.
        # Without this, the first target Q-values are random noise.
        self._sync_target()

        # Target network never receives gradients — eval mode + no_grad
        # is enforced inside update(). We set eval here as a safeguard.
        self.target_net.eval()

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # ── Loss ──────────────────────────────────────────────────────────
        # Huber loss (SmoothL1) is more robust to outlier TD errors
        # than MSE — large errors are clipped to linear rather than
        # growing quadratically. Standard in DQN implementations.
        self.loss_fn = nn.SmoothL1Loss()

        # ── Step counter for target sync ──────────────────────────────────
        self._update_count = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon: random action (exploration).
        Otherwise: greedy action from online network (exploitation).

        Parameters
        ----------
        state : np.ndarray
            Flattened observation vector, shape ``(state_dim,)``.
            This is a single state, not a batch.
        epsilon : float
            Exploration probability in [0, 1].

        Returns
        -------
        int
            Selected action index.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))

        # Greedy: get Q-values from online network
        # unsqueeze(0) adds the batch dimension: (state_dim,) → (1, state_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)     # (1, action_dim)
        self.online_net.train()

        return int(q_values.argmax(dim=1).item())   # scalar int

    # ── Learning update ───────────────────────────────────────────────────────

    def update(
        self,
        states: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        next_states: torch.FloatTensor,
        dones: torch.FloatTensor,
    ) -> float:
        """
        Perform one gradient update step on the online network.

        Computes the TD target using either Double DQN or vanilla rule,
        calculates Huber loss, and backpropagates.

        Automatically syncs target network every target_update_freq calls.

        Parameters
        ----------
        states : torch.FloatTensor
            Shape ``(batch_size, state_dim)``.
        actions : torch.LongTensor
            Shape ``(batch_size,)``.
        rewards : torch.FloatTensor
            Shape ``(batch_size,)``.
        next_states : torch.FloatTensor
            Shape ``(batch_size, state_dim)``.
        dones : torch.FloatTensor
            Shape ``(batch_size,)``. Values 0.0 or 1.0.

        Returns
        -------
        float
            Scalar loss value for logging.
        """

        # ── Current Q-values ──────────────────────────────────────────────
        # online_net(states) → (batch, action_dim)
        # .gather picks the Q-value for the action actually taken.
        # actions.unsqueeze(1) → (batch, 1) so gather works along dim=1
        # result after gather: (batch, 1) → squeeze → (batch,)
        current_q = self.online_net(states) \
                        .gather(1, actions.unsqueeze(1)) \
                        .squeeze(1)                         # (batch,)

        # ── TD Target ─────────────────────────────────────────────────────
        with torch.no_grad():
            if self.use_double:
                # Double DQN:
                # Step 1 — online network picks the best next action
                next_actions = self.online_net(next_states) \
                                   .argmax(dim=1, keepdim=True)     # (batch, 1)
                # Step 2 — target network evaluates that action's value
                next_q = self.target_net(next_states) \
                             .gather(1, next_actions) \
                             .squeeze(1)                            # (batch,)
            else:
                # Vanilla DQN:
                # Target network both picks and evaluates
                next_q = self.target_net(next_states) \
                             .max(dim=1).values                     # (batch,)

            # If done=1.0 the episode ended — no future reward.
            # (1 - dones) zeroes out next_q for terminal transitions.
            target = rewards + self.gamma * next_q * (1.0 - dones)  # (batch,)

        # ── Loss and backprop ─────────────────────────────────────────────
        loss = self.loss_fn(current_q, target)

        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients, especially
        # early in training when Q-values are noisy.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimiser.step()

        # ── Target sync ───────────────────────────────────────────────────
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())

    # ── Target network sync ───────────────────────────────────────────────────

    def _sync_target(self) -> None:
        """
        Hard update: copy online network weights into target network.

        Called at initialisation and every target_update_freq steps.
        load_state_dict with a deep copy ensures no shared references
        between online and target parameters.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return serialisable state for checkpointing."""
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore agent state from a checkpoint dict."""
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimiser.load_state_dict(state["optimiser"])
        self._update_count = state["update_count"]