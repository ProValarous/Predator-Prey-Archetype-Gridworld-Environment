# src/baselines/AC/return_normalizer.py
"""
Running mean/std normalizer for bootstrapped return targets -- a PopArt-style
(van Hasselt et al., 2016, "Learning Values Across Many Orders of Magnitude")
fix for entropy collapse in ActorCriticNetwork's shared-trunk architecture,
used by both ActorCritic and A3C. ActorCritic uses the full technique
(`update_and_rescale_value_head`, including PopArt's "preserving outputs
precisely" weight rescale); A3C uses the plain `update()` -- see that
method's docstring for why the full version doesn't fit A3C's per-worker
design.

Root cause this addresses (see docs/algorithms/actor-critic.md's "The instant
collapse" section for the full diagnostic trail): the critic's regression
target on this environment reaches into the hundreds/thousands, and nothing
bounds the shared trunk's output feature magnitude, so the trunk grows
without limit to let the critic represent targets that large. Since
`policy_head` is a fixed linear readout on those SAME trunk features, its
logits inherit that growth for free, saturating the softmax and collapsing
entropy -- independent of any actor-specific gradient. Normalizing the
*instantaneous reward* alone doesn't fix this: even an O(1) per-step reward
compounds to O(1 / (1 - gamma)) in the bootstrapped return. This normalizer
tracks the *return's own* running scale instead, so the critic is always
trained to predict an O(1) quantity regardless of gamma or reward magnitude.
"""

import math

import torch


class RunningReturnNormalizer:
    """Tracks a bias-corrected running mean/std of observed raw return
    values (Adam-style EMA + bias correction, applied to a scalar instead of
    gradients). `stats()` reflects whatever has been seen so far -- call it
    once before `update()` to un-normalize a bootstrap value against the
    *prior* estimate, then again after `update()` to normalize the freshly
    computed target against the *updated* estimate.

    Plain `update()` alone is only really safe with a slow-adapting `decay`
    (confirmed empirically: `decay=0.99` diverges to a numerical overflow
    within ~1600 episodes on this env's A3C config, `decay=0.999` does not).
    The reason: the value head's weights only move via slow gradient steps,
    so if the running scale shifts meaningfully between updates, the same
    weights get reinterpreted against a different target each time --
    usually a mild, self-correcting drift, but fast enough decay lets it
    compound into runaway divergence. `update_and_rescale_value_head()`
    below is the actual PopArt fix for this (the "preserving outputs
    precisely" component this class's docstring's simplification omits).
    """

    def __init__(self, decay: float = 0.999, eps: float = 1e-4):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.eps = eps
        self._mean = 0.0
        self._sq = 0.0
        self._t = 0

    def stats(self) -> tuple:
        """Bias-corrected (mean, std) estimate from observations so far.
        Before any update(), returns (0.0, eps) -- a neutral starting point
        that doesn't blow up the first bootstrap value it's used against."""
        if self._t == 0:
            return 0.0, self.eps
        bias_corr = 1 - self.decay**self._t
        mean_hat = self._mean / bias_corr
        sq_hat = self._sq / bias_corr
        std_hat = math.sqrt(max(sq_hat - mean_hat**2, 0.0)) + self.eps
        return mean_hat, std_hat

    def update(self, raw_value: float) -> None:
        self._t += 1
        self._mean = self.decay * self._mean + (1 - self.decay) * raw_value
        self._sq = self.decay * self._sq + (1 - self.decay) * (raw_value**2)

    def update_and_rescale_value_head(self, raw_value: float, value_head) -> None:
        """PopArt-style update (van Hasselt et al., 2016): advances the
        running mean/std, then analytically rescales `value_head`'s
        weight/bias so its DENORMALIZED prediction is unchanged at the
        instant of this update -- "preserving outputs precisely" while the
        normalization target itself keeps moving. This decouples what the
        network has already learned from a shifting normalization scale,
        which is what plain `update()` (see class docstring) doesn't do and
        why it can diverge with a fast-adapting `decay`.

        `value_head` must be the exact `nn.Linear(..., 1)` layer whose
        output is trained against this normalizer's targets. Only valid
        with ONE consistent caller/estimate per value_head -- multiple
        independent normalizers rescaling the same shared layer (e.g.
        several async workers) would fight each other; A3C's per-worker
        normalizers intentionally use plain `update()` instead for this
        reason (see docs/algorithms/a3c.md).
        """
        old_mean, old_std = self.stats()
        self.update(raw_value)
        new_mean, new_std = self.stats()
        with torch.no_grad():
            ratio = old_std / new_std
            value_head.weight.mul_(ratio)
            value_head.bias.mul_(ratio).add_((old_mean - new_mean) / new_std)

    def normalize(self, raw_value: float) -> float:
        mean, std = self.stats()
        return (raw_value - mean) / std

    def denormalize(self, normalized_value: float) -> float:
        mean, std = self.stats()
        return normalized_value * std + mean

    def state_dict(self) -> dict:
        return {
            "mean": self._mean,
            "sq": self._sq,
            "t": self._t,
            "decay": self.decay,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._mean = state["mean"]
        self._sq = state["sq"]
        self._t = state["t"]
        self.decay = state["decay"]
        self.eps = state["eps"]
