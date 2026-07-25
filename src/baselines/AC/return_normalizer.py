# src/baselines/AC/return_normalizer.py
"""
Running mean/std normalizer for bootstrapped return targets -- a PopArt-style
(van Hasselt et al., 2016, "Learning Values Across Many Orders of Magnitude")
fix for entropy collapse in ActorCriticNetwork's shared-trunk architecture,
used by both ActorCritic and A3C.

Two classes: `RunningReturnNormalizer` (process-local, used by ActorCritic
and by A3C's `evaluate()`/single-process paths) and `SharedReturnNormalizer`
(process-shared via `multiprocessing.Value` + `Lock`, used by A3C's worker
processes so every worker updates and rescales against ONE consistent
estimate instead of independent per-worker ones). Both support the full
PopArt technique -- `update_and_rescale_value_head`, including "preserving
outputs precisely" -- since collapsing A3C onto one shared estimate is what
makes that rescale well-defined there too (see SharedReturnNormalizer's own
docstring for why the earlier per-worker-local design couldn't use it).

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
import torch.multiprocessing as mp


def _bias_corrected_stats(mean: float, sq: float, t: int, decay: float, eps: float):
    """Shared math between RunningReturnNormalizer and SharedReturnNormalizer:
    bias-corrected (mean, std) from raw EMA accumulators."""
    if t == 0:
        return 0.0, eps
    bias_corr = 1 - decay**t
    mean_hat = mean / bias_corr
    sq_hat = sq / bias_corr
    std_hat = math.sqrt(max(sq_hat - mean_hat**2, 0.0)) + eps
    return mean_hat, std_hat


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
        return _bias_corrected_stats(self._mean, self._sq, self._t, self.decay, self.eps)

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
        several async workers each with their own local estimate) would
        fight each other. A3C uses `SharedReturnNormalizer` (below) instead,
        which gives every worker one consistent, lock-protected estimate so
        this same method can be used safely there too.
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


class SharedReturnNormalizer:
    """Process-shared counterpart to RunningReturnNormalizer -- ONE running
    mean/std estimate, backed by `multiprocessing.Value`s and guarded by a
    `multiprocessing.Lock`, usable safely from A3C's multiple worker
    processes at once (same primitives already used for `episode_counter`/
    `counter_lock` elsewhere in this file's caller).

    Why this exists: A3C originally gave each worker its own LOCAL
    RunningReturnNormalizer, each estimating the same underlying return
    distribution independently. That worked well overall (see
    docs/algorithms/a3c.md's verification numbers) but left one open
    wrinkle -- a real, sharp late-training partial re-collapse shared
    identically across all workers. Root cause: with no weight-rescale
    (RunningReturnNormalizer.update_and_rescale_value_head can't be used
    safely with multiple independent estimates over one shared network --
    see that method's docstring), the shared `value_head`'s weights
    (updated only by slow gradient steps) can drift out of sync with
    whichever worker's locally-tracked scale happens to be current at any
    given moment. Collapsing every worker onto ONE shared, synchronized
    estimate makes the rescale well-defined again -- there is only ever one
    (mean, std) the shared value_head needs to stay consistent with.

    Read pattern: `stats()` is a lock-free read (consistent with A3C's
    Hogwild philosophy -- staleness across workers is expected and fine,
    only lost/corrupted updates are not). Take ONE snapshot per rollout
    (right after syncing the local network from the global one) and reuse
    it for that whole rollout's bootstrap-denormalize and return-normalize
    steps, rather than re-reading mid-rollout -- the local network's
    collected values reflect whatever scale was true at sync time, so the
    returns computed against them must use that exact same snapshot to
    stay internally consistent, even though the shared estimate may keep
    moving (from other workers) while this rollout is still in flight.
    """

    def __init__(self, decay: float = 0.999, eps: float = 1e-4):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.eps = eps
        self._mean = mp.Value("d", 0.0)
        self._sq = mp.Value("d", 0.0)
        self._t = mp.Value("l", 0)
        self._lock = mp.Lock()

    def stats(self) -> tuple:
        """Bias-corrected (mean, std) snapshot from observations so far.
        Lock-free -- see class docstring."""
        return _bias_corrected_stats(
            self._mean.value, self._sq.value, self._t.value, self.decay, self.eps
        )

    def normalize(self, raw_value: float) -> float:
        mean, std = self.stats()
        return (raw_value - mean) / std

    def denormalize(self, normalized_value: float) -> float:
        mean, std = self.stats()
        return normalized_value * std + mean

    def update_and_rescale_value_head(self, raw_value: float, value_head) -> None:
        """Locked update + PopArt rescale -- see
        RunningReturnNormalizer.update_and_rescale_value_head for the
        rescale math itself (identical here, just against shared state).
        `value_head` should be the SHARED GLOBAL network's value_head (not
        a worker's local copy), since that's the persistent state every
        worker eventually syncs from.
        """
        with self._lock:
            old_mean, old_std = _bias_corrected_stats(
                self._mean.value, self._sq.value, self._t.value, self.decay, self.eps
            )

            self._t.value += 1
            self._mean.value = self.decay * self._mean.value + (1 - self.decay) * raw_value
            self._sq.value = (
                self.decay * self._sq.value + (1 - self.decay) * (raw_value**2)
            )

            new_mean, new_std = _bias_corrected_stats(
                self._mean.value, self._sq.value, self._t.value, self.decay, self.eps
            )

            with torch.no_grad():
                ratio = old_std / new_std
                value_head.weight.mul_(ratio)
                value_head.bias.mul_(ratio).add_((old_mean - new_mean) / new_std)

    def state_dict(self) -> dict:
        return {
            "mean": self._mean.value,
            "sq": self._sq.value,
            "t": self._t.value,
            "decay": self.decay,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self._mean.value = state["mean"]
            self._sq.value = state["sq"]
            self._t.value = state["t"]
            self.decay = state["decay"]
            self.eps = state["eps"]
