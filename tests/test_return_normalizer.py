"""
Tests for RunningReturnNormalizer and SharedReturnNormalizer -- the running
mean/std normalizers used by AC/A3C's normalize_returns fix (see
baselines/AC/return_normalizer.py and docs/algorithms/actor-critic.md's
"instant collapse" writeup).
"""

import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from baselines.AC.return_normalizer import (
    RunningReturnNormalizer,
    SharedReturnNormalizer,
)


class TestRunningReturnNormalizer:
    def test_rejects_invalid_decay(self):
        with pytest.raises(ValueError):
            RunningReturnNormalizer(decay=0.0)
        with pytest.raises(ValueError):
            RunningReturnNormalizer(decay=1.0)
        with pytest.raises(ValueError):
            RunningReturnNormalizer(decay=1.5)

    def test_starts_neutral(self):
        normalizer = RunningReturnNormalizer()
        mean, std = normalizer.stats()
        assert mean == 0.0
        assert std == normalizer.eps

    def test_mean_converges_toward_constant_observations(self):
        normalizer = RunningReturnNormalizer(decay=0.99)
        for _ in range(500):
            normalizer.update(42.0)
        mean, _ = normalizer.stats()
        assert mean == pytest.approx(42.0, abs=0.5)

    def test_std_shrinks_for_constant_observations(self):
        normalizer = RunningReturnNormalizer(decay=0.99)
        for _ in range(500):
            normalizer.update(42.0)
        _, std = normalizer.stats()
        assert std < 1.0  # a constant series has ~zero true variance

    def test_std_reflects_spread_for_varying_observations(self):
        normalizer = RunningReturnNormalizer(decay=0.99)
        values = [10.0, -10.0] * 500
        for v in values:
            normalizer.update(v)
        mean, std = normalizer.stats()
        assert mean == pytest.approx(0.0, abs=1.0)
        assert std == pytest.approx(10.0, rel=0.2)

    def test_normalize_denormalize_roundtrip(self):
        normalizer = RunningReturnNormalizer(decay=0.99)
        for v in [5.0, 15.0, -3.0, 22.0]:
            normalizer.update(v)
        raw = 7.5
        normalized = normalizer.normalize(raw)
        assert normalizer.denormalize(normalized) == pytest.approx(raw)

    def test_first_normalize_does_not_raise_or_blow_up(self):
        normalizer = RunningReturnNormalizer()
        # Before any update(), stats() falls back to (0.0, eps) -- normalizing
        # a large raw value here divides by a tiny eps, so this is expected to
        # produce a large number, but it must not raise or return NaN/inf.
        result = normalizer.normalize(100.0)
        assert result == result  # not NaN
        assert abs(result) != float("inf")

    def test_state_dict_roundtrip(self):
        normalizer = RunningReturnNormalizer(decay=0.995, eps=1e-3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            normalizer.update(v)

        restored = RunningReturnNormalizer(decay=0.999, eps=1e-6)
        restored.load_state_dict(normalizer.state_dict())

        assert restored.decay == normalizer.decay
        assert restored.eps == normalizer.eps
        assert restored.stats() == normalizer.stats()


class TestUpdateAndRescaleValueHead:
    def test_advances_stats_same_as_plain_update(self):
        """The rescale is an additional side effect -- the stats bookkeeping
        itself must behave identically to plain update()."""
        normalizer_a = RunningReturnNormalizer(decay=0.99)
        normalizer_b = RunningReturnNormalizer(decay=0.99)
        value_head = nn.Linear(4, 1)

        for v in [10.0, -5.0, 20.0]:
            normalizer_a.update(v)
            normalizer_b.update_and_rescale_value_head(v, value_head)

        assert normalizer_a.stats() == normalizer_b.stats()

    def test_denormalized_output_is_preserved_across_the_update(self):
        """The whole point of the rescale: for a FIXED input, the value
        head's DENORMALIZED prediction must be unchanged immediately before
        vs. immediately after update_and_rescale_value_head -- "preserving
        outputs precisely" while the normalization target shifts underneath
        it, which is what prevents the divergence plain update() risks."""
        torch.manual_seed(0)
        normalizer = RunningReturnNormalizer(decay=0.9)
        value_head = nn.Linear(4, 1)
        h = torch.randn(1, 4)

        # Prime the normalizer with some observations first so mean/std
        # aren't both at their trivial (0.0, eps) starting point.
        for v in [50.0, -30.0, 80.0]:
            normalizer.update_and_rescale_value_head(v, value_head)

        with torch.no_grad():
            pred_before_normalized = value_head(h).item()
        denorm_before = normalizer.denormalize(pred_before_normalized)

        normalizer.update_and_rescale_value_head(120.0, value_head)

        with torch.no_grad():
            pred_after_normalized = value_head(h).item()
        denorm_after = normalizer.denormalize(pred_after_normalized)

        # abs= matters here, not just rel=: denorm_before/after can land
        # arbitrarily close to zero (this is the difference of two
        # O(100)-scale quantities), where a purely relative tolerance
        # becomes sub-float32-precision-tight and fails on ordinary rounding
        # noise rather than a real mismatch.
        assert denorm_after == pytest.approx(denorm_before, rel=1e-4, abs=1e-3)

    def test_rejects_fast_decay_divergence_is_avoided(self):
        """Sanity check for the actual bug this method fixes: plain update()
        with a fast decay and a network that can't track the shifting scale
        can diverge to overflow (confirmed on the real A3C training loop at
        decay=0.99); the rescale variant must stay finite under the same
        adversarial-ish sequence of large, growing observations."""
        normalizer = RunningReturnNormalizer(decay=0.99)
        value_head = nn.Linear(4, 1)

        value = 10.0
        for _ in range(2000):
            normalizer.update_and_rescale_value_head(value, value_head)
            value *= 1.05  # steadily growing raw target, stresses the rescale

        mean, std = normalizer.stats()
        assert mean == mean and std == std  # not NaN
        assert abs(mean) != float("inf")
        assert abs(std) != float("inf")
        assert torch.isfinite(value_head.weight).all()
        assert torch.isfinite(value_head.bias).all()


class TestSharedReturnNormalizer:
    """SharedReturnNormalizer is RunningReturnNormalizer's process-shared
    counterpart, used by A3C so every worker rescales against ONE consistent
    estimate instead of independent per-worker ones (see the class's own
    docstring for why that matters -- it's what let A3C safely use the
    PopArt rescale at all)."""

    def test_rejects_invalid_decay(self):
        with pytest.raises(ValueError):
            SharedReturnNormalizer(decay=0.0)
        with pytest.raises(ValueError):
            SharedReturnNormalizer(decay=1.0)

    def test_starts_neutral(self):
        normalizer = SharedReturnNormalizer()
        mean, std = normalizer.stats()
        assert mean == 0.0
        assert std == normalizer.eps

    def test_mean_converges_toward_constant_observations(self):
        normalizer = SharedReturnNormalizer(decay=0.99)
        value_head = nn.Linear(4, 1)
        for _ in range(500):
            normalizer.update_and_rescale_value_head(42.0, value_head)
        mean, _ = normalizer.stats()
        assert mean == pytest.approx(42.0, abs=0.5)

    def test_denormalized_output_is_preserved_across_the_update(self):
        """Same output-preservation property as RunningReturnNormalizer's
        equivalent test -- SharedReturnNormalizer must give the identical
        guarantee, just backed by shared memory instead of plain floats."""
        torch.manual_seed(0)
        normalizer = SharedReturnNormalizer(decay=0.9)
        value_head = nn.Linear(4, 1)
        h = torch.randn(1, 4)

        for v in [50.0, -30.0, 80.0]:
            normalizer.update_and_rescale_value_head(v, value_head)

        with torch.no_grad():
            denorm_before = normalizer.denormalize(value_head(h).item())

        normalizer.update_and_rescale_value_head(120.0, value_head)

        with torch.no_grad():
            denorm_after = normalizer.denormalize(value_head(h).item())

        assert denorm_after == pytest.approx(denorm_before, rel=1e-4, abs=1e-3)

    def test_state_dict_roundtrip(self):
        normalizer = SharedReturnNormalizer(decay=0.995, eps=1e-3)
        value_head = nn.Linear(4, 1)
        for v in [1.0, 2.0, 3.0, 4.0]:
            normalizer.update_and_rescale_value_head(v, value_head)

        restored = SharedReturnNormalizer(decay=0.999, eps=1e-6)
        restored.load_state_dict(normalizer.state_dict())

        assert restored.decay == normalizer.decay
        assert restored.eps == normalizer.eps
        assert restored.stats() == normalizer.stats()

    def test_matches_running_normalizer_on_identical_sequence(self):
        """Single-process sanity check that the shared-memory bookkeeping
        computes the exact same stats as the plain-float version, before
        trusting it under real multiprocessing."""
        decay = 0.97
        values = [12.0, -8.0, 30.5, -15.0, 6.0]

        plain = RunningReturnNormalizer(decay=decay)
        shared = SharedReturnNormalizer(decay=decay)
        plain_head = nn.Linear(4, 1)
        shared_head = nn.Linear(4, 1)
        shared_head.load_state_dict(plain_head.state_dict())

        for v in values:
            plain.update_and_rescale_value_head(v, plain_head)
            shared.update_and_rescale_value_head(v, shared_head)

        assert plain.stats() == pytest.approx(shared.stats())
        assert torch.allclose(plain_head.weight, shared_head.weight)
        assert torch.allclose(plain_head.bias, shared_head.bias)


def _concurrent_update_worker(normalizer, value_head, n_updates, barrier):
    barrier.wait()  # start all worker processes at roughly the same instant
    for i in range(n_updates):
        normalizer.update_and_rescale_value_head(float(i + 1), value_head)


class TestSharedReturnNormalizerConcurrency:
    def test_concurrent_updates_from_real_processes_are_not_lost_or_corrupted(self):
        """The actual risk this class exists to manage: multiple real OS
        processes calling update_and_rescale_value_head on the SAME shared
        estimate and the SAME shared value_head at once. Verifies the lock
        prevents lost updates (final count matches total observations) and
        that the value_head's weights survive concurrent rescaling intact
        (finite, not corrupted by an interleaved partial write)."""
        normalizer = SharedReturnNormalizer(decay=0.999)
        value_head = nn.Linear(8, 1)
        value_head.share_memory()

        n_workers = 4
        n_updates = 50
        barrier = mp.Barrier(n_workers)

        processes = [
            mp.Process(
                target=_concurrent_update_worker,
                args=(normalizer, value_head, n_updates, barrier),
            )
            for _ in range(n_workers)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        assert normalizer._t.value == n_workers * n_updates
        assert torch.isfinite(value_head.weight).all()
        assert torch.isfinite(value_head.bias).all()
        mean, std = normalizer.stats()
        assert mean == mean and std == std  # not NaN
        assert std > 0
