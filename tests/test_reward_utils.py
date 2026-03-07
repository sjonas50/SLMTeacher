"""Tests for reward utility functions."""
import numpy as np
import pytest
from src.rewards.reward_utils import (
    normalize_rewards,
    compute_advantage_estimates,
)


class TestNormalizeRewards:
    def test_standardize(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_rewards(rewards, method='standardize')
        assert np.abs(np.mean(normalized)) < 1e-6
        assert np.abs(np.std(normalized) - 1.0) < 0.1

    def test_minmax(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_rewards(rewards, method='minmax')
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_softmax_sums_to_one(self):
        rewards = np.array([1.0, 2.0, 3.0])
        normalized = normalize_rewards(rewards, method='softmax')
        assert np.sum(normalized) == pytest.approx(1.0, abs=1e-6)

    def test_softmax_numerical_stability(self):
        # Large values that would overflow naive exp()
        rewards = np.array([1000.0, 1001.0, 1002.0])
        normalized = normalize_rewards(rewards, method='softmax')
        assert np.all(np.isfinite(normalized))
        assert np.sum(normalized) == pytest.approx(1.0, abs=1e-6)

    def test_softmax_negative_values(self):
        rewards = np.array([-1000.0, -999.0, -998.0])
        normalized = normalize_rewards(rewards, method='softmax')
        assert np.all(np.isfinite(normalized))
        assert np.sum(normalized) == pytest.approx(1.0, abs=1e-6)

    def test_robust_normalization(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier
        normalized = normalize_rewards(rewards, method='robust')
        # Robust normalization should handle outliers better
        assert np.all(np.isfinite(normalized))

    def test_unknown_method_raises(self):
        rewards = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_rewards(rewards, method='bogus')

    def test_single_value_standardize(self):
        rewards = np.array([5.0])
        normalized = normalize_rewards(rewards, method='standardize')
        assert np.all(np.isfinite(normalized))

    def test_constant_values_standardize(self):
        rewards = np.array([3.0, 3.0, 3.0])
        normalized = normalize_rewards(rewards, method='standardize')
        assert np.all(np.isfinite(normalized))


class TestComputeAdvantageEstimates:
    def test_simple_advantage(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        advantages = compute_advantage_estimates(rewards)
        # Simple advantage: rewards - mean
        assert np.mean(advantages) == pytest.approx(0.0, abs=1e-6)

    def test_with_values(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        advantages = compute_advantage_estimates(rewards, values)
        assert len(advantages) == len(rewards)
        assert np.all(np.isfinite(advantages))
