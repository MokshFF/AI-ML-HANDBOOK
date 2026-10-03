"""
Tests for Probability and Statistics module.
"""

import numpy as np
import pytest
from scipy import stats
from stats_engine import (
    normal_pdf,
    bayes_posterior,
    bayes_binary_update,
    sample_mean,
    sample_variance,
    sample_covariance,
    pearson_correlation,
    spearman_correlation,
    simulate_clt,
    normal_confidence_interval,
    bootstrap_confidence_interval,
    two_sample_welch_t_test,
    permutation_test_difference,
)


def test_normal_pdf():
    x = 0.0
    val = normal_pdf(x, mu=0.0, sigma=1.0)
    expected = 1.0 / np.sqrt(2.0 * np.pi)
    assert np.isclose(val, expected)


def test_bayes_rule():
    # Rare disease: Prevalence P(D) = 0.01
    # Sensitivity: P(+ | D) = 0.99
    # False Positive: P(+ | H) = 0.05
    post = bayes_binary_update(prior=0.01, p_pos_given_disease=0.99, p_pos_given_healthy=0.05)
    # Expected: (0.99 * 0.01) / (0.99 * 0.01 + 0.05 * 0.99) = 0.0099 / (0.0099 + 0.0495) = 0.16666...
    assert np.isclose(post, 1.0 / 6.0, atol=1e-3)


def test_sample_moments():
    np.random.seed(42)
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert np.isclose(sample_mean(x), np.mean(x))
    assert np.isclose(sample_variance(x, ddof=1), np.var(x, ddof=1))

    y = 2.0 * x + 1.0
    assert np.isclose(pearson_correlation(x, y), 1.0)
    assert np.isclose(spearman_correlation(x, y), 1.0)


def test_clt_simulation():
    # Exponential distribution (heavily skewed, non-Gaussian)
    sampler = lambda n: np.random.exponential(scale=2.0, size=n)
    sample_means = simulate_clt(sampler, sample_size=50, num_trials=500)

    # By CLT, sample means should be approximately normal with mean ~2.0
    assert np.isclose(np.mean(sample_means), 2.0, atol=0.1)
    # Standard error should be ~ 2.0 / sqrt(50) = 0.2828
    assert np.isclose(np.std(sample_means), 2.0 / np.sqrt(50), atol=0.05)


def test_confidence_intervals():
    np.random.seed(0)
    data = np.random.normal(loc=10.0, scale=2.0, size=100)

    mu, low, high = normal_confidence_interval(data, confidence=0.95)
    assert low < mu < high
    assert low < 10.0 < high

    mu_b, low_b, high_b = bootstrap_confidence_interval(data, confidence=0.95, seed=0)
    assert low_b < mu_b < high_b
    assert np.isclose(low, low_b, atol=0.5)


def test_hypothesis_testing():
    np.random.seed(42)
    # Two distinct groups
    group_a = np.random.normal(loc=5.0, scale=1.0, size=50)
    group_b = np.random.normal(loc=6.0, scale=1.0, size=50)

    t_stat, df, p_val = two_sample_welch_t_test(group_a, group_b)
    scipy_res = stats.ttest_ind(group_a, group_b, equal_var=False)

    assert np.isclose(t_stat, scipy_res.statistic)
    assert np.isclose(p_val, scipy_res.pvalue)
    assert p_val < 0.05  # Should reject null

    # Permutation test
    diff, perm_p = permutation_test_difference(group_a, group_b, num_permutations=1000, seed=42)
    assert perm_p < 0.05
